#include "gpu_postprocess.hpp"

#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ppe {
namespace {

static_assert(sizeof(GpuCandidate) == 28, "GpuCandidate ABI must remain 28 bytes");

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ float clamp_value(float value, float lower, float upper) {
    return fminf(fmaxf(value, lower), upper);
}

__device__ bool decode_candidate(
    const float* raw,
    int index,
    LetterboxGeometry geometry,
    float confidence_threshold,
    GpuCandidate* candidate) {
    int class_id = 0;
    float confidence = raw[4 * kYoloCandidateCount + index];
    for (int category = 1; category < kYoloClassCount; ++category) {
        const float value = raw[(4 + category) * kYoloCandidateCount + index];
        if (value > confidence) {
            confidence = value;
            class_id = category;
        }
    }
    if (!isfinite(confidence) || confidence < confidence_threshold) {
        return false;
    }

    const float center_x = raw[index];
    const float center_y = raw[kYoloCandidateCount + index];
    const float width = raw[2 * kYoloCandidateCount + index];
    const float height = raw[3 * kYoloCandidateCount + index];
    if (!isfinite(center_x) || !isfinite(center_y) || !isfinite(width) ||
        !isfinite(height) || width <= 0.0F || height <= 0.0F) {
        return false;
    }

    GpuCandidate decoded{};
    decoded.candidate_index = index;
    decoded.class_id = class_id;
    decoded.confidence = confidence;
    decoded.x1 = clamp_value(
        (center_x - width * 0.5F - geometry.padding_left) / geometry.ratio,
        0.0F, static_cast<float>(geometry.source_width));
    decoded.y1 = clamp_value(
        (center_y - height * 0.5F - geometry.padding_top) / geometry.ratio,
        0.0F, static_cast<float>(geometry.source_height));
    decoded.x2 = clamp_value(
        (center_x + width * 0.5F - geometry.padding_left) / geometry.ratio,
        0.0F, static_cast<float>(geometry.source_width));
    decoded.y2 = clamp_value(
        (center_y + height * 0.5F - geometry.padding_top) / geometry.ratio,
        0.0F, static_cast<float>(geometry.source_height));
    if (decoded.x2 <= decoded.x1 || decoded.y2 <= decoded.y1) {
        return false;
    }
    *candidate = decoded;
    return true;
}

__global__ void decode_filter_atomic_kernel(
    const float* raw,
    LetterboxGeometry geometry,
    float confidence_threshold,
    GpuCandidate* compacted,
    int capacity,
    int* count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kYoloCandidateCount) {
        return;
    }
    GpuCandidate candidate{};
    if (!decode_candidate(
            raw, index, geometry, confidence_threshold, &candidate)) {
        return;
    }
    const int position = atomicAdd(count, 1);
    if (position < capacity) {
        compacted[position] = candidate;
    }
}

__global__ void decode_flag_kernel(
    const float* raw,
    LetterboxGeometry geometry,
    float confidence_threshold,
    GpuCandidate* candidates_by_index,
    int* flags) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kYoloCandidateCount) {
        return;
    }
    GpuCandidate candidate{};
    const bool valid = decode_candidate(
        raw, index, geometry, confidence_threshold, &candidate);
    flags[index] = valid ? 1 : 0;
    if (valid) {
        candidates_by_index[index] = candidate;
    }
}

}  // namespace

class GpuPostprocessor::Impl {
public:
    explicit Impl(int capacity) : capacity_(capacity) {
        if (capacity_ <= 0 || capacity_ > kYoloCandidateCount) {
            throw std::runtime_error("GPU postprocess capacity must be in [1, 8400]");
        }
        check_cuda(
            cudaMalloc(&device_compacted_,
                       static_cast<std::size_t>(capacity_) * sizeof(GpuCandidate)),
            "cudaMalloc compacted candidates");
        check_cuda(
            cudaMalloc(&device_candidates_by_index_,
                       kYoloCandidateCount * sizeof(GpuCandidate)),
            "cudaMalloc candidates by index");
        check_cuda(
            cudaMalloc(&device_flags_, kYoloCandidateCount * sizeof(int)),
            "cudaMalloc candidate flags");
        check_cuda(cudaMalloc(&device_count_, sizeof(int)), "cudaMalloc count");

        check_cuda(
            cub::DeviceSelect::Flagged(
                nullptr, cub_temporary_bytes_, device_candidates_by_index_,
                device_flags_, device_compacted_, device_count_,
                kYoloCandidateCount),
            "CUB temporary storage query");
        check_cuda(
            cudaMalloc(&device_cub_temporary_, cub_temporary_bytes_),
            "cudaMalloc CUB temporary storage");
    }

    ~Impl() {
        cudaFree(device_cub_temporary_);
        cudaFree(device_count_);
        cudaFree(device_flags_);
        cudaFree(device_candidates_by_index_);
        cudaFree(device_compacted_);
    }

    void launch(
        const float* raw,
        const LetterboxGeometry& geometry,
        float confidence_threshold,
        GpuCompactionMode mode,
        cudaStream_t stream) {
        if (raw == nullptr || stream == nullptr || !std::isfinite(confidence_threshold) ||
            confidence_threshold < 0.0F || confidence_threshold > 1.0F) {
            throw std::runtime_error("invalid GPU postprocess launch argument");
        }
        check_cuda(
            cudaMemsetAsync(device_count_, 0, sizeof(int), stream),
            "cudaMemsetAsync candidate count");
        constexpr int threads = 256;
        constexpr int blocks =
            (kYoloCandidateCount + threads - 1) / threads;
        if (mode == GpuCompactionMode::kAtomic) {
            decode_filter_atomic_kernel<<<blocks, threads, 0, stream>>>(
                raw, geometry, confidence_threshold, device_compacted_,
                capacity_, device_count_);
            check_cuda(cudaGetLastError(), "launch atomic decode/filter kernel");
            return;
        }

        decode_flag_kernel<<<blocks, threads, 0, stream>>>(
            raw, geometry, confidence_threshold, device_candidates_by_index_,
            device_flags_);
        check_cuda(cudaGetLastError(), "launch stable decode/flag kernel");
        check_cuda(
            cub::DeviceSelect::Flagged(
                device_cub_temporary_, cub_temporary_bytes_,
                device_candidates_by_index_, device_flags_, device_compacted_,
                device_count_, kYoloCandidateCount, stream),
            "CUB stable candidate compaction");
    }

    const int* count() const { return device_count_; }
    const GpuCandidate* candidates() const { return device_compacted_; }
    int capacity() const { return capacity_; }
    std::size_t cub_bytes() const { return cub_temporary_bytes_; }

private:
    int capacity_{};
    GpuCandidate* device_compacted_{nullptr};
    GpuCandidate* device_candidates_by_index_{nullptr};
    int* device_flags_{nullptr};
    int* device_count_{nullptr};
    void* device_cub_temporary_{nullptr};
    std::size_t cub_temporary_bytes_{};
};

GpuPostprocessor::GpuPostprocessor(int capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}
GpuPostprocessor::~GpuPostprocessor() = default;
GpuPostprocessor::GpuPostprocessor(GpuPostprocessor&&) noexcept = default;
GpuPostprocessor& GpuPostprocessor::operator=(GpuPostprocessor&&) noexcept = default;

void GpuPostprocessor::launch(
    const float* device_raw_output,
    const LetterboxGeometry& geometry,
    float confidence_threshold,
    GpuCompactionMode mode,
    cudaStream_t stream) {
    impl_->launch(
        device_raw_output, geometry, confidence_threshold, mode, stream);
}

const int* GpuPostprocessor::device_count() const { return impl_->count(); }
const GpuCandidate* GpuPostprocessor::device_candidates() const {
    return impl_->candidates();
}
int GpuPostprocessor::capacity() const { return impl_->capacity(); }
std::size_t GpuPostprocessor::cub_temporary_storage_bytes() const {
    return impl_->cub_bytes();
}

}  // namespace ppe
