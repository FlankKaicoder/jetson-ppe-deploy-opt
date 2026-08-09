#include "yolo_decode_plugin.hpp"

#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ppe::exp16 {
namespace {

struct NetworkCandidate {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t class_id;
};

constexpr std::size_t kAlignment = 256;

constexpr std::size_t align_up(std::size_t value) {
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

std::size_t cub_storage_bytes() noexcept {
    std::size_t bytes = 0;
    auto input = cub::CountingInputIterator<int32_t>(0);
    static_cast<void>(cub::DeviceSelect::Flagged(
        nullptr, bytes, input, static_cast<int32_t const*>(nullptr),
        static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
        kCandidateCount));
    return bytes;
}

__device__ float clamp_network(float value) {
    return fminf(fmaxf(value, 0.0F), static_cast<float>(kNetworkSize));
}

__global__ void decode_flag_kernel(
    float const* raw,
    float confidence_threshold,
    NetworkCandidate* candidates,
    int32_t* flags) {
    int32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kCandidateCount) {
        return;
    }

    int32_t class_id = 0;
    float confidence = raw[4 * kCandidateCount + index];
    for (int32_t category = 1; category < kClassCount; ++category) {
        float const value = raw[(4 + category) * kCandidateCount + index];
        if (value > confidence) {
            confidence = value;
            class_id = category;
        }
    }

    float const cx = raw[index];
    float const cy = raw[kCandidateCount + index];
    float const width = raw[2 * kCandidateCount + index];
    float const height = raw[3 * kCandidateCount + index];
    bool valid = isfinite(confidence) && confidence >= confidence_threshold &&
        isfinite(cx) && isfinite(cy) && isfinite(width) && isfinite(height) &&
        width > 0.0F && height > 0.0F;

    NetworkCandidate candidate{};
    if (valid) {
        candidate.x1 = clamp_network(cx - 0.5F * width);
        candidate.y1 = clamp_network(cy - 0.5F * height);
        candidate.x2 = clamp_network(cx + 0.5F * width);
        candidate.y2 = clamp_network(cy + 0.5F * height);
        candidate.confidence = confidence;
        candidate.class_id = class_id;
        valid = candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1;
    }
    candidates[index] = candidate;
    flags[index] = valid ? 1 : 0;
}

__global__ void gather_kernel(
    NetworkCandidate const* candidates,
    int32_t const* indices,
    int32_t const* count,
    float* boxes_scores,
    int32_t* classes) {
    int32_t const position = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const valid_count = *count;
    if (position >= valid_count || position >= kCandidateCount) {
        return;
    }
    int32_t const source_index = indices[position];
    NetworkCandidate const candidate = candidates[source_index];
    float* output = boxes_scores + static_cast<std::size_t>(position) * 5;
    output[0] = candidate.x1;
    output[1] = candidate.y1;
    output[2] = candidate.x2;
    output[3] = candidate.y2;
    output[4] = candidate.confidence;
    classes[position] = candidate.class_id;
}

}  // namespace

std::size_t plugin_workspace_size() noexcept {
    return align_up(sizeof(NetworkCandidate) * kCandidateCount) +
        align_up(sizeof(int32_t) * kCandidateCount) +
        align_up(cub_storage_bytes());
}

int launch_decode_compact(
    float const* raw,
    float* boxes_scores,
    int32_t* classes,
    int32_t* indices,
    int32_t* count,
    void* workspace,
    std::size_t workspace_bytes,
    float confidence_threshold,
    cudaStream_t stream) noexcept {
    if (raw == nullptr || boxes_scores == nullptr || classes == nullptr ||
        indices == nullptr || count == nullptr || workspace == nullptr ||
        stream == nullptr || workspace_bytes < plugin_workspace_size()) {
        return -1;
    }

    auto* base = static_cast<unsigned char*>(workspace);
    auto* candidates = reinterpret_cast<NetworkCandidate*>(base);
    base += align_up(sizeof(NetworkCandidate) * kCandidateCount);
    auto* flags = reinterpret_cast<int32_t*>(base);
    base += align_up(sizeof(int32_t) * kCandidateCount);
    void* cub_storage = base;
    std::size_t cub_bytes = cub_storage_bytes();

    constexpr int32_t threads = 256;
    constexpr int32_t blocks = (kCandidateCount + threads - 1) / threads;
    decode_flag_kernel<<<blocks, threads, 0, stream>>>(
        raw, confidence_threshold, candidates, flags);
    if (cudaGetLastError() != cudaSuccess) {
        return -2;
    }

    auto input = cub::CountingInputIterator<int32_t>(0);
    cudaError_t const select_status = cub::DeviceSelect::Flagged(
        cub_storage, cub_bytes, input, flags, indices, count,
        kCandidateCount, stream);
    if (select_status != cudaSuccess) {
        return -3;
    }

    gather_kernel<<<blocks, threads, 0, stream>>>(
        candidates, indices, count, boxes_scores, classes);
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

}  // namespace ppe::exp16
