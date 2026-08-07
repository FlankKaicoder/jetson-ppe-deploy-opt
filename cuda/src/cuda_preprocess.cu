#include "cuda_preprocess.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppe {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return pointer_; }

private:
    void* pointer_{nullptr};
};

class CudaStream {
public:
    CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

double percentile(const std::vector<double>& sorted, double quantile) {
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

TimingStats summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("no CUDA timing values");
    }
    std::sort(values.begin(), values.end());
    TimingStats result;
    result.mean_ms =
        std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    result.p50_ms = percentile(values, 0.50);
    result.p95_ms = percentile(values, 0.95);
    result.p99_ms = percentile(values, 0.99);
    result.min_ms = values.front();
    result.max_ms = values.back();
    return result;
}

__device__ float interpolate_channel(
    const std::uint8_t* image,
    int width,
    int height,
    float source_x,
    float source_y,
    int channel) {
    source_x = fminf(fmaxf(source_x, 0.0F), static_cast<float>(width - 1));
    source_y = fminf(fmaxf(source_y, 0.0F), static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float wx = source_x - static_cast<float>(x0);
    const float wy = source_y - static_cast<float>(y0);
    const float top =
        static_cast<float>(image[(y0 * width + x0) * 3 + channel]) * (1.0F - wx) +
        static_cast<float>(image[(y0 * width + x1) * 3 + channel]) * wx;
    const float bottom =
        static_cast<float>(image[(y1 * width + x0) * 3 + channel]) * (1.0F - wx) +
        static_cast<float>(image[(y1 * width + x1) * 3 + channel]) * wx;
    return top * (1.0F - wy) + bottom * wy;
}

__global__ void fused_preprocess_kernel(
    const std::uint8_t* input,
    float* output,
    int source_width,
    int source_height,
    int target_size,
    int resized_width,
    int resized_height,
    int padding_left,
    int padding_top) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= target_size || y >= target_size) {
        return;
    }
    const int output_index = y * target_size + x;
    const int plane = target_size * target_size;
    if (x < padding_left || x >= padding_left + resized_width ||
        y < padding_top || y >= padding_top + resized_height) {
        const float padding = 114.0F / 255.0F;
        output[output_index] = padding;
        output[plane + output_index] = padding;
        output[2 * plane + output_index] = padding;
        return;
    }

    const int resized_x = x - padding_left;
    const int resized_y = y - padding_top;
    const float source_x =
        (static_cast<float>(resized_x) + 0.5F) *
            static_cast<float>(source_width) / static_cast<float>(resized_width) -
        0.5F;
    const float source_y =
        (static_cast<float>(resized_y) + 0.5F) *
            static_cast<float>(source_height) / static_cast<float>(resized_height) -
        0.5F;
    const float blue = floorf(
        interpolate_channel(
            input, source_width, source_height, source_x, source_y, 0) +
        0.5F);
    const float green = floorf(
        interpolate_channel(
            input, source_width, source_height, source_x, source_y, 1) +
        0.5F);
    const float red = floorf(
        interpolate_channel(
            input, source_width, source_height, source_x, source_y, 2) +
        0.5F);
    output[output_index] = red / 255.0F;
    output[plane + output_index] = green / 255.0F;
    output[2 * plane + output_index] = blue / 255.0F;
}

void launch(
    const DeviceBuffer& input,
    const DeviceBuffer& output,
    const LetterboxGeometry& geometry,
    cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(
        (geometry.target_size + block.x - 1) / block.x,
        (geometry.target_size + block.y - 1) / block.y);
    fused_preprocess_kernel<<<grid, block, 0, stream>>>(
        static_cast<const std::uint8_t*>(input.data()),
        static_cast<float*>(output.data()),
        geometry.source_width,
        geometry.source_height,
        geometry.target_size,
        geometry.resized_width,
        geometry.resized_height,
        geometry.padding_left,
        geometry.padding_top);
    check_cuda(cudaGetLastError(), "fused_preprocess_kernel launch");
}

double elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
        "cudaEventElapsedTime");
    return static_cast<double>(milliseconds);
}

}  // namespace

LetterboxGeometry make_letterbox_geometry(
    int source_width,
    int source_height,
    int target_size) {
    if (source_width <= 0 || source_height <= 0 || target_size <= 0) {
        throw std::runtime_error("invalid image dimensions");
    }
    const double ratio = std::min(
        static_cast<double>(target_size) / static_cast<double>(source_height),
        static_cast<double>(target_size) / static_cast<double>(source_width));
    const int resized_width = static_cast<int>(
        std::round(static_cast<double>(source_width) * ratio));
    const int resized_height = static_cast<int>(
        std::round(static_cast<double>(source_height) * ratio));
    const double half_width =
        static_cast<double>(target_size - resized_width) / 2.0;
    const double half_height =
        static_cast<double>(target_size - resized_height) / 2.0;
    LetterboxGeometry geometry;
    geometry.source_width = source_width;
    geometry.source_height = source_height;
    geometry.target_size = target_size;
    geometry.resized_width = resized_width;
    geometry.resized_height = resized_height;
    geometry.padding_left = static_cast<int>(std::round(half_width - 0.1));
    geometry.padding_right = static_cast<int>(std::round(half_width + 0.1));
    geometry.padding_top = static_cast<int>(std::round(half_height - 0.1));
    geometry.padding_bottom = static_cast<int>(std::round(half_height + 0.1));
    geometry.ratio = static_cast<float>(ratio);
    if (geometry.padding_left + geometry.resized_width +
            geometry.padding_right != target_size ||
        geometry.padding_top + geometry.resized_height +
            geometry.padding_bottom != target_size) {
        throw std::runtime_error("letterbox geometry does not reach target size");
    }
    return geometry;
}

CudaPreprocessResult run_cuda_preprocess(
    const std::uint8_t* host_bgr,
    const LetterboxGeometry& geometry,
    int warmup_iterations,
    int timed_iterations) {
    if (host_bgr == nullptr || warmup_iterations < 0 || timed_iterations <= 0) {
        throw std::runtime_error("invalid CUDA preprocess arguments");
    }
    const std::size_t input_bytes =
        static_cast<std::size_t>(geometry.source_width) *
        static_cast<std::size_t>(geometry.source_height) * 3;
    const std::size_t output_elements =
        static_cast<std::size_t>(geometry.target_size) *
        static_cast<std::size_t>(geometry.target_size) * 3;
    const std::size_t output_bytes = output_elements * sizeof(float);
    DeviceBuffer device_input(input_bytes);
    DeviceBuffer device_output(output_bytes);
    CudaStream stream;
    CudaEvent start;
    CudaEvent end;
    std::vector<float> output(output_elements);

    check_cuda(
        cudaMemcpyAsync(
            device_input.data(), host_bgr, input_bytes,
            cudaMemcpyHostToDevice, stream.get()),
        "initial cudaMemcpyAsync H2D");
    for (int index = 0; index < warmup_iterations; ++index) {
        launch(device_input, device_output, geometry, stream.get());
    }
    check_cuda(cudaStreamSynchronize(stream.get()), "warmup synchronize");

    std::vector<double> kernel_values;
    kernel_values.reserve(static_cast<std::size_t>(timed_iterations));
    for (int index = 0; index < timed_iterations; ++index) {
        check_cuda(cudaEventRecord(start.get(), stream.get()), "kernel start event");
        launch(device_input, device_output, geometry, stream.get());
        check_cuda(cudaEventRecord(end.get(), stream.get()), "kernel end event");
        check_cuda(cudaEventSynchronize(end.get()), "kernel event synchronize");
        kernel_values.push_back(elapsed_ms(start, end));
    }

    std::vector<double> total_values;
    total_values.reserve(static_cast<std::size_t>(timed_iterations));
    for (int index = 0; index < timed_iterations; ++index) {
        check_cuda(cudaEventRecord(start.get(), stream.get()), "total start event");
        check_cuda(
            cudaMemcpyAsync(
                device_input.data(), host_bgr, input_bytes,
                cudaMemcpyHostToDevice, stream.get()),
            "timed cudaMemcpyAsync H2D");
        launch(device_input, device_output, geometry, stream.get());
        check_cuda(
            cudaMemcpyAsync(
                output.data(), device_output.data(), output_bytes,
                cudaMemcpyDeviceToHost, stream.get()),
            "timed cudaMemcpyAsync D2H");
        check_cuda(cudaEventRecord(end.get(), stream.get()), "total end event");
        check_cuda(cudaEventSynchronize(end.get()), "total event synchronize");
        total_values.push_back(elapsed_ms(start, end));
    }
    check_cuda(cudaGetLastError(), "CUDA preprocess final status");
    return {std::move(output), summarize(kernel_values), summarize(total_values)};
}

}  // namespace ppe
