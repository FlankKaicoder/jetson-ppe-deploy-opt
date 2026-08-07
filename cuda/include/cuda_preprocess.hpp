#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace ppe {

struct LetterboxGeometry {
    int source_width{};
    int source_height{};
    int target_size{};
    int resized_width{};
    int resized_height{};
    int padding_left{};
    int padding_right{};
    int padding_top{};
    int padding_bottom{};
    float ratio{};
};

struct PreprocessTimingStats {
    double mean_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};
    double min_ms{};
    double max_ms{};
};

struct CudaPreprocessResult {
    std::vector<float> output;
    PreprocessTimingStats kernel_only;
    PreprocessTimingStats total_with_transfers;
};

struct DevicePreprocessResult {
    const float* device_output{};
    double host_total_ms{};
    double cuda_total_ms{};
};

class CudaPreprocessor {
public:
    explicit CudaPreprocessor(const LetterboxGeometry& geometry);
    ~CudaPreprocessor();

    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;
    CudaPreprocessor(CudaPreprocessor&&) noexcept;
    CudaPreprocessor& operator=(CudaPreprocessor&&) noexcept;

    const LetterboxGeometry& geometry() const;
    cudaStream_t stream() const;
    DevicePreprocessResult process(const std::uint8_t* host_bgr);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

LetterboxGeometry make_letterbox_geometry(
    int source_width,
    int source_height,
    int target_size);

CudaPreprocessResult run_cuda_preprocess(
    const std::uint8_t* host_bgr,
    const LetterboxGeometry& geometry,
    int warmup_iterations,
    int timed_iterations);

}  // namespace ppe
