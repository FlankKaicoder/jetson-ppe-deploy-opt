#pragma once

#include <cstdint>
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

struct TimingStats {
    double mean_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};
    double min_ms{};
    double max_ms{};
};

struct CudaPreprocessResult {
    std::vector<float> output;
    TimingStats kernel_only;
    TimingStats total_with_transfers;
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
