#pragma once

#include "cuda_preprocess.hpp"
#include "gpu_postprocess.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ppe {

struct PluginFrameResult {
    std::vector<GpuCandidate> candidates;
    int plugin_count{};
    std::size_t d2h_bytes{};
    double host_total_ms{};
    double inference_cuda_ms{};
    double count_copy_cuda_ms{};
    double count_sync_host_ms{};
    double candidate_copy_cuda_ms{};
    double candidate_sync_host_ms{};
    double cpu_inverse_letterbox_ms{};
};

class TrtPluginRuntime {
public:
    TrtPluginRuntime(
        const std::string& engine_path,
        const std::string& plugin_path,
        LetterboxGeometry geometry);
    ~TrtPluginRuntime();

    TrtPluginRuntime(const TrtPluginRuntime&) = delete;
    TrtPluginRuntime& operator=(const TrtPluginRuntime&) = delete;
    TrtPluginRuntime(TrtPluginRuntime&&) noexcept;
    TrtPluginRuntime& operator=(TrtPluginRuntime&&) noexcept;

    PluginFrameResult process(
        const float* device_input,
        cudaStream_t stream,
        std::vector<GpuCandidate>* network_candidates = nullptr);
    void set_geometry(LetterboxGeometry geometry);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ppe
