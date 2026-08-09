#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ppe::exp16 {

inline constexpr char kPluginName[] = "PpeYoloDecodeCompact";
inline constexpr char kPluginVersion[] = "1";
inline constexpr char kPluginNamespace[] = "com.flankkaicoder.ppe";
inline constexpr int32_t kCandidateCount = 8400;
inline constexpr int32_t kClassCount = 3;
inline constexpr int32_t kInputChannels = 7;
inline constexpr int32_t kNetworkSize = 640;

std::size_t plugin_workspace_size() noexcept;

int launch_decode_compact(
    float const* raw,
    float* boxes_scores,
    int32_t* classes,
    int32_t* indices,
    int32_t* count,
    void* workspace,
    std::size_t workspace_bytes,
    float confidence_threshold,
    cudaStream_t stream) noexcept;

}  // namespace ppe::exp16

extern "C" __attribute__((visibility("default"))) bool ppeInitPlugin();
