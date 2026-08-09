#pragma once

#include "cuda_preprocess.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>

namespace ppe {

constexpr int kYoloCandidateCount = 8400;
constexpr int kYoloClassCount = 3;
constexpr int kYoloOutputChannels = 7;

struct GpuCandidate {
    int candidate_index{};
    int class_id{};
    float confidence{};
    float x1{};
    float y1{};
    float x2{};
    float y2{};
};

enum class GpuCompactionMode {
    kAtomic,
    kCubStable,
    kFixed,
};

class GpuPostprocessor {
public:
    explicit GpuPostprocessor(int capacity = kYoloCandidateCount);
    ~GpuPostprocessor();

    GpuPostprocessor(const GpuPostprocessor&) = delete;
    GpuPostprocessor& operator=(const GpuPostprocessor&) = delete;
    GpuPostprocessor(GpuPostprocessor&&) noexcept;
    GpuPostprocessor& operator=(GpuPostprocessor&&) noexcept;

    void launch(
        const float* device_raw_output,
        const LetterboxGeometry& geometry,
        float confidence_threshold,
        GpuCompactionMode mode,
        cudaStream_t stream);

    const int* device_count() const;
    const GpuCandidate* device_candidates() const;
    const GpuCandidate* device_fixed_candidates() const;
    int capacity() const;
    std::size_t cub_temporary_storage_bytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ppe
