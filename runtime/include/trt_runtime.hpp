#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ppe {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> dimensions;
    std::string data_type;
    std::string mode;
    std::size_t elements{};
    std::size_t bytes{};
};

struct TimingStats {
    double mean_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};
    double min_ms{};
    double max_ms{};
};

struct InferenceResult {
    std::vector<float> output;
    TimingStats host_total;
    TimingStats cuda_total;
};

class TrtRuntime {
public:
    explicit TrtRuntime(const std::string& engine_path);
    ~TrtRuntime();

    TrtRuntime(const TrtRuntime&) = delete;
    TrtRuntime& operator=(const TrtRuntime&) = delete;
    TrtRuntime(TrtRuntime&&) noexcept;
    TrtRuntime& operator=(TrtRuntime&&) noexcept;

    const TensorInfo& input_info() const;
    const TensorInfo& output_info() const;
    InferenceResult infer(
        const std::vector<float>& input,
        int warmup_iterations,
        int timed_iterations);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ppe
