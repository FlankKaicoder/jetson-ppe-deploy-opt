#include "cuda_preprocess.hpp"
#include "gpu_postprocess.hpp"

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceRaw {
public:
    DeviceRaw() {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes()),
            "cudaMalloc synthetic raw tensor");
    }
    ~DeviceRaw() { cudaFree(pointer_); }
    DeviceRaw(const DeviceRaw&) = delete;
    DeviceRaw& operator=(const DeviceRaw&) = delete;
    float* data() const { return pointer_; }
    static constexpr std::size_t elements() {
        return static_cast<std::size_t>(ppe::kYoloOutputChannels) *
               ppe::kYoloCandidateCount;
    }
    static constexpr std::size_t bytes() { return elements() * sizeof(float); }

private:
    float* pointer_{nullptr};
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate test"); }
    ~Stream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

struct TestCase {
    std::string name;
    std::vector<float> raw;
};

void set_candidate(
    std::vector<float>& raw,
    int index,
    float center_x,
    float center_y,
    float width,
    float height,
    int class_id,
    float confidence) {
    raw[index] = center_x;
    raw[ppe::kYoloCandidateCount + index] = center_y;
    raw[2 * ppe::kYoloCandidateCount + index] = width;
    raw[3 * ppe::kYoloCandidateCount + index] = height;
    raw[(4 + class_id) * ppe::kYoloCandidateCount + index] = confidence;
}

std::vector<ppe::GpuCandidate> cpu_reference(
    const std::vector<float>& raw,
    const ppe::LetterboxGeometry& geometry,
    float threshold) {
    std::vector<ppe::GpuCandidate> result;
    for (int index = 0; index < ppe::kYoloCandidateCount; ++index) {
        int class_id = 0;
        float confidence = raw[4 * ppe::kYoloCandidateCount + index];
        for (int category = 1; category < ppe::kYoloClassCount; ++category) {
            const float value =
                raw[(4 + category) * ppe::kYoloCandidateCount + index];
            if (value > confidence) {
                confidence = value;
                class_id = category;
            }
        }
        if (!std::isfinite(confidence) || confidence < threshold) {
            continue;
        }
        const float center_x = raw[index];
        const float center_y = raw[ppe::kYoloCandidateCount + index];
        const float width = raw[2 * ppe::kYoloCandidateCount + index];
        const float height = raw[3 * ppe::kYoloCandidateCount + index];
        if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
            !std::isfinite(width) || !std::isfinite(height) ||
            width <= 0.0F || height <= 0.0F) {
            continue;
        }
        ppe::GpuCandidate candidate{};
        candidate.candidate_index = index;
        candidate.class_id = class_id;
        candidate.confidence = confidence;
        candidate.x1 = std::clamp(
            (center_x - width * 0.5F - geometry.padding_left) / geometry.ratio,
            0.0F, static_cast<float>(geometry.source_width));
        candidate.y1 = std::clamp(
            (center_y - height * 0.5F - geometry.padding_top) / geometry.ratio,
            0.0F, static_cast<float>(geometry.source_height));
        candidate.x2 = std::clamp(
            (center_x + width * 0.5F - geometry.padding_left) / geometry.ratio,
            0.0F, static_cast<float>(geometry.source_width));
        candidate.y2 = std::clamp(
            (center_y + height * 0.5F - geometry.padding_top) / geometry.ratio,
            0.0F, static_cast<float>(geometry.source_height));
        if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1) {
            result.push_back(candidate);
        }
    }
    return result;
}

float max_error(
    const ppe::GpuCandidate& left,
    const ppe::GpuCandidate& right,
    bool boxes) {
    if (!boxes) {
        return std::abs(left.confidence - right.confidence);
    }
    return std::max({
        std::abs(left.x1 - right.x1), std::abs(left.y1 - right.y1),
        std::abs(left.x2 - right.x2), std::abs(left.y2 - right.y2)});
}

struct ModeResult {
    int count{};
    float confidence_error{};
    float box_error{};
    bool stable_order{};
};

ModeResult run_mode(
    const TestCase& test,
    const ppe::LetterboxGeometry& geometry,
    ppe::GpuCompactionMode mode,
    DeviceRaw& device_raw,
    Stream& stream) {
    const auto reference = cpu_reference(test.raw, geometry, 0.25F);
    ppe::GpuPostprocessor postprocessor;
    check_cuda(
        cudaMemcpyAsync(
            device_raw.data(), test.raw.data(), DeviceRaw::bytes(),
            cudaMemcpyHostToDevice, stream.get()),
        "copy synthetic raw tensor");
    postprocessor.launch(device_raw.data(), geometry, 0.25F, mode, stream.get());
    int count = -1;
    check_cuda(
        cudaMemcpyAsync(
            &count, postprocessor.device_count(), sizeof(count),
            cudaMemcpyDeviceToHost, stream.get()),
        "copy synthetic candidate count");
    check_cuda(cudaStreamSynchronize(stream.get()), "sync candidate count");
    if (count < 0 || count > postprocessor.capacity()) {
        throw std::runtime_error(test.name + ": invalid candidate count");
    }
    std::vector<ppe::GpuCandidate> actual(static_cast<std::size_t>(count));
    if (count > 0) {
        check_cuda(
            cudaMemcpyAsync(
                actual.data(), postprocessor.device_candidates(),
                actual.size() * sizeof(ppe::GpuCandidate),
                cudaMemcpyDeviceToHost, stream.get()),
            "copy synthetic candidates");
        check_cuda(cudaStreamSynchronize(stream.get()), "sync candidates");
    }
    const bool stable_order = std::is_sorted(
        actual.begin(), actual.end(),
        [](const auto& left, const auto& right) {
            return left.candidate_index < right.candidate_index;
        });
    std::sort(
        actual.begin(), actual.end(),
        [](const auto& left, const auto& right) {
            return left.candidate_index < right.candidate_index;
        });
    if (actual.size() != reference.size()) {
        throw std::runtime_error(test.name + ": CPU/GPU count mismatch");
    }
    float confidence_error = 0.0F;
    float box_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].candidate_index != reference[index].candidate_index ||
            actual[index].class_id != reference[index].class_id) {
            throw std::runtime_error(test.name + ": index/class mismatch");
        }
        confidence_error = std::max(
            confidence_error, max_error(actual[index], reference[index], false));
        box_error = std::max(
            box_error, max_error(actual[index], reference[index], true));
    }
    if (confidence_error > 1.0e-6F || box_error > 1.0e-3F) {
        throw std::runtime_error(test.name + ": numeric tolerance exceeded");
    }
    if (mode == ppe::GpuCompactionMode::kCubStable && !stable_order) {
        throw std::runtime_error(test.name + ": CUB output is not stable");
    }
    return {count, confidence_error, box_error, stable_order};
}

std::vector<TestCase> make_cases() {
    const std::size_t elements = DeviceRaw::elements();
    std::vector<TestCase> cases;
    cases.push_back({"zero_candidates", std::vector<float>(elements, 0.0F)});

    TestCase single{"single_candidate", std::vector<float>(elements, 0.0F)};
    set_candidate(single.raw, 17, 320.0F, 320.0F, 40.0F, 60.0F, 2, 0.9F);
    cases.push_back(std::move(single));

    TestCase boundaries{"threshold_tie_invalid", std::vector<float>(elements, 0.0F)};
    set_candidate(boundaries.raw, 1, 300.0F, 300.0F, 20.0F, 30.0F, 0, 0.25F);
    boundaries.raw[(4 + 1) * ppe::kYoloCandidateCount + 1] = 0.25F;
    set_candidate(boundaries.raw, 2, 320.0F, 320.0F, -1.0F, 30.0F, 1, 0.8F);
    set_candidate(boundaries.raw, 3, 320.0F, 320.0F, 20.0F, 30.0F, 1, 0.8F);
    boundaries.raw[(4 + 1) * ppe::kYoloCandidateCount + 3] =
        std::numeric_limits<float>::quiet_NaN();
    set_candidate(boundaries.raw, 4, 0.0F, 0.0F, 40.0F, 40.0F, 1, 0.7F);
    set_candidate(boundaries.raw, 5, 640.0F, 640.0F, 40.0F, 40.0F, 1, 0.7F);
    cases.push_back(std::move(boundaries));

    TestCase all{"all_candidates", std::vector<float>(elements, 0.0F)};
    for (int index = 0; index < ppe::kYoloCandidateCount; ++index) {
        set_candidate(
            all.raw, index, 320.0F, 320.0F, 20.0F, 30.0F,
            index % ppe::kYoloClassCount,
            0.5F + static_cast<float>(index % 100) * 0.001F);
    }
    cases.push_back(std::move(all));
    return cases;
}

int profile_fixture(
    const std::filesystem::path& fixture_path,
    ppe::GpuCompactionMode mode,
    int iterations) {
    if (iterations <= 0 || !std::filesystem::is_regular_file(fixture_path) ||
        std::filesystem::file_size(fixture_path) != DeviceRaw::bytes()) {
        throw std::runtime_error("invalid profiling fixture or iteration count");
    }
    std::vector<float> raw(DeviceRaw::elements());
    std::ifstream fixture(fixture_path, std::ios::binary);
    fixture.read(
        reinterpret_cast<char*>(raw.data()),
        static_cast<std::streamsize>(DeviceRaw::bytes()));
    if (!fixture) {
        throw std::runtime_error("failed to read profiling fixture");
    }
    const auto geometry = ppe::make_letterbox_geometry(1920, 1080, 640);
    DeviceRaw device_raw;
    Stream stream;
    ppe::GpuPostprocessor postprocessor;
    check_cuda(
        cudaMemcpyAsync(
            device_raw.data(), raw.data(), DeviceRaw::bytes(),
            cudaMemcpyHostToDevice, stream.get()),
        "copy profiling fixture");
    for (int index = 0; index < 5; ++index) {
        postprocessor.launch(
            device_raw.data(), geometry, 0.25F, mode, stream.get());
    }
    check_cuda(cudaStreamSynchronize(stream.get()), "sync profiling warmup");
    check_cuda(cudaProfilerStart(), "cudaProfilerStart");
    for (int index = 0; index < iterations; ++index) {
        postprocessor.launch(
            device_raw.data(), geometry, 0.25F, mode, stream.get());
    }
    check_cuda(cudaStreamSynchronize(stream.get()), "sync profiling kernels");
    check_cuda(cudaProfilerStop(), "cudaProfilerStop");
    int count = -1;
    check_cuda(
        cudaMemcpy(
            &count, postprocessor.device_count(), sizeof(count),
            cudaMemcpyDeviceToHost),
        "copy profiling result count");
    if (count < 0 || count > ppe::kYoloCandidateCount) {
        throw std::runtime_error("invalid profiling result count");
    }
    std::cout << "result=PASS mode="
              << (mode == ppe::GpuCompactionMode::kAtomic ? "atomic" : "cub")
              << " iterations=" << iterations << " count=" << count << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string(argv[1]) == "--profile-fixture" &&
            std::string(argv[3]) == "--mode" &&
            std::string(argv[5]) == "--iterations") {
            const std::string mode_name(argv[4]);
            if (mode_name != "atomic" && mode_name != "cub") {
                throw std::runtime_error("profile mode must be atomic or cub");
            }
            return profile_fixture(
                argv[2],
                mode_name == "atomic" ? ppe::GpuCompactionMode::kAtomic
                                       : ppe::GpuCompactionMode::kCubStable,
                std::stoi(argv[6]));
        }
        if (argc != 3 || std::string(argv[1]) != "--output-dir") {
            throw std::runtime_error(
                "usage: exp15_gpu_postprocess_test --output-dir DIR | "
                "--profile-fixture FILE --mode atomic|cub --iterations N");
        }
        const std::filesystem::path output_dir(argv[2]);
        std::filesystem::create_directories(output_dir);
        const auto geometry = ppe::make_letterbox_geometry(1920, 1080, 640);
        DeviceRaw device_raw;
        Stream stream;
        const auto cases = make_cases();
        std::ofstream csv(output_dir / "synthetic_results.csv");
        csv << "case,mode,count,confidence_max_abs_error,box_max_abs_error,stable_order\n";
        csv << std::scientific << std::setprecision(9);
        for (const auto& test : cases) {
            for (const auto mode : {
                     ppe::GpuCompactionMode::kAtomic,
                     ppe::GpuCompactionMode::kCubStable}) {
                const auto result = run_mode(
                    test, geometry, mode, device_raw, stream);
                csv << test.name << ','
                    << (mode == ppe::GpuCompactionMode::kAtomic ? "atomic" : "cub")
                    << ',' << result.count << ',' << result.confidence_error << ','
                    << result.box_error << ',' << (result.stable_order ? 1 : 0) << '\n';
            }
        }
        ppe::GpuPostprocessor guard_workspace(8);
        const auto& all = cases.back();
        check_cuda(
            cudaMemcpyAsync(
                device_raw.data(), all.raw.data(), DeviceRaw::bytes(),
                cudaMemcpyHostToDevice, stream.get()),
            "copy overflow guard tensor");
        guard_workspace.launch(
            device_raw.data(), geometry, 0.25F,
            ppe::GpuCompactionMode::kAtomic, stream.get());
        int overflow_count = 0;
        check_cuda(
            cudaMemcpyAsync(
                &overflow_count, guard_workspace.device_count(), sizeof(int),
                cudaMemcpyDeviceToHost, stream.get()),
            "copy overflow guard count");
        check_cuda(cudaStreamSynchronize(stream.get()), "sync overflow guard");
        if (overflow_count != ppe::kYoloCandidateCount) {
            throw std::runtime_error("capacity guard did not preserve full count");
        }
        std::ofstream summary(output_dir / "summary.json");
        summary << "{\n"
                << "  \"result\": \"PASS\",\n"
                << "  \"cases\": " << cases.size() << ",\n"
                << "  \"modes\": 2,\n"
                << "  \"candidate_size_bytes\": " << sizeof(ppe::GpuCandidate) << ",\n"
                << "  \"capacity_guard_count\": " << overflow_count << "\n"
                << "}\n";
        std::cout << "result=PASS cases=" << cases.size()
                  << " modes=2 overflow_count=" << overflow_count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
