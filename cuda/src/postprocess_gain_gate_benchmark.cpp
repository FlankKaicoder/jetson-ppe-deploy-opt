#include "cuda_preprocess.hpp"
#include "gpu_postprocess.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

class DeviceRaw {
public:
    DeviceRaw() {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes()),
                   "cudaMalloc fixture raw");
    }
    ~DeviceRaw() { cudaFree(pointer_); }
    float* data() const { return pointer_; }
    static constexpr std::size_t elements() {
        return static_cast<std::size_t>(ppe::kYoloOutputChannels) *
               ppe::kYoloCandidateCount;
    }
    static constexpr std::size_t bytes() { return elements() * sizeof(float); }

private:
    float* pointer_{nullptr};
};

class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) {
        check_cuda(cudaHostAlloc(&pointer_, bytes, cudaHostAllocDefault),
                   "cudaHostAlloc gain gate");
    }
    ~PinnedBuffer() { cudaFreeHost(pointer_); }
    void* data() const { return pointer_; }

private:
    void* pointer_{nullptr};
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate gate"); }
    ~Stream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate gate"); }
    ~Event() { cudaEventDestroy(event_); }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

float iou(const ppe::GpuCandidate& left, const ppe::GpuCandidate& right) {
    const float x1 = std::max(left.x1, right.x1);
    const float y1 = std::max(left.y1, right.y1);
    const float x2 = std::min(left.x2, right.x2);
    const float y2 = std::min(left.y2, right.y2);
    const float intersection = std::max(0.0F, x2 - x1) *
                               std::max(0.0F, y2 - y1);
    const float left_area = (left.x2 - left.x1) * (left.y2 - left.y1);
    const float right_area = (right.x2 - right.x1) * (right.y2 - right.y1);
    const float union_area = left_area + right_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<ppe::GpuCandidate> cpu_decode(
    const float* raw,
    const ppe::LetterboxGeometry& geometry,
    float threshold) {
    std::vector<ppe::GpuCandidate> result;
    result.reserve(256);
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

std::vector<ppe::GpuCandidate> nms(
    std::vector<ppe::GpuCandidate> candidates,
    float threshold) {
    std::sort(
        candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) {
            if (left.confidence != right.confidence) {
                return left.confidence > right.confidence;
            }
            return left.candidate_index < right.candidate_index;
        });
    std::vector<ppe::GpuCandidate> kept;
    kept.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& accepted : kept) {
            if (candidate.class_id == accepted.class_id &&
                iou(candidate, accepted) > threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

void validate_candidates(
    const std::vector<ppe::GpuCandidate>& actual,
    const std::vector<ppe::GpuCandidate>& reference) {
    if (actual.size() != reference.size()) {
        throw std::runtime_error("fixture candidate count mismatch");
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto& left = actual[index];
        const auto& right = reference[index];
        if (left.candidate_index != right.candidate_index ||
            left.class_id != right.class_id ||
            std::abs(left.confidence - right.confidence) > 1.0e-6F ||
            std::max({std::abs(left.x1 - right.x1),
                      std::abs(left.y1 - right.y1),
                      std::abs(left.x2 - right.x2),
                      std::abs(left.y2 - right.y2)}) > 1.0e-3F) {
            throw std::runtime_error("fixture candidate mismatch");
        }
    }
}

struct Sample {
    std::string path;
    int iteration{};
    double gpu_postprocess_cuda_ms{};
    double count_sync_host_ms{};
    double payload_sync_host_ms{};
    double cpu_decode_filter_ms{};
    double cpu_candidate_scan_ms{};
    double cpu_nms_ms{};
    double total_ms{};
    std::size_t d2h_bytes{};
    int candidate_count{};
    int detection_count{};
};

class Benchmark {
public:
    Benchmark(const std::vector<float>& fixture,
              const ppe::LetterboxGeometry& geometry)
        : geometry_(geometry),
          host_raw_(DeviceRaw::bytes()),
          host_candidates_(ppe::kYoloCandidateCount * sizeof(ppe::GpuCandidate)),
          host_count_(sizeof(int)),
          reference_(cpu_decode(fixture.data(), geometry_, 0.25F)),
          reference_nms_(nms(reference_, 0.70F)) {
        check_cuda(cudaMemcpyAsync(
            device_raw_.data(), fixture.data(), DeviceRaw::bytes(),
            cudaMemcpyHostToDevice, stream_.get()), "copy fixture to device");
        check_cuda(cudaStreamSynchronize(stream_.get()), "sync fixture H2D");
    }

    Sample run(const std::string& path, int iteration, bool validate) {
        const auto total_start = Clock::now();
        std::vector<ppe::GpuCandidate> candidates;
        double gpu_ms = 0.0;
        double count_sync_ms = 0.0;
        double payload_sync_ms = 0.0;
        double decode_ms = 0.0;
        double scan_ms = 0.0;
        std::size_t d2h_bytes = 0;

        if (path == "P0") {
            const auto copy_start = Clock::now();
            check_cuda(cudaMemcpyAsync(
                host_raw_.data(), device_raw_.data(), DeviceRaw::bytes(),
                cudaMemcpyDeviceToHost, stream_.get()), "P0 raw D2H");
            check_cuda(cudaStreamSynchronize(stream_.get()), "P0 raw sync");
            payload_sync_ms = elapsed_ms(copy_start, Clock::now());
            const auto decode_start = Clock::now();
            candidates = cpu_decode(
                static_cast<const float*>(host_raw_.data()), geometry_, 0.25F);
            decode_ms = elapsed_ms(decode_start, Clock::now());
            d2h_bytes = DeviceRaw::bytes();
        } else {
            const auto mode = path == "P1" ? ppe::GpuCompactionMode::kFixed
                                            : ppe::GpuCompactionMode::kCubStable;
            check_cuda(cudaEventRecord(gpu_start_.get(), stream_.get()),
                       "record GPU postprocess start");
            postprocessor_.launch(
                device_raw_.data(), geometry_, 0.25F, mode, stream_.get());
            check_cuda(cudaEventRecord(gpu_end_.get(), stream_.get()),
                       "record GPU postprocess end");
            int count = ppe::kYoloCandidateCount;
            if (path == "P2") {
                const auto count_start = Clock::now();
                check_cuda(cudaMemcpyAsync(
                    host_count_.data(), postprocessor_.device_count(), sizeof(int),
                    cudaMemcpyDeviceToHost, stream_.get()), "P2 count D2H");
                check_cuda(cudaStreamSynchronize(stream_.get()), "P2 count sync");
                count_sync_ms = elapsed_ms(count_start, Clock::now());
                count = *static_cast<const int*>(host_count_.data());
                if (count < 0 || count > ppe::kYoloCandidateCount) {
                    throw std::runtime_error("P2 count overflow");
                }
            }
            const std::size_t candidate_bytes =
                static_cast<std::size_t>(count) * sizeof(ppe::GpuCandidate);
            const auto payload_start = Clock::now();
            if (candidate_bytes > 0) {
                check_cuda(cudaMemcpyAsync(
                    host_candidates_.data(),
                    path == "P1" ? postprocessor_.device_fixed_candidates()
                                 : postprocessor_.device_candidates(),
                    candidate_bytes, cudaMemcpyDeviceToHost, stream_.get()),
                    "candidate payload D2H");
            }
            check_cuda(cudaStreamSynchronize(stream_.get()), "candidate payload sync");
            payload_sync_ms = elapsed_ms(payload_start, Clock::now());
            float measured_gpu_ms = 0.0F;
            check_cuda(cudaEventElapsedTime(
                &measured_gpu_ms, gpu_start_.get(), gpu_end_.get()),
                "measure GPU postprocess");
            gpu_ms = measured_gpu_ms;
            const auto scan_start = Clock::now();
            const auto* host = static_cast<const ppe::GpuCandidate*>(
                host_candidates_.data());
            candidates.reserve(reference_.size());
            for (int index = 0; index < count; ++index) {
                const auto& candidate = host[index];
                if (candidate.candidate_index == -1) {
                    if (path != "P1" || candidate.class_id != -1 ||
                        candidate.confidence != 0.0F || candidate.x1 != 0.0F ||
                        candidate.y1 != 0.0F || candidate.x2 != 0.0F ||
                        candidate.y2 != 0.0F) {
                        throw std::runtime_error("invalid fixture sentinel");
                    }
                    continue;
                }
                candidates.push_back(candidate);
            }
            scan_ms = elapsed_ms(scan_start, Clock::now());
            d2h_bytes = candidate_bytes + (path == "P2" ? sizeof(int) : 0U);
        }

        const auto nms_start = Clock::now();
        const auto detections = nms(candidates, 0.70F);
        const double nms_ms = elapsed_ms(nms_start, Clock::now());
        if (validate) {
            validate_candidates(candidates, reference_);
            validate_candidates(detections, reference_nms_);
        }
        return {path, iteration, gpu_ms, count_sync_ms, payload_sync_ms,
                decode_ms, scan_ms, nms_ms,
                elapsed_ms(total_start, Clock::now()), d2h_bytes,
                static_cast<int>(candidates.size()),
                static_cast<int>(detections.size())};
    }

private:
    ppe::LetterboxGeometry geometry_;
    DeviceRaw device_raw_;
    Stream stream_;
    Event gpu_start_;
    Event gpu_end_;
    PinnedBuffer host_raw_;
    PinnedBuffer host_candidates_;
    PinnedBuffer host_count_;
    ppe::GpuPostprocessor postprocessor_;
    std::vector<ppe::GpuCandidate> reference_;
    std::vector<ppe::GpuCandidate> reference_nms_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 9 || std::string(argv[1]) != "--fixture" ||
            std::string(argv[3]) != "--output-dir" ||
            std::string(argv[5]) != "--warmup" ||
            std::string(argv[7]) != "--iterations") {
            throw std::runtime_error(
                "usage: postprocess_gain_gate_benchmark --fixture FILE "
                "--output-dir DIR --warmup N --iterations N");
        }
        const std::filesystem::path fixture_path(argv[2]);
        const std::filesystem::path output_dir(argv[4]);
        const int warmup = std::stoi(argv[6]);
        const int iterations = std::stoi(argv[8]);
        if (warmup < 0 || iterations <= 0 ||
            !std::filesystem::is_regular_file(fixture_path) ||
            std::filesystem::file_size(fixture_path) != DeviceRaw::bytes()) {
            throw std::runtime_error("invalid fixture or iteration arguments");
        }
        std::vector<float> fixture(DeviceRaw::elements());
        std::ifstream input(fixture_path, std::ios::binary);
        input.read(reinterpret_cast<char*>(fixture.data()),
                   static_cast<std::streamsize>(DeviceRaw::bytes()));
        if (!input) {
            throw std::runtime_error("failed to read raw fixture");
        }
        std::filesystem::create_directories(output_dir);
        Benchmark benchmark(
            fixture, ppe::make_letterbox_geometry(1920, 1080, 640));
        for (int index = 0; index < warmup; ++index) {
            for (const auto* path : {"P0", "P1", "P2"}) {
                static_cast<void>(benchmark.run(path, -1, true));
            }
        }
        std::ofstream csv(output_dir / "samples.csv");
        csv << "iteration,path,gpu_postprocess_cuda_ms,count_sync_host_ms,"
               "payload_sync_host_ms,cpu_decode_filter_ms,"
               "cpu_candidate_scan_ms,cpu_nms_ms,total_ms,d2h_bytes,"
               "candidate_count,detection_count\n";
        csv << std::fixed << std::setprecision(9);
        const std::array<std::array<const char*, 3>, 3> orders{{
            {{"P0", "P1", "P2"}},
            {{"P2", "P1", "P0"}},
            {{"P1", "P0", "P2"}},
        }};
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (const auto* path : orders[iteration % orders.size()]) {
                const auto sample = benchmark.run(path, iteration, true);
                csv << sample.iteration << ',' << sample.path << ','
                    << sample.gpu_postprocess_cuda_ms << ','
                    << sample.count_sync_host_ms << ','
                    << sample.payload_sync_host_ms << ','
                    << sample.cpu_decode_filter_ms << ','
                    << sample.cpu_candidate_scan_ms << ','
                    << sample.cpu_nms_ms << ',' << sample.total_ms << ','
                    << sample.d2h_bytes << ',' << sample.candidate_count << ','
                    << sample.detection_count << '\n';
            }
        }
        std::ofstream summary(output_dir / "summary.txt");
        summary << "result=PASS\nwarmup=" << warmup
                << "\niterations_per_path=" << iterations
                << "\nraw_bytes=" << DeviceRaw::bytes() << '\n';
        std::cout << "result=PASS iterations_per_path=" << iterations
                  << " output_dir=" << output_dir << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
