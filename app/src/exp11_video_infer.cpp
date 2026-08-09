#include "cuda_preprocess.hpp"
#ifdef PPE_ENABLE_GPU_POSTPROCESS
#include "gpu_postprocess.hpp"
#endif
#include "ppe_nvtx.hpp"
#include "trt_runtime.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Detection {
    int class_id{};
    int candidate_index{};
    float confidence{};
    float x1{};
    float y1{};
    float x2{};
    float y2{};
};

struct Stats {
    double mean{};
    double p50{};
    double p95{};
    double p99{};
    double minimum{};
    double maximum{};
};

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
            throw std::runtime_error("arguments must be --key value pairs");
        }
        args[argv[index]] = argv[index + 1];
    }
    for (const auto* required : {"--engine", "--source-type", "--output-dir"}) {
        if (!args.count(required)) {
            throw std::runtime_error(std::string("missing argument: ") + required);
        }
    }
    if (args.at("--source-type") == "file" && !args.count("--source")) {
        throw std::runtime_error("file input requires --source");
    }
    if (args.at("--source-type") != "file" &&
        args.at("--source-type") != "camera") {
        throw std::runtime_error("--source-type must be file or camera");
    }
    return args;
}

int integer_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    int fallback) {
    return args.count(key) ? std::stoi(args.at(key)) : fallback;
}

double double_arg(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    double fallback) {
    return args.count(key) ? std::stod(args.at(key)) : fallback;
}

#ifdef PPE_ENABLE_GPU_POSTPROCESS
enum class PostprocessMode { kBaseline, kRawPinned, kAtomic, kCub, kFixed };

PostprocessMode parse_postprocess_mode(
    const std::map<std::string, std::string>& args) {
    const std::string value =
        args.count("--postprocess") ? args.at("--postprocess") : "baseline";
    if (value == "baseline") {
        return PostprocessMode::kBaseline;
    }
    if (value == "raw_pinned") {
        return PostprocessMode::kRawPinned;
    }
    if (value == "atomic") {
        return PostprocessMode::kAtomic;
    }
    if (value == "cub") {
        return PostprocessMode::kCub;
    }
    if (value == "fixed") {
        return PostprocessMode::kFixed;
    }
    throw std::runtime_error(
        "--postprocess must be baseline, raw_pinned, atomic, cub, or fixed");
}

const char* postprocess_mode_name(PostprocessMode mode) {
    switch (mode) {
        case PostprocessMode::kBaseline:
            return "baseline";
        case PostprocessMode::kRawPinned:
            return "raw_pinned";
        case PostprocessMode::kAtomic:
            return "atomic";
        case PostprocessMode::kCub:
            return "cub";
        case PostprocessMode::kFixed:
            return "fixed";
    }
    throw std::runtime_error("invalid postprocess mode");
}
#endif

std::string camera_pipeline(int sensor, int width, int height, int fps) {
    std::ostringstream stream;
    stream << "nvarguscamerasrc sensor-id=" << sensor
           << " ! video/x-raw(memory:NVMM),width=" << width
           << ",height=" << height << ",framerate=" << fps
           << "/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx"
           << " ! videoconvert ! video/x-raw,format=BGR"
           << " ! appsink max-buffers=1 drop=true sync=false";
    return stream.str();
}

std::string gst_quote(const std::string& value) {
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

std::string file_pipeline(const std::filesystem::path& source) {
    return "filesrc location=" + gst_quote(source.string()) +
           " ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv"
           " ! video/x-raw,format=BGRx ! videoconvert"
           " ! video/x-raw,format=BGR"
           " ! appsink max-buffers=1 drop=false sync=false";
}

double percentile(const std::vector<double>& sorted, double quantile) {
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

Stats summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("cannot summarize empty timing vector");
    }
    std::sort(values.begin(), values.end());
    return {
        std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size()),
        percentile(values, 0.50),
        percentile(values, 0.95),
        percentile(values, 0.99),
        values.front(),
        values.back()};
}

float intersection_over_union(const Detection& left, const Detection& right) {
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

std::vector<Detection> decode_detections(
    const float* output,
    std::size_t output_size,
    const ppe::LetterboxGeometry& geometry,
    float confidence_threshold,
    float nms_threshold,
    int* decoded_count = nullptr,
    double* decode_filter_ms = nullptr,
    double* nms_ms = nullptr) {
    constexpr int candidates = 8400;
    constexpr int classes = 3;
    constexpr int channels = 7;
    if (output == nullptr ||
        output_size != static_cast<std::size_t>(channels * candidates)) {
        throw std::runtime_error("unexpected YOLO11 output element count");
    }
    std::vector<Detection> decoded;
    decoded.reserve(256);
    const auto decode_start = std::chrono::steady_clock::now();
    {
        PPE_NVTX_RANGE("decode");
        for (int index = 0; index < candidates; ++index) {
            int class_id = 0;
            float confidence = output[4 * candidates + index];
            for (int category = 1; category < classes; ++category) {
                const float value = output[(4 + category) * candidates + index];
                if (value > confidence) {
                    confidence = value;
                    class_id = category;
                }
            }
            if (!std::isfinite(confidence) || confidence < confidence_threshold) {
                continue;
            }
            const float center_x = output[index];
            const float center_y = output[candidates + index];
            const float width = output[2 * candidates + index];
            const float height = output[3 * candidates + index];
            if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
                !std::isfinite(width) || !std::isfinite(height) ||
                width <= 0.0F || height <= 0.0F) {
                continue;
            }
            Detection detection;
            detection.class_id = class_id;
            detection.candidate_index = index;
            detection.confidence = confidence;
            detection.x1 = std::clamp(
                (center_x - width * 0.5F - geometry.padding_left) / geometry.ratio,
                0.0F, static_cast<float>(geometry.source_width));
            detection.y1 = std::clamp(
                (center_y - height * 0.5F - geometry.padding_top) / geometry.ratio,
                0.0F, static_cast<float>(geometry.source_height));
            detection.x2 = std::clamp(
                (center_x + width * 0.5F - geometry.padding_left) / geometry.ratio,
                0.0F, static_cast<float>(geometry.source_width));
            detection.y2 = std::clamp(
                (center_y + height * 0.5F - geometry.padding_top) / geometry.ratio,
                0.0F, static_cast<float>(geometry.source_height));
            if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
                decoded.push_back(detection);
            }
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    std::vector<Detection> kept;
    if (decoded_count != nullptr) {
        *decoded_count = static_cast<int>(decoded.size());
    }
    const auto nms_start = std::chrono::steady_clock::now();
    {
        PPE_NVTX_RANGE("nms");
        std::sort(
            decoded.begin(), decoded.end(),
            [](const Detection& left, const Detection& right) {
                if (left.confidence != right.confidence) {
                    return left.confidence > right.confidence;
                }
                return left.candidate_index < right.candidate_index;
            });
        kept.reserve(decoded.size());
        for (const auto& candidate : decoded) {
            bool suppressed = false;
            for (const auto& accepted : kept) {
                if (candidate.class_id == accepted.class_id &&
                    intersection_over_union(candidate, accepted) > nms_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.push_back(candidate);
            }
        }
    }
    const auto nms_end = std::chrono::steady_clock::now();
    if (decode_filter_ms != nullptr) {
        *decode_filter_ms = std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
    }
    if (nms_ms != nullptr) {
        *nms_ms = std::chrono::duration<double, std::milli>(
            nms_end - nms_start).count();
    }
    return kept;
}

std::vector<Detection> decode_detections(
    const std::vector<float>& output,
    const ppe::LetterboxGeometry& geometry,
    float confidence_threshold,
    float nms_threshold,
    int* decoded_count = nullptr,
    double* decode_filter_ms = nullptr,
    double* nms_ms = nullptr) {
    return decode_detections(
        output.data(), output.size(), geometry, confidence_threshold,
        nms_threshold, decoded_count, decode_filter_ms, nms_ms);
}

cv::Mat annotate(const cv::Mat& frame, const std::vector<Detection>& detections) {
    static const std::vector<std::string> names = {
        "person", "helmet", "safety_vest"};
    static const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 128, 0)};
    cv::Mat result = frame.clone();
    for (const auto& detection : detections) {
        const cv::Rect rectangle(
            cv::Point(
                static_cast<int>(std::round(detection.x1)),
                static_cast<int>(std::round(detection.y1))),
            cv::Point(
                static_cast<int>(std::round(detection.x2)),
                static_cast<int>(std::round(detection.y2))));
        cv::rectangle(result, rectangle, colors[detection.class_id], 2);
        std::ostringstream label;
        label << names[detection.class_id] << ' ' << std::fixed
              << std::setprecision(2) << detection.confidence;
        cv::putText(
            result, label.str(),
            cv::Point(rectangle.x, std::max(20, rectangle.y - 4)),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, colors[detection.class_id], 2);
    }
    return result;
}

void write_stats(std::ostream& stream, const std::string& name, const Stats& stats) {
    stream << "    \"" << name << "\": {\"mean\": " << stats.mean
           << ", \"p50\": " << stats.p50
           << ", \"p95\": " << stats.p95
           << ", \"p99\": " << stats.p99
           << ", \"min\": " << stats.minimum
           << ", \"max\": " << stats.maximum << "}";
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    return escaped;
}

#ifdef PPE_ENABLE_GPU_POSTPROCESS
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes_), "cudaMalloc Exp15 buffer");
    }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return pointer_; }

private:
    void* pointer_{nullptr};
    std::size_t bytes_{};
};

class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) {
        check_cuda(
            cudaHostAlloc(&pointer_, bytes, cudaHostAllocDefault),
            "cudaHostAlloc Exp15 buffer");
    }
    ~PinnedBuffer() { cudaFreeHost(pointer_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void* data() const { return pointer_; }

private:
    void* pointer_{nullptr};
};

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate Exp15"); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

double event_elapsed_ms(const CudaEvent& begin, const CudaEvent& end) {
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
        "cudaEventElapsedTime Exp15");
    return static_cast<double>(milliseconds);
}

struct RawPinnedFrameResult {
    const float* output{};
    std::size_t elements{};
    double host_total_ms{};
    double inference_cuda_ms{};
    double raw_copy_cuda_ms{};
    double raw_sync_host_ms{};
};

class RawPinnedPostprocessPath {
public:
    explicit RawPinnedPostprocessPath(ppe::TrtRuntime& runtime)
        : runtime_(runtime),
          raw_output_(runtime.output_info().bytes),
          host_raw_(runtime.output_info().bytes) {}

    RawPinnedFrameResult process(
        const float* device_model_input,
        cudaStream_t stream) {
        const auto host_start = std::chrono::steady_clock::now();
        check_cuda(
            cudaEventRecord(inference_start_.get(), stream),
            "record pinned raw inference start");
        runtime_.enqueue_device_async(
            device_model_input, static_cast<float*>(raw_output_.data()), stream);
        check_cuda(
            cudaEventRecord(inference_end_.get(), stream),
            "record pinned raw inference end");
        check_cuda(
            cudaEventRecord(raw_copy_start_.get(), stream),
            "record pinned raw copy start");
        {
            PPE_NVTX_RANGE("raw_pinned_d2h");
            check_cuda(
                cudaMemcpyAsync(
                    host_raw_.data(), raw_output_.data(),
                    runtime_.output_info().bytes, cudaMemcpyDeviceToHost, stream),
                "copy pinned raw output");
        }
        check_cuda(
            cudaEventRecord(raw_copy_end_.get(), stream),
            "record pinned raw copy end");
        const auto sync_start = std::chrono::steady_clock::now();
        {
            PPE_NVTX_RANGE("raw_pinned_sync");
            check_cuda(
                cudaEventSynchronize(raw_copy_end_.get()),
                "synchronize pinned raw output");
        }
        const double sync_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sync_start).count();
        return {
            static_cast<const float*>(host_raw_.data()),
            runtime_.output_info().elements,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - host_start).count(),
            event_elapsed_ms(inference_start_, inference_end_),
            event_elapsed_ms(raw_copy_start_, raw_copy_end_),
            sync_ms};
    }

private:
    ppe::TrtRuntime& runtime_;
    DeviceBuffer raw_output_;
    PinnedBuffer host_raw_;
    CudaEvent inference_start_;
    CudaEvent inference_end_;
    CudaEvent raw_copy_start_;
    CudaEvent raw_copy_end_;
};

struct GpuFrameResult {
    std::vector<ppe::GpuCandidate> candidates;
    int candidate_count{};
    std::size_t d2h_bytes{};
    double host_total_ms{};
    double inference_cuda_ms{};
    double gpu_postprocess_cuda_ms{};
    double count_copy_cuda_ms{};
    double count_sync_host_ms{};
    double candidate_copy_cuda_ms{};
    double candidate_sync_host_ms{};
};

class GpuPostprocessPath {
public:
    explicit GpuPostprocessPath(ppe::TrtRuntime& runtime)
        : runtime_(runtime),
          raw_output_(runtime.output_info().bytes),
          host_count_(sizeof(int)),
          host_candidates_(
              static_cast<std::size_t>(ppe::kYoloCandidateCount) *
              sizeof(ppe::GpuCandidate)) {
        if (runtime.output_info().elements !=
            static_cast<std::size_t>(ppe::kYoloOutputChannels) *
                ppe::kYoloCandidateCount) {
            throw std::runtime_error("unexpected raw tensor shape for Exp15");
        }
    }

    GpuFrameResult process(
        const float* device_model_input,
        cudaStream_t stream,
        const ppe::LetterboxGeometry& geometry,
        float confidence_threshold,
        PostprocessMode mode) {
        if (mode == PostprocessMode::kBaseline) {
            throw std::runtime_error("baseline cannot use GPU postprocess path");
        }
        const auto host_start = std::chrono::steady_clock::now();
        check_cuda(
            cudaEventRecord(inference_start_.get(), stream),
            "record Exp15 inference start");
        runtime_.enqueue_device_async(
            device_model_input, static_cast<float*>(raw_output_.data()), stream);
        check_cuda(
            cudaEventRecord(inference_end_.get(), stream),
            "record Exp15 inference end");
        check_cuda(
            cudaEventRecord(gpu_postprocess_start_.get(), stream),
            "record Exp15 GPU postprocess start");
        {
            PPE_NVTX_RANGE("gpu_decode_filter_compaction");
            postprocessor_.launch(
                static_cast<const float*>(raw_output_.data()), geometry,
                confidence_threshold,
                mode == PostprocessMode::kAtomic
                    ? ppe::GpuCompactionMode::kAtomic
                    : (mode == PostprocessMode::kFixed
                           ? ppe::GpuCompactionMode::kFixed
                           : ppe::GpuCompactionMode::kCubStable),
                stream);
        }
        check_cuda(
            cudaEventRecord(gpu_postprocess_end_.get(), stream),
            "record Exp15 GPU postprocess end");
        int count = ppe::kYoloCandidateCount;
        double count_copy_cuda_ms = 0.0;
        double count_sync_host_ms = 0.0;
        if (mode != PostprocessMode::kFixed) {
            check_cuda(
                cudaEventRecord(count_copy_start_.get(), stream),
                "record Exp15 count copy start");
            {
                PPE_NVTX_RANGE("candidate_count_d2h");
                check_cuda(
                    cudaMemcpyAsync(
                        host_count_.data(), postprocessor_.device_count(),
                        sizeof(int), cudaMemcpyDeviceToHost, stream),
                    "copy Exp15 candidate count");
            }
            check_cuda(
                cudaEventRecord(count_copy_end_.get(), stream),
                "record Exp15 count copy end");
            const auto count_wait_start = std::chrono::steady_clock::now();
            {
                PPE_NVTX_RANGE("candidate_count_sync");
                check_cuda(
                    cudaEventSynchronize(count_copy_end_.get()),
                    "synchronize Exp15 candidate count");
            }
            count_sync_host_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - count_wait_start).count();
            count = *static_cast<const int*>(host_count_.data());
            if (count < 0 || count > postprocessor_.capacity()) {
                throw std::runtime_error("Exp15 candidate count overflow");
            }
            count_copy_cuda_ms =
                event_elapsed_ms(count_copy_start_, count_copy_end_);
        }

        const std::size_t candidate_bytes =
            static_cast<std::size_t>(count) * sizeof(ppe::GpuCandidate);
        check_cuda(
            cudaEventRecord(candidate_copy_start_.get(), stream),
            "record Exp15 candidate copy start");
        if (candidate_bytes > 0) {
            PPE_NVTX_RANGE("candidate_payload_d2h");
            check_cuda(
                cudaMemcpyAsync(
                    host_candidates_.data(),
                    mode == PostprocessMode::kFixed
                        ? postprocessor_.device_fixed_candidates()
                        : postprocessor_.device_candidates(),
                    candidate_bytes, cudaMemcpyDeviceToHost, stream),
                "copy Exp15 candidates");
        }
        check_cuda(
            cudaEventRecord(candidate_copy_end_.get(), stream),
            "record Exp15 candidate copy end");
        const auto candidate_wait_start = std::chrono::steady_clock::now();
        {
            PPE_NVTX_RANGE("candidate_payload_sync");
            check_cuda(
                cudaEventSynchronize(candidate_copy_end_.get()),
                "synchronize Exp15 candidates");
        }
        const double candidate_sync_host_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - candidate_wait_start).count();
        check_cuda(cudaGetLastError(), "Exp15 postprocess CUDA status");

        const auto* begin =
            static_cast<const ppe::GpuCandidate*>(host_candidates_.data());
        GpuFrameResult result;
        result.candidates.assign(begin, begin + count);
        result.candidate_count = count;
        result.d2h_bytes =
            (mode == PostprocessMode::kFixed ? 0U : sizeof(int)) +
            candidate_bytes;
        result.host_total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_start).count();
        result.inference_cuda_ms =
            event_elapsed_ms(inference_start_, inference_end_);
        result.gpu_postprocess_cuda_ms =
            event_elapsed_ms(gpu_postprocess_start_, gpu_postprocess_end_);
        result.count_copy_cuda_ms = count_copy_cuda_ms;
        result.count_sync_host_ms = count_sync_host_ms;
        result.candidate_copy_cuda_ms =
            event_elapsed_ms(candidate_copy_start_, candidate_copy_end_);
        result.candidate_sync_host_ms = candidate_sync_host_ms;
        return result;
    }

private:
    ppe::TrtRuntime& runtime_;
    DeviceBuffer raw_output_;
    ppe::GpuPostprocessor postprocessor_;
    PinnedBuffer host_count_;
    PinnedBuffer host_candidates_;
    CudaEvent inference_start_;
    CudaEvent inference_end_;
    CudaEvent gpu_postprocess_start_;
    CudaEvent gpu_postprocess_end_;
    CudaEvent count_copy_start_;
    CudaEvent count_copy_end_;
    CudaEvent candidate_copy_start_;
    CudaEvent candidate_copy_end_;
};

std::vector<Detection> nms_gpu_candidates(
    const std::vector<ppe::GpuCandidate>& candidates,
    float nms_threshold,
    bool allow_invalid_sentinel = false,
    int* valid_count = nullptr,
    double* candidate_scan_ms = nullptr,
    double* nms_ms = nullptr) {
    std::vector<Detection> decoded;
    decoded.reserve(candidates.size());
    const auto scan_start = std::chrono::steady_clock::now();
    for (const auto& candidate : candidates) {
        if (candidate.candidate_index == -1) {
            if (!allow_invalid_sentinel || candidate.class_id != -1 ||
                candidate.confidence != 0.0F || candidate.x1 != 0.0F ||
                candidate.y1 != 0.0F || candidate.x2 != 0.0F ||
                candidate.y2 != 0.0F) {
                throw std::runtime_error("invalid fixed candidate sentinel");
            }
            continue;
        }
        if (candidate.class_id < 0 || candidate.class_id >= ppe::kYoloClassCount ||
            candidate.candidate_index < 0 ||
            candidate.candidate_index >= ppe::kYoloCandidateCount ||
            !std::isfinite(candidate.confidence) ||
            !std::isfinite(candidate.x1) || !std::isfinite(candidate.y1) ||
            !std::isfinite(candidate.x2) || !std::isfinite(candidate.y2) ||
            candidate.x2 <= candidate.x1 || candidate.y2 <= candidate.y1) {
            throw std::runtime_error("invalid compacted GPU candidate");
        }
        decoded.push_back({
            candidate.class_id, candidate.candidate_index,
            candidate.confidence, candidate.x1, candidate.y1,
            candidate.x2, candidate.y2});
    }
    const auto scan_end = std::chrono::steady_clock::now();
    if (valid_count != nullptr) {
        *valid_count = static_cast<int>(decoded.size());
    }
    const auto nms_start = std::chrono::steady_clock::now();
    std::sort(
        decoded.begin(), decoded.end(),
        [](const Detection& left, const Detection& right) {
            if (left.confidence != right.confidence) {
                return left.confidence > right.confidence;
            }
            return left.candidate_index < right.candidate_index;
        });
    std::vector<Detection> kept;
    kept.reserve(decoded.size());
    for (const auto& candidate : decoded) {
        bool suppressed = false;
        for (const auto& accepted : kept) {
            if (candidate.class_id == accepted.class_id &&
                intersection_over_union(candidate, accepted) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
        }
    }
    const auto nms_end = std::chrono::steady_clock::now();
    if (candidate_scan_ms != nullptr) {
        *candidate_scan_ms = std::chrono::duration<double, std::milli>(
            scan_end - scan_start).count();
    }
    if (nms_ms != nullptr) {
        *nms_ms = std::chrono::duration<double, std::milli>(
            nms_end - nms_start).count();
    }
    return kept;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const std::string source_type = args.at("--source-type");
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        const PostprocessMode postprocess_mode = parse_postprocess_mode(args);
#endif
        const int max_frames = integer_arg(args, "--max-frames", 0);
        const int warmup = integer_arg(args, "--warmup", 2);
        const float confidence = static_cast<float>(
            double_arg(args, "--confidence", 0.25));
        const float nms_iou = static_cast<float>(
            double_arg(args, "--nms-iou", 0.70));
        if (max_frames < 0 || warmup < 0 || confidence < 0.0F ||
            confidence > 1.0F || nms_iou < 0.0F || nms_iou > 1.0F) {
            throw std::runtime_error("invalid numeric arguments");
        }
        const std::filesystem::path output_dir(args.at("--output-dir"));
        std::filesystem::create_directories(output_dir);

        cv::VideoCapture capture;
        std::string source_description;
        if (source_type == "file") {
            const std::filesystem::path source(args.at("--source"));
            if (!std::filesystem::is_regular_file(source)) {
                throw std::runtime_error("video file does not exist: " + source.string());
            }
            source_description = file_pipeline(source);
            capture.open(source_description, cv::CAP_GSTREAMER);
        } else {
            const int sensor = integer_arg(args, "--sensor-id", 0);
            const int width = integer_arg(args, "--width", 1920);
            const int height = integer_arg(args, "--height", 1080);
            const int fps = integer_arg(args, "--fps", 30);
            source_description = camera_pipeline(sensor, width, height, fps);
            capture.open(source_description, cv::CAP_GSTREAMER);
        }
        if (!capture.isOpened()) {
            throw std::runtime_error("GStreamer VideoCapture open failed");
        }

        cv::Mat frame;
        const auto first_capture_start = std::chrono::steady_clock::now();
        {
            PPE_NVTX_RANGE("capture");
            if (!capture.read(frame) || frame.empty()) {
                throw std::runtime_error("failed to acquire first frame");
            }
        }
        const auto first_capture_end = std::chrono::steady_clock::now();
        if (frame.type() != CV_8UC3) {
            throw std::runtime_error("captured frame is not CV_8UC3 BGR");
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }
        const auto geometry = ppe::make_letterbox_geometry(frame.cols, frame.rows, 640);
        ppe::TrtRuntime runtime(args.at("--engine"));
        ppe::CudaPreprocessor preprocessor(geometry);
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        std::unique_ptr<RawPinnedPostprocessPath> raw_pinned_path;
        std::unique_ptr<GpuPostprocessPath> gpu_postprocess_path;
        if (postprocess_mode == PostprocessMode::kRawPinned) {
            raw_pinned_path = std::make_unique<RawPinnedPostprocessPath>(runtime);
        } else if (postprocess_mode != PostprocessMode::kBaseline) {
            gpu_postprocess_path = std::make_unique<GpuPostprocessPath>(runtime);
        }
#endif
        for (int index = 0; index < warmup; ++index) {
            const auto prepared = preprocessor.process(frame.ptr<std::uint8_t>());
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            if (postprocess_mode == PostprocessMode::kBaseline) {
#endif
            const auto inferred = runtime.infer_device(
                prepared.device_output, preprocessor.stream());
            if (!std::all_of(
                    inferred.output.begin(), inferred.output.end(),
                    [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("warmup output contains NaN or Inf");
            }
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            } else if (postprocess_mode == PostprocessMode::kRawPinned) {
                const auto inferred = raw_pinned_path->process(
                    prepared.device_output, preprocessor.stream());
                static_cast<void>(decode_detections(
                    inferred.output, inferred.elements, geometry,
                    confidence, nms_iou));
            } else {
                const auto compacted = gpu_postprocess_path->process(
                    prepared.device_output, preprocessor.stream(), geometry,
                    confidence, postprocess_mode);
                static_cast<void>(nms_gpu_candidates(
                    compacted.candidates, nms_iou,
                    postprocess_mode == PostprocessMode::kFixed));
            }
#endif
        }

        std::ofstream detections_csv(output_dir / "detections.csv");
        std::ofstream frames_csv(output_dir / "frames.csv");
        if (!detections_csv || !frames_csv) {
            throw std::runtime_error("cannot open CSV outputs");
        }
        detections_csv << "frame_index,detection_index,class_id,class_name,confidence,x1,y1,x2,y2\n";
        frames_csv << "frame_index,detection_count,capture_ms,preprocess_host_ms,"
                      "preprocess_cuda_ms,inference_host_ms,inference_cuda_ms,"
                      "postprocess_ms,end_to_end_ms";
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        frames_csv << ",postprocess_mode,candidate_count,d2h_bytes,"
                      "gpu_postprocess_cuda_ms,count_copy_cuda_ms,"
                      "count_sync_host_ms,candidate_copy_cuda_ms,"
                      "candidate_sync_host_ms,cpu_decode_filter_ms,"
                      "cpu_candidate_scan_ms,cpu_nms_ms\n";
#else
        frames_csv << '\n';
#endif
        detections_csv << std::fixed << std::setprecision(9);
        frames_csv << std::fixed << std::setprecision(9);
        const std::vector<std::string> class_names = {
            "person", "helmet", "safety_vest"};
        std::vector<double> capture_times;
        std::vector<double> preprocess_host_times;
        std::vector<double> preprocess_cuda_times;
        std::vector<double> inference_host_times;
        std::vector<double> inference_cuda_times;
        std::vector<double> postprocess_times;
        std::vector<double> end_to_end_times;
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        std::vector<double> candidate_counts;
        std::vector<double> d2h_bytes_per_frame;
        std::vector<double> gpu_postprocess_cuda_times;
        std::vector<double> count_copy_cuda_times;
        std::vector<double> count_sync_host_times;
        std::vector<double> candidate_copy_cuda_times;
        std::vector<double> candidate_sync_host_times;
        std::vector<double> cpu_decode_filter_times;
        std::vector<double> cpu_candidate_scan_times;
        std::vector<double> cpu_nms_times;
#endif
        std::size_t total_detections = 0;
        int frame_index = 0;
        double current_capture_ms = std::chrono::duration<double, std::milli>(
            first_capture_end - first_capture_start).count();
        cv::Mat first_annotated;
        cv::Mat last_annotated;

        const auto pipeline_start = std::chrono::steady_clock::now();
        while (true) {
            PPE_NVTX_RANGE("frame_total");
            const auto frame_start = std::chrono::steady_clock::now();
            const auto prepared = preprocessor.process(frame.ptr<std::uint8_t>());
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            double inference_host_ms = 0.0;
            double inference_cuda_ms = 0.0;
            double gpu_postprocess_cuda_ms = 0.0;
            double count_copy_cuda_ms = 0.0;
            double count_sync_host_ms = 0.0;
            double candidate_copy_cuda_ms = 0.0;
            double candidate_sync_host_ms = 0.0;
            double cpu_decode_filter_ms = 0.0;
            double cpu_candidate_scan_ms = 0.0;
            double cpu_nms_ms = 0.0;
            int candidate_count = 0;
            std::size_t d2h_bytes = runtime.output_info().bytes;
            std::vector<Detection> detections;
            double post_ms = 0.0;
            if (postprocess_mode == PostprocessMode::kBaseline) {
#endif
            const auto inferred = runtime.infer_device(
                prepared.device_output, preprocessor.stream());
            const bool finite = std::all_of(
                inferred.output.begin(), inferred.output.end(),
                [](float value) { return std::isfinite(value); });
            if (!finite) {
                throw std::runtime_error("inference output contains NaN or Inf");
            }
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            if (frame_index == 0 && args.count("--raw-fixture")) {
                const std::filesystem::path fixture_path(args.at("--raw-fixture"));
                if (fixture_path.has_parent_path()) {
                    std::filesystem::create_directories(fixture_path.parent_path());
                }
                std::ofstream fixture(fixture_path, std::ios::binary);
                if (!fixture) {
                    throw std::runtime_error("cannot open raw fixture output");
                }
                fixture.write(
                    reinterpret_cast<const char*>(inferred.output.data()),
                    static_cast<std::streamsize>(
                        inferred.output.size() * sizeof(float)));
                if (!fixture) {
                    throw std::runtime_error("cannot write raw fixture output");
                }
            }
#endif
            const auto post_start = std::chrono::steady_clock::now();
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            detections = decode_detections(
                inferred.output, geometry, confidence, nms_iou,
                &candidate_count, &cpu_decode_filter_ms, &cpu_nms_ms);
#else
            const auto detections = decode_detections(
                inferred.output, geometry, confidence, nms_iou);
#endif
            const auto post_end = std::chrono::steady_clock::now();
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            post_ms = std::chrono::duration<double, std::milli>(
                post_end - post_start).count();
#else
            const double post_ms = std::chrono::duration<double, std::milli>(
                post_end - post_start).count();
#endif
#ifdef PPE_ENABLE_GPU_POSTPROCESS
            inference_host_ms = inferred.host_total_ms;
            inference_cuda_ms = inferred.cuda_total_ms;
            } else if (postprocess_mode == PostprocessMode::kRawPinned) {
                const auto inferred = raw_pinned_path->process(
                    prepared.device_output, preprocessor.stream());
                if (!std::all_of(
                        inferred.output, inferred.output + inferred.elements,
                        [](float value) { return std::isfinite(value); })) {
                    throw std::runtime_error(
                        "pinned raw inference output contains NaN or Inf");
                }
                if (frame_index == 0 && args.count("--raw-fixture")) {
                    const std::filesystem::path fixture_path(args.at("--raw-fixture"));
                    if (fixture_path.has_parent_path()) {
                        std::filesystem::create_directories(
                            fixture_path.parent_path());
                    }
                    std::ofstream fixture(fixture_path, std::ios::binary);
                    fixture.write(
                        reinterpret_cast<const char*>(inferred.output),
                        static_cast<std::streamsize>(
                            inferred.elements * sizeof(float)));
                    if (!fixture) {
                        throw std::runtime_error(
                            "cannot write pinned raw fixture output");
                    }
                }
                const auto post_start = std::chrono::steady_clock::now();
                detections = decode_detections(
                    inferred.output, inferred.elements, geometry, confidence,
                    nms_iou, &candidate_count, &cpu_decode_filter_ms,
                    &cpu_nms_ms);
                post_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - post_start).count();
                inference_host_ms = inferred.host_total_ms;
                inference_cuda_ms = inferred.inference_cuda_ms;
                candidate_copy_cuda_ms = inferred.raw_copy_cuda_ms;
                candidate_sync_host_ms = inferred.raw_sync_host_ms;
                d2h_bytes = runtime.output_info().bytes;
            } else {
                const auto compacted = gpu_postprocess_path->process(
                    prepared.device_output, preprocessor.stream(), geometry,
                    confidence, postprocess_mode);
                const auto nms_start = std::chrono::steady_clock::now();
                detections = nms_gpu_candidates(
                    compacted.candidates, nms_iou,
                    postprocess_mode == PostprocessMode::kFixed,
                    &candidate_count, &cpu_candidate_scan_ms, &cpu_nms_ms);
                const auto nms_end = std::chrono::steady_clock::now();
                post_ms = std::chrono::duration<double, std::milli>(
                    nms_end - nms_start).count();
                inference_host_ms = compacted.host_total_ms;
                inference_cuda_ms = compacted.inference_cuda_ms;
                gpu_postprocess_cuda_ms = compacted.gpu_postprocess_cuda_ms;
                count_copy_cuda_ms = compacted.count_copy_cuda_ms;
                count_sync_host_ms = compacted.count_sync_host_ms;
                candidate_copy_cuda_ms = compacted.candidate_copy_cuda_ms;
                candidate_sync_host_ms = compacted.candidate_sync_host_ms;
                d2h_bytes = compacted.d2h_bytes;
            }
#endif
            const auto frame_end = std::chrono::steady_clock::now();
            const double end_to_end_ms = current_capture_ms +
                std::chrono::duration<double, std::milli>(
                    frame_end - frame_start).count();

            {
                PPE_NVTX_RANGE("output");
                for (std::size_t index = 0; index < detections.size(); ++index) {
                    const auto& detection = detections[index];
                    if (detection.class_id < 0 || detection.class_id >= 3 ||
                        detection.confidence < confidence ||
                        detection.confidence > 1.0F ||
                        detection.x1 < 0.0F || detection.y1 < 0.0F ||
                        detection.x2 > frame.cols || detection.y2 > frame.rows ||
                        detection.x2 <= detection.x1 ||
                        detection.y2 <= detection.y1) {
                        throw std::runtime_error("invalid decoded detection");
                    }
                    detections_csv << frame_index << ',' << index << ','
                        << detection.class_id << ',' << class_names[detection.class_id]
                        << ',' << detection.confidence << ',' << detection.x1 << ','
                        << detection.y1 << ',' << detection.x2 << ',' << detection.y2
                        << '\n';
                }
                frames_csv << frame_index << ',' << detections.size() << ','
                    << current_capture_ms << ',' << prepared.host_total_ms << ','
                    << prepared.cuda_total_ms << ','
#ifdef PPE_ENABLE_GPU_POSTPROCESS
                    << inference_host_ms << ',' << inference_cuda_ms << ','
#else
                    << inferred.host_total_ms << ',' << inferred.cuda_total_ms << ','
#endif
                    << post_ms << ',' << end_to_end_ms;
#ifdef PPE_ENABLE_GPU_POSTPROCESS
                frames_csv << ',' << postprocess_mode_name(postprocess_mode)
                    << ',' << candidate_count << ',' << d2h_bytes
                    << ',' << gpu_postprocess_cuda_ms
                    << ',' << count_copy_cuda_ms
                    << ',' << count_sync_host_ms
                    << ',' << candidate_copy_cuda_ms
                    << ',' << candidate_sync_host_ms
                    << ',' << cpu_decode_filter_ms
                    << ',' << cpu_candidate_scan_ms
                    << ',' << cpu_nms_ms << '\n';
#else
                frames_csv << '\n';
#endif
                total_detections += detections.size();
                capture_times.push_back(current_capture_ms);
                preprocess_host_times.push_back(prepared.host_total_ms);
                preprocess_cuda_times.push_back(prepared.cuda_total_ms);
#ifdef PPE_ENABLE_GPU_POSTPROCESS
                inference_host_times.push_back(inference_host_ms);
                inference_cuda_times.push_back(inference_cuda_ms);
                candidate_counts.push_back(candidate_count);
                d2h_bytes_per_frame.push_back(static_cast<double>(d2h_bytes));
                gpu_postprocess_cuda_times.push_back(gpu_postprocess_cuda_ms);
                count_copy_cuda_times.push_back(count_copy_cuda_ms);
                count_sync_host_times.push_back(count_sync_host_ms);
                candidate_copy_cuda_times.push_back(candidate_copy_cuda_ms);
                candidate_sync_host_times.push_back(candidate_sync_host_ms);
                cpu_decode_filter_times.push_back(cpu_decode_filter_ms);
                cpu_candidate_scan_times.push_back(cpu_candidate_scan_ms);
                cpu_nms_times.push_back(cpu_nms_ms);
#else
                inference_host_times.push_back(inferred.host_total_ms);
                inference_cuda_times.push_back(inferred.cuda_total_ms);
#endif
                postprocess_times.push_back(post_ms);
                end_to_end_times.push_back(end_to_end_ms);
                last_annotated = annotate(frame, detections);
                if (frame_index == 0) {
                    first_annotated = last_annotated.clone();
                }
            }
            ++frame_index;
            if (max_frames > 0 && frame_index >= max_frames) {
                break;
            }
            const auto capture_start = std::chrono::steady_clock::now();
            bool acquired = false;
            {
                PPE_NVTX_RANGE("capture");
                acquired = capture.read(frame);
            }
            const auto capture_end = std::chrono::steady_clock::now();
            if (!acquired || frame.empty()) {
                if (source_type == "camera") {
                    throw std::runtime_error("camera frame acquisition ended early");
                }
                break;
            }
            if (frame.type() != CV_8UC3 || frame.cols != geometry.source_width ||
                frame.rows != geometry.source_height) {
                throw std::runtime_error("frame format or dimensions changed");
            }
            if (!frame.isContinuous()) {
                frame = frame.clone();
            }
            current_capture_ms = std::chrono::duration<double, std::milli>(
                capture_end - capture_start).count();
        }
        const auto pipeline_end = std::chrono::steady_clock::now();
        if (frame_index <= 0 ||
            (source_type == "camera" && max_frames > 0 && frame_index != max_frames)) {
            throw std::runtime_error("processed frame count failed acceptance");
        }
        detections_csv.close();
        frames_csv.close();
        if (!cv::imwrite((output_dir / "first_annotated.jpg").string(), first_annotated) ||
            !cv::imwrite((output_dir / "last_annotated.jpg").string(), last_annotated)) {
            throw std::runtime_error("cannot write annotated evidence frames");
        }

        const auto capture_stats = summarize(capture_times);
        const auto preprocess_host_stats = summarize(preprocess_host_times);
        const auto preprocess_cuda_stats = summarize(preprocess_cuda_times);
        const auto inference_host_stats = summarize(inference_host_times);
        const auto inference_cuda_stats = summarize(inference_cuda_times);
        const auto postprocess_stats = summarize(postprocess_times);
        const auto end_to_end_stats = summarize(end_to_end_times);
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        const auto candidate_count_stats = summarize(candidate_counts);
        const auto d2h_bytes_stats = summarize(d2h_bytes_per_frame);
        const auto gpu_postprocess_cuda_stats = summarize(gpu_postprocess_cuda_times);
        const auto count_copy_cuda_stats = summarize(count_copy_cuda_times);
        const auto count_sync_host_stats = summarize(count_sync_host_times);
        const auto candidate_copy_cuda_stats = summarize(candidate_copy_cuda_times);
        const auto candidate_sync_host_stats = summarize(candidate_sync_host_times);
        const auto cpu_decode_filter_stats = summarize(cpu_decode_filter_times);
        const auto cpu_candidate_scan_stats = summarize(cpu_candidate_scan_times);
        const auto cpu_nms_stats = summarize(cpu_nms_times);
#endif
        const double elapsed_seconds = std::accumulate(
            end_to_end_times.begin(), end_to_end_times.end(), 0.0) / 1000.0;
        const double effective_fps = static_cast<double>(frame_index) / elapsed_seconds;
        const double pipeline_wall_seconds =
            std::chrono::duration<double>(pipeline_end - pipeline_start).count();
        const double pipeline_wall_fps =
            static_cast<double>(frame_index) / pipeline_wall_seconds;
        std::ofstream summary(output_dir / "summary.json");
        if (!summary) {
            throw std::runtime_error("cannot open summary.json");
        }
        summary << std::fixed << std::setprecision(9)
                << "{\n"
                << "  \"result\": \"PASS\",\n"
                << "  \"source_type\": \"" << source_type << "\",\n"
#ifdef PPE_ENABLE_GPU_POSTPROCESS
                << "  \"postprocess_mode\": \""
                << postprocess_mode_name(postprocess_mode) << "\",\n"
                << "  \"gpu_candidate_bytes\": "
                << sizeof(ppe::GpuCandidate) << ",\n"
                << "  \"raw_output_d2h_bytes_per_frame\": "
                << runtime.output_info().bytes << ",\n"
#endif
                << "  \"source\": \"" << json_escape(source_description) << "\",\n"
                << "  \"engine\": \"" << json_escape(args.at("--engine")) << "\",\n"
                << "  \"frame_width\": " << geometry.source_width << ",\n"
                << "  \"frame_height\": " << geometry.source_height << ",\n"
                << "  \"processed_frames\": " << frame_index << ",\n"
                << "  \"total_detections\": " << total_detections << ",\n"
                << "  \"warmup_iterations\": " << warmup << ",\n"
                << "  \"confidence_threshold\": " << confidence << ",\n"
                << "  \"nms_iou_threshold\": " << nms_iou << ",\n"
                << "  \"effective_fps\": " << effective_fps << ",\n"
                << "  \"pipeline_wall_seconds\": " << pipeline_wall_seconds << ",\n"
                << "  \"pipeline_wall_fps\": " << pipeline_wall_fps << ",\n"
                << "  \"timing_scope\": \"capture+H2D+CUDA_preprocess+TensorRT+D2H+NMS\",\n"
                << "  \"timings_ms\": {\n";
        write_stats(summary, "capture", capture_stats); summary << ",\n";
        write_stats(summary, "preprocess_host", preprocess_host_stats); summary << ",\n";
        write_stats(summary, "preprocess_cuda", preprocess_cuda_stats); summary << ",\n";
        write_stats(summary, "inference_host", inference_host_stats); summary << ",\n";
        write_stats(summary, "inference_cuda", inference_cuda_stats); summary << ",\n";
        write_stats(summary, "postprocess", postprocess_stats); summary << ",\n";
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        write_stats(summary, "gpu_postprocess_cuda", gpu_postprocess_cuda_stats);
        summary << ",\n";
        write_stats(summary, "count_copy_cuda", count_copy_cuda_stats);
        summary << ",\n";
        write_stats(summary, "count_sync_host", count_sync_host_stats);
        summary << ",\n";
        write_stats(summary, "candidate_copy_cuda", candidate_copy_cuda_stats);
        summary << ",\n";
        write_stats(summary, "candidate_sync_host", candidate_sync_host_stats);
        summary << ",\n";
        write_stats(summary, "cpu_decode_filter", cpu_decode_filter_stats);
        summary << ",\n";
        write_stats(summary, "cpu_candidate_scan", cpu_candidate_scan_stats);
        summary << ",\n";
        write_stats(summary, "cpu_nms", cpu_nms_stats);
        summary << ",\n";
#endif
        write_stats(summary, "end_to_end", end_to_end_stats); summary << "\n";
#ifdef PPE_ENABLE_GPU_POSTPROCESS
        summary << "  },\n  \"transfer\": {\n";
        write_stats(summary, "candidate_count", candidate_count_stats);
        summary << ",\n";
        write_stats(summary, "d2h_bytes_per_frame", d2h_bytes_stats);
        summary << "\n";
#endif
        summary << "  }\n}\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "result=PASS source_type=" << source_type
#ifdef PPE_ENABLE_GPU_POSTPROCESS
                  << " postprocess=" << postprocess_mode_name(postprocess_mode)
#endif
                  << " frames=" << frame_index
                  << " detections=" << total_detections
                  << " e2e_p95_ms=" << end_to_end_stats.p95
                  << " effective_fps=" << effective_fps << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
