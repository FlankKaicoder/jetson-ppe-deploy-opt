#include "cuda_preprocess.hpp"
#include "ppe_nvtx.hpp"
#include "trt_runtime.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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

enum class Variant { kA, kB, kC };

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
            throw std::runtime_error("arguments must be --key value pairs");
        }
        args[argv[index]] = argv[index + 1];
    }
    for (const auto* required :
         {"--engine", "--source-type", "--output-dir", "--variant"}) {
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

Variant parse_variant(const std::string& text) {
    if (text == "A" || text == "a") {
        return Variant::kA;
    }
    if (text == "B" || text == "b") {
        return Variant::kB;
    }
    if (text == "C" || text == "c") {
        return Variant::kC;
    }
    throw std::runtime_error("--variant must be A, B, or C");
}

const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::kA:
            return "A";
        case Variant::kB:
            return "B";
        case Variant::kC:
            return "C";
    }
    throw std::runtime_error("invalid variant");
}

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
    const std::vector<float>& output,
    const ppe::LetterboxGeometry& geometry,
    float confidence_threshold,
    float nms_threshold) {
    constexpr int candidates = 8400;
    constexpr int classes = 3;
    constexpr int channels = 7;
    if (output.size() != static_cast<std::size_t>(channels * candidates)) {
        throw std::runtime_error("unexpected YOLO11 output element count");
    }
    std::vector<Detection> decoded;
    decoded.reserve(256);
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
    std::vector<Detection> kept;
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
    return kept;
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

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes_), "cudaMalloc Exp14 buffer");
    }
    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return pointer_; }
    std::size_t bytes() const { return bytes_; }

private:
    void* pointer_{nullptr};
    std::size_t bytes_{};
};

class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
        check_cuda(
            cudaHostAlloc(&pointer_, bytes_, cudaHostAllocDefault),
            "cudaHostAlloc Exp14 staging");
    }
    ~PinnedBuffer() {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void* data() const { return pointer_; }
    std::size_t bytes() const { return bytes_; }

private:
    void* pointer_{nullptr};
    std::size_t bytes_{};
};

class CudaStream {
public:
    CudaStream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags Exp14");
    }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class CudaEvent {
public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate Exp14");
    }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

double event_elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
        "cudaEventElapsedTime Exp14");
    return static_cast<double>(milliseconds);
}

struct Slot {
    Slot(std::size_t input_bytes, std::size_t model_input_bytes,
         std::size_t output_bytes)
        : host_input(input_bytes),
          host_output(output_bytes),
          device_input(input_bytes),
          device_model_input(model_input_bytes),
          device_output(output_bytes) {}

    PinnedBuffer host_input;
    PinnedBuffer host_output;
    DeviceBuffer device_input;
    DeviceBuffer device_model_input;
    DeviceBuffer device_output;
    CudaEvent upload_start;
    CudaEvent upload_end;
    CudaEvent inference_start;
    CudaEvent inference_end;
    CudaEvent download_start;
    CudaEvent download_end;
    bool busy{};
    int frame_index{-1};
    double capture_ms{};
    double staging_ms{};
    double submit_host_ms{};
    Clock::time_point submit_start{};
};

struct Completion {
    std::vector<float> output;
    double upload_cuda_ms{};
    double inference_cuda_ms{};
    double download_cuda_ms{};
    double wait_host_ms{};
};

class AsyncPipeline {
public:
    AsyncPipeline(
        ppe::TrtRuntime& runtime,
        const ppe::LetterboxGeometry& geometry,
        Variant variant)
        : runtime_(runtime), geometry_(geometry), variant_(variant) {
        const std::size_t input_bytes =
            static_cast<std::size_t>(geometry.source_width) *
            static_cast<std::size_t>(geometry.source_height) * 3;
        const std::size_t slot_count = variant == Variant::kC ? 2 : 1;
        slots_.reserve(slot_count);
        for (std::size_t index = 0; index < slot_count; ++index) {
            slots_.push_back(std::make_unique<Slot>(
                input_bytes, runtime_.input_info().bytes,
                runtime_.output_info().bytes));
        }
    }

    ~AsyncPipeline() {
        cudaStreamSynchronize(upload_stream_.get());
        cudaStreamSynchronize(inference_stream_.get());
        cudaStreamSynchronize(download_stream_.get());
    }

    std::size_t capacity() const { return slots_.size(); }
    Slot& slot(std::size_t index) { return *slots_.at(index); }

    void submit(
        Slot& slot,
        const cv::Mat& frame,
        int frame_index,
        double capture_ms) {
        if (slot.busy) {
            throw std::runtime_error("attempted to reuse an in-flight slot");
        }
        if (!frame.isContinuous() || frame.type() != CV_8UC3 ||
            frame.cols != geometry_.source_width ||
            frame.rows != geometry_.source_height) {
            throw std::runtime_error("invalid frame supplied to Exp14 pipeline");
        }
        slot.busy = true;
        slot.frame_index = frame_index;
        slot.capture_ms = capture_ms;
        slot.submit_start = Clock::now();
        const auto staging_start = Clock::now();
        {
            PPE_NVTX_RANGE("pinned_staging");
            std::memcpy(slot.host_input.data(), frame.data, slot.host_input.bytes());
        }
        const auto staging_end = Clock::now();
        slot.staging_ms = std::chrono::duration<double, std::milli>(
            staging_end - staging_start).count();

        check_cuda(
            cudaEventRecord(slot.upload_start.get(), upload_stream_.get()),
            "record upload start");
        {
            PPE_NVTX_RANGE("h2d");
            check_cuda(
                cudaMemcpyAsync(
                    slot.device_input.data(), slot.host_input.data(),
                    slot.host_input.bytes(), cudaMemcpyHostToDevice,
                    upload_stream_.get()),
                "Exp14 pinned H2D");
        }
        {
            PPE_NVTX_RANGE("preprocess_kernel");
            ppe::launch_cuda_preprocess_async(
                static_cast<const std::uint8_t*>(slot.device_input.data()),
                static_cast<float*>(slot.device_model_input.data()),
                geometry_, upload_stream_.get());
        }
        check_cuda(
            cudaEventRecord(slot.upload_end.get(), upload_stream_.get()),
            "record upload end");

        if (variant_ == Variant::kA) {
            {
                PPE_NVTX_RANGE("preprocess_sync");
                check_cuda(
                    cudaEventSynchronize(slot.upload_end.get()),
                    "Variant A preprocess synchronize");
            }
            check_cuda(
                cudaEventRecord(slot.inference_start.get(), upload_stream_.get()),
                "record Variant A inference start");
            runtime_.enqueue_device_async(
                static_cast<const float*>(slot.device_model_input.data()),
                static_cast<float*>(slot.device_output.data()),
                upload_stream_.get());
            check_cuda(
                cudaEventRecord(slot.inference_end.get(), upload_stream_.get()),
                "record Variant A inference end");
            check_cuda(
                cudaEventRecord(slot.download_start.get(), upload_stream_.get()),
                "record Variant A download start");
            {
                PPE_NVTX_RANGE("d2h");
                check_cuda(
                    cudaMemcpyAsync(
                        slot.host_output.data(), slot.device_output.data(),
                        slot.host_output.bytes(), cudaMemcpyDeviceToHost,
                        upload_stream_.get()),
                    "Variant A pinned D2H");
            }
            check_cuda(
                cudaEventRecord(slot.download_end.get(), upload_stream_.get()),
                "record Variant A download end");
            {
                PPE_NVTX_RANGE("inference_sync");
                check_cuda(
                    cudaEventSynchronize(slot.download_end.get()),
                    "Variant A inference synchronize");
            }
        } else {
            check_cuda(
                cudaStreamWaitEvent(
                    inference_stream_.get(), slot.upload_end.get(), 0),
                "inference stream wait upload event");
            check_cuda(
                cudaEventRecord(
                    slot.inference_start.get(), inference_stream_.get()),
                "record asynchronous inference start");
            runtime_.enqueue_device_async(
                static_cast<const float*>(slot.device_model_input.data()),
                static_cast<float*>(slot.device_output.data()),
                inference_stream_.get());
            check_cuda(
                cudaEventRecord(
                    slot.inference_end.get(), inference_stream_.get()),
                "record asynchronous inference end");
            check_cuda(
                cudaStreamWaitEvent(
                    download_stream_.get(), slot.inference_end.get(), 0),
                "download stream wait inference event");
            check_cuda(
                cudaEventRecord(
                    slot.download_start.get(), download_stream_.get()),
                "record asynchronous download start");
            {
                PPE_NVTX_RANGE("d2h");
                check_cuda(
                    cudaMemcpyAsync(
                        slot.host_output.data(), slot.device_output.data(),
                        slot.host_output.bytes(), cudaMemcpyDeviceToHost,
                        download_stream_.get()),
                    "Exp14 pinned D2H");
            }
            check_cuda(
                cudaEventRecord(
                    slot.download_end.get(), download_stream_.get()),
                "record asynchronous download end");
        }
        slot.submit_host_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - slot.submit_start).count();
    }

    Completion wait(Slot& slot) {
        if (!slot.busy) {
            throw std::runtime_error("attempted to wait on a free slot");
        }
        const auto wait_start = Clock::now();
        {
            PPE_NVTX_RANGE("pipeline_wait");
            check_cuda(
                cudaEventSynchronize(slot.download_end.get()),
                "Exp14 terminal event synchronize");
        }
        const auto wait_end = Clock::now();
        check_cuda(cudaGetLastError(), "Exp14 post-pipeline CUDA status");
        Completion result;
        const auto* begin = static_cast<const float*>(slot.host_output.data());
        result.output.assign(begin, begin + runtime_.output_info().elements);
        result.upload_cuda_ms =
            event_elapsed_ms(slot.upload_start, slot.upload_end);
        result.inference_cuda_ms =
            event_elapsed_ms(slot.inference_start, slot.inference_end);
        result.download_cuda_ms =
            event_elapsed_ms(slot.download_start, slot.download_end);
        result.wait_host_ms = std::chrono::duration<double, std::milli>(
            wait_end - wait_start).count();
        return result;
    }

    void release(Slot& slot) {
        if (!slot.busy) {
            throw std::runtime_error("attempted to release a free slot");
        }
        slot.busy = false;
        slot.frame_index = -1;
    }

private:
    ppe::TrtRuntime& runtime_;
    ppe::LetterboxGeometry geometry_;
    Variant variant_;
    std::vector<std::unique_ptr<Slot>> slots_;
    CudaStream upload_stream_;
    CudaStream inference_stream_;
    CudaStream download_stream_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const Variant variant = parse_variant(args.at("--variant"));
        const std::string source_type = args.at("--source-type");
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
                throw std::runtime_error(
                    "video file does not exist: " + source.string());
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
        const auto first_capture_start = Clock::now();
        {
            PPE_NVTX_RANGE("capture");
            if (!capture.read(frame) || frame.empty()) {
                throw std::runtime_error("failed to acquire first frame");
            }
        }
        const auto first_capture_end = Clock::now();
        if (frame.type() != CV_8UC3) {
            throw std::runtime_error("captured frame is not CV_8UC3 BGR");
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }
        const auto geometry =
            ppe::make_letterbox_geometry(frame.cols, frame.rows, 640);
        ppe::TrtRuntime runtime(args.at("--engine"));
        AsyncPipeline pipeline(runtime, geometry, variant);

        for (int index = 0; index < warmup; ++index) {
            Slot& slot = pipeline.slot(0);
            pipeline.submit(slot, frame, -1, 0.0);
            const auto completed = pipeline.wait(slot);
            if (!std::all_of(
                    completed.output.begin(), completed.output.end(),
                    [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("warmup output contains NaN or Inf");
            }
            pipeline.release(slot);
        }

        std::ofstream detections_csv(output_dir / "detections.csv");
        std::ofstream frames_csv(output_dir / "frames.csv");
        if (!detections_csv || !frames_csv) {
            throw std::runtime_error("cannot open CSV outputs");
        }
        detections_csv << "frame_index,detection_index,class_id,class_name,"
                          "confidence,x1,y1,x2,y2\n";
        frames_csv << "frame_index,detection_count,capture_ms,staging_ms,"
                      "submit_host_ms,wait_host_ms,upload_cuda_ms,"
                      "inference_cuda_ms,download_cuda_ms,postprocess_ms,"
                      "end_to_end_ms,slot_index\n";
        detections_csv << std::fixed << std::setprecision(9);
        frames_csv << std::fixed << std::setprecision(9);
        const std::vector<std::string> class_names = {
            "person", "helmet", "safety_vest"};
        std::vector<double> capture_times;
        std::vector<double> staging_times;
        std::vector<double> submit_host_times;
        std::vector<double> wait_host_times;
        std::vector<double> upload_cuda_times;
        std::vector<double> inference_cuda_times;
        std::vector<double> download_cuda_times;
        std::vector<double> postprocess_times;
        std::vector<double> end_to_end_times;
        std::size_t total_detections = 0;
        int submitted_frames = 0;
        int completed_frames = 0;
        double current_capture_ms = std::chrono::duration<double, std::milli>(
            first_capture_end - first_capture_start).count();
        cv::Mat first_annotated;
        cv::Mat last_annotated;
        std::deque<std::size_t> in_flight;

        const auto complete_front = [&]() {
            if (in_flight.empty()) {
                throw std::runtime_error("completion queue is empty");
            }
            const std::size_t slot_index = in_flight.front();
            in_flight.pop_front();
            Slot& slot = pipeline.slot(slot_index);
            PPE_NVTX_RANGE("frame_complete");
            const auto completed = pipeline.wait(slot);
            if (!std::all_of(
                    completed.output.begin(), completed.output.end(),
                    [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("inference output contains NaN or Inf");
            }
            const auto post_start = Clock::now();
            const auto detections = decode_detections(
                completed.output, geometry, confidence, nms_iou);
            const auto post_end = Clock::now();
            const double post_ms = std::chrono::duration<double, std::milli>(
                post_end - post_start).count();
            const double end_to_end_ms = slot.capture_ms +
                std::chrono::duration<double, std::milli>(
                    post_end - slot.submit_start).count();

            {
                PPE_NVTX_RANGE("output");
                for (std::size_t index = 0; index < detections.size(); ++index) {
                    const auto& detection = detections[index];
                    if (detection.class_id < 0 || detection.class_id >= 3 ||
                        detection.confidence < confidence ||
                        detection.confidence > 1.0F ||
                        detection.x1 < 0.0F || detection.y1 < 0.0F ||
                        detection.x2 > geometry.source_width ||
                        detection.y2 > geometry.source_height ||
                        detection.x2 <= detection.x1 ||
                        detection.y2 <= detection.y1) {
                        throw std::runtime_error("invalid decoded detection");
                    }
                    detections_csv << slot.frame_index << ',' << index << ','
                        << detection.class_id << ','
                        << class_names[detection.class_id] << ','
                        << detection.confidence << ',' << detection.x1 << ','
                        << detection.y1 << ',' << detection.x2 << ','
                        << detection.y2 << '\n';
                }
                frames_csv << slot.frame_index << ',' << detections.size() << ','
                    << slot.capture_ms << ',' << slot.staging_ms << ','
                    << slot.submit_host_ms << ',' << completed.wait_host_ms << ','
                    << completed.upload_cuda_ms << ','
                    << completed.inference_cuda_ms << ','
                    << completed.download_cuda_ms << ',' << post_ms << ','
                    << end_to_end_ms << ',' << slot_index << '\n';
                total_detections += detections.size();
                capture_times.push_back(slot.capture_ms);
                staging_times.push_back(slot.staging_ms);
                submit_host_times.push_back(slot.submit_host_ms);
                wait_host_times.push_back(completed.wait_host_ms);
                upload_cuda_times.push_back(completed.upload_cuda_ms);
                inference_cuda_times.push_back(completed.inference_cuda_ms);
                download_cuda_times.push_back(completed.download_cuda_ms);
                postprocess_times.push_back(post_ms);
                end_to_end_times.push_back(end_to_end_ms);
                cv::Mat staged_frame(
                    geometry.source_height, geometry.source_width, CV_8UC3,
                    slot.host_input.data());
                last_annotated = annotate(staged_frame, detections);
                if (slot.frame_index == 0) {
                    first_annotated = last_annotated.clone();
                }
            }
            if (slot.frame_index != completed_frames) {
                throw std::runtime_error("frame completion order mismatch");
            }
            ++completed_frames;
            pipeline.release(slot);
        };

        const auto pipeline_start = Clock::now();
        while (true) {
            if (in_flight.size() == pipeline.capacity()) {
                complete_front();
            }
            const std::size_t slot_index =
                static_cast<std::size_t>(submitted_frames) % pipeline.capacity();
            {
                PPE_NVTX_RANGE("frame_submit");
                pipeline.submit(
                    pipeline.slot(slot_index), frame, submitted_frames,
                    current_capture_ms);
            }
            in_flight.push_back(slot_index);
            ++submitted_frames;
            if (max_frames > 0 && submitted_frames >= max_frames) {
                break;
            }

            const auto capture_start = Clock::now();
            bool acquired = false;
            {
                PPE_NVTX_RANGE("capture");
                acquired = capture.read(frame);
            }
            const auto capture_end = Clock::now();
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
        while (!in_flight.empty()) {
            complete_front();
        }
        const auto pipeline_end = Clock::now();

        if (completed_frames <= 0 || completed_frames != submitted_frames ||
            (max_frames > 0 && completed_frames != max_frames)) {
            throw std::runtime_error("processed frame count failed acceptance");
        }
        detections_csv.close();
        frames_csv.close();
        if (!cv::imwrite(
                (output_dir / "first_annotated.jpg").string(), first_annotated) ||
            !cv::imwrite(
                (output_dir / "last_annotated.jpg").string(), last_annotated)) {
            throw std::runtime_error("cannot write annotated evidence frames");
        }

        const auto capture_stats = summarize(capture_times);
        const auto staging_stats = summarize(staging_times);
        const auto submit_host_stats = summarize(submit_host_times);
        const auto wait_host_stats = summarize(wait_host_times);
        const auto upload_cuda_stats = summarize(upload_cuda_times);
        const auto inference_cuda_stats = summarize(inference_cuda_times);
        const auto download_cuda_stats = summarize(download_cuda_times);
        const auto postprocess_stats = summarize(postprocess_times);
        const auto end_to_end_stats = summarize(end_to_end_times);
        const double elapsed_seconds = std::accumulate(
            end_to_end_times.begin(), end_to_end_times.end(), 0.0) / 1000.0;
        const double effective_fps =
            static_cast<double>(completed_frames) / elapsed_seconds;
        const double pipeline_wall_seconds =
            std::chrono::duration<double>(pipeline_end - pipeline_start).count();
        const double pipeline_wall_fps =
            static_cast<double>(completed_frames) / pipeline_wall_seconds;
        std::ofstream summary(output_dir / "summary.json");
        if (!summary) {
            throw std::runtime_error("cannot open summary.json");
        }
        summary << std::fixed << std::setprecision(9)
                << "{\n"
                << "  \"result\": \"PASS\",\n"
                << "  \"experiment\": \"Exp14\",\n"
                << "  \"variant\": \"" << variant_name(variant) << "\",\n"
                << "  \"host_memory\": \"pinned\",\n"
                << "  \"buffer_count\": " << pipeline.capacity() << ",\n"
                << "  \"source_type\": \"" << source_type << "\",\n"
                << "  \"source\": \"" << json_escape(source_description)
                << "\",\n"
                << "  \"engine\": \"" << json_escape(args.at("--engine"))
                << "\",\n"
                << "  \"frame_width\": " << geometry.source_width << ",\n"
                << "  \"frame_height\": " << geometry.source_height << ",\n"
                << "  \"processed_frames\": " << completed_frames << ",\n"
                << "  \"total_detections\": " << total_detections << ",\n"
                << "  \"warmup_iterations\": " << warmup << ",\n"
                << "  \"confidence_threshold\": " << confidence << ",\n"
                << "  \"nms_iou_threshold\": " << nms_iou << ",\n"
                << "  \"effective_fps\": " << effective_fps << ",\n"
                << "  \"pipeline_wall_seconds\": " << pipeline_wall_seconds
                << ",\n"
                << "  \"pipeline_wall_fps\": " << pipeline_wall_fps << ",\n"
                << "  \"timing_scope\": "
                   "\"capture+pinned_staging+H2D+CUDA_preprocess+TensorRT+"
                   "D2H+NMS\",\n"
                << "  \"timings_ms\": {\n";
        write_stats(summary, "capture", capture_stats); summary << ",\n";
        write_stats(summary, "staging", staging_stats); summary << ",\n";
        write_stats(summary, "submit_host", submit_host_stats); summary << ",\n";
        write_stats(summary, "wait_host", wait_host_stats); summary << ",\n";
        write_stats(summary, "upload_cuda", upload_cuda_stats); summary << ",\n";
        write_stats(summary, "inference_cuda", inference_cuda_stats);
        summary << ",\n";
        write_stats(summary, "download_cuda", download_cuda_stats);
        summary << ",\n";
        write_stats(summary, "postprocess", postprocess_stats); summary << ",\n";
        write_stats(summary, "end_to_end", end_to_end_stats); summary << "\n";
        summary << "  }\n}\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "result=PASS variant=" << variant_name(variant)
                  << " source_type=" << source_type
                  << " frames=" << completed_frames
                  << " detections=" << total_detections
                  << " e2e_p95_ms=" << end_to_end_stats.p95
                  << " pipeline_wall_fps=" << pipeline_wall_fps << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
