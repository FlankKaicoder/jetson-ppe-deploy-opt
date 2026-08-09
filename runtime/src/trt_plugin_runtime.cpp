#include "trt_plugin_runtime.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppe {
namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

void check_cuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<char> read_binary(std::string const& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open engine: " + path);
    }
    auto const end = stream.tellg();
    if (end <= 0) {
        throw std::runtime_error("engine is empty: " + path);
    }
    std::vector<char> data(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!stream.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("cannot read engine: " + path);
    }
    return data;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc Plugin output");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    void* data() const { return data_; }
private:
    void* data_{nullptr};
};

class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) {
        check_cuda(cudaHostAlloc(&data_, bytes, cudaHostAllocDefault), "cudaHostAlloc Plugin output");
    }
    ~PinnedBuffer() { cudaFreeHost(data_); }
    void* data() const { return data_; }
private:
    void* data_{nullptr};
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(event_); }
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_{};
};

double elapsed(Event const& start, Event const& end) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

bool shape_equals(nvinfer1::Dims const& dims, std::initializer_list<int32_t> expected) {
    if (dims.nbDims != static_cast<int32_t>(expected.size())) {
        return false;
    }
    int32_t index = 0;
    for (int32_t value : expected) {
        if (dims.d[index++] != value) {
            return false;
        }
    }
    return true;
}

using InitFunction = bool (*)();

}  // namespace

class TrtPluginRuntime::Impl {
public:
    Impl(std::string const& engine_path, std::string const& plugin_path,
         LetterboxGeometry geometry)
        : geometry_(geometry), boxes_device_(8400U * 5U * sizeof(float)),
          classes_device_(8400U * sizeof(int32_t)),
          indices_device_(8400U * sizeof(int32_t)), count_device_(sizeof(int32_t)),
          boxes_host_(8400U * 5U * sizeof(float)),
          classes_host_(8400U * sizeof(int32_t)),
          indices_host_(8400U * sizeof(int32_t)), count_host_(sizeof(int32_t)) {
        library_ = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (library_ == nullptr) {
            throw std::runtime_error(std::string("dlopen Plugin failed: ") + dlerror());
        }
        auto init = reinterpret_cast<InitFunction>(dlsym(library_, "ppeInitPlugin"));
        if (init == nullptr || !init()) {
            throw std::runtime_error("Plugin registration failed");
        }
        auto const serialized = read_binary(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_ ? runtime_->deserializeCudaEngine(
            serialized.data(), serialized.size()) : nullptr);
        context_.reset(engine_ ? engine_->createExecutionContext() : nullptr);
        if (!runtime_ || !engine_ || !context_) {
            throw std::runtime_error("Plugin Engine deserialize/context failed");
        }
        validate_contract();
        if (!context_->setTensorAddress("boxes_scores", boxes_device_.data()) ||
            !context_->setTensorAddress("classes", classes_device_.data()) ||
            !context_->setTensorAddress("indices", indices_device_.data()) ||
            !context_->setTensorAddress("count", count_device_.data())) {
            throw std::runtime_error("Plugin output setTensorAddress failed");
        }
    }

    ~Impl() {
        context_.reset();
        engine_.reset();
        runtime_.reset();
        if (library_ != nullptr) {
            dlclose(library_);
        }
    }

    PluginFrameResult process(
        float const* input,
        cudaStream_t stream,
        std::vector<GpuCandidate>* network_candidates) {
        if (input == nullptr || stream == nullptr ||
            !context_->setTensorAddress("images", const_cast<float*>(input))) {
            throw std::runtime_error("Plugin input setTensorAddress failed");
        }
        auto const host_start = std::chrono::steady_clock::now();
        check_cuda(cudaEventRecord(inference_start_.get(), stream), "Plugin inference start");
        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error("Plugin enqueueV3 returned false");
        }
        check_cuda(cudaEventRecord(inference_end_.get(), stream), "Plugin inference end");
        check_cuda(cudaEventRecord(count_start_.get(), stream), "Plugin count copy start");
        check_cuda(cudaMemcpyAsync(count_host_.data(), count_device_.data(), sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream), "Plugin count D2H");
        check_cuda(cudaEventRecord(count_end_.get(), stream), "Plugin count copy end");
        auto const count_wait = std::chrono::steady_clock::now();
        check_cuda(cudaEventSynchronize(count_end_.get()), "Plugin count synchronize");
        double const count_sync = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - count_wait).count();
        int32_t const count = *static_cast<int32_t const*>(count_host_.data());
        if (count < 0 || count > 8400) {
            throw std::runtime_error("Plugin count out of range");
        }

        std::size_t const count_size = static_cast<std::size_t>(count);
        check_cuda(cudaEventRecord(payload_start_.get(), stream), "Plugin payload copy start");
        if (count > 0) {
            check_cuda(cudaMemcpyAsync(boxes_host_.data(), boxes_device_.data(),
                count_size * 5U * sizeof(float), cudaMemcpyDeviceToHost, stream), "Plugin boxes D2H");
            check_cuda(cudaMemcpyAsync(classes_host_.data(), classes_device_.data(),
                count_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream), "Plugin classes D2H");
            check_cuda(cudaMemcpyAsync(indices_host_.data(), indices_device_.data(),
                count_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream), "Plugin indices D2H");
        }
        check_cuda(cudaEventRecord(payload_end_.get(), stream), "Plugin payload copy end");
        auto const payload_wait = std::chrono::steady_clock::now();
        check_cuda(cudaEventSynchronize(payload_end_.get()), "Plugin payload synchronize");
        double const payload_sync = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - payload_wait).count();

        auto const inverse_start = std::chrono::steady_clock::now();
        auto const* boxes = static_cast<float const*>(boxes_host_.data());
        auto const* classes = static_cast<int32_t const*>(classes_host_.data());
        auto const* indices = static_cast<int32_t const*>(indices_host_.data());
        if (network_candidates != nullptr) {
            network_candidates->clear();
            network_candidates->reserve(count_size);
        }
        std::vector<GpuCandidate> candidates;
        candidates.reserve(count_size);
        for (int32_t position = 0; position < count; ++position) {
            float const* box = boxes + static_cast<std::size_t>(position) * 5U;
            GpuCandidate candidate{};
            candidate.candidate_index = indices[position];
            candidate.class_id = classes[position];
            candidate.confidence = box[4];
            candidate.x1 = std::clamp((box[0] - geometry_.padding_left) / geometry_.ratio,
                                      0.0F, static_cast<float>(geometry_.source_width));
            candidate.y1 = std::clamp((box[1] - geometry_.padding_top) / geometry_.ratio,
                                      0.0F, static_cast<float>(geometry_.source_height));
            candidate.x2 = std::clamp((box[2] - geometry_.padding_left) / geometry_.ratio,
                                      0.0F, static_cast<float>(geometry_.source_width));
            candidate.y2 = std::clamp((box[3] - geometry_.padding_top) / geometry_.ratio,
                                      0.0F, static_cast<float>(geometry_.source_height));
            if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1) {
                if (network_candidates != nullptr) {
                    network_candidates->push_back({
                        candidate.candidate_index,
                        candidate.class_id,
                        candidate.confidence,
                        box[0], box[1], box[2], box[3]});
                }
                candidates.push_back(candidate);
            }
        }
        double const inverse_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inverse_start).count();

        PluginFrameResult result;
        result.candidates = std::move(candidates);
        result.plugin_count = count;
        result.d2h_bytes = sizeof(int32_t) + count_size * 28U;
        result.host_total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_start).count();
        result.inference_cuda_ms = elapsed(inference_start_, inference_end_);
        result.count_copy_cuda_ms = elapsed(count_start_, count_end_);
        result.count_sync_host_ms = count_sync;
        result.candidate_copy_cuda_ms = elapsed(payload_start_, payload_end_);
        result.candidate_sync_host_ms = payload_sync;
        result.cpu_inverse_letterbox_ms = inverse_ms;
        return result;
    }

    void set_geometry(LetterboxGeometry geometry) {
        if (geometry.source_width <= 0 || geometry.source_height <= 0 ||
            geometry.target_size <= 0 || geometry.ratio <= 0.0F) {
            throw std::runtime_error("invalid Plugin letterbox geometry");
        }
        geometry_ = geometry;
    }

private:
    void validate_contract() {
        if (engine_->getNbIOTensors() != 5 ||
            engine_->getTensorDataType("images") != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType("boxes_scores") != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType("classes") != nvinfer1::DataType::kINT32 ||
            engine_->getTensorDataType("indices") != nvinfer1::DataType::kINT32 ||
            engine_->getTensorDataType("count") != nvinfer1::DataType::kINT32 ||
            !shape_equals(engine_->getTensorShape("images"), {1, 3, 640, 640}) ||
            !shape_equals(engine_->getTensorShape("boxes_scores"), {1, 8400, 5}) ||
            !shape_equals(engine_->getTensorShape("classes"), {1, 8400}) ||
            !shape_equals(engine_->getTensorShape("indices"), {1, 8400}) ||
            !shape_equals(engine_->getTensorShape("count"), {1})) {
            throw std::runtime_error("Plugin Engine ABI mismatch");
        }
    }

    LetterboxGeometry geometry_;
    void* library_{nullptr};
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    DeviceBuffer boxes_device_, classes_device_, indices_device_, count_device_;
    PinnedBuffer boxes_host_, classes_host_, indices_host_, count_host_;
    Event inference_start_, inference_end_, count_start_, count_end_,
          payload_start_, payload_end_;
};

TrtPluginRuntime::TrtPluginRuntime(
    std::string const& engine_path, std::string const& plugin_path,
    LetterboxGeometry geometry)
    : impl_(std::make_unique<Impl>(engine_path, plugin_path, geometry)) {}
TrtPluginRuntime::~TrtPluginRuntime() = default;
TrtPluginRuntime::TrtPluginRuntime(TrtPluginRuntime&&) noexcept = default;
TrtPluginRuntime& TrtPluginRuntime::operator=(TrtPluginRuntime&&) noexcept = default;
PluginFrameResult TrtPluginRuntime::process(
    float const* input,
    cudaStream_t stream,
    std::vector<GpuCandidate>* network_candidates) {
    return impl_->process(input, stream, network_candidates);
}
void TrtPluginRuntime::set_geometry(LetterboxGeometry geometry) {
    impl_->set_geometry(geometry);
}

}  // namespace ppe
