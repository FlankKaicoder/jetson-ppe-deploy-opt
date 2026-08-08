#include "trt_runtime.hpp"
#include "ppe_nvtx.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ppe {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<char> read_binary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open engine: " + path);
    }
    const auto end = stream.tellg();
    if (end <= 0) {
        throw std::runtime_error("engine is empty: " + path);
    }
    std::vector<char> data(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("cannot read complete engine: " + path);
    }
    return data;
}

std::vector<int64_t> dimensions(const nvinfer1::Dims& dims) {
    std::vector<int64_t> result;
    result.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error("dynamic or invalid tensor dimension");
        }
        result.push_back(dims.d[index]);
    }
    return result;
}

std::size_t volume(const std::vector<int64_t>& dims) {
    std::size_t result = 1;
    for (const auto value : dims) {
        const auto dimension = static_cast<std::size_t>(value);
        if (result > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor volume overflow");
        }
        result *= dimension;
    }
    return result;
}

std::string dtype_name(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
        default:
            return "UNSUPPORTED";
    }
}

std::string mode_name(nvinfer1::TensorIOMode mode) {
    return mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT";
}

TensorInfo tensor_info(const nvinfer1::ICudaEngine& engine, const char* name) {
    const auto type = engine.getTensorDataType(name);
    if (type != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(
            std::string("only FP32 I/O tensors are supported: ") + name);
    }
    TensorInfo info;
    info.name = name;
    info.dimensions = dimensions(engine.getTensorShape(name));
    info.data_type = dtype_name(type);
    info.mode = mode_name(engine.getTensorIOMode(name));
    info.elements = volume(info.dimensions);
    info.bytes = info.elements * sizeof(float);
    return info;
}

double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) {
        throw std::runtime_error("cannot summarize empty timing list");
    }
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

TimingStats summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("no timing samples");
    }
    std::sort(values.begin(), values.end());
    TimingStats result;
    result.mean_ms =
        std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    result.p50_ms = percentile(values, 0.50);
    result.p95_ms = percentile(values, 0.95);
    result.p99_ms = percentile(values, 0.99);
    result.min_ms = values.front();
    result.max_ms = values.back();
    return result;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes_), "cudaMalloc");
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

class CudaStream {
public:
    CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
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
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
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

}  // namespace

class TrtRuntime::Impl {
public:
    explicit Impl(const std::string& engine_path) {
        const auto serialized = read_binary(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("createInferRuntime failed");
        }
        engine_.reset(runtime_->deserializeCudaEngine(
            serialized.data(), serialized.size()));
        if (!engine_) {
            throw std::runtime_error("deserializeCudaEngine failed");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("createExecutionContext failed");
        }
        if (engine_->getNbIOTensors() != 2) {
            throw std::runtime_error("expected exactly two I/O tensors");
        }
        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (name == nullptr) {
                throw std::runtime_error("null TensorRT tensor name");
            }
            auto info = tensor_info(*engine_, name);
            if (info.mode == "INPUT") {
                if (!input_.name.empty()) {
                    throw std::runtime_error("multiple input tensors");
                }
                input_ = std::move(info);
            } else {
                if (!output_.name.empty()) {
                    throw std::runtime_error("multiple output tensors");
                }
                output_ = std::move(info);
            }
        }
        if (input_.name.empty() || output_.name.empty()) {
            throw std::runtime_error("missing input or output tensor");
        }
        if (input_.dimensions != std::vector<int64_t>({1, 3, 640, 640}) ||
            output_.dimensions != std::vector<int64_t>({1, 7, 8400})) {
            throw std::runtime_error("unexpected frozen engine tensor shapes");
        }
        device_input_ = std::make_unique<DeviceBuffer>(input_.bytes);
        device_output_ = std::make_unique<DeviceBuffer>(output_.bytes);
        if (!context_->setTensorAddress(input_.name.c_str(), device_input_->data()) ||
            !context_->setTensorAddress(output_.name.c_str(), device_output_->data())) {
            throw std::runtime_error("setTensorAddress failed");
        }
    }

    InferenceResult infer(
        const std::vector<float>& input,
        int warmup_iterations,
        int timed_iterations) {
        if (input.size() != input_.elements) {
            throw std::runtime_error("input element count mismatch");
        }
        if (warmup_iterations < 0 || timed_iterations <= 0) {
            throw std::runtime_error("invalid iteration counts");
        }
        std::vector<float> output(output_.elements);
        if (!context_->setTensorAddress(input_.name.c_str(), device_input_->data()) ||
            !context_->setTensorAddress(output_.name.c_str(), device_output_->data())) {
            throw std::runtime_error("setTensorAddress for host inference failed");
        }
        const auto execute_once = [&]() {
            check_cuda(
                cudaMemcpyAsync(
                    device_input_->data(), input.data(), input_.bytes,
                    cudaMemcpyHostToDevice, stream_.get()),
                "cudaMemcpyAsync H2D");
            if (!context_->enqueueV3(stream_.get())) {
                throw std::runtime_error("enqueueV3 returned false");
            }
            check_cuda(
                cudaMemcpyAsync(
                    output.data(), device_output_->data(), output_.bytes,
                    cudaMemcpyDeviceToHost, stream_.get()),
                "cudaMemcpyAsync D2H");
            check_cuda(cudaStreamSynchronize(stream_.get()), "cudaStreamSynchronize");
        };
        for (int index = 0; index < warmup_iterations; ++index) {
            execute_once();
        }

        CudaEvent start_event;
        CudaEvent end_event;
        std::vector<double> host_values;
        std::vector<double> cuda_values;
        host_values.reserve(static_cast<std::size_t>(timed_iterations));
        cuda_values.reserve(static_cast<std::size_t>(timed_iterations));
        for (int index = 0; index < timed_iterations; ++index) {
            const auto host_start = std::chrono::steady_clock::now();
            check_cuda(
                cudaEventRecord(start_event.get(), stream_.get()),
                "cudaEventRecord start");
            check_cuda(
                cudaMemcpyAsync(
                    device_input_->data(), input.data(), input_.bytes,
                    cudaMemcpyHostToDevice, stream_.get()),
                "cudaMemcpyAsync H2D");
            if (!context_->enqueueV3(stream_.get())) {
                throw std::runtime_error("enqueueV3 returned false");
            }
            check_cuda(
                cudaMemcpyAsync(
                    output.data(), device_output_->data(), output_.bytes,
                    cudaMemcpyDeviceToHost, stream_.get()),
                "cudaMemcpyAsync D2H");
            check_cuda(
                cudaEventRecord(end_event.get(), stream_.get()),
                "cudaEventRecord end");
            check_cuda(
                cudaEventSynchronize(end_event.get()),
                "cudaEventSynchronize end");
            const auto host_end = std::chrono::steady_clock::now();
            float cuda_ms = 0.0F;
            check_cuda(
                cudaEventElapsedTime(&cuda_ms, start_event.get(), end_event.get()),
                "cudaEventElapsedTime");
            const auto host_ms = std::chrono::duration<double, std::milli>(
                host_end - host_start).count();
            host_values.push_back(host_ms);
            cuda_values.push_back(static_cast<double>(cuda_ms));
        }
        check_cuda(cudaGetLastError(), "CUDA post-inference status");
        return {std::move(output), summarize(host_values), summarize(cuda_values)};
    }

    DeviceInferenceResult infer_device(
        const float* device_input,
        cudaStream_t stream) {
        if (device_input == nullptr || stream == nullptr) {
            throw std::runtime_error("invalid external device input or CUDA stream");
        }
        std::vector<float> output(output_.elements);
        if (!context_->setTensorAddress(
                input_.name.c_str(), const_cast<float*>(device_input)) ||
            !context_->setTensorAddress(
                output_.name.c_str(), device_output_->data())) {
            throw std::runtime_error("setTensorAddress for device inference failed");
        }
        CudaEvent start_event;
        CudaEvent end_event;
        const auto host_start = std::chrono::steady_clock::now();
        check_cuda(
            cudaEventRecord(start_event.get(), stream),
            "device inference start event");
        {
            PPE_NVTX_RANGE("tensorrt_enqueue");
            if (!context_->enqueueV3(stream)) {
                throw std::runtime_error(
                    "enqueueV3 with external device input returned false");
            }
        }
        {
            PPE_NVTX_RANGE("d2h");
            check_cuda(
                cudaMemcpyAsync(
                    output.data(), device_output_->data(), output_.bytes,
                    cudaMemcpyDeviceToHost, stream),
                "device inference cudaMemcpyAsync D2H");
        }
        check_cuda(
            cudaEventRecord(end_event.get(), stream),
            "device inference end event");
        {
            PPE_NVTX_RANGE("inference_sync");
            check_cuda(
                cudaEventSynchronize(end_event.get()),
                "device inference event synchronize");
        }
        const auto host_end = std::chrono::steady_clock::now();
        float cuda_ms = 0.0F;
        check_cuda(
            cudaEventElapsedTime(
                &cuda_ms, start_event.get(), end_event.get()),
            "device inference elapsed time");
        check_cuda(cudaGetLastError(), "CUDA post-device-inference status");
        return {
            std::move(output),
            std::chrono::duration<double, std::milli>(
                host_end - host_start).count(),
            static_cast<double>(cuda_ms)};
    }

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    TensorInfo input_;
    TensorInfo output_;
    CudaStream stream_;
    std::unique_ptr<DeviceBuffer> device_input_;
    std::unique_ptr<DeviceBuffer> device_output_;
};

TrtRuntime::TrtRuntime(const std::string& engine_path)
    : impl_(std::make_unique<Impl>(engine_path)) {}
TrtRuntime::~TrtRuntime() = default;
TrtRuntime::TrtRuntime(TrtRuntime&&) noexcept = default;
TrtRuntime& TrtRuntime::operator=(TrtRuntime&&) noexcept = default;
const TensorInfo& TrtRuntime::input_info() const { return impl_->input_; }
const TensorInfo& TrtRuntime::output_info() const { return impl_->output_; }
InferenceResult TrtRuntime::infer(
    const std::vector<float>& input,
    int warmup_iterations,
    int timed_iterations) {
    return impl_->infer(input, warmup_iterations, timed_iterations);
}
DeviceInferenceResult TrtRuntime::infer_device(
    const float* device_input,
    cudaStream_t stream) {
    return impl_->infer_device(device_input, stream);
}

}  // namespace ppe
