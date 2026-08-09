#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

std::vector<char> read_binary(std::filesystem::path const& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open: " + path.string());
    }
    auto const end = stream.tellg();
    if (end <= 0) {
        throw std::runtime_error("empty file: " + path.string());
    }
    std::vector<char> data(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!stream.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("incomplete read: " + path.string());
    }
    return data;
}

void write_binary(
    std::filesystem::path const& path, void const* data, std::size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.write(
            static_cast<char const*>(data), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("write failed: " + path.string());
    }
}

std::size_t element_size(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kINT32) {
        return 4;
    }
    throw std::runtime_error("unsupported tensor data type");
}

std::size_t tensor_bytes(nvinfer1::ICudaEngine const& engine, char const* name) {
    nvinfer1::Dims const dims = engine.getTensorShape(name);
    std::size_t elements = 1;
    for (int32_t index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error(std::string("dynamic tensor: ") + name);
        }
        elements *= static_cast<std::size_t>(dims.d[index]);
    }
    return elements * element_size(engine.getTensorDataType(name));
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    void* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }

private:
    void* data_{nullptr};
    std::size_t bytes_{};
};

using InitFunction = bool (*)();

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            throw std::runtime_error(
                "usage: exp16_deserialize_infer PLUGIN_SO ENGINE INPUT_FP32 OUTPUT_DIR");
        }
        std::filesystem::path const output_dir(argv[4]);
        if (std::filesystem::exists(output_dir)) {
            throw std::runtime_error("output directory already exists");
        }
        std::filesystem::create_directories(output_dir);

        void* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (library == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }
        auto init = reinterpret_cast<InitFunction>(dlsym(library, "ppeInitPlugin"));
        if (init == nullptr || !init()) {
            throw std::runtime_error("plugin initialization failed");
        }

        auto const serialized = read_binary(argv[2]);
        Logger logger;
        std::unique_ptr<nvinfer1::IRuntime> runtime(
            nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("createInferRuntime failed");
        }
        std::unique_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (!engine) {
            throw std::runtime_error("deserializeCudaEngine failed");
        }
        std::unique_ptr<nvinfer1::IExecutionContext> context(
            engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("createExecutionContext failed");
        }

        std::unordered_map<std::string, std::unique_ptr<DeviceBuffer>> buffers;
        for (int32_t index = 0; index < engine->getNbIOTensors(); ++index) {
            char const* name = engine->getIOTensorName(index);
            if (name == nullptr) {
                throw std::runtime_error("null tensor name");
            }
            auto buffer = std::make_unique<DeviceBuffer>(tensor_bytes(*engine, name));
            if (!context->setTensorAddress(name, buffer->data())) {
                throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
            }
            buffers.emplace(name, std::move(buffer));
        }
        for (char const* required : {"boxes_scores", "classes", "indices", "count"}) {
            if (buffers.find(required) == buffers.end()) {
                throw std::runtime_error(std::string("missing tensor: ") + required);
            }
        }

        std::string input_name;
        for (int32_t index = 0; index < engine->getNbIOTensors(); ++index) {
            char const* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name.empty()) {
                    throw std::runtime_error("multiple input tensors");
                }
                input_name = name;
            }
        }
        if (input_name != "images" && input_name != "output0") {
            throw std::runtime_error("unexpected input tensor: " + input_name);
        }
        bool const has_diagnostic_raw = input_name == "images" &&
            buffers.find("output0") != buffers.end() &&
            engine->getTensorIOMode("output0") == nvinfer1::TensorIOMode::kOUTPUT;

        auto const input = read_binary(argv[3]);
        if (input.size() != buffers.at(input_name)->bytes()) {
            throw std::runtime_error("input byte count mismatch");
        }
        cudaStream_t stream{};
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cuda(cudaMemcpyAsync(
            buffers.at(input_name)->data(), input.data(), input.size(),
            cudaMemcpyHostToDevice, stream), "input H2D");
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("enqueueV3 returned false");
        }
        int32_t count = -1;
        check_cuda(cudaMemcpyAsync(
            &count, buffers.at("count")->data(), sizeof(count),
            cudaMemcpyDeviceToHost, stream), "count D2H");
        check_cuda(cudaStreamSynchronize(stream), "count synchronize");
        if (count < 0 || count > 8400) {
            throw std::runtime_error("invalid candidate count");
        }

        std::vector<float> boxes(static_cast<std::size_t>(count) * 5);
        std::vector<int32_t> classes(static_cast<std::size_t>(count));
        std::vector<int32_t> indices(static_cast<std::size_t>(count));
        if (count > 0) {
            check_cuda(cudaMemcpyAsync(
                boxes.data(), buffers.at("boxes_scores")->data(),
                boxes.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "boxes D2H");
            check_cuda(cudaMemcpyAsync(
                classes.data(), buffers.at("classes")->data(),
                classes.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
                "classes D2H");
            check_cuda(cudaMemcpyAsync(
                indices.data(), buffers.at("indices")->data(),
                indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
                "indices D2H");
            check_cuda(cudaStreamSynchronize(stream), "payload synchronize");
        }
        std::vector<float> diagnostic_raw;
        if (has_diagnostic_raw) {
            diagnostic_raw.resize(7U * 8400U);
            check_cuda(cudaMemcpyAsync(
                diagnostic_raw.data(), buffers.at("output0")->data(),
                diagnostic_raw.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "diagnostic raw D2H");
            check_cuda(cudaStreamSynchronize(stream), "diagnostic raw synchronize");
        }
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        write_binary(output_dir / "count.bin", &count, sizeof(count));
        write_binary(output_dir / "boxes_scores.bin", boxes.data(), boxes.size() * sizeof(float));
        write_binary(output_dir / "classes.bin", classes.data(), classes.size() * sizeof(int32_t));
        write_binary(output_dir / "indices.bin", indices.data(), indices.size() * sizeof(int32_t));
        if (has_diagnostic_raw) {
            write_binary(
                output_dir / "raw_output0.bin", diagnostic_raw.data(),
                diagnostic_raw.size() * sizeof(float));
        }
        std::ofstream summary(output_dir / "summary.json");
        summary << "{\n  \"status\": \"PASS\",\n  \"count\": " << count
                << ",\n  \"payload_bytes\": "
                << (boxes.size() * sizeof(float) + classes.size() * sizeof(int32_t) +
                    indices.size() * sizeof(int32_t) + sizeof(count))
                << ",\n  \"diagnostic_raw_output\": "
                << (has_diagnostic_raw ? "true" : "false") << "\n}\n";
        std::cout << "deserialize_infer=PASS\ncount=" << count << '\n';
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "deserialize_infer=FAIL\nerror=" << error.what() << '\n';
        return 1;
    }
}
