#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <dlfcn.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

using InitFunction = bool (*)();

template <typename T>
using TrtPtr = std::unique_ptr<T>;

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::runtime_error(
                "usage: exp16_build_engine PLUGIN_SO MODIFIED_ONNX OUTPUT_ENGINE");
        }
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (library == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }
        auto init = reinterpret_cast<InitFunction>(dlsym(library, "ppeInitPlugin"));
        if (init == nullptr || !init()) {
            throw std::runtime_error("plugin initialization failed");
        }

        Logger logger;
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            throw std::runtime_error("createInferBuilder failed");
        }
        TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!network || !config) {
            throw std::runtime_error("TensorRT network/config creation failed");
        }
        TrtPtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, logger));
        if (!parser || !parser->parseFromFile(
                argv[2], static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO))) {
            throw std::runtime_error("ONNX parse failed");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        config->setBuilderOptimizationLevel(3);
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30U);
        TrtPtr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            throw std::runtime_error("buildSerializedNetwork failed");
        }
        std::ofstream output(argv[3], std::ios::binary);
        if (!output.write(
                static_cast<char const*>(serialized->data()),
                static_cast<std::streamsize>(serialized->size()))) {
            throw std::runtime_error("engine write failed");
        }
        std::cout << "engine_build=PASS\nbytes=" << serialized->size() << '\n';
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "engine_build=FAIL\nerror=" << error.what() << '\n';
        return 1;
    }
}
