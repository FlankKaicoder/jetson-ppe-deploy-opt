#include <NvInfer.h>

#include <dlfcn.h>

#include <cstdint>
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

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error(
                "usage: exp16_build_fixture_engine PLUGIN_SO OUTPUT_ENGINE");
        }
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (library == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }
        auto init = reinterpret_cast<InitFunction>(dlsym(library, "ppeInitPlugin"));
        if (init == nullptr || !init()) {
            throw std::runtime_error("plugin initialization failed");
        }
        auto* creator_interface = getPluginRegistry()->getCreator(
            "PpeYoloDecodeCompact", "1", "com.flankkaicoder.ppe");
        auto* creator = dynamic_cast<nvinfer1::IPluginCreatorV3One*>(creator_interface);
        if (creator == nullptr) {
            throw std::runtime_error("IPluginCreatorV3One lookup failed");
        }
        float threshold = 0.25F;
        nvinfer1::PluginField field(
            "confidence_threshold", &threshold,
            nvinfer1::PluginFieldType::kFLOAT32, 1);
        nvinfer1::PluginFieldCollection fields{1, &field};
        std::unique_ptr<nvinfer1::IPluginV3> plugin(creator->createPlugin(
            "exp16_postprocess", &fields, nvinfer1::TensorRTPhase::kBUILD));
        if (!plugin) {
            throw std::runtime_error("createPlugin build phase failed");
        }

        Logger logger;
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        std::unique_ptr<nvinfer1::INetworkDefinition> network(
            builder ? builder->createNetworkV2(0U) : nullptr);
        std::unique_ptr<nvinfer1::IBuilderConfig> config(
            builder ? builder->createBuilderConfig() : nullptr);
        if (!builder || !network || !config) {
            throw std::runtime_error("builder/network/config creation failed");
        }
        nvinfer1::ITensor* input = network->addInput(
            "output0", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims3{1, 7, 8400});
        if (input == nullptr) {
            throw std::runtime_error("addInput failed");
        }
        nvinfer1::ITensor* inputs[]{input};
        nvinfer1::IPluginV3Layer* layer = network->addPluginV3(inputs, 1, nullptr, 0, *plugin);
        if (layer == nullptr || layer->getNbOutputs() != 4) {
            throw std::runtime_error("addPluginV3 failed");
        }
        char const* names[]{"boxes_scores", "classes", "indices", "count"};
        for (int32_t index = 0; index < 4; ++index) {
            layer->getOutput(index)->setName(names[index]);
            network->markOutput(*layer->getOutput(index));
        }
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, 64ULL << 20U);
        std::unique_ptr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            throw std::runtime_error("fixture buildSerializedNetwork failed");
        }
        std::ofstream output(argv[2], std::ios::binary);
        if (!output.write(
                static_cast<char const*>(serialized->data()),
                static_cast<std::streamsize>(serialized->size()))) {
            throw std::runtime_error("fixture engine write failed");
        }
        std::cout << "fixture_engine_build=PASS\nbytes=" << serialized->size() << '\n';
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "fixture_engine_build=FAIL\nerror=" << error.what() << '\n';
        return 1;
    }
}
