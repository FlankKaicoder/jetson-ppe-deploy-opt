#include "yolo_decode_plugin.hpp"

#include <NvInferRuntime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace ppe::exp16 {
namespace {

using namespace nvinfer1;

class YoloDecodePlugin final : public IPluginV3,
                               public IPluginV3OneCore,
                               public IPluginV3OneBuild,
                               public IPluginV3OneRuntime {
public:
    explicit YoloDecodePlugin(float confidence_threshold)
        : confidence_threshold_(confidence_threshold) {
        initialize_fields();
    }

    YoloDecodePlugin(YoloDecodePlugin const& other)
        : confidence_threshold_(other.confidence_threshold_) {
        initialize_fields();
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) {
            return static_cast<IPluginV3OneCore*>(this);
        }
        if (type == PluginCapabilityType::kBUILD) {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME) {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override {
        try {
            return new YoloDecodePlugin(*this);
        } catch (...) {
            return nullptr;
        }
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    int32_t getNbOutputs() const noexcept override { return 4; }

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* in, int32_t nb_inputs,
        DynamicPluginTensorDesc const*, int32_t nb_outputs) noexcept override {
        if (in == nullptr || nb_inputs != 1 || nb_outputs != 4) {
            return -1;
        }
        Dims const& dims = in[0].desc.dims;
        return dims.nbDims == 3 && dims.d[0] == 1 &&
                dims.d[1] == kInputChannels && dims.d[2] == kCandidateCount
            ? 0
            : -1;
    }

    bool supportsFormatCombination(
        int32_t position, DynamicPluginTensorDesc const* in_out,
        int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (in_out == nullptr || nb_inputs != 1 || nb_outputs != 4 ||
            position < 0 || position >= 5) {
            return false;
        }
        DataType const expected = position <= 1 ? DataType::kFLOAT : DataType::kINT32;
        return in_out[position].desc.type == expected &&
            in_out[position].desc.format == PluginFormat::kLINEAR;
    }

    int32_t getOutputDataTypes(
        DataType* output_types, int32_t nb_outputs,
        DataType const* input_types, int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 4 ||
            nb_inputs != 1 || input_types[0] != DataType::kFLOAT) {
            return -1;
        }
        output_types[0] = DataType::kFLOAT;
        output_types[1] = DataType::kINT32;
        output_types[2] = DataType::kINT32;
        output_types[3] = DataType::kINT32;
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const*, int32_t nb_inputs, DimsExprs const*, int32_t nb_shape_inputs,
        DimsExprs* outputs, int32_t nb_outputs, IExprBuilder& builder) noexcept override {
        if (outputs == nullptr || nb_inputs != 1 || nb_shape_inputs != 0 || nb_outputs != 4) {
            return -1;
        }
        outputs[0].nbDims = 3;
        outputs[0].d[0] = builder.constant(1);
        outputs[0].d[1] = builder.constant(kCandidateCount);
        outputs[0].d[2] = builder.constant(5);
        for (int output = 1; output <= 2; ++output) {
            outputs[output].nbDims = 2;
            outputs[output].d[0] = builder.constant(1);
            outputs[output].d[1] = builder.constant(kCandidateCount);
        }
        outputs[3].nbDims = 1;
        outputs[3].d[0] = builder.constant(1);
        return 0;
    }

    std::size_t getWorkspaceSize(
        DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        return plugin_workspace_size();
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nb_inputs,
        PluginTensorDesc const*, int32_t nb_outputs) noexcept override {
        if (in == nullptr || nb_inputs != 1 || nb_outputs != 4) {
            return -1;
        }
        Dims const& dims = in[0].dims;
        return in[0].type == DataType::kFLOAT && dims.nbDims == 3 &&
                dims.d[0] == 1 && dims.d[1] == kInputChannels &&
                dims.d[2] == kCandidateCount
            ? 0
            : -1;
    }

    int32_t enqueue(
        PluginTensorDesc const*, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        if (inputs == nullptr || outputs == nullptr) {
            return -1;
        }
        return launch_decode_compact(
            static_cast<float const*>(inputs[0]),
            static_cast<float*>(outputs[0]),
            static_cast<int32_t*>(outputs[1]),
            static_cast<int32_t*>(outputs[2]),
            static_cast<int32_t*>(outputs[3]),
            workspace, plugin_workspace_size(), confidence_threshold_, stream);
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &fields_to_serialize_;
    }

private:
    void initialize_fields() {
        serialized_fields_.clear();
        serialized_fields_.emplace_back(
            "confidence_threshold", &confidence_threshold_,
            PluginFieldType::kFLOAT32, 1);
        fields_to_serialize_.nbFields = static_cast<int32_t>(serialized_fields_.size());
        fields_to_serialize_.fields = serialized_fields_.data();
    }

    float confidence_threshold_{0.25F};
    std::vector<PluginField> serialized_fields_;
    PluginFieldCollection fields_to_serialize_{};
};

class YoloDecodePluginCreator final : public IPluginCreatorV3One {
public:
    YoloDecodePluginCreator() {
        advertised_fields_.emplace_back(
            "confidence_threshold", nullptr, PluginFieldType::kFLOAT32, 1);
        fields_.nbFields = static_cast<int32_t>(advertised_fields_.size());
        fields_.fields = advertised_fields_.data();
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return kPluginNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }

    IPluginV3* createPlugin(
        char const*, PluginFieldCollection const* fields,
        TensorRTPhase) noexcept override {
        try {
            float threshold = 0.25F;
            if (fields != nullptr) {
                for (int32_t index = 0; index < fields->nbFields; ++index) {
                    PluginField const& field = fields->fields[index];
                    if (field.name != nullptr &&
                        std::strcmp(field.name, "confidence_threshold") == 0) {
                        if (field.data == nullptr || field.type != PluginFieldType::kFLOAT32 ||
                            field.length != 1) {
                            return nullptr;
                        }
                        threshold = *static_cast<float const*>(field.data);
                    }
                }
            }
            if (!std::isfinite(threshold) || threshold < 0.0F || threshold > 1.0F) {
                return nullptr;
            }
            return new YoloDecodePlugin(threshold);
        } catch (...) {
            return nullptr;
        }
    }

private:
    std::vector<PluginField> advertised_fields_;
    PluginFieldCollection fields_{};
};

YoloDecodePluginCreator g_creator;
std::once_flag g_registration_once;
bool g_registration_result = false;

}  // namespace
}  // namespace ppe::exp16

extern "C" __attribute__((visibility("default"))) bool ppeInitPlugin() {
    std::call_once(ppe::exp16::g_registration_once, [] {
        ppe::exp16::g_registration_result =
            getPluginRegistry()->registerCreator(
                ppe::exp16::g_creator, ppe::exp16::kPluginNamespace);
    });
    return ppe::exp16::g_registration_result;
}
