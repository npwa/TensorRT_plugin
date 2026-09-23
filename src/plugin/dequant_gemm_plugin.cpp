#include "dequant_gemm_plugin.h"
#include "dequant_gemm_optimized.cuh"

#include <stdexcept>

using namespace nvinfer1;

namespace trt_quant_plugin {

namespace {

std::shared_ptr<uint8_t> uploadPacked(const uint8_t* host, size_t bytes) {
    uint8_t* dev = nullptr;
    if (cudaMalloc(&dev, bytes) != cudaSuccess) {
        throw std::runtime_error("DequantGemmPluginV3: cudaMalloc(packed) failed");
    }
    if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        throw std::runtime_error("DequantGemmPluginV3: cudaMemcpy(packed) failed");
    }
    return std::shared_ptr<uint8_t>(dev, [](uint8_t* p) { cudaFree(p); });
}

std::shared_ptr<float> uploadScale(const float* host, size_t count) {
    float* dev = nullptr;
    const size_t bytes = count * sizeof(float);
    if (cudaMalloc(&dev, bytes) != cudaSuccess) {
        throw std::runtime_error("DequantGemmPluginV3: cudaMalloc(scale) failed");
    }
    if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        throw std::runtime_error("DequantGemmPluginV3: cudaMemcpy(scale) failed");
    }
    return std::shared_ptr<float>(dev, [](float* p) { cudaFree(p); });
}

}  // namespace

DequantGemmPluginV3::DequantGemmPluginV3(int32_t N, int32_t K, int32_t group_size,
                                          const uint8_t* packed_host, size_t packed_bytes,
                                          const float* scale_host, size_t scale_count)
    : DequantGemmPluginV3(N, K, group_size,
                           std::make_shared<std::vector<uint8_t>>(packed_host, packed_host + packed_bytes),
                           std::make_shared<std::vector<float>>(scale_host, scale_host + scale_count),
                           uploadPacked(packed_host, packed_bytes),
                           uploadScale(scale_host, scale_count),
                           std::string("")) {}

DequantGemmPluginV3::DequantGemmPluginV3(int32_t N, int32_t K, int32_t group_size,
                                          std::shared_ptr<std::vector<uint8_t>> host_packed,
                                          std::shared_ptr<std::vector<float>> host_scale,
                                          std::shared_ptr<uint8_t> dev_packed,
                                          std::shared_ptr<float> dev_scale,
                                          std::string plugin_namespace)
    : mN(N),
      mK(K),
      mGroupSize(group_size),
      mHostPacked(std::move(host_packed)),
      mHostScale(std::move(host_scale)),
      mDevPacked(std::move(dev_packed)),
      mDevScale(std::move(dev_scale)),
      mNamespace(std::move(plugin_namespace)) {
    buildSerializeFields();
}

void DequantGemmPluginV3::buildSerializeFields() {
    mSerializeFields.clear();
    mSerializeFields.push_back(PluginField("N", &mN, PluginFieldType::kINT32, 1));
    mSerializeFields.push_back(PluginField("K", &mK, PluginFieldType::kINT32, 1));
    mSerializeFields.push_back(PluginField("group_size", &mGroupSize, PluginFieldType::kINT32, 1));
    mSerializeFields.push_back(PluginField(
        "packed", mHostPacked->data(), PluginFieldType::kINT8, static_cast<int32_t>(mHostPacked->size())));
    mSerializeFields.push_back(PluginField(
        "scale", mHostScale->data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mHostScale->size())));
    mSerializeFC.nbFields = static_cast<int32_t>(mSerializeFields.size());
    mSerializeFC.fields = mSerializeFields.data();
}

IPluginCapability* DequantGemmPluginV3::getCapabilityInterface(PluginCapabilityType type) noexcept {
    switch (type) {
        case PluginCapabilityType::kCORE:
            return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:
            return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:
            return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

IPluginV3* DequantGemmPluginV3::clone() noexcept {
    try {
        return new DequantGemmPluginV3(mN, mK, mGroupSize, mHostPacked, mHostScale, mDevPacked, mDevScale, mNamespace);
    } catch (...) {
        return nullptr;
    }
}

int32_t DequantGemmPluginV3::configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                                              int32_t) noexcept {
    return 0;
}

int32_t DequantGemmPluginV3::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, const DataType*,
                                                 int32_t) const noexcept {
    if (nbOutputs != 1) return -1;
    outputTypes[0] = DataType::kFLOAT;
    return 0;
}

int32_t DequantGemmPluginV3::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t,
                                              DimsExprs* outputs, int32_t nbOutputs,
                                              IExprBuilder& exprBuilder) noexcept {
    if (nbInputs != 1 || nbOutputs != 1) return -1;
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];             // M, carried through from the activation input
    outputs[0].d[1] = exprBuilder.constant(mN);   // N is fixed at plugin-creation time
    return 0;
}

bool DequantGemmPluginV3::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t,
                                                     int32_t) noexcept {
    // Single input (activation), single output, both fp32 linear -- the naive/optimized
    // kernels were written and validated fp32-only (Phase 1 note: deliberately, to avoid
    // __half-handling complexity during the correctness-first phase).
    return inOut[pos].desc.format == TensorFormat::kLINEAR && inOut[pos].desc.type == DataType::kFLOAT;
}

int32_t DequantGemmPluginV3::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const*,
                                            int32_t) noexcept {
    if (nbInputs != 1) return -1;
    if (in[0].dims.nbDims != 2 || in[0].dims.d[1] != mK) return -1;
    return 0;
}

int32_t DequantGemmPluginV3::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
                                      void const* const* inputs, void* const* outputs, void*,
                                      cudaStream_t stream) noexcept {
    const int32_t M = inputDesc[0].dims.d[0];
    launch_dequant_gemm_optimized(static_cast<const float*>(inputs[0]), mDevPacked.get(), mDevScale.get(),
                                   static_cast<float*>(outputs[0]), M, mN, mK, mGroupSize, stream);
    return 0;
}

IPluginV3* DequantGemmPluginV3::attachToContext(IPluginResourceContext*) noexcept {
    // No per-context resource is needed (the weight buffers are read-only and already
    // shared across clones via shared_ptr) -- a full clone satisfies the contract.
    return clone();
}

PluginFieldCollection const* DequantGemmPluginV3::getFieldsToSerialize() noexcept { return &mSerializeFC; }

DequantGemmPluginCreator::DequantGemmPluginCreator() {
    mFields.push_back(PluginField("N", nullptr, PluginFieldType::kINT32, 1));
    mFields.push_back(PluginField("K", nullptr, PluginFieldType::kINT32, 1));
    mFields.push_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));
    mFields.push_back(PluginField("packed", nullptr, PluginFieldType::kINT8, 0));
    mFields.push_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

IPluginV3* DequantGemmPluginCreator::createPlugin(char const*, PluginFieldCollection const* fc,
                                                   TensorRTPhase) noexcept {
    try {
        int32_t N = 0, K = 0, group_size = 0;
        const uint8_t* packed = nullptr;
        size_t packed_len = 0;
        const float* scale = nullptr;
        size_t scale_len = 0;
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const PluginField& f = fc->fields[i];
            const std::string name(f.name);
            if (name == "N") {
                N = *static_cast<const int32_t*>(f.data);
            } else if (name == "K") {
                K = *static_cast<const int32_t*>(f.data);
            } else if (name == "group_size") {
                group_size = *static_cast<const int32_t*>(f.data);
            } else if (name == "packed") {
                packed = static_cast<const uint8_t*>(f.data);
                packed_len = static_cast<size_t>(f.length);
            } else if (name == "scale") {
                scale = static_cast<const float*>(f.data);
                scale_len = static_cast<size_t>(f.length);
            }
        }
        if (N <= 0 || K <= 0 || group_size <= 0 || packed == nullptr || scale == nullptr) {
            return nullptr;
        }
        return new DequantGemmPluginV3(N, K, group_size, packed, packed_len, scale, scale_len);
    } catch (...) {
        return nullptr;
    }
}

}  // namespace trt_quant_plugin
