#pragma once
// Phase 3: IPluginV3 wrapper around the Phase 2 optimized kernel
// (src/kernels/dequant_gemm_optimized.cu). Weights (packed INT4 + per-group scale) are
// baked in as plugin attributes at network-build time, not runtime tensor inputs -- this
// is a weight-only quantized linear layer, so the only runtime input is the activation
// `x` [M, K] and the only output is `out` [M, N]. N, K and group_size are fixed once the
// plugin is created (Phase 3 step 4: static shapes for v1).
//
// API surface (class names, method signatures, capability-split design) taken directly
// from the installed TensorRT 11.3.0.99 headers
// (/usr/include/x86_64-linux-gnu/NvInferRuntime.h, NvInferPluginBase.h) rather than
// assumed -- the implementation_plan's Phase 0 note already flagged this API as having
// shifted across TensorRT versions.

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

namespace trt_quant_plugin {

// One IPluginV3 object implementing all three capability interfaces itself (the "One"
// pattern TensorRT's own headers name and expect), rather than three separate objects.
class DequantGemmPluginV3 : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime {
public:
    // Takes ownership of nothing; copies host data into owned buffers and uploads to
    // freshly cudaMalloc'd device buffers shared (via refcount) with any clone.
    DequantGemmPluginV3(int32_t N, int32_t K, int32_t group_size,
                         const uint8_t* packed_host, size_t packed_bytes,
                         const float* scale_host, size_t scale_count);

    // Cheap-clone constructor: shares existing host/device buffers (refcounted) instead
    // of re-uploading. Used by clone() so attachToContext()/clone() don't repeat a
    // cudaMalloc+cudaMemcpy of the (potentially large) weight tensors on every context.
    DequantGemmPluginV3(int32_t N, int32_t K, int32_t group_size,
                         std::shared_ptr<std::vector<uint8_t>> host_packed,
                         std::shared_ptr<std::vector<float>> host_scale,
                         std::shared_ptr<uint8_t> dev_packed,
                         std::shared_ptr<float> dev_scale,
                         std::string plugin_namespace);

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    const char* getPluginName() const noexcept override { return kPluginName; }
    const char* getPluginVersion() const noexcept override { return kPluginVersion; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // IPluginV3OneBuild
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                             nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                                const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                             nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                             nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
                             nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
                                    int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getNbOutputs() const noexcept override { return 1; }

    // IPluginV3OneRuntime
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
                           nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
                     void const* const* inputs, void* const* outputs, void* workspace,
                     cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    static constexpr const char* kPluginName = "DequantGemmInt4";
    static constexpr const char* kPluginVersion = "1";

private:
    int32_t mN, mK, mGroupSize;
    std::shared_ptr<std::vector<uint8_t>> mHostPacked;
    std::shared_ptr<std::vector<float>> mHostScale;
    std::shared_ptr<uint8_t> mDevPacked;  // cudaMalloc'd, custom cudaFree deleter
    std::shared_ptr<float> mDevScale;     // cudaMalloc'd, custom cudaFree deleter
    std::string mNamespace;

    // Backing storage for getFieldsToSerialize()'s returned PluginFieldCollection --
    // must stay alive and pointer-stable as long as this plugin object does. Built once
    // per-instance (not copied from another instance) so its pointers reference this
    // object's own members, never a clone source's.
    void buildSerializeFields();
    std::vector<nvinfer1::PluginField> mSerializeFields;
    nvinfer1::PluginFieldCollection mSerializeFC{};
};

class DequantGemmPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    DequantGemmPluginCreator();

    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc,
                                       nvinfer1::TensorRTPhase phase) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    char const* getPluginName() const noexcept override { return DequantGemmPluginV3::kPluginName; }
    char const* getPluginVersion() const noexcept override { return DequantGemmPluginV3::kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFC{};
};

}  // namespace trt_quant_plugin
