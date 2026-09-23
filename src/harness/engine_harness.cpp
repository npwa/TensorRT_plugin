#include "engine_harness.h"
#include "dequant_gemm_plugin.h"

#include <fstream>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;

namespace trt_quant_plugin {

namespace {

// Registered once per process (function-local static init is thread-safe in C++11) --
// the deserializer looks this creator up by (name, version, namespace) to resolve the
// DequantGemmInt4 layer serialized inside the plan.
void ensurePluginRegistered() {
    static DequantGemmPluginCreator creator;
    static const bool registered = getPluginRegistry()->registerCreator(creator, "");
    (void)registered;
}

std::vector<char> readPlanFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("EngineHarness: cannot open plan file " + path);
    const std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(size));
    f.read(buf.data(), size);
    return buf;
}

}  // namespace

EngineHarness::EngineHarness(const std::string& plan_path) {
    ensurePluginRegistered();

    const auto plan = readPlanFile(plan_path);
    mRuntime.reset(createInferRuntime(mLogger));
    if (!mRuntime) throw std::runtime_error("EngineHarness: createInferRuntime failed");
    mEngine.reset(mRuntime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine) throw std::runtime_error("EngineHarness: deserializeCudaEngine failed");

    const Dims in_dims = mEngine->getTensorShape(kInputName);
    const Dims out_dims = mEngine->getTensorShape(kOutputName);
    if (in_dims.nbDims != 2 || out_dims.nbDims != 2) {
        throw std::runtime_error("EngineHarness: unexpected tensor rank in loaded plan");
    }
    mShape = EngineShape{in_dims.d[0], in_dims.d[1], out_dims.d[1]};
}

std::unique_ptr<IExecutionContext> EngineHarness::createContext() const {
    std::unique_ptr<IExecutionContext> context(mEngine->createExecutionContext());
    if (!context) throw std::runtime_error("EngineHarness: createExecutionContext failed");
    return context;
}

}  // namespace trt_quant_plugin
