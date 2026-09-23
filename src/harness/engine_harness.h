#pragma once
// Phase 4: production-style C++ harness around a serialized TensorRT engine (produced by
// src/plugin/validate_plugin_engine.cpp once it's confirmed correct against the Python
// reference). Loads the plan, registers the plugin creator the deserializer needs to
// resolve the DequantGemmInt4 layer inside it, and hands out execution contexts.
//
// RAII note: plain std::unique_ptr for IRuntime/ICudaEngine/IExecutionContext, no custom
// deleter -- Phase 0 confirmed TensorRT 11.3.0.99's objects have plain virtual
// destructors, not the legacy ->destroy() pattern some older TensorRT major versions
// used (Doc/implementation_plan.md, Phase 0 completion note).
#include "trt_logger.h"

#include <NvInfer.h>
// NvInferRuntime.h (pulled in via trt_logger.h) only drags in the plain-C
// cuda_runtime_api.h, which lacks the templated cudaMalloc<T>(T**, size_t) host-side
// convenience overloads -- include the full header so callers can write
// cudaMalloc(&typed_ptr, ...) without a manual reinterpret_cast<void**>.
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace trt_quant_plugin {

struct EngineShape {
    int64_t M, K, N;
};

class EngineHarness {
public:
    // Loads the plan file, creates the IRuntime, deserializes the engine. Throws
    // std::runtime_error on any failure (missing file, deserialize failure) rather than
    // returning a null/half-built object.
    explicit EngineHarness(const std::string& plan_path);

    // The engine is safe to share read-only across threads (TensorRT's own guarantee);
    // each thread must create and use its own IExecutionContext, never share one.
    std::unique_ptr<nvinfer1::IExecutionContext> createContext() const;

    // Queried from the engine's bindings, not hardcoded -- the plugin's static-shapes-v1
    // scope decision (Phase 3 step 4) means this is fixed once the plan was built, but
    // the harness shouldn't assume which shape without checking.
    EngineShape shape() const { return mShape; }

    static constexpr const char* kInputName = "x";
    static constexpr const char* kOutputName = "out";

private:
    StderrLogger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    EngineShape mShape{};
};

}  // namespace trt_quant_plugin
