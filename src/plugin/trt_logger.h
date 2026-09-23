#pragma once
// Shared between the Phase 3 engine-build/validate driver and the Phase 4 harness --
// both need to construct an IRuntime/IBuilder, both need some ILogger.
#include <NvInferRuntime.h>

#include <cstdio>

namespace trt_quant_plugin {

class StderrLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};

}  // namespace trt_quant_plugin
