// Phase 3 step 3: build a minimal single-layer TensorRT engine around
// DequantGemmPluginV3, run it, and check its output against Python-generated reference
// vectors (python/reference/gen_plugin_test_vectors.py) -- the numerical ground truth
// every earlier phase (naive kernel, optimized kernel) was already validated against, so
// a match here confirms the plugin wrapping didn't change behavior, not just that it runs.
#include "dequant_gemm_plugin.h"

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;
using trt_quant_plugin::DequantGemmPluginCreator;

namespace {

class StderrLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    const std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

template <typename T>
std::vector<T> readBinary(const std::string& path) {
    auto raw = readFile(path);
    std::vector<T> out(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

struct Meta { int64_t M, N, K, group_size; };

Meta readMeta(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    Meta meta{};
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string key;
        if (!std::getline(iss, key, '=')) continue;
        std::string value;
        std::getline(iss, value);
        if (key == "M") meta.M = std::stoll(value);
        else if (key == "N") meta.N = std::stoll(value);
        else if (key == "K") meta.K = std::stoll(value);
        else if (key == "group_size") meta.group_size = std::stoll(value);
    }
    return meta;
}

#define CUDA_CHECK(expr)                                                                     \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                    cudaGetErrorString(err_));                                                \
            exit(1);                                                                          \
        }                                                                                      \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "build/plugin_test_vectors";
    const Meta meta = readMeta(dir + "/meta.txt");
    printf("loaded test vectors: M=%ld N=%ld K=%ld group_size=%ld\n", (long)meta.M, (long)meta.N, (long)meta.K,
           (long)meta.group_size);

    auto h_x = readBinary<float>(dir + "/x.bin");
    auto h_packed = readBinary<uint8_t>(dir + "/packed.bin");
    auto h_scale = readBinary<float>(dir + "/scale.bin");
    auto h_reference = readBinary<float>(dir + "/reference.bin");

    StderrLogger logger;

    DequantGemmPluginCreator creator;
    getPluginRegistry()->registerCreator(creator, "");

    const int32_t N32 = static_cast<int32_t>(meta.N);
    const int32_t K32 = static_cast<int32_t>(meta.K);
    const int32_t group_size32 = static_cast<int32_t>(meta.group_size);
    std::vector<PluginField> fields;
    fields.push_back(PluginField("N", &N32, PluginFieldType::kINT32, 1));
    fields.push_back(PluginField("K", &K32, PluginFieldType::kINT32, 1));
    fields.push_back(PluginField("group_size", &group_size32, PluginFieldType::kINT32, 1));
    fields.push_back(PluginField("packed", h_packed.data(), PluginFieldType::kINT8,
                                  static_cast<int32_t>(h_packed.size())));
    fields.push_back(PluginField("scale", h_scale.data(), PluginFieldType::kFLOAT32,
                                  static_cast<int32_t>(h_scale.size())));
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};

    IPluginV3* plugin = creator.createPlugin("dequant_gemm_0", &fc, TensorRTPhase::kBUILD);
    if (!plugin) { fprintf(stderr, "createPlugin failed\n"); return 1; }

    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));

    ITensor* x = network->addInput("x", DataType::kFLOAT, Dims{2, {meta.M, meta.K}});
    ITensor* inputs[] = {x};
    IPluginV3Layer* layer = network->addPluginV3(inputs, 1, nullptr, 0, *plugin);
    if (!layer) { fprintf(stderr, "addPluginV3 failed\n"); return 1; }
    ITensor* out = layer->getOutput(0);
    out->setName("out");
    network->markOutput(*out);

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) { fprintf(stderr, "buildSerializedNetwork failed\n"); return 1; }

    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine) { fprintf(stderr, "deserializeCudaEngine failed\n"); return 1; }
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    if (!context) { fprintf(stderr, "createExecutionContext failed\n"); return 1; }

    float *d_x = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, meta.M * meta.N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    context->setTensorAddress("x", d_x);
    context->setTensorAddress("out", d_out);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (!context->enqueueV3(stream)) { fprintf(stderr, "enqueueV3 failed\n"); return 1; }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_out(meta.M * meta.N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, (double)std::abs(h_out[i] - h_reference[i]));
        max_ref = std::max(max_ref, (double)std::abs(h_reference[i]));
    }
    const double rel_err = max_abs_diff / std::max(max_ref, 1e-8);
    const bool pass = max_abs_diff < 1e-3 + 1e-3 * max_ref;
    printf("engine vs Python reference: max_abs_diff=%.6f rel_err=%.6f -> %s\n", max_abs_diff, rel_err,
           pass ? "PASS" : "FAIL");

    cudaStreamDestroy(stream);
    cudaFree(d_x);
    cudaFree(d_out);
    delete plugin;  // user-created (kBUILD phase): ours to delete, per IPluginCreatorV3One::createPlugin's contract

    return pass ? 0 : 1;
}
