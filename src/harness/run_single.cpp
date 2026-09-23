// Phase 4 step 2: single-threaded correctness pass. Load the serialized engine
// (produced by src/plugin/validate_plugin_engine.cpp), run one inference through the
// harness, diff against the same reference.bin Phase 3 validated the in-process-built
// engine against. Get this right before any concurrency is added (run_concurrent.cpp).
#include "engine_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;
using trt_quant_plugin::EngineHarness;

namespace {

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(size));
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

#define CUDA_CHECK(expr)                                                                   \
    do {                                                                                    \
        cudaError_t err_ = (expr);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err_));                                              \
            exit(1);                                                                        \
        }                                                                                    \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    const std::string plan_path = argc > 1 ? argv[1] : "build/plugin_test_vectors/engine.plan";
    const std::string vectors_dir = argc > 2 ? argv[2] : "build/plugin_test_vectors";

    EngineHarness harness(plan_path);
    const auto shape = harness.shape();
    printf("loaded engine: M=%ld K=%ld N=%ld\n", (long)shape.M, (long)shape.K, (long)shape.N);

    auto h_x = readBinary<float>(vectors_dir + "/x.bin");
    auto h_reference = readBinary<float>(vectors_dir + "/reference.bin");
    if ((int64_t)h_x.size() != shape.M * shape.K || (int64_t)h_reference.size() != shape.M * shape.N) {
        fprintf(stderr, "test vectors in %s don't match the loaded engine's shape\n", vectors_dir.c_str());
        return 1;
    }

    auto context = harness.createContext();

    float *d_x = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, shape.M * shape.N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    context->setTensorAddress(EngineHarness::kInputName, d_x);
    context->setTensorAddress(EngineHarness::kOutputName, d_out);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (!context->enqueueV3(stream)) { fprintf(stderr, "enqueueV3 failed\n"); return 1; }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_out(shape.M * shape.N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, (double)std::abs(h_out[i] - h_reference[i]));
        max_ref = std::max(max_ref, (double)std::abs(h_reference[i]));
    }
    const double rel_err = max_abs_diff / std::max(max_ref, 1e-8);
    const bool pass = max_abs_diff < 1e-3 + 1e-3 * max_ref;
    printf("harness (single-threaded) vs Python reference: max_abs_diff=%.6f rel_err=%.6f -> %s\n", max_abs_diff,
           rel_err, pass ? "PASS" : "FAIL");

    cudaStreamDestroy(stream);
    cudaFree(d_x);
    cudaFree(d_out);
    return pass ? 0 : 1;
}
