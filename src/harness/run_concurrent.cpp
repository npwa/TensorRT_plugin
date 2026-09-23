// Phase 4 steps 3-4: concurrency and timing. Each worker owns its own IExecutionContext
// and cudaStream_t (TensorRT engines are safe to share read-only across threads, but a
// single IExecutionContext is not safe for concurrent enqueue calls from multiple
// threads) plus its own pinned host buffers, so H2D copy, compute, and D2H copy for one
// worker's request can genuinely overlap with another worker's -- not just "multiple
// threads call the same function" on shared state.
#include "engine_harness.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using trt_quant_plugin::EngineHarness;
using Clock = std::chrono::high_resolution_clock;

namespace {

#define CUDA_CHECK(expr)                                                                   \
    do {                                                                                    \
        cudaError_t err_ = (expr);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err_));                                              \
            exit(1);                                                                        \
        }                                                                                    \
    } while (0)

struct Worker {
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream{};
    float* h_x = nullptr;
    float* h_out = nullptr;
    float* d_x = nullptr;
    float* d_out = nullptr;
    int64_t M, K, N;

    explicit Worker(const EngineHarness& harness) {
        context = harness.createContext();
        const auto shape = harness.shape();
        M = shape.M;
        K = shape.K;
        N = shape.N;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMallocHost(&h_x, M * K * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_out, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(float)));
        for (int64_t i = 0; i < M * K; ++i) h_x[i] = 0.01f;  // content is irrelevant for a timing-only workload
        context->setTensorAddress(EngineHarness::kInputName, d_x);
        context->setTensorAddress(EngineHarness::kOutputName, d_out);
    }

    Worker(const Worker&) = delete;

    ~Worker() {
        cudaFreeHost(h_x);
        cudaFreeHost(h_out);
        cudaFree(d_x);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
    }

    // No sync inside -- callers control when (or whether) to wait, so independent
    // requests on independent streams can be outstanding at the same time.
    void submitOnce() {
        cudaMemcpyAsync(d_x, h_x, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV3(stream);
        cudaMemcpyAsync(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    void sync() { cudaStreamSynchronize(stream); }
};

double medianOfTimedRuns(Worker& w, int n_warmup, int n_runs) {
    for (int i = 0; i < n_warmup; ++i) {
        w.submitOnce();
        w.sync();
    }
    std::vector<double> timings_ms(n_runs);
    for (int i = 0; i < n_runs; ++i) {
        const auto start = Clock::now();
        w.submitOnce();
        w.sync();
        timings_ms[i] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    std::sort(timings_ms.begin(), timings_ms.end());
    return timings_ms[n_runs / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::string plan_path = argc > 1 ? argv[1] : "build/plugin_test_vectors/engine.plan";
    const int num_workers = argc > 2 ? std::atoi(argv[2]) : 4;
    const int requests_per_worker = argc > 3 ? std::atoi(argv[3]) : 50;

    EngineHarness harness(plan_path);
    const auto shape = harness.shape();
    printf("loaded engine: M=%ld K=%ld N=%ld\n", (long)shape.M, (long)shape.K, (long)shape.N);

    // Step 4: warmup + median-of-N single-request timing, same methodology as
    // evol_inference.fitness.measure_latency_ms (warmup runs, then N individually-timed
    // and individually-synchronized runs, median rather than mean) and this project's
    // own kernel-level benchmarks (bench_naive.cu / bench_optimized.cu).
    Worker solo(harness);
    const double solo_median_ms = medianOfTimedRuns(solo, /*n_warmup=*/3, /*n_runs=*/50);
    const double serial_throughput_req_s = 1000.0 / solo_median_ms;
    printf("single-stream baseline: median %.4f ms/request over 50 runs (%.1f req/s if fully serialized)\n",
           solo_median_ms, serial_throughput_req_s);

    // Step 3: num_workers independent streams, each with its own context + pinned
    // buffers, each submitting requests_per_worker requests back-to-back without
    // synchronizing between them -- only synchronizing once, after all are submitted.
    std::vector<std::unique_ptr<Worker>> workers;
    for (int i = 0; i < num_workers; ++i) workers.push_back(std::make_unique<Worker>(harness));
    for (auto& w : workers) {  // warmup
        w->submitOnce();
        w->sync();
    }

    const auto t0 = Clock::now();
    {
        std::vector<std::jthread> threads;
        for (int i = 0; i < num_workers; ++i) {
            threads.emplace_back([&workers, i, requests_per_worker] {
                for (int r = 0; r < requests_per_worker; ++r) workers[i]->submitOnce();
                workers[i]->sync();
            });
        }
    }  // jthreads join here
    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    const int64_t total_requests = static_cast<int64_t>(num_workers) * requests_per_worker;
    const double aggregate_throughput_req_s = total_requests / (total_ms / 1000.0);
    const double overlap_speedup = aggregate_throughput_req_s / serial_throughput_req_s;
    printf("concurrent (%d streams x %d requests): %.2f ms total, %.1f req/s aggregate\n", num_workers,
           requests_per_worker, total_ms, aggregate_throughput_req_s);
    printf("overlap speedup over fully-serial single-stream throughput: %.2fx\n", overlap_speedup);

    return 0;
}
