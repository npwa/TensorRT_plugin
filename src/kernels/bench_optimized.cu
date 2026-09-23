// Phase 2 steps 2-3: standalone timing driver for the optimized kernel, same shape and
// methodology as bench_naive.cu so the two numbers are directly comparable.
#include "dequant_gemm_optimized.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    const int M = 1, N = 16384, K = 3072, group_size = 128;

    std::vector<float> h_x(M * K);
    std::vector<float> h_scale(N * (K / group_size));
    std::vector<uint8_t> h_packed(N * (K / 2));

    srand(0);
    for (auto& v : h_x) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto& v : h_scale) v = (rand() % 1000) / 10000.0f + 0.001f;
    for (auto& v : h_packed) v = static_cast<uint8_t>(rand() % 256);

    float *d_x, *d_scale, *d_out;
    uint8_t* d_packed;
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_packed, h_packed.size() * sizeof(uint8_t));
    cudaMalloc(&d_scale, h_scale.size() * sizeof(float));
    cudaMalloc(&d_out, (size_t)M * N * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packed, h_packed.data(), h_packed.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 3; i++) launch_dequant_gemm_optimized(d_x, d_packed, d_scale, d_out, M, N, K, group_size, 0);
    cudaDeviceSynchronize();

    const int n_runs = 50;
    std::vector<double> timings_ms(n_runs);
    for (int i = 0; i < n_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_dequant_gemm_optimized(d_x, d_packed, d_scale, d_out, M, N, K, group_size, 0);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        timings_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::sort(timings_ms.begin(), timings_ms.end());
    double median_ms = timings_ms[n_runs / 2];

    printf("optimized kernel: M=%d N=%d K=%d group_size=%d -> median %.4f ms/call over %d runs\n",
           M, N, K, group_size, median_ms, n_runs);

    cudaFree(d_x);
    cudaFree(d_packed);
    cudaFree(d_scale);
    cudaFree(d_out);
    return 0;
}
