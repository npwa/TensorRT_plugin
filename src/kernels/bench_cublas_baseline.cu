// Phase 2 step 3: "dequantize the full weight matrix, then call cuBLAS" baseline.
// cuBLAS is column-major; our data is row-major, so the transpose bookkeeping is easy
// to get subtly wrong -- this program checks its own answer against the already-proven
// optimized kernel (dequant_gemm_optimized.cu) on the same random input before trusting
// any timing number, rather than assuming the cuBLAS call is correct just because it
// compiles and runs.
#include "dequant_gemm_optimized.cuh"
#include "dequant_only.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cublas_v2.h>

namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main() {
    const int M = 1, N = 16384, K = 3072, group_size = 128;

    std::vector<float> h_x(M * K);
    std::vector<float> h_scale(N * (K / group_size));
    std::vector<uint8_t> h_packed(N * (K / 2));
    srand(0);
    for (auto& v : h_x) v = (rand() % 2000 - 1000) / 1000.0f;
    for (auto& v : h_scale) v = (rand() % 1000) / 10000.0f + 0.001f;
    for (auto& v : h_packed) v = static_cast<uint8_t>(rand() % 256);

    float *d_x, *d_scale, *d_out_fused, *d_out_cublas, *d_w_dense;
    uint8_t* d_packed;
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_packed, h_packed.size() * sizeof(uint8_t));
    cudaMalloc(&d_scale, h_scale.size() * sizeof(float));
    cudaMalloc(&d_out_fused, (size_t)M * N * sizeof(float));
    cudaMalloc(&d_out_cublas, (size_t)M * N * sizeof(float));
    cudaMalloc(&d_w_dense, (size_t)N * K * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packed, h_packed.data(), h_packed.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    // --- correctness check: fused kernel vs dequant-then-cublasSgemv, M=1 only ---
    // W is stored row-major [N,K]; that byte layout equals a column-major [K,N] matrix
    // (i.e. W^T). Declaring it to cuBLAS as (m=K, n=N, lda=K) with CUBLAS_OP_T gives
    // (W^T)^T @ x = W @ x -- see src/kernels/bench_cublas_baseline.cu's derivation
    // note in Doc/implementation_plan.md Phase 2, checked numerically below, not just
    // derived on paper.
    launch_dequant_gemm_optimized(d_x, d_packed, d_scale, d_out_fused, M, N, K, group_size, 0);
    launch_dequant_only(d_packed, d_scale, d_w_dense, N, K, group_size, 0);
    cublasSgemv(handle, CUBLAS_OP_T, K, N, &alpha, d_w_dense, K, d_x, 1, &beta, d_out_cublas, 1);
    cudaDeviceSynchronize();

    std::vector<float> h_out_fused(N), h_out_cublas(N);
    cudaMemcpy(h_out_fused.data(), d_out_fused, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_cublas.data(), d_out_cublas, N * sizeof(float), cudaMemcpyDeviceToHost);

    double max_abs_diff = 0.0, max_val = 0.0;
    for (int i = 0; i < N; i++) {
        max_abs_diff = std::max(max_abs_diff, (double)std::abs(h_out_fused[i] - h_out_cublas[i]));
        max_val = std::max(max_val, (double)std::abs(h_out_fused[i]));
    }
    const double rel_err = max_abs_diff / std::max(max_val, 1e-8);
    printf("cuBLAS-path vs fused-kernel: max_abs_diff=%.6f rel_err=%.6f -> %s\n",
           max_abs_diff, rel_err, rel_err < 1e-3 ? "MATCH" : "MISMATCH");
    if (rel_err >= 1e-3) {
        printf("ABORTING: cuBLAS baseline does not agree with the validated kernel; not reporting timings.\n");
        return 1;
    }

    // --- benchmark: dequant-then-cublasSgemv, full pipeline timed together ---
    for (int i = 0; i < 3; i++) {
        launch_dequant_only(d_packed, d_scale, d_w_dense, N, K, group_size, 0);
        cublasSgemv(handle, CUBLAS_OP_T, K, N, &alpha, d_w_dense, K, d_x, 1, &beta, d_out_cublas, 1);
    }
    cudaDeviceSynchronize();

    const int n_runs = 50;
    std::vector<double> timings_ms(n_runs);
    for (int i = 0; i < n_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_dequant_only(d_packed, d_scale, d_w_dense, N, K, group_size, 0);
        cublasSgemv(handle, CUBLAS_OP_T, K, N, &alpha, d_w_dense, K, d_x, 1, &beta, d_out_cublas, 1);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        timings_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    printf("dequant-then-cuBLAS baseline: M=%d N=%d K=%d -> median %.4f ms/call over %d runs\n",
           M, N, K, median(timings_ms), n_runs);

    cublasDestroy(handle);
    cudaFree(d_x); cudaFree(d_packed); cudaFree(d_scale);
    cudaFree(d_out_fused); cudaFree(d_out_cublas); cudaFree(d_w_dense);
    return 0;
}
