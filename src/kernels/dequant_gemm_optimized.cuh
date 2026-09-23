#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Phase 2 steps 1-2: optimized kernel, built from an ncu-measured diagnosis of the
// naive kernel (src/kernels/dequant_gemm_naive.cu), not a blind rewrite.
//
// Finding: the naive kernel indexes threads by output row `n`, but the weight matrix
// is stored row-major -- so adjacent threads in a warp (adjacent `n`) read addresses
// `K/2` bytes apart simultaneously. Measured: 86% of memory sectors wasted (only 3 of
// 32 bytes/sector used), ~46% estimated speedup available from fixing this alone.
//
// Fix, standard for a batch-1 (GEMV-shaped) workload: one warp per output row. Lanes
// within a warp read *consecutive* bytes of that row (properly coalesced), vectorized
// as uint32_t (4 packed bytes = 8 weight values per load), each lane accumulates a
// partial dot product, then a warp shuffle reduces to the final scalar. The activation
// vector `x` (reused by every output row) is cached once per thread block in shared
// memory rather than re-read from global memory per warp.
//
// Same interface contract as the naive kernel: same packing/grouping layout
// (python/reference/quant.py), same fp32-throughout precision choice.
void launch_dequant_gemm_optimized(
    const float* x,
    const uint8_t* packed,
    const float* scale,
    float* out,
    int M, int N, int K, int group_size,
    cudaStream_t stream);
