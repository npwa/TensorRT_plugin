#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// The "obvious alternative" to a fused dequant+GEMM kernel: unpack INT4 -> a dense fp32
// weight matrix in global memory, then let a vendor GEMM library do the matmul
// (bench_cublas_baseline.cu). Not fused, not optimized -- it exists purely as the
// comparison baseline for Phase 2 step 3, so it should be a plain, unremarkable
// implementation, not a competitor to the fused kernel in its own right.
//
//   packed: [N, K/2]              uint8
//   scale:  [N, K/group_size]     fp32
//   w_out:  [N, K]                fp32, dense dequantized output
void launch_dequant_only(
    const uint8_t* packed,
    const float* scale,
    float* w_out,
    int N, int K, int group_size,
    cudaStream_t stream);
