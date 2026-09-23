#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Phase 1 step 4: naive INT4 groupwise dequant + GEMM. One thread per output element,
// correctness over performance -- Phase 2 is where the memory-access pattern (tiling,
// coalesced/vectorized loads) gets optimized, not this file.
//
// Deliberately fp32 throughout (x, scale, and accumulation), not fp16: half-precision
// arithmetic in raw CUDA C++ needs __half-specific handling that would obscure the
// naive/correctness-first version; that's a Phase 2 concern, not a Phase 1 one.
//
// Packing/grouping layout must exactly match python/reference/quant.py's
// pack_int4/quantize_groupwise_int4 -- this is the kernel's ground truth, and the two
// are validated against each other directly (tests/test_dequant_gemm_naive_kernel.py).
//
//   x:      [M, K]         row-major, fp32
//   packed: [N, K/2]       row-major, uint8 -- two signed 4-bit values per byte
//                          (low nibble = even k, high nibble = odd k, within each row)
//   scale:  [N, K/group_size]  row-major, fp32 -- one scale per (output row, group)
//   out:    [M, N]         row-major, fp32
void launch_dequant_gemm_naive(
    const float* x,
    const uint8_t* packed,
    const float* scale,
    float* out,
    int M, int N, int K, int group_size,
    cudaStream_t stream);
