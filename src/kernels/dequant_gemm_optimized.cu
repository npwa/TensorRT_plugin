#include "dequant_gemm_optimized.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block

__global__ void dequant_gemm_optimized_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int M, int N, int K, int group_size)
{
    extern __shared__ float x_shared[];  // [K] -- one activation row, shared by every warp in this block

    const int m = blockIdx.y;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;

    // Cooperative, coalesced load of x[m, :] into shared memory -- every thread in the
    // block participates regardless of whether its warp's `n` is in range, since all
    // threads must reach __syncthreads() uniformly.
    const float* x_row = x + (size_t)m * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        x_shared[k] = x_row[k];
    }
    __syncthreads();

    if (n >= N) return;

    const int half_k = K / 2;
    const int n_groups = K / group_size;
    const uint8_t* row_packed = packed + (size_t)n * half_k;
    const float* row_scale = scale + (size_t)n * n_groups;

    // half_k is 1536 or 4096 for every real Phi-3 shape (K = 3072 or 8192) -- both
    // divisible by 4, so the vectorized path below covers every group/lane exactly,
    // no scalar remainder loop needed.
    const int half_k_vec4 = half_k / 4;
    const uint32_t* row_packed_vec = reinterpret_cast<const uint32_t*>(row_packed);

    float partial = 0.0f;
    for (int chunk = lane; chunk < half_k_vec4; chunk += kWarpSize) {
        const uint32_t four_bytes = row_packed_vec[chunk];  // 4 packed bytes = 8 weight values
        const int k0 = chunk * 8;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint8_t byte = (four_bytes >> (b * 8)) & 0xFF;
            const int nib_lo = byte & 0xF;
            const int nib_hi = (byte >> 4) & 0xF;
            const int q_lo = (nib_lo >= 8) ? (nib_lo - 16) : nib_lo;
            const int q_hi = (nib_hi >= 8) ? (nib_hi - 16) : nib_hi;
            const int k_lo = k0 + b * 2;
            const int k_hi = k_lo + 1;
            const float w_lo = q_lo * row_scale[k_lo / group_size];
            const float w_hi = q_hi * row_scale[k_hi / group_size];
            partial += x_shared[k_lo] * w_lo + x_shared[k_hi] * w_hi;
        }
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFFu, partial, offset);
    }

    if (lane == 0) {
        out[(size_t)m * N + n] = partial;
    }
}

}  // namespace

void launch_dequant_gemm_optimized(
    const float* x, const uint8_t* packed, const float* scale, float* out,
    int M, int N, int K, int group_size, cudaStream_t stream)
{
    const dim3 block(kWarpsPerBlock * kWarpSize);
    const dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock, M);
    const size_t shared_bytes = (size_t)K * sizeof(float);
    dequant_gemm_optimized_kernel<<<grid, block, shared_bytes, stream>>>(x, packed, scale, out, M, N, K, group_size);
}
