#include "dequant_gemm_naive.cuh"

namespace {

__global__ void dequant_gemm_naive_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int M, int N, int K, int group_size)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const int half_k = K / 2;
    const int n_groups = K / group_size;
    const uint8_t* row_packed = packed + (size_t)n * half_k;
    const float* row_scale = scale + (size_t)n * n_groups;
    const float* row_x = x + (size_t)m * K;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const uint8_t byte = row_packed[k / 2];
        const int nibble = (k % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
        const int q = (nibble >= 8) ? (nibble - 16) : nibble;  // sign-extend 4-bit -> int
        const float w = q * row_scale[k / group_size];
        acc += row_x[k] * w;
    }
    out[(size_t)m * N + n] = acc;
}

}  // namespace

void launch_dequant_gemm_naive(
    const float* x, const uint8_t* packed, const float* scale, float* out,
    int M, int N, int K, int group_size, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    dequant_gemm_naive_kernel<<<grid, block, 0, stream>>>(x, packed, scale, out, M, N, K, group_size);
}
