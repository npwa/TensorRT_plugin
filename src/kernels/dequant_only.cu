#include "dequant_only.cuh"

namespace {

__global__ void dequant_only_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    float* __restrict__ w_out,
    int N, int K, int group_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // flat index into [N, K]
    if (idx >= N * K) return;
    const int n = idx / K;
    const int k = idx % K;

    const int half_k = K / 2;
    const int n_groups = K / group_size;
    const uint8_t byte = packed[(size_t)n * half_k + k / 2];
    const int nibble = (k % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
    const int q = (nibble >= 8) ? (nibble - 16) : nibble;
    w_out[idx] = q * scale[(size_t)n * n_groups + k / group_size];
}

}  // namespace

void launch_dequant_only(const uint8_t* packed, const float* scale, float* w_out, int N, int K, int group_size, cudaStream_t stream) {
    const int total = N * K;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    dequant_only_kernel<<<grid, block, 0, stream>>>(packed, scale, w_out, N, K, group_size);
}
