// Thin torch-extension wrapper around the pure CUDA kernel (dequant_gemm_naive.cu/.cuh
// have no torch dependency, and never will -- Phase 3's TensorRT plugin links the pure
// kernel directly). This file exists purely so Phase 1 can validate the kernel against
// python/reference/quant.py numerically in the same test process, instead of a manual
// serialize-to-file round trip through a separate executable. Phase 5's planned
// pybind11 kernel-benchmark wrapper is effectively this file, hardened; this is that
// task pulled forward out of necessity, not a scope change.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "dequant_gemm_naive.cuh"
#include "dequant_gemm_optimized.cuh"

namespace {

void check_inputs(const torch::Tensor& x, const torch::Tensor& packed, const torch::Tensor& scale, int64_t group_size) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scale.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && scale.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.dim() == 2 && packed.dim() == 2 && scale.dim() == 2, "expected 2D tensors");
    const int64_t K = x.size(1);
    TORCH_CHECK(packed.size(1) * 2 == K, "packed.size(1)*2 must equal x.size(1) (K)");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(scale.size(0) == packed.size(0) && scale.size(1) == K / group_size, "scale shape mismatch");
}

}  // namespace

torch::Tensor dequant_gemm_naive(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, int64_t group_size) {
    check_inputs(x, packed, scale, group_size);
    const int64_t M = x.size(0), K = x.size(1), N = packed.size(0);
    auto out = torch::empty({M, N}, x.options());
    launch_dequant_gemm_naive(
        x.data_ptr<float>(), packed.data_ptr<uint8_t>(), scale.data_ptr<float>(), out.data_ptr<float>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(group_size),
        at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor dequant_gemm_optimized(torch::Tensor x, torch::Tensor packed, torch::Tensor scale, int64_t group_size) {
    check_inputs(x, packed, scale, group_size);
    const int64_t M = x.size(0), K = x.size(1), N = packed.size(0);
    auto out = torch::empty({M, N}, x.options());
    launch_dequant_gemm_optimized(
        x.data_ptr<float>(), packed.data_ptr<uint8_t>(), scale.data_ptr<float>(), out.data_ptr<float>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(group_size),
        at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_gemm_naive", &dequant_gemm_naive, "Naive INT4 groupwise dequant + GEMM (Phase 1)");
    m.def("dequant_gemm_optimized", &dequant_gemm_optimized, "Optimized (warp-per-row, coalesced+vectorized) INT4 groupwise dequant + GEMM (Phase 2)");
}
