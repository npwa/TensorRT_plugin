# Phase 1: PyTorch reference + naive kernel; Phase 2: optimized kernel.
# Pure CUDA, no torch dependency -- Phase 3's plugin links this directly.
add_library(dequant_kernels STATIC
    dequant_gemm_naive.cu
    dequant_gemm_optimized.cu
)
target_include_directories(dequant_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# No cross-translation-unit device calls here, so leave separable compilation (-rdc=true)
# off -- turning it on requires an explicit device-link step for every consumer, which a
# pure-C++ executable linking this as a static lib (src/plugin/validate_plugin_engine)
# does not get automatically and fails with an undefined __cudaRegisterLinkedBinary
# reference.
