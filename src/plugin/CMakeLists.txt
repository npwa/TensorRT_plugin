# Phase 3: IPluginV3 wrapper around the kernel from src/kernels.
add_library(trt_dequant_gemm_plugin STATIC
    dequant_gemm_plugin.cpp
)
target_include_directories(trt_dequant_gemm_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trt_dequant_gemm_plugin PUBLIC dequant_kernels ${NVINFER_LIB})

# Phase 3 step 3: standalone driver that builds a minimal single-layer network around
# this plugin and validates its output against Python-generated reference vectors
# (python/reference/gen_plugin_test_vectors.py). Not wired into ctest yet -- Phase 5
# formalizes the full kernel/plugin/engine validation ladder into the real test suite.
add_executable(validate_plugin_engine validate_plugin_engine.cpp)
target_link_libraries(validate_plugin_engine PRIVATE trt_dequant_gemm_plugin ${NVINFER_LIB})
