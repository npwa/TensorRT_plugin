# Phase 4: C++20 inference harness (RAII TensorRT ownership, concurrent execution).
find_package(Threads REQUIRED)

add_library(engine_harness STATIC
    engine_harness.cpp
)
target_include_directories(engine_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(engine_harness PUBLIC trt_dequant_gemm_plugin ${NVINFER_LIB})

add_executable(run_single run_single.cpp)
target_link_libraries(run_single PRIVATE engine_harness)

add_executable(run_concurrent run_concurrent.cpp)
target_link_libraries(run_concurrent PRIVATE engine_harness Threads::Threads)
