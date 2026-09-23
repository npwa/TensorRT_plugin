cmake_minimum_required(VERSION 3.28)
project(trt_quant_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3080 = compute capability 8.6

# TensorRT ships as plain system libs/headers here (installed via apt from NVIDIA's
# CUDA repo, matched to the pip tensorrt-cu13==11.3.0.99 Python bindings so the C++
# and Python paths agree on version) -- no vendored SDK directory to point at.
find_library(NVINFER_LIB nvinfer REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin REQUIRED)

add_subdirectory(src/kernels)
add_subdirectory(src/plugin)
add_subdirectory(src/harness)

enable_testing()
add_subdirectory(tests)
