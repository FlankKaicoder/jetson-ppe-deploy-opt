cmake_minimum_required(VERSION 3.18)

project(jetson_ppe_trt_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or nvinfer library not found")
endif()

add_library(ppe_trt_runtime STATIC
    src/trt_runtime.cpp
    src/trt_plugin_runtime.cpp)
target_include_directories(ppe_trt_runtime
    PUBLIC include
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../profiling/include)
if(PPE_ENABLE_NVTX)
    target_compile_definitions(ppe_trt_runtime PRIVATE PPE_ENABLE_NVTX=1)
endif()
target_link_libraries(ppe_trt_runtime
    PUBLIC CUDA::cudart ${TENSORRT_LIBRARY}
    PRIVATE ${CMAKE_DL_LIBS})
target_compile_options(ppe_trt_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

add_executable(exp09_trt_infer
    src/exp09_infer.cpp)
target_include_directories(exp09_trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(exp09_trt_infer PRIVATE ppe_trt_runtime)
target_compile_options(exp09_trt_infer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
