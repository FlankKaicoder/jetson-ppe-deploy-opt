cmake_minimum_required(VERSION 3.18)

project(
    exp01_compile_smoke
    VERSION 1.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Jetson Orin Nano Ampere GPU, compute capability 8.7.
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(PkgConfig REQUIRED)

pkg_check_modules(
    GSTREAMER
    REQUIRED
    IMPORTED_TARGET
    gstreamer-1.0
)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    PATHS
        /usr/include/aarch64-linux-gnu
        /usr/include
    REQUIRED
)

find_library(
    TENSORRT_LIBRARY
    nvinfer
    PATHS
        /usr/lib/aarch64-linux-gnu
        /usr/lib
    REQUIRED
)

add_executable(
    exp01_compile_smoke
    src/main.cpp
    src/cuda_smoke.cu
)

target_include_directories(
    exp01_compile_smoke
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${TENSORRT_INCLUDE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    exp01_compile_smoke
    PRIVATE
        CUDA::cudart
        "${TENSORRT_LIBRARY}"
        ${OpenCV_LIBS}
        PkgConfig::GSTREAMER
)

target_compile_options(
    exp01_compile_smoke
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
            -Wpedantic
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=-Wall,-Wextra
        >
)
