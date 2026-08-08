cmake_minimum_required(VERSION 3.18)

project(jetson_ppe_cuda_preprocess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)

add_library(ppe_cuda_preprocess STATIC
    src/cuda_preprocess.cu)
target_include_directories(ppe_cuda_preprocess PUBLIC include)
if(PPE_ENABLE_NVTX)
    target_include_directories(ppe_cuda_preprocess PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../profiling/include)
    target_compile_definitions(ppe_cuda_preprocess PRIVATE PPE_ENABLE_NVTX=1)
endif()
target_link_libraries(ppe_cuda_preprocess PUBLIC CUDA::cudart)
target_compile_options(ppe_cuda_preprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(exp10_preprocess
    src/exp10_preprocess.cpp)
target_include_directories(exp10_preprocess PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(exp10_preprocess PRIVATE
    ppe_cuda_preprocess ${OpenCV_LIBS})
target_compile_options(exp10_preprocess PRIVATE -Wall -Wextra -Wpedantic)
