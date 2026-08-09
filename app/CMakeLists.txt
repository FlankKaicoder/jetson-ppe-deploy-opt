cmake_minimum_required(VERSION 3.18)

project(jetson_ppe_video_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

option(PPE_ENABLE_NVTX "Enable NVTX ranges for Nsight Systems" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc videoio)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cuda
                 ${CMAKE_CURRENT_BINARY_DIR}/cuda)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../runtime
                 ${CMAKE_CURRENT_BINARY_DIR}/runtime)

add_executable(exp11_video_infer src/exp11_video_infer.cpp)
target_include_directories(exp11_video_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../profiling/include)
if(PPE_ENABLE_NVTX)
    target_compile_definitions(exp11_video_infer PRIVATE PPE_ENABLE_NVTX=1)
    set_target_properties(exp11_video_infer PROPERTIES
        OUTPUT_NAME exp13_profiled_video_infer)
endif()
target_link_libraries(exp11_video_infer PRIVATE
    ppe_cuda_preprocess ppe_trt_runtime ${OpenCV_LIBS})
target_compile_options(exp11_video_infer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(exp15_gpu_postprocess src/exp11_video_infer.cpp)
target_include_directories(exp15_gpu_postprocess PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../profiling/include)
target_compile_definitions(exp15_gpu_postprocess PRIVATE
    PPE_ENABLE_GPU_POSTPROCESS=1)
if(PPE_ENABLE_NVTX)
    target_compile_definitions(exp15_gpu_postprocess PRIVATE PPE_ENABLE_NVTX=1)
endif()
target_link_libraries(exp15_gpu_postprocess PRIVATE
    ppe_cuda_preprocess ppe_gpu_postprocess ppe_trt_runtime ${OpenCV_LIBS})
target_compile_options(exp15_gpu_postprocess PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(exp14_async_pipeline src/exp14_async_pipeline.cpp)
target_include_directories(exp14_async_pipeline PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../profiling/include)
if(PPE_ENABLE_NVTX)
    target_compile_definitions(exp14_async_pipeline PRIVATE PPE_ENABLE_NVTX=1)
endif()
target_link_libraries(exp14_async_pipeline PRIVATE
    ppe_cuda_preprocess ppe_trt_runtime ${OpenCV_LIBS})
target_compile_options(exp14_async_pipeline PRIVATE
    -Wall -Wextra -Wpedantic)
