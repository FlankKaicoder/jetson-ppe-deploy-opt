cmake_minimum_required(VERSION 3.18)

project(jetson_ppe_video_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc videoio)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cuda
                 ${CMAKE_CURRENT_BINARY_DIR}/cuda)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../runtime
                 ${CMAKE_CURRENT_BINARY_DIR}/runtime)

add_executable(exp11_video_infer src/exp11_video_infer.cpp)
target_include_directories(exp11_video_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(exp11_video_infer PRIVATE
    ppe_cuda_preprocess ppe_trt_runtime ${OpenCV_LIBS})
target_compile_options(exp11_video_infer PRIVATE -Wall -Wextra -Wpedantic)
