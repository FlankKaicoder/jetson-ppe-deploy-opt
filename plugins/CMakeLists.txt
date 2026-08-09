cmake_minimum_required(VERSION 3.18)

project(ppe_exp16_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or nvinfer library not found")
endif()

add_library(ppe_yolo_decode_plugin SHARED
    src/yolo_decode_plugin.cpp
    src/yolo_decode_plugin.cu)
target_include_directories(ppe_yolo_decode_plugin
    PUBLIC include
    PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(ppe_yolo_decode_plugin PRIVATE
    CUDA::cudart ${TENSORRT_LIBRARY})
target_compile_options(ppe_yolo_decode_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(ppe_yolo_decode_plugin PROPERTIES
    OUTPUT_NAME ppe_yolo_decode_plugin
    POSITION_INDEPENDENT_CODE ON)

add_executable(exp16_plugin_probe src/exp16_plugin_probe.cpp)
target_include_directories(exp16_plugin_probe PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(exp16_plugin_probe PRIVATE
    CUDA::cudart ${TENSORRT_LIBRARY} dl)
target_compile_options(exp16_plugin_probe PRIVATE -Wall -Wextra -Wpedantic)

add_executable(exp16_deserialize_infer src/exp16_deserialize_infer.cpp)
target_include_directories(exp16_deserialize_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(exp16_deserialize_infer PRIVATE
    CUDA::cudart ${TENSORRT_LIBRARY} dl)
target_compile_options(exp16_deserialize_infer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(exp16_build_fixture_engine src/exp16_build_fixture_engine.cpp)
target_include_directories(exp16_build_fixture_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(exp16_build_fixture_engine PRIVATE
    CUDA::cudart ${TENSORRT_LIBRARY} dl)
target_compile_options(exp16_build_fixture_engine PRIVATE -Wall -Wextra -Wpedantic)

if(TENSORRT_ONNX_LIBRARY)
    add_executable(exp16_build_engine src/exp16_build_engine.cpp)
    target_include_directories(exp16_build_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(exp16_build_engine PRIVATE
        CUDA::cudart ${TENSORRT_LIBRARY} ${TENSORRT_ONNX_LIBRARY} dl)
    target_compile_options(exp16_build_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()
