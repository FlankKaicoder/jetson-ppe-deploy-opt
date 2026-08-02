# Exp01：Jetson 环境审计与 C++/CUDA/TensorRT 编译验证总结

## 1. 实验目标

Exp01 用于确认 Jetson Orin Nano Super 是否具备后续 PPE 小目标检测、ONNX 导出、TensorRT 推理和 CUDA 优化所需的端侧开发环境。

本实验不进行模型训练，也不升级系统软件，重点验证：

- Jetson 硬件与 L4T 系统识别；
- CUDA 编译器、CUDA Runtime 和 GPU Kernel 执行；
- TensorRT C++ 头文件、动态库和 Runtime 初始化；
- OpenCV C++ 开发环境；
- GStreamer C++ 开发环境；
- CMake、GCC、G++、NVCC 联合编译；
- 普通 C++ 源码与 CUDA 源码分离编译；
- 实验日志、异常记录和非覆盖式输出目录。

训练服务器环境不在 Exp01 中审计，正式开始训练时再单独冻结。

---

## 2. Jetson 端环境结果

| 项目 | 实际结果 |
|---|---|
| 开发板 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| CPU 架构 | aarch64 |
| L4T | R36.4.3 |
| CUDA Toolkit | 12.6.68 |
| cuDNN | 9.3.0.75 |
| TensorRT | 10.3.0.30 |
| GCC / G++ | 11.4.0 |
| CMake | 3.22.1 |
| GNU Make | 4.3 |
| OpenCV | 4.10.0 |
| GStreamer | 1.20.3 |
| 功耗模式 | 25W |

补充说明：

1. `nvidia-jetpack` 汇总元包未安装，但 CUDA、cuDNN、TensorRT 等核心组件已经独立安装并验证可用。
2. `/usr/src/tensorrt/bin/trtexec --version` 会打印 TensorRT 版本和帮助信息，随后因没有提供模型而返回失败，因此不能只依据该命令的退出码判断 TensorRT 是否正常。
3. TensorRT 版本通过 `libnvinfer` 软件包和 C++ 编译宏共同确认。

---

## 3. GStreamer 与多媒体组件

| 组件 | 结果 |
|---|---|
| `nvarguscamerasrc` | PASS |
| `nvvidconv` | PASS |
| `nvv4l2decoder` | PASS |
| `nvv4l2h264enc` | NOT_APPLICABLE_NO_NVENC |
| `nvv4l2h265enc` | NOT_APPLICABLE_NO_NVENC |

Jetson Orin Nano 不提供 NVENC 硬件编码器，因此没有 H.264/H.265 NVENC 编码插件不属于环境异常。

本项目以 TensorRT/CUDA 推理优化为主，不依赖 NVENC，所以上述两项不会阻塞后续实验。

---

## 4. Exp01.2 联合编译验证

最终工程结构：

```text
experiments/exp01_compile_smoke/
├── CMakeLists.txt
├── include/
│   └── cuda_smoke.h
└── src/
    ├── main.cpp
    └── cuda_smoke.cu
```

职责划分：

```text
main.cpp
├── TensorRT Runtime 初始化
├── OpenCV cv::Mat 测试
├── GStreamer gst_init 测试
└── 汇总各模块结果

cuda_smoke.cu
├── CUDA Device 查询
├── Device Memory 分配
├── Host/Device 数据复制
├── CUDA Kernel 启动
├── cudaDeviceSynchronize
└── 数值正确性验证
```

普通 C++ 与 CUDA 源码分离编译，是为了避免让 NVCC 直接预处理 GLib/GStreamer 等普通宿主端头文件。

---

## 5. CUDA 验证结果

最终运行结果：

```text
cuda_device_count=1
cuda_device_name=Orin
cuda_compute_capability=8.7
cuda_global_memory_bytes=7989972992
cuda_kernel_max_abs_error=0
cuda_kernel_test=PASS
```

CUDA Kernel 完成向量加法：

```text
output[i] = input_a[i] + input_b[i]
```

验证流程：

1. CPU 构造输入数组；
2. 使用 `cudaMalloc` 分配 GPU 内存；
3. 使用 `cudaMemcpy` 将输入复制到 GPU；
4. 启动 CUDA Kernel；
5. 使用 `cudaDeviceSynchronize` 等待执行完成；
6. 将结果复制回 CPU；
7. 与 CPU 参考结果逐元素比较。

最大绝对误差为 0，说明 CUDA 编译、Kernel 启动、GPU 执行和内存传输链路均正常。

Jetson 使用统一内存架构，因此这里显示的约 8 GB 不是独立显卡显存。

---

## 6. TensorRT 验证结果

最终运行结果：

```text
tensorrt_compile_version=10.3.0
tensorrt_runtime_create=PASS
```

测试程序成功执行：

```cpp
nvinfer1::createInferRuntime(logger)
```

这证明：

- TensorRT 头文件可以被 C++ 编译器使用；
- `libnvinfer` 能够正确链接；
- TensorRT Runtime 能够在 Jetson 上初始化；
- 后续可以继续实现 Engine 反序列化、ExecutionContext 和 `enqueueV3`。

本实验尚未加载真实 TensorRT Engine，该部分将在后续 ONNX 与 Engine 构建实验中验证。

---

## 7. OpenCV 与 GStreamer 验证结果

OpenCV：

```text
opencv_version=4.10.0
opencv_mat_test=PASS
```

测试程序成功创建并检查三通道 `cv::Mat`。

GStreamer：

```text
gstreamer_runtime_version=1.20.3
gstreamer_init_test=PASS
```

测试程序成功调用：

```cpp
gst_init(nullptr, nullptr);
```

说明 OpenCV 和 GStreamer 的头文件、动态库及 CMake 链接均可用。

---

## 8. 问题与排查过程

### 8.1 第一次失败：NVCC 与 GLib 头文件冲突

最初将所有代码写入 `main.cu`，并同时包含：

```cpp
#include <gst/gst.h>
#include <opencv2/core.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
```

NVCC 在处理 GLib 宏时出现：

```text
glib/gmacros.h: error: missing ')' after "__has_attribute"
```

原因不是 GStreamer 缺失，而是 NVCC 预处理普通宿主端头文件时发生兼容性问题。

修复方式：

```text
main.cpp       → 由 g++ 编译
cuda_smoke.cu  → 由 nvcc 编译
```

最终由 CMake 链接成一个可执行程序。

### 8.2 第二次失败：CUDA 源码语法错误

错误代码：

```cpp
host_a[index] + host_b[index);
```

正确代码：

```cpp
host_a[index] + host_b[index];
```

修复后 CUDA 源码正常编译。

### 8.3 编译警告过多

NVCC 宿主编译阶段使用 `-Wpedantic` 后，产生大量：

```text
style of line directive is a GCC extension
```

这些警告来自 NVCC 生成的中间代码，并不代表项目源码错误。

最终配置：

```text
普通 C++：-Wall -Wextra -Wpedantic
CUDA：    -Wall -Wextra
```

### 8.4 abnormal.txt 误报

最初异常扫描使用过宽的关键词：

```text
error|failed|fatal
```

因此会错误匹配：

```text
cudaError_t
max_abs_error
```

调整匹配规则后，最终结果为：

```text
No abnormal messages detected.
```

---

## 9. 最终结果

最终返回码：

```text
cmake_return_code : 0
build_return_code : 0
run_return_code   : 0
```

各模块结果：

```text
CUDA Kernel       : PASS
TensorRT Runtime  : PASS
OpenCV C++        : PASS
GStreamer C++     : PASS
Overall           : PASS
```

最终成功实验目录：

```text
results/environment_audit/exp01_2_compile_smoke_20260802_212701
```

最终结论：

> Jetson Orin Nano Super 已具备 CUDA 12.6、TensorRT 10.3、OpenCV 4.10 和 GStreamer 1.20 的 C++ 开发环境。C++ 与 CUDA 分离编译、CUDA Kernel 执行、TensorRT Runtime 创建以及 OpenCV/GStreamer 链接均验证通过，可以进入后续模型、ONNX、TensorRT 和 CUDA 优化实验。

---

## 10. Exp01 后保留的核心文件

```text
experiments/exp01_compile_smoke/CMakeLists.txt
experiments/exp01_compile_smoke/include/cuda_smoke.h
experiments/exp01_compile_smoke/src/cuda_smoke.cu
experiments/exp01_compile_smoke/src/main.cpp
tools/exp01_1_jetson_environment_audit.sh
tools/exp01_2_compile_smoke.sh
results/environment_audit/
```

建议将本文件保存到仓库：

```text
docs/01_environment.md
```
