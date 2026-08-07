# Exp09：TensorRT C++ Runtime 部署与验证总结

## 当前状态

```text
PASS
```

Exp09 已完成 CMake/G++ 编译、TensorRT 10.3 C++ Runtime、Python/C++ 一致性、三独立
进程生命周期和诊断计时验证。图像预处理与后处理仍不属于本实验。

## 实验目标

1. 在目标 Jetson 上使用 TensorRT 10.3 C++ API 反序列化 Exp07 冻结 FP16 Engine；
2. 正确检查 I/O 名称、模式、数据类型和静态维度；
3. 通过 RAII 管理 Runtime、Engine、Context、CUDA Stream 与 Device Buffer；
4. 使用 `setTensorAddress` 和 `enqueueV3` 完成 H2D、推理、D2H 与同步；
5. 与同一 Engine、同一预处理张量的 Python TensorRT 输出比较；
6. 记录 C++ Runtime 的诊断延迟分布和重复执行稳定性。

## 冻结边界

Exp09 只验证“预处理完成的 FP32 NCHW Tensor → TensorRT Engine → 原始输出 Tensor”。

本实验不实现：

- JPEG/摄像头解码；
- OpenCV 或 CUDA letterbox；
- BGR→RGB、归一化和 HWC→CHW；
- C++ NMS、绘制、GStreamer 或摄像头；
- 功耗、温度和长时间稳定性结论。

上述内容分别属于 Exp10～Exp12。这样可以把 Runtime 错误与预处理/后处理错误隔离。

## 冻结输入

| 输入 | 值 |
|---|---|
| Jetson 仓库 | `/home/nvidia/projects/jetson-ppe-deploy-opt` |
| FP16 Engine | `/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine` |
| Engine SHA256 | `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83` |
| Probe image | `/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg` |
| Probe SHA256 | `39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e` |
| 输入 | FP32、NCHW、`[1,3,640,640]` |
| 输出 | FP32、`[1,7,8400]` |

Python 准备器必须复用 Exp07 的 letterbox 预处理，并输出无容器头的连续 FP32 binary。
输入 binary 的字节数和 SHA256 在首次生成后记录，正式 C++ 运行复用同一文件。

## 预冻结验收条件

### 编译与接口

- CMake configure、build 返回码均为 0；
- 不依赖 Python、Ultralytics 或 OpenCV C++；
- Engine SHA256 必须匹配冻结值；
- 必须发现且仅发现一个输入与一个输出；
- 输入/输出模式、FP32 类型和静态形状必须完全匹配；
- 任一 CUDA/TensorRT 调用失败时返回非零，不吞掉异常。

### 正确性

同一输入 Tensor 下，C++ 输出相对 Python TensorRT 参考必须满足：

| 指标 | 门槛 |
|---|---:|
| 输出形状 | 完全等于 `[1,7,8400]` |
| NaN/Inf | 0 |
| raw max absolute error | `<= 0.001` |
| raw mean absolute error | `<= 0.00001` |
| raw relative L2 error | `<= 0.00001` |
| NMS 后检测数 | 完全一致 |
| NMS 后类别序列 | 完全一致 |
| box max absolute error | `<= 0.1 px` |
| confidence max absolute error | `<= 0.001` |

NMS 只由 Python 审计器对两份原始输出执行，固定 `confidence=0.25`、`IoU=0.70`；
这不是在 Exp09 提前实现 C++ 后处理。

### 生命周期与诊断计时

- 最小 Smoke Test：warmup=2、iterations=5；
- 正式诊断：warmup=20、iterations=200；
- 至少连续启动 3 个独立进程，每次返回码为 0、输出 SHA256 一致；
- 正式运行记录 host wall 与 CUDA Event 口径的 mean/P50/P95/P99；
- 计时范围必须明确是否包含 H2D/D2H，不与 Exp07 GPU-only 数值混称；
- Exp09 不用单一延迟值决定 PASS，正确性与资源生命周期是强制门槛。

## 推荐实现

```text
runtime/CMakeLists.txt
runtime/include/trt_runtime.hpp
runtime/src/trt_runtime.cpp
runtime/src/exp09_infer.cpp
tools/exp09_prepare_reference.py
tools/exp09_compare_outputs.py
tools/exp09_0_cpp_runtime_smoke.sh
tools/exp09_1_cpp_runtime_formal.sh
```

所有 run 使用 `results/runtime/exp09_*_YYYYMMDD_HHMMSS/`，不得覆盖历史结果。

## 风险与停止条件

- TensorRT 10 的 tensor API 与旧 binding API 混用；
- 把 Engine 声明的 Tensor dtype 误当成网络计算精度；
- 字节数、维度乘积或主机文件大小计算错误；
- `enqueueV3` 前遗漏 `setTensorAddress`；
- 异步复制后在数据就绪前写文件；
- 把包含传输的 C++ 延迟与 Exp07 关闭传输的 GPU-only 延迟直接比较；
- 为通过测试而放宽预冻结误差阈值。

Smoke Test 失败时保留目录、命令、返回码和失败摘要，不进入正式运行。

## 执行清单

- [x] Windows 规划审查并合并；
- [x] Jetson 创建 `exp/09-trt-cpp-runtime`；
- [x] C++ Runtime 与参考/比较工具实现；
- [x] CMake configure/build；
- [x] Smoke Test；
- [x] 正式正确性、三进程生命周期与诊断计时；
- [x] 文档与学习复盘。

## 实现结果

核心实现：

- `runtime/src/trt_runtime.cpp`：Engine 反序列化、I/O 元数据、Device Buffer、CUDA Stream/Event、
  `setTensorAddress`、`enqueueV3` 和异常传播；
- `runtime/src/exp09_infer.cpp`：严格的 binary 输入大小检查、重复执行、有限值检查、原始输出写盘；
- `tools/exp09_prepare_reference.py`：复用 Exp07 letterbox，生成固定输入与 Python TensorRT 参考；
- `tools/exp09_compare_outputs.py`：raw tensor 与统一 Python NMS 检查；
- `tools/exp09_0_cpp_runtime_smoke.sh` / `exp09_1_cpp_runtime_formal.sh`：非覆盖运行与返回码收集。

Runtime 仅接受 FP32 I/O tensor，要求恰好一个输入和一个输出，并强制静态维度
`images=[1,3,640,640]`、`output0=[1,7,8400]`。Engine 内部仍是 FP16 优化，FP32 是 Engine
外部 I/O 类型，两者不能混淆。

## Smoke Test

正式 smoke 目录：

```text
results/runtime/exp09_0_cpp_runtime_smoke_20260807_164202
```

结果：CMake configure、build、参考生成、C++ 执行和比较器返回码全部为 0；warmup=2、
iterations=5。输入 binary 为 4,915,200 bytes，SHA256：

```text
963a674701c7bc3bcf26b121646fd7754e1e87d7db5f14eb8d24fe07984fe90f
```

Python/C++ 输出均为 235,200 bytes，SHA256 完全相同：

```text
29ae405ef3ca01c826e01982d3469848eaa17d5325d89d2380732a127bd62c5d
```

raw max/mean/relative-L2 均为 0；NMS 后 30 个检测，类别、框和置信度误差均为 0。
5 次小样本的 host mean 为 10.3266 ms，因样本数过少且包含 pageable H2D/D2H，仅用于放行正式测试。

## 正式三进程验证

正式目录：

```text
results/runtime/exp09_1_cpp_runtime_formal_20260807_164714
```

配置：3 个独立进程；每个 warmup=20、iterations=200；每次重新创建并销毁 Runtime、Engine、
Context、Stream、Event 与 Device Buffer。全部进程返回 0，输出 SHA256 相同且与 Python 参考一致。

| 进程 | host mean | host P50 | host P95 | host P99 | CUDA total mean |
|---|---:|---:|---:|---:|---:|
| 1 | 6.606200 ms | 6.612337 ms | 8.364161 ms | 9.915610 ms | 6.591960 ms |
| 2 | 6.618855 ms | 6.609452 ms | 8.759622 ms | 10.619855 ms | 6.603031 ms |
| 3 | 6.638301 ms | 6.665779 ms | 8.824101 ms | 9.969482 ms | 6.623321 ms |

三个进程的 host mean 范围为 6.606200～6.638301 ms。计时包含 pageable host memory 的
H2D、`enqueueV3`、D2H 和同步；Exp07 的 3.479713 ms 关闭 H2D/D2H 且只统计 GPU compute，
因此两者不能直接用来声称 C++ 变慢或变快。

## 验收结论

| 验收项 | 结果 |
|---|---|
| CMake/G++ 编译 | PASS，无编译警告 |
| Engine/I/O/形状/类型检查 | PASS |
| 输出 `[1,7,8400]` 与有限值 | PASS |
| raw tensor 全部误差门槛 | PASS，实际为 0 |
| NMS 后检测数、类别、框、置信度 | PASS，实际误差为 0 |
| 三独立进程与输出 SHA256 | PASS |
| 诊断分位数记录 | PASS |

Exp09 状态为 `PASS`。已证实 C++ Runtime 能正确、可重复地执行冻结 FP16 Engine；尚未证实
图像预处理、C++ NMS、摄像头端到端、功耗温度和长时间稳定性。

下一步进入 Exp10：用 CPU OpenCV/Exp07 Python 预处理作为 Reference，实现并验证 CUDA 融合
letterbox、padding、BGR→RGB、归一化和 HWC→CHW，输出直接供本 Runtime 使用。
