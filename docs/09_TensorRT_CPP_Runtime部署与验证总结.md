# Exp09：TensorRT C++ Runtime 部署与验证总结

## 当前状态

```text
IN_PROGRESS
```

本文档先冻结实验边界与验收条件。尚未执行的结果不得表述为完成。

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

## 尚待执行

- [ ] Windows 规划审查并合并；
- [ ] Jetson 创建 `exp/09-trt-cpp-runtime`；
- [ ] C++ Runtime 与参考/比较工具实现；
- [ ] CMake configure/build；
- [ ] Smoke Test；
- [ ] 正式正确性、三进程生命周期与诊断计时；
- [ ] 文档、学习复盘与三端 Git 收口。
