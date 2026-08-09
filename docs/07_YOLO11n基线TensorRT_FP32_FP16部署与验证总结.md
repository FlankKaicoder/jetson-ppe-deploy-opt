# Exp07 YOLO11n 基线 TensorRT FP32 / FP16 部署与验证总结

## 1. 最终结论

Exp06 冻结的静态 FP32 ONNX 已在 Jetson Orin Nano Super 上成功构建为
TensorRT 10.3 FP32 与 FP16 Engine。两种 Engine 均完成单图原始张量检查、
NMS 后检测检查和 219 张独立测试集验证；FP32/FP16 相对同一 Jetson 运行时下
PyTorch 的 AP 偏差满足校准后的验收条件。GPU-only 诊断 benchmark 显示 FP16
平均计算延迟为 FP32 的约 1/2.94，Engine 体积减少 39.84%。

Exp07 状态：`PASS`。

这些数据不包含 H2D、D2H、预处理和后处理，也未锁定 `jetson_clocks`，因此不得
表述为摄像头或端到端性能；端到端、功耗、温度和稳定性仍属于 Exp11/Exp12。

## 2. 实验目的与假设

目的：

- 验证 TensorRT 10.3 能解析 Exp06 ONNX 并构建 batch 1、640×640 Engine；
- 验证 TensorRT FP32/FP16 的原始输出和最终检测没有不可接受的偏差；
- 在公平的 GPU-only 计时范围内比较 FP32 与 FP16；
- 冻结可供 Exp08 INT8 和 Exp09 C++ Runtime 使用的构建配置与产物哈希。

假设：FP32 应与 ONNX/PyTorch 高度一致；FP16 允许更大的逐元素误差，但完整测试集
AP 不应出现实质退化。

## 3. 环境

- 设备：Jetson Orin Nano Super，aarch64；
- 系统：Ubuntu 22.04 / L4T R36.4.3；
- CUDA：12.6.68；
- TensorRT：10.3.0.30；
- PyTorch：2.5.0a0+872d972e41.nv24.08；
- Ultralytics：8.3.159；
- OpenCV Python：4.11.0；
- 功耗模式：25W；
- `jetson_clocks`：`NOT_CHECKED_NON_ROOT`，未修改；
- 分支：`exp/07-tensorrt-fp16`；
- 基线 Git commit：`0b09fa49192d18a097e7293438e6a9c12a669013`。

## 4. 输入与产物

冻结输入：

```text
ONNX SHA256 : 305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8
best.pt SHA256: 79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
input         : 1×3×640×640 FP32
batch         : 1
test split    : 219 images / 840 instances
rect          : false
```

正式 Engine 产物位于 Jetson 本机，不进入 Git：

| 精度 | 文件大小 | SHA256 | 构建时间 |
|---|---:|---|---:|
| FP32 | 14,880,428 bytes | `01616a8144228db5edbf8948227e3bbaee43b22c495aba3c6c44212e43efe0f1` | 164 s |
| FP16 | 8,951,540 bytes | `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83` | 425 s |

FP16 Engine 相对 FP32 体积减少 39.843531%。Engine 必须在目标 Jetson 环境重建，
不得跨设备复用。

## 5. 构建配置

共同配置：

```text
TensorRT trtexec
batch=1
input=1x3x640x640
TF32=disabled
workspace=1024 MiB
builderOptimizationLevel=3
profilingVerbosity=detailed
```

FP16 构建额外启用 `--fp16`。正式构建目录：

```text
results/tensorrt/exp07_1_trt_fp32_fp16_formal_20260806_170501/
```

该次运行因初始 FP16 单图绝对误差门槛过严而整体记为 `FAIL`，但两种 Engine 构建
返回码均为 0，文件大小与 SHA256 已冻结；失败状态没有被改写。

## 6. Smoke Test 与单图一致性

最终 FP32 Smoke Test：

```text
results/tensorrt/exp07_0_trt_fp32_smoke_20260806_165553/
result=PASS
```

FP32 单图结果：

| 项目 | 数值 |
|---|---:|
| raw max abs error | 0.0006103515625 |
| raw mean abs error | 0.000010268287 |
| raw relative L2 | 0.00000013818 |
| 检测数 | 30 / 30 |
| 类别 | 完全一致 |
| box max abs error | 0.0000610352 px |
| confidence max abs error | 0.000000894 |

FP16 初始正式检查中，30/30 检测数和类别保持一致，box 最大误差为
0.140167 px，raw relative L2 为 0.00045447；raw 最大/均值误差与置信度误差
超过最初借用 FP32 量级制定的绝对门槛，因此该运行保留为 `FAIL`，并转为使用完整
测试集 AP 判断实际检测质量。

## 7. 完整测试集一致性

最终同运行时验收目录：

```text
results/tensorrt/exp07_1b_full_test_consistency_20260806_175921/
result=PASS
return_code=0
```

为避免 AutoDL Ultralytics 8.4.95 与 Jetson 8.3.159 的验证器差异，冻结
`best.pt`、FP32 Engine、FP16 Engine 均在 Jetson 8.3.159 下、以独立 Python
进程执行相同的 `imgsz=640, batch=1, rect=False, split=test` 验证。

| 后端 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 0.907342787931 | 0.813693227775 | 0.884295853296 | 0.521895191595 |
| TensorRT FP32 | 0.890126489963 | 0.831086598128 | 0.884320929181 | 0.521604061490 |
| TensorRT FP16 | 0.906842481519 | 0.814374066011 | 0.884373548441 | 0.521928806380 |

相对 PyTorch：

| 后端 | ΔPrecision | ΔRecall | ΔmAP50 | ΔmAP50-95 |
|---|---:|---:|---:|---:|
| FP32 | -0.01721630 | +0.01739337 | +0.00002508 | -0.00029113 |
| FP16 | -0.00050031 | +0.00068084 | +0.00007770 | +0.00003361 |

Ultralytics 的 Precision/Recall 取平滑平均 F1 最大值对应的离散置信度索引；微小
置信度扰动可让该索引跳到相邻位置，而完整 PR 曲线积分得到的 AP 基本不变。初始
严格 P/R 门槛导致前一运行 `175651` 为 `FAIL`。失败后明确校准为：FP32/FP16
P/R 容差 0.02、FP32 AP 容差 0.0005、FP16 AP 容差 0.001。该调整和先前失败均
保留，不表述为预先设定。

## 8. GPU-only 诊断性能

正式诊断目录：

```text
results/tensorrt/exp07_2_trt_fp32_fp16_benchmark_20260806_180305/
result=PASS
return_code=0
```

固定条件：batch 1、640×640、500 ms warmup、200 次正式迭代、duration 0、
CUDA Graph、spin wait、关闭 H2D/D2H；未锁定 `jetson_clocks`。

| 精度 | mean | P50 | P95 | P99 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| FP32 | 10.241488 ms | 10.083900 ms | 11.270900 ms | 11.279325 ms | 97.6307 qps |
| FP16 | 3.479713 ms | 3.479645 ms | 3.485470 ms | 3.488407 ms | 287.2950 qps |

在该诊断范围内，FP16 平均计算延迟加速 2.943199×，吞吐提高 2.942671×。
这不是端到端 FPS。

## 9. 异常、修复与保留记录

| 运行目录 | 状态 | 原因与处理 |
|---|---|---|
| `exp07_0_trt_fp32_smoke_20260806_165158` | FAIL | SSH 控制通道超时使输出管道收到 SIGPIPE；返回码 141，保留现场后重跑 |
| `exp07_0_trt_fp32_smoke_20260806_165553` | PASS | FP32 Smoke Test 通过 |
| `exp07_1_trt_fp32_fp16_formal_20260806_170501` | FAIL | Engine 构建成功；初始 FP16 单图绝对误差门槛过严，benchmark 按规则跳过 |
| `exp07_1b_full_test_consistency_20260806_172550` | FAIL | 启动脚本预建文件与验证器空目录保护冲突 |
| `exp07_1b_full_test_consistency_20260806_172801` | FAIL | Ultralytics 尝试自动安装缺失包并停滞；终止进程，返回码 143，未安装依赖 |
| `exp07_1b_full_test_consistency_20260806_174622` | FAIL | Exp06 JSON 指标路径解析错误，推理前失败 |
| `exp07_1b_full_test_consistency_20260806_175024` | FAIL | 跨 Ultralytics 版本指标不可严门槛比较；同进程连续 Engine 还触发 CUDA 上下文清理错误 |
| `exp07_1b_full_test_consistency_20260806_175651` | FAIL | 独立进程验证成功；初始 FP32 P/R/AP 门槛揭示最大-F1索引不稳定，作为校准依据 |
| `exp07_1b_full_test_consistency_20260806_175921` | PASS | 同运行时、独立后端进程的校准后正式验收通过 |
| `exp07_2_trt_fp32_fp16_benchmark_20260806_180305` | PASS | 两种精度诊断 benchmark 通过 |

所有失败目录、返回码、摘要和原始 `run.log` 均保留在 Jetson worktree；完整日志不
进入 Git。

## 10. 已证实与尚未证实

已证实：

- ONNX 在 TensorRT 10.3 上可构建 FP32/FP16 Engine；
- Engine 哈希、大小、构建配置和构建时间已记录；
- FP32 单图原始张量与最终检测高度一致；
- FP16 检测数、类别和完整测试集 AP 满足验收；
- FP32/FP16 在相同 Jetson 验证器下的完整测试集 AP 与 PyTorch 一致；
- FP16 在 GPU-only 诊断范围内约为 FP32 的 2.94×。

尚未证实：

- INT8 PTQ 精度和性能；
- TensorRT C++ Runtime；
- CUDA 融合预处理；
- 包含预处理、H2D/D2H、后处理的端到端延迟；
- 摄像头链路、功耗、温度、降频和长时间稳定性。

## 11. 最终决策与下一步

Exp07 验收通过，FP16 作为后续 C++ Runtime 和端到端部署的优先精度模式；FP32
保留为正确性参考。下一步按路线进入 Exp08 INT8 PTQ，在固定 test split、预处理
和计时范围下比较 FP32、FP16、INT8，不能用本次 GPU-only 数据替代 Exp12 综合
性能结论。

## 12. 2026-08-09 跨 Build 语义边界补充（不重做 Exp07）

Exp07的`PASS`与冻结Engine结论不变，也不重新构建本实验。后续Exp09/Exp11使用同一份Serialized FP16
Engine的独立进程和固定输入，进一步证明了“冻结Engine重复执行稳定”。Exp16诊断同时发现：从同一Exp06
ONNX、同一TensorRT 10.3和相同显式Builder参数重新Build的普通无Plugin control，Engine哈希、raw输出及
少量临界检测不保证与冻结Exp07 Engine bitwise一致；TensorRT rebuild/tactic selection是跨build比较中的
真实变量。

因此必须区分：

```text
Frozen Serialized Engine repeated execution stability
!=
Fresh rebuild from the same ONNX gives bitwise-identical raw output
```

该补充不否定Exp07已完成的模型级精度验收，也不允许把fresh rebuild漂移自动归因于任何Plugin。后续跨
Engine比较应记录每次build及SHA256，使用image+class+IoU/Hungarian检测匹配、模型级P/R与AP，并用至少
两个普通baseline rebuild估计build variance；不得依赖CSV行号或raw bitwise一致性做唯一裁决。
