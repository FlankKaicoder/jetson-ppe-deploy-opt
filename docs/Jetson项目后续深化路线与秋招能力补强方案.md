# Jetson PPE 项目后续深化路线与秋招能力补强方案

> 文档用途：提供给 Codex 作为当前 Jetson PPE 项目后续实验规划、实现边界、技术优先级和验收依据。
> 当前时间：2026-08-08
> 当前阶段：Exp00～Exp12 已完成，不立即进入原计划中的“Exp13 项目收尾”，而是继续补齐推理优化、CUDA、TensorRT 扩展和量化优化能力。
> 核心原则：**停止继续横向堆模型实验，开始从 Runtime、内存、并发、算子、计算图和量化失败分析方向向下深入。**

---

# 1. 当前项目状态判断

当前项目并不是“简单把 RK3588 上做过的 YOLO 部署搬到 Jetson 上重新做一遍”。

截至 Exp12，已经完成：

```text
数据集审计
→ YOLO11n 基线训练
→ P2 小目标结构消融
→ 部署可重参数化结构消融
→ CBAM-Lite / Focal Loss 消融
→ 冻结原始 YOLO11n baseline
→ PyTorch → ONNX
→ ONNX Runtime 一致性验证
→ TensorRT FP32 / FP16
→ INT8 PTQ
→ TensorRT C++ Runtime
→ CUDA 融合预处理
→ GStreamer / IMX219 摄像头端到端推理
→ 延迟 / P95 / P99 / FPS / 功耗 / 温度 / 稳定性
```

已有成果明显区别于普通的“YOLO 导出 TensorRT 后跑起来”：

- PyTorch / ONNX / TensorRT 有完整一致性验证；
- FP32 / FP16 / INT8 有公平对照；
- INT8 不只是构建成功，而是根据精度和 tiny/small recall 主动 REJECT；
- C++ TensorRT Runtime 已实现；
- CUDA 已实现真实融合预处理 Kernel；
- IMX219 摄像头端到端链路已跑通；
- 30 分钟稳定性、P95/P99、功耗、温度和 RSS 趋势已经验证。

因此：

> 当前项目的“宽度”已经足够，真正不足的是向 GPU 推理优化底层继续深入的“深度”。

---

# 2. 当前最大的能力缺口

当前数据流基本为：

```text
IMX219 / Video
↓
GStreamer / OpenCV BGR Host Frame
↓
pageable Host → Device Copy
↓
CUDA Preprocess
↓
TensorRT enqueueV3
↓
Raw Output Device → Host
↓
CPU Decode / NMS
↓
Result
```

当前主要问题不是“功能不完整”，而是：

```text
数据传输仍偏同步
Host / Device 之间仍存在较多搬运
流水线未充分重叠
后处理仍主要在 CPU
没有系统级 Nsight Profiling
没有真正完成 TensorRT Plugin
没有 ONNX GraphSurgeon 图级扩展
INT8 失败后没有继续做敏感层分析
没有形成 Mixed Precision / QAT 优化闭环
```

所以当前项目仍主要处于：

```text
TensorRT SDK 使用
+
C++ Runtime
+
单个 CUDA 融合 Kernel
+
端到端工程验证
```

还没有完全进入：

```text
性能瓶颈定位
→ 内存优化
→ GPU 并发调度
→ CUDA 算子优化
→ TensorRT 自定义 Plugin
→ 计算图修改
→ 量化敏感层分析
→ 混合精度
```

这正是后续最需要补齐的部分。

---

# 3. 与 RK3588 项目的能力边界重新划分

后续不要让 Jetson 项目继续重复 RK3588 项目的内容。

## 3.1 RK3588 项目主要体现

```text
V4L2
RGA
RKNN
NPU 多核
MPP
H.264
ALSA
AAC
FFmpeg
RTSP
MP4
多线程
音视频同步
嵌入式 Linux 音视频系统工程
```

## 3.2 Jetson 项目今后主要体现

```text
ONNX
TensorRT
TensorRT C++ Runtime
FP16 / INT8 / Mixed Precision / QAT
CUDA
GPU Memory
Pinned Memory
CUDA Stream
CUDA Event
Double Buffer
CUDA Graph
Nsight Systems
Nsight Compute
Custom CUDA Kernel
TensorRT IPluginV3
ONNX GraphSurgeon
GPU Postprocess
性能建模
端侧 GPU 推理优化
```

以后不要再把 Jetson 项目做成：

```text
摄像头
→ 模型
→ 画框
```

而要把项目转成：

> **Jetson Orin Nano 端侧视觉推理引擎优化项目。**

PPE 只是 workload。

YOLO11n 只是测试模型。

真正研究对象变成：

```text
模型
↓
计算图
↓
数值精度
↓
算子
↓
内存
↓
Runtime
↓
GPU 调度
↓
Profiling
↓
系统性能
```

---

# 4. 当前能力矩阵

| 能力 | 当前状态 | 后续决策 |
|---|---|---|
| 数据集 / 模型训练 | 已完成 | 不再扩展 |
| 模型结构消融 | 已完成 P2 / Rep / Attention / Focal | 不再扩展 |
| PyTorch → ONNX | 已完成 | 保留 |
| ONNX 一致性 | 已完成 | 保留 |
| TensorRT FP32 / FP16 | 已完成 | 保留 |
| TensorRT INT8 PTQ | 工程完成，但候选 REJECT | 继续深入 |
| TensorRT C++ Runtime | 已完成 | 作为后续基础 |
| CUDA Kernel | 已有融合预处理 | 继续深入 |
| Pinned Memory | 未完成 | 必做 |
| cudaMemcpyAsync | 未完整形成优化链 | 必做 |
| CUDA Stream | 已使用，但没有真正流水优化 | 必做 |
| Double / Triple Buffer | 未完成 | 必做 |
| CUDA Graph | trtexec 用过，自己 Runtime 未集成 | 推荐 |
| Nsight Systems | 未形成正式实验 | 必做 |
| Nsight Compute | 未形成 Kernel 深度分析 | 推荐 |
| GPU 后处理 | 未完成 | 推荐 |
| TensorRT IPluginV3 | 未完成 | 高优先级 |
| ONNX GraphSurgeon | 未完成 | 高优先级 |
| INT8 Layer Sensitivity | 未完成 | 高优先级 |
| Mixed Precision | 未完成 | 高优先级 |
| QAT | 未完成 | 视 Mixed Precision 结果 |
| TVM / MLIR | 未完成 | 本项目暂不做 |

---

# 5. 后续总体路线

原计划：

```text
Exp13
项目收尾 / README / 简历 / 面试
```

暂时取消。

重新规划为：

```text
Exp13  Nsight Systems 端到端性能瓶颈画像
↓
Exp14  Pinned Memory + Async + Double Buffer 流水线
↓
Exp15  CUDA GPU 后处理与 D2H 压缩
↓
Exp16  TensorRT IPluginV3 + GraphSurgeon
↓
Exp17  INT8 Layer Sensitivity + Mixed Precision
↓
Exp18  CUDA Graph / 最终 Runtime 优化
↓
Exp19  最终综合 Benchmark
↓
Exp20  README / 简历 / 面试材料
```

如果时间不足，则优先保留：

```text
Exp13
Exp14
Exp16
Exp17
```

这是秋招前最高价值组合。

---

# 6. Exp13：Nsight Systems 端到端瓶颈分析

## 6.1 核心目标

不要先修改性能代码。

先回答：

> 当前 Exp12 同步版 Runtime 到底慢在哪里？

需要对完整 C++ 推理程序加入 NVTX Range：

```text
capture
h2d
preprocess
tensorrt
d2h
decode
nms
output
```

然后使用：

```text
Nsight Systems
CUDA Event
TensorRT profiling
trtexec --dumpProfile
trtexec --dumpLayerInfo
```

形成完整时间线。

---

## 6.2 两种测试场景

### 场景 A：文件视频无限速

用于测试系统最大吞吐：

```text
sync=false
不限制原视频 FPS
尽快消费
```

主要关注：

```text
最大 throughput
GPU 空洞
CPU 等待
Memcpy 与 Kernel 是否重叠
enqueue 间隔
```

### 场景 B：IMX219 30 FPS

用于测试真实实时链路：

```text
1920×1080 @ 30 FPS
appsink max-buffers=1
drop=true
```

主要关注：

```text
P50
P95
P99
jitter
摄像头等待占比
GPU 是否实际成为瓶颈
```

---

## 6.3 必须回答的问题

最终总结必须回答：

```text
1. CPU 在哪里阻塞？
2. CUDA Stream 是否存在长时间 idle？
3. H2D 与 Kernel 是否重叠？
4. D2H 是否阻塞下一帧？
5. TensorRT enqueue 前后是否存在 GPU bubble？
6. CPU 后处理占多少比例？
7. 摄像头模式为什么 FPS 接近 30？
8. 文件模式下系统真正最大吞吐是多少？
9. 当前瓶颈属于：
   - compute-bound
   - memory-bound
   - synchronization-bound
   - input-rate-bound
   中的哪一种？
```

---

## 6.4 预期输出

```text
results/profiling/exp13_*/
├── nsys_report.qdrep
├── nsys_stats.txt
├── nvtx_summary.csv
├── trt_layer_profile.txt
├── timeline_summary.json
├── benchmark.csv
└── summary.md
```

---

## 6.5 验收标准

Exp13 不要求“变快”。

PASS 条件是：

```text
可以明确定位主要同步点
可以明确定位主要 memcpy
可以明确各阶段比例
可以识别 GPU idle 区域
可以形成后续优化假设
```

Exp13 是后续所有优化实验的基准。

---

# 7. Exp14：Pinned Memory + Async + Double Buffer

## 7.1 实验目的

将当前相对串行的数据流：

```text
Frame N:
H2D
→ preprocess
→ TensorRT
→ D2H
→ CPU postprocess
→ Frame N+1
```

改造成部分流水：

```text
Frame N
H2D → Preprocess → TensorRT → D2H

Frame N+1
       H2D → Preprocess → TensorRT → D2H

CPU
Capture / Postprocess 与 GPU 工作尽可能重叠
```

---

## 7.2 必须实现

### Host 内存

对比：

```text
pageable malloc/new
vs
cudaHostAlloc / cudaHostRegister
```

### 异步传输

使用：

```cpp
cudaMemcpyAsync()
```

### CUDA Stream

至少形成：

```text
stream0
stream1
```

或者：

```text
capture thread
+
GPU inference stream
+
postprocess thread
```

### Buffer

实现：

```text
double buffer
```

如果结构清晰，可扩展：

```text
triple buffer
```

---

## 7.3 必须理解的概念

不能只让 Codex 写代码。

需要通过实验理解：

```text
pageable memory
pinned memory
DMA
cudaMemcpy
cudaMemcpyAsync
default stream
non-default stream
CUDA Event
stream dependency
cudaStreamSynchronize
cudaEventSynchronize
buffer ownership
race condition
producer-consumer
```

---

## 7.4 性能比较

至少比较：

```text
V0 当前 Exp12 同步 baseline
V1 pinned memory
V2 pinned + async
V3 pinned + async + double buffer
```

分别测试：

```text
文件视频最大吞吐
IMX219 30 FPS 实时延迟
```

---

## 7.5 重点指标

```text
mean
P50
P95
P99
FPS
GPU utilization
CPU utilization
H2D time
D2H time
GPU idle ratio
VDD_IN
temperature
```

不要只比较 FPS。

摄像头本身是 30 FPS，因此：

```text
文件模式 → 看 throughput
摄像头模式 → 看 latency / jitter / resource
```

---

# 8. Exp15：CUDA GPU 后处理与 D2H 压缩

## 8.1 当前问题

TensorRT 当前输出：

```text
[1, 7, 8400]
```

当前做法：

```text
整个 Raw Tensor D2H
↓
CPU decode
↓
confidence filter
↓
NMS
```

目标是减少：

```text
GPU → CPU
```

的数据量，并继续学习 CUDA Kernel。

---

## 8.2 第一阶段

先不要求完整 GPU NMS。

实现：

```text
Raw Tensor
↓
CUDA Decode
↓
Class Max
↓
Confidence Filter
↓
Candidate Compaction
↓
仅将候选框 D2H
↓
CPU NMS
```

这样能够直接比较：

```text
raw output D2H bytes
vs
filtered candidate D2H bytes
```

---

## 8.3 CUDA 学习重点

该实验需要重点学习：

```text
thread mapping
global memory access
coalescing
warp divergence
atomic operations
prefix sum / compaction
shared memory（若需要）
occupancy
branch behavior
```

---

## 8.4 第二阶段（可选）

如果第一阶段稳定：

```text
CUDA NMS
```

再比较：

```text
CPU NMS
vs
GPU NMS
```

---

## 8.5 Nsight Compute

至少对核心 Kernel 使用一次 Nsight Compute，观察：

```text
SM utilization
Memory Throughput
DRAM Throughput
Achieved Occupancy
Warp Stall
Branch Efficiency
L1 / L2 behavior
```

目标不是为了追求某个漂亮数字，而是学会：

> 用硬件计数器解释 Kernel 为什么快或为什么慢。

---

# 9. Exp16：TensorRT IPluginV3 + ONNX GraphSurgeon

这是项目后半段最重要的高级实验之一。

---

## 9.1 不建议的路线

不要为了“有 Plugin”重新训练一个 DCNv2 模型。

原因：

```text
训练风险高
模型收益不确定
时间不足
容易再次回到模型魔改
```

---

## 9.2 推荐 Plugin

直接使用 Exp15 已经验证的 CUDA 后处理算子：

```text
YOLO Decode
+
Confidence Filter
```

封装为：

```text
PPEDecodeFilterPlugin
```

---

## 9.3 Plugin 需要完成

至少覆盖：

```text
IPluginV3
Plugin Creator
Plugin Registry
输入输出 Shape
FP32 / FP16
Tensor Format
Workspace
enqueue()
serialize
deserialize
clone / lifecycle
CMake
.so
Engine build
Engine load
```

---

## 9.4 ONNX GraphSurgeon

使用 GraphSurgeon 修改 Exp06 ONNX：

原始：

```text
YOLO
↓
output0 [1,7,8400]
```

修改：

```text
YOLO
↓
PPEDecodeFilterPlugin
↓
compact detections
```

形成：

```text
PyTorch
↓
ONNX
↓
GraphSurgeon
↓
Custom Plugin Node
↓
TensorRT Parser
↓
Plugin Registry
↓
CUDA Kernel
↓
TensorRT Engine
```

---

## 9.5 必须验证

正确性：

```text
Python / CPU Reference
vs
CUDA Kernel
vs
TensorRT Plugin
```

至少验证：

```text
candidate count
class
confidence
bbox
max error
NaN / Inf
```

性能：

```text
原始 Runtime
vs
Plugin Runtime
```

关注：

```text
D2H bytes
postprocess latency
CPU usage
P95
P99
throughput
```

---

## 9.6 为什么 Exp16 很重要

完成前：

> 会使用 TensorRT。

完成后：

> 可以扩展 TensorRT。

这两者在简历和面试中的技术含量明显不同。

---

# 10. Exp17：INT8 Layer Sensitivity + Mixed Precision

Exp08 已经出现一个非常有价值的负向结果。

当前结果：

```text
FP16 GPU-only mean:
3.640914 ms

INT8 GPU-only mean:
2.714804 ms

INT8 latency:
-25.4362%

Engine size:
-39.8177%
```

但：

```text
mAP50-95:
下降 0.01391398

tiny+small recall:
0.79020979
→
0.48951049
```

因此 INT8 被 REJECT。

后续不能停在：

> INT8 精度掉了，所以不用。

真正需要回答：

> 哪些层对 INT8 最敏感？

---

## 10.1 第一阶段：层级敏感性分析

目标：

```text
逐层或逐模块测试高精度 fallback
```

重点模块：

```text
Backbone
Neck
Detect Head
最后若干 Conv
分类分支
回归分支
```

得到类似：

```text
Layer / Module
↓
FP16 fallback 后
↓
mAP 恢复量
tiny recall 恢复量
latency 损失
```

形成：

```text
accuracy sensitivity ranking
```

---

## 10.2 第二阶段：Mixed Precision

构建：

```text
Full FP16
Full INT8
Mixed INT8/FP16
```

例如：

```text
Backbone INT8
Detect Head FP16
```

或者：

```text
敏感层 FP16
其他层 INT8
```

---

## 10.3 最终目标

绘制：

```text
Accuracy
↑
│          FP16
│
│      Mixed
│
│
│ INT8
└──────────────→ Performance
```

找到：

```text
accuracy-performance Pareto point
```

---

## 10.4 如果 Mixed Precision 仍失败

再考虑：

```text
QAT
```

而不是一开始就直接做 QAT。

QAT 必须有明确问题驱动：

```text
PTQ 失败
↓
定位量化敏感层
↓
Mixed Precision 仍不够
↓
QAT
```

---

# 11. Exp18：CUDA Graph 与最终 Runtime 优化

## 11.1 目标

当以下执行链稳定后：

```text
H2D
↓
CUDA preprocess
↓
TensorRT
↓
CUDA postprocess
↓
D2H
```

尝试 CUDA Graph Capture。

---

## 11.2 对比

```text
Normal enqueue
vs
CUDA Graph
```

观察：

```text
CPU launch overhead
mean latency
P95
P99
jitter
throughput
```

---

## 11.3 注意

CUDA Graph 不保证一定有明显收益。

如果当前 workload：

```text
Kernel 较长
CPU launch overhead 很小
```

收益可能有限。

负向结果也允许 PASS，只要：

```text
测试公平
结论清楚
能解释原因
```

---

# 12. Exp19：最终综合 Benchmark

最终形成：

```text
Baseline
V1 FP16 C++ Runtime
V2 CUDA Preprocess
V3 Async Pipeline
V4 GPU Postprocess
V5 Plugin
V6 Mixed INT8/FP16
V7 CUDA Graph
```

对比：

| Version | mean | P95 | P99 | FPS | CPU | GPU | Power | Temp | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

还要记录：

```text
Git Commit
Engine SHA256
Precision Mode
Power Mode
Clock State
Input
Batch
Warmup
Iterations
```

---

# 13. Exp20：最终 README、简历与面试材料

等后续优化真正完成后，再统一收尾。

不要现在就写死最终简历结论。

---

# 14. 秋招前最小可执行路线

考虑时间有限，最低建议只做：

```text
Exp13
Nsight Profiling
↓
Exp14
Pinned + Async + Double Buffer
↓
Exp16
CUDA Kernel + IPluginV3 + GraphSurgeon
↓
Exp17
INT8 Sensitivity + Mixed Precision
```

CUDA Graph 可以合并到 Exp14 或 Exp18。

GPU 后处理 Kernel 可以直接作为 Exp16 Plugin 的底层实现。

也就是最终只要形成三条完整故事：

---

## 14.1 性能优化故事

```text
Nsight
↓
发现同步 / Memcpy / GPU Bubble
↓
Pinned Memory
↓
cudaMemcpyAsync
↓
Double Buffer
↓
CUDA Stream / Event
↓
CUDA Graph
↓
Benchmark
```

---

## 14.2 算子优化故事

```text
CPU Postprocess
↓
CUDA Kernel
↓
GPU Candidate Filter
↓
TensorRT IPluginV3
↓
ONNX GraphSurgeon
↓
Engine Integration
```

---

## 14.3 量化优化故事

```text
FP16
↓
PTQ INT8
↓
发现 tiny/small 精度严重下降
↓
Layer Sensitivity
↓
Mixed Precision
↓
必要时 QAT
↓
Accuracy / Performance Pareto
```

这三条路线比继续训练多个 YOLO 变体更有价值。

---

# 15. 后续明确停止的方向

## 15.1 不继续 YOLO 模型魔改

当前已经完成：

```text
P2
Rep
Attention
Focal
```

足以证明：

```text
模型结构理解
公平消融
业务指标判断
失败候选拒绝
```

不要继续：

```text
新的 Attention
新的 Neck
新的 Loss
新的 YOLO 小改动
```

除非未来有非常明确的数据问题驱动。

---

## 15.2 不做 DeepStream 多路视频作为当前主线

原因：

```text
RK3588 项目已经充分覆盖音视频系统
Jetson 当前最缺的是 GPU 推理优化深度
```

DeepStream 可作为以后扩展，不作为秋招前高优先级。

---

## 15.3 不机械做 Pruning + Distillation

现在已有真实 INT8 精度问题。

应优先：

```text
分析 INT8 为什么失败
```

而不是继续堆：

```text
pruning
distillation
```

---

## 15.4 不在当前项目强行加入 TVM / MLIR

TVM / MLIR / AI Compiler 是另一条能力树：

```text
Graph IR
Lowering
Scheduling
Codegen
Compiler Pass
```

短期硬塞进当前 PPE 项目容易流于表面。

后续可以单独规划：

```text
CUDA
↓
Triton
↓
TVM TensorIR
↓
MLIR
```

但不作为本项目秋招前必做内容。

---

# 16. Codex 后续工作方式要求

Codex 可以继续负责大量工程实现，但后续不允许“黑盒式一次性把实验做完”。

每个实验必须包含：

```text
实验前规划
↓
当前瓶颈 / 假设
↓
设计
↓
Smoke Test
↓
正式实验
↓
Profiling
↓
正确性验证
↓
性能验证
↓
失败记录
↓
结论
↓
学习复盘
```

---

## 16.1 Codex 在每个实验开始前必须先输出

```text
1. 当前代码路径
2. 当前数据流
3. 本实验唯一主要变量
4. 预计修改哪些文件
5. 为什么这些修改可能有效
6. 正确性验收条件
7. 性能验收条件
8. 失败停止条件
```

未经审查不要直接进行大规模重构。

---

## 16.2 每次实验必须保留

```text
失败目录
run.log
return code
benchmark JSON / CSV
Git commit
SHA256
配置
正式 summary.md
```

不得：

```text
删除失败现场
覆盖旧结果
看到结果后偷偷修改验收阈值
```

---

## 16.3 代码结构要求

尽量模块化：

```text
runtime/
cuda/
plugins/
profiling/
tools/
tests/
docs/
results/
```

不要把所有逻辑塞进一个：

```text
main.cpp
```

---

# 17. 用户必须亲自掌握的内容

后续实验不能只由 Codex “完成”。

每个实验结束后，必须能够不用代码回答：

---

## CUDA 内存

```text
pageable memory 和 pinned memory 有什么区别？
为什么 pinned memory 才容易真正实现异步 DMA？
cudaMemcpy 和 cudaMemcpyAsync 有什么区别？
```

---

## Stream / Event

```text
为什么两个 Stream 可以 overlap？
什么时候其实仍然不能 overlap？
CUDA Event 和 cudaDeviceSynchronize 有什么区别？
```

---

## Double Buffer

```text
为什么需要两个 buffer？
CPU 和 GPU 怎么避免同时写一个 buffer？
什么时候会 race？
```

---

## Kernel

```text
thread / block / grid 如何划分？
global memory 是否 coalesced？
为什么 warp divergence 会影响效率？
occupancy 是不是越高越好？
```

---

## Profiling

```text
Nsight Systems 和 Nsight Compute 的区别？
timeline 上的 GPU bubble 是什么？
怎么判断 compute-bound / memory-bound？
```

---

## TensorRT

```text
Builder
Engine
Runtime
ExecutionContext
enqueueV3
setTensorAddress
Plugin
分别是什么？
```

---

## Plugin

```text
为什么需要 serialize / deserialize？
Plugin Creator 干什么？
Plugin 的 enqueue 在什么时候执行？
GraphSurgeon 为什么需要插入自定义节点？
```

---

## INT8

```text
PTQ 是什么？
Calibration 在估计什么？
为什么某些层量化敏感？
Q/DQ 节点有什么意义？
Mixed Precision 为什么可以恢复精度？
QAT 与 PTQ 的区别？
```

---

# 18. 项目最终定位

原来的名称：

> 基于 Jetson Orin Nano Super 的 PPE 小目标检测与 TensorRT/CUDA 推理优化

后续可以继续保留业务标题，但技术定位建议升级为：

> **基于 Jetson Orin Nano Super 的端侧视觉推理引擎与 TensorRT/CUDA 性能优化**

或者：

> **Jetson Orin Nano 端侧视觉推理优化：TensorRT、CUDA Kernel、异步流水与低精度量化**

项目核心不再是：

```text
PPE 检测做得多准
```

而是：

```text
如何在 Jetson GPU 上
将一个训练模型
逐层变成
高效、可解释、可验证的端侧推理系统
```

---

# 19. 后续简历能力目标

完成上述关键实验后，希望项目可以真实支撑以下能力描述：

```text
PyTorch / ONNX / TensorRT
TensorRT C++ Runtime
FP32 / FP16 / INT8
PTQ / Mixed Precision / QAT
CUDA Kernel
CUDA Stream / Event
Pinned Memory
cudaMemcpyAsync
Double Buffer
CUDA Graph
Nsight Systems / Nsight Compute
TensorRT IPluginV3
ONNX GraphSurgeon
C++17 / CMake
Jetson Orin
GStreamer / IMX219
性能 / 功耗 / 温度 / 稳定性 Benchmark
```

只有真正完成后才能写入简历。

---

# 20. 当前项目与目标岗位映射

## 20.1 嵌入式 AI / Edge AI

当前已经可以投。

重点补强：

```text
CUDA
性能分析
内存
异步流水
```

---

## 20.2 TensorRT / 模型部署工程师

当前已经比较接近。

完成：

```text
Nsight
Async
Plugin
Mixed Precision
```

后竞争力会明显提高。

---

## 20.3 AI Infra / 推理优化

当前属于有基础但深度不足。

需要重点补：

```text
Profiler
Memory
Concurrency
Kernel
Plugin
Quantization Sensitivity
```

---

## 20.4 CUDA 算子工程师

当前不作为主投方向。

后续至少还需要多个真实 Kernel 优化案例，并真正掌握：

```text
coalescing
shared memory
bank conflict
warp divergence
occupancy
arithmetic intensity
memory-bound
compute-bound
```

---

## 20.5 AI Compiler

当前不作为主投方向。

不能因为使用 TensorRT 就表述为：

```text
AI Compiler
Compiler Optimization
```

后续若学习 TVM / MLIR，再单独形成新项目或学习路线。

---

# 21. 最终执行原则

后续所有实验遵循：

> **Measure → Identify Bottleneck → Optimize → Verify Correctness → Re-profile → Decide**

不能变成：

```text
看到一个 CUDA API
↓
加进去
↓
FPS 高一点
↓
宣布优化成功
```

任何优化必须回答：

```text
为什么做？
瓶颈证据是什么？
改了哪一层？
性能为什么变化？
正确性是否保持？
有没有新的 trade-off？
```

---

# 22. 当前下一步

不要开始原来的“Exp13 项目收尾”。

下一实验直接定义为：

# Exp13：Nsight Systems 端到端性能瓶颈分析与优化基线

以 Exp12 当前同步版本作为冻结 baseline。

先：

```text
不优化
只 Profiling
```

得到真正的 GPU/CPU 时间线之后，再决定 Exp14 中：

```text
Pinned Memory
Async
Double Buffer
Stream
CUDA Graph
```

的具体改动顺序。

后续所有性能优化必须由 Exp13 的实际 Profiling 数据驱动，而不是凭经验直接修改。

---

# 23. 参考当前项目文档

Codex 在执行前应重点阅读：

```text
00_project_scope.md
01_environment.md
02_YOLO11n基线训练与评估总结.md
03_YOLO11n-P2小目标结构消融总结.md
04_YOLO11n部署可重参数化结构消融总结.md
05_YOLO11n轻量注意力与Focal损失消融总结.md
06_YOLO11n基线ONNX导出与一致性验证总结.md
07_YOLO11n基线TensorRT_FP32_FP16部署与验证总结.md
08_YOLO11n基线INT8_PTQ部署与验证总结.md
09_TensorRT_CPP_Runtime部署与验证总结.md
10_CUDA融合预处理与验证总结.md
11_视频摄像头端到端推理总结.md
12_Jetson性能功耗温度稳定性总结.md
experiment_index.md
项目全流程快速学习手册.md
```

并以现有代码、日志、Git 历史和正式实验结果为事实来源。

---

# 24. 一句话给 Codex 的总指令

> Exp00～Exp12 已经完成“模型选择 → ONNX → TensorRT → C++ → CUDA 预处理 → 摄像头 → 稳定性”的基础部署链。后续不要继续堆模型功能，也不要立即收尾。请从 Exp13 开始，把项目切换到真正的 GPU 推理性能工程阶段：先用 Nsight 对现有同步 Runtime 做瓶颈画像，然后围绕 Pinned Memory、Async、Double Buffer、GPU 后处理、TensorRT IPluginV3、GraphSurgeon、INT8 敏感层分析和 Mixed Precision 逐层深入；任何优化都必须以 Profiling 数据、正确性对照和正式 Benchmark 为依据。
