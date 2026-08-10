# Jetson PPE 项目后续深化路线与秋招能力补强方案

> 文档版本：V3（2026-08-09依据Exp16真实结果重校准）。
> 文档地位：本文件是仓库中后续技术路线、能力边界、优先级和验收原则的唯一 canonical 文档。
> 当前阶段：Exp00～Exp16、Postprocess Gain Attribution Gate及Exp16 Deployment Semantic
> Revalidation Gate已完成；Exp17～Exp20尚未执行。
> 核心方法：**Measure → Identify → Optimize → Verify → Re-profile → Accept/Reject**。
> 事实边界：计划、实现、验证和主线采用必须分开表述；没有代码和实验产物支撑的能力不得写成成果。

---

# 1. 文档用途与事实来源

本文件用于统一：

```text
项目技术定位
当前真实状态
后续实验顺序
工程与性能验收
能力—证据映射
秋招表达边界
Codex 执行约束
```

事实优先级为：

```text
代码、正式日志、指标、SHA256、Git 历史
↓
各实验正式总结与项目全流程快速学习手册
↓
README / ROADMAP / experiment_index
↓
本路线文档中的规划与解释
```

发生冲突时，不得用路线文档覆盖真实实验结果。旧版 V1、V2 和 supplement 不再作为同时有效的
路线来源；本文件完成合并后，仓库只保留这一份 canonical 路线文档。

---

# 2. 项目定位与能力边界

业务名称继续使用：

> 基于 Jetson Orin Nano Super 的 PPE 小目标检测与 TensorRT/CUDA 推理优化。

技术定位升级为：

> **Jetson Orin Nano 端侧视觉推理优化：TensorRT、CUDA Kernel、Profiling、Plugin 与低精度推理。**

PPE 是 workload，YOLO11n 是冻结模型。真正研究对象是：

```text
Model
↓
Graph
↓
Precision
↓
CUDA Operator
↓
Memory
↓
Runtime
↓
Scheduling
↓
Profiling
↓
System
```

## 2.1 与 RK3588 项目的边界

RK3588 项目重点体现：

```text
V4L2 / RGA / RKNN / NPU 多核 / MPP
H.264 / ALSA / AAC / FFmpeg / RTSP / MP4
多线程 / 音视频同步 / 嵌入式 Linux 音视频系统
```

Jetson 项目重点体现：

```text
ONNX / TensorRT 10 / TensorRT C++ Runtime
FP16 / INT8 / Explicit Q/DQ / Mixed Precision
CUDA Kernel / CUB / GPU Memory
CUDA Stream / Event / Graph
Nsight Systems / Nsight Compute / NVTX
IPluginV3 / ONNX GraphSurgeon
性能建模、正确性验证和端侧 GPU 推理优化
```

不再把 Jetson 项目扩张为另一个摄像头—模型—画框或多路音视频系统项目。

---

# 3. 当前已完成链路与部署主线

截至 Exp16，已完成：

```text
数据集审计
→ YOLO11n baseline 与 P2/Rep/Attention/Focal 消融
→ 冻结原始 YOLO11n
→ PyTorch → ONNX → ONNX Runtime 一致性
→ TensorRT FP32 / FP16 / INT8 PTQ
→ TensorRT C++ Runtime
→ CUDA 融合预处理
→ 文件视频 / IMX219 端到端推理
→ 延迟 / P95 / P99 / 功耗 / 温度 / 30 分钟稳定性
→ Nsight Systems 端到端瓶颈画像
→ Pinned / Stream / Event / Double Buffer 消融
→ CUDA Atomic/CUB Decode/Filter/Compaction
→ Nsight Compute Kernel 分析
→ TensorRT IPluginV3 / Creator / Registry / 显式workspace
→ ONNX GraphSurgeon自定义图与独立新进程Plugin加载
→ synthetic / fixture / dual-output同Engine数学零误差
→ 跨独立Engine部署语义Gate与rebuild/tactic混杂变量诊断
```

当前模型主线：

```text
Exp02 原始 YOLO11n baseline
→ Exp06 静态 ONNX
→ Exp07 Jetson 本机构建 FP16 Engine
```

当前 Runtime 主线：

```text
GStreamer / OpenCV Host Frame
→ CUDA 融合预处理
→ TensorRT FP16 enqueueV3
→ Exp15 CUB stable GPU Decode/Filter/Compaction
→ count D2H + 可变长 Candidate D2H
→ CPU class-aware NMS
```

Exp08 Full INT8 因精度退化 `REJECT`；Exp14 双缓冲异步路径因尾延迟退化 `REJECT`。二者保留代码和
负向证据，但不进入当前部署主线。

Exp16 Plugin组件为`IMPLEMENTED + VERIFIED`，但原四输出全图候选在跨Engine部署语义Gate中
`REJECTED`，未进入当前Runtime主线。普通无Plugin rebuild也相对冻结Exp07 Engine出现raw漂移，说明
跨build比较必须显式控制TensorRT rebuild/tactic selection变量；这既不能把Exp16简化成“Plugin失败”，
也不能让候选自动通过系统级采用Gate。

---

# 4. Exp13～Exp15 性能工程证据链

## 4.1 Exp13：先测量，再优化

Exp13 通过 NVTX、Nsight Systems、CUDA Event 和 TensorRT layer profile 建立同步 Runtime 基线：

```text
文件：synchronization-bound
相机：input-rate-bound + synchronization-bound
Kernel/Memcpy overlap：0
```

文件 wall FPS 约61.583；相机约30.174 FPS。文件 GPU idle 约33.81%，相机约63.31%。

必须保留的解释边界：

```text
Host API Duration
≠ GPU Activity Duration
≠ Critical Path Contribution
```

例如 D2H Host Range 可包含等待前序 Stream 工作的时间，不能把整个 Host Range 都解释为物理复制耗时。

## 4.2 Exp14：工程实现成功，性能采用失败

Exp14 的状态应分层表达：

```text
Engineering Implementation : IMPLEMENTED
Dependency Correctness      : VERIFIED
Cross-frame Overlap         : VERIFIED
Performance Gate            : REJECTED
Mainline Adoption           : REJECTED
```

Variant C 文件吞吐只提升约4.51%，但 P95 退化159.28%；相机 P95 退化173.51%。原因包括：

```text
OpenCV pageable frame → pinned staging 的额外 CPU memcpy
YOLO11n batch=1 可重叠窗口较小
Stream/Event/slot/retirement 调度成本
排队提高 throughput 机会但扩大单帧驻留时间
```

Exp14 isolation audit 降为 optional/post-resume，不阻塞Revalidation Gate或Exp17～Exp20，也不改变当前
主线 `REJECTED` 结论。

## 4.3 Exp15：系统最优不等于单 Kernel 最优

Exp15 将 raw `[1,7,8400]` 的 class-max、filter、box decode 和 compaction 移到 GPU，保留 CPU NMS。

三轮文件结果：

| Variant | Wall FPS | E2E mean | E2E P95 | D2H B/frame |
|---|---:|---:|---:|---:|
| Baseline | 60.270 | 15.502 ms | 16.940 ms | 235,200.00 |
| Atomic | 70.201 | 13.310 ms | 14.668 ms | 263.84 |
| CUB | 71.838 | 12.939 ms | 14.403 ms | 263.84 |

CUB 相对 baseline：文件 FPS +19.19%，mean -16.53%，P95 -14.97%，D2H -99.89%；相机 P95 仅退化
1.61%。文件检测 digest 保持冻结值，因此 CUB 路径 `ACCEPTED`。

Nsight Compute 显示 Atomic 单 Kernel 约23.90 µs；CUB decode/init/select 约21.12/11.94/35.01 µs。
Atomic 局部 Kernel 更短，但 CUB 保持稳定顺序，避免 CPU 恢复排序，最终 Runtime 更快。主要 Kernel
waves/SM不足1且 SM/Memory throughput未饱和，当前更接近小 workload/固定 launch 成本约束。

不得把全部19.19%提升简单归因于235 KB D2H缩减。Exp13 的真实 copy与CPU decode/NMS量级远小于
Exp15 的同日 E2E差值；该因果边界已由第7节完成的Postprocess Gain Attribution Gate进一步校准。

---

# 5. 状态模型与能力—证据矩阵

## 5.1 四种能力状态

- `IMPLEMENTED`：已有代码或工程链路，但不代表正确或适合主线；
- `VERIFIED`：已通过冻结输入、正确性、生命周期或 Profiling 验证；
- `ACCEPTED`：已满足预先冻结的采用条件，进入当前主线；
- `REJECTED`：工程或验证可能完成，但未满足采用条件，不进入主线。

单项能力可同时具有多个维度，例如 Exp14 为 `IMPLEMENTED + VERIFIED + REJECTED`。不得用一个笼统的
`PASS` 隐藏“实现成功但主线拒绝”的事实。

## 5.2 当前能力—证据矩阵

| 能力 | 状态 | 主要证据 | 简历成果边界 |
|---|---|---|---|
| PyTorch→ONNX一致性 | VERIFIED / ACCEPTED | Exp06张量与检测一致性 | 可写 |
| TensorRT FP16 | VERIFIED / ACCEPTED | Exp07精度与GPU benchmark | 可写 |
| Full INT8 PTQ | IMPLEMENTED / VERIFIED / REJECTED | Exp08速度、体积、tiny/small退化 | 可写负向决策，不可写INT8主线 |
| TensorRT C++ Runtime | VERIFIED / ACCEPTED | Exp09原始输出与三进程生命周期 | 可写 |
| CUDA融合预处理 | VERIFIED / ACCEPTED | Exp10五形状逐元素对照 | 可写 |
| 视频/IMX219链路 | VERIFIED / ACCEPTED | Exp11文件确定性与相机300帧 | 可写 |
| 性能/功耗/稳定性 | VERIFIED | Exp12三轮与54,000帧 | 可写 |
| Nsight Systems/NVTX | VERIFIED | Exp13时间线与瓶颈分类 | 可写 |
| Pinned/Stream/Event/Double Buffer | IMPLEMENTED / VERIFIED / REJECTED | Exp14 overlap与P95退化 | 可写实现和拒绝原因 |
| CUDA Atomic/CUB后处理 | VERIFIED | Exp15 synthetic、视频、NCU | 可写 |
| CUB stable后处理主线 | ACCEPTED | Exp15三轮性能、digest与相机P95 | 可写 |
| IPluginV3、Creator/Registry、显式workspace | IMPLEMENTED / VERIFIED | Plugin `.so`、Engine、fixture/synthetic/dual同Engine | 可写工程闭环，不可写已采用加速 |
| ONNX GraphSurgeon自定义图 | IMPLEMENTED / VERIFIED | 四输出ONNX、Parser/Engine构建、独立进程加载 | 可写工程闭环，不可写主线采用 |
| Plugin全图替换Exp15 B | REJECTED | 原150帧Gate 151 vs 153及超限框差 | 只能写拒绝决策 |
| Exp16 Deployment Semantic Revalidation | VERIFIED / REJECTED | candidate forensic、Hungarian匹配、219图build variance与三轮动态调频性能 | 可写严谨验证与拒绝决策，不可写已采用加速 |
| Explicit Q/DQ / Mixed Precision | —（尚无IMPLEMENTED/VERIFIED证据） | 尚无敏感性与Pareto结果 | 不可写成果 |
| Runtime CUDA Graph | —（尚无IMPLEMENTED/VERIFIED证据） | trtexec使用不能替代自有Runtime集成 | 不可写成果 |
| GPU NMS / NVMM zero-copy | —（停止扩展） | 无正式证据且不进入当前范围 | 不可写成果 |

此矩阵必须随实验更新。只有存在可追溯代码、日志、指标或哈希的能力，才允许进入最终简历成果。

---

# 6. 当前能力缺口与岗位映射

## 6.1 当前最高价值缺口

```text
跨Engine检测语义匹配与TensorRT build/tactic variance估计
Plugin系统级部署采用决策与复杂度评估
量化 activation/dynamic-range/clipping 诊断
P3/P4/P5 与 cls/reg/DFL 模块敏感性
Explicit Q/DQ 与 Mixed Precision Pareto
由 Nsight 证据驱动的 CUDA Graph
最终统一条件下的系统 Benchmark
```

## 6.2 岗位映射

- 嵌入式 AI / Edge AI：当前已经具备投递基础，继续强化 GPU Runtime、正确性和系统性能解释；
- TensorRT / 模型部署：Exp16组件和Revalidation闭环已完成；Exp17是下一关键补强；
- AI Infra / 推理优化：重点是 critical path、内存、调度、Kernel、Plugin、量化和可重复 Benchmark；
- CUDA 算子：当前只有融合预处理和后处理两个案例，不应夸大为通用算子专家；
- AI Compiler：使用 TensorRT/GraphSurgeon 不等于掌握编译器 pass、lowering、scheduling 或 codegen。

---

# 7. Postprocess Gain Attribution Gate（不新增 Exp15.1 编号）

## 7.1 目的

在 Exp16 前用小型审计拆分：

```text
GPU decode/filter 收益
vs
CUB compaction + D2H缩减收益
vs
动态调频、视频解码和运行顺序噪声
```

该 Gate 补充 Exp15 因果解释，不改变 Exp15 `ACCEPTED` 状态，也不创建新的正式实验编号。

## 7.2 三条路径

```text
P0 Raw Baseline
raw 235,200 B D2H → CPU class-max/filter/decode/NMS

P1 GPU Decode Only
GPU class-max/filter/decode
→ 固定 8400×28 B Candidate Buffer D2H（恰好235,200 B）
→ invalid candidate 使用 index=-1 等冻结语义表示
→ CPU跳过invalid并执行同一NMS

P2 GPU Decode + Compact
GPU class-max/filter/decode
→ CUB stable compaction
→ count + 可变长 Candidate D2H
→ 同一CPU NMS
```

P1 不增加独立 valid array，避免改变传输字节口径。必须初始化所有无效位置，验证 index=-1 不会进入
排序/NMS，且 P0/P1/P2 最终检测 digest一致。

## 7.3 方法与证据

先用固定 raw fixture 做低方差 microbenchmark，再做同日 E2E paired/interleaved 测试。推荐顺序：

```text
P0 → P1 → P2 → P2 → P1 → P0
```

记录：Kernel、raw/fixed/variable D2H、count copy、count-sync、payload copy、CPU decode/filter、CPU NMS、
host blocking、E2E、P95、wall FPS、温度和频率状态。报告每组配对差值、均值、median、CV和min/max，
不挑最好一轮。

## 7.4 2026-08-09 实际结果

Gate已完成且不新增实验编号。最终公平版本令P0/P1都使用pinned Host buffer并都复制235,200 B；P2平均
复制263.84 B。P0→P1和P1→P2的P95三轮方向一致，paired平均分别为−3.05%和−1.11%；但FPS和mean
均出现跨轮正负混合，不能精确分摊Exp15的19.19%。冻结fixture中P0/P1/P2 total mean为
0.2295/0.3634/0.1201 ms，说明fixed 8400项Host扫描有明显局部成本，CUB compact路径局部最优。

早期pageable P0与pinned P1的比较存在Host memory混杂，已保留但不用于最终归因。当前事实边界为：
Exp15的系统收益来自pageable raw路径、CPU全量decode与完整payload被联合替换；P2仍为ACCEPTED主线，
但不得把全部收益归因于D2H缩减，也不得声称已得到严格可加的因果百分比。Exp16及其Deployment
Semantic Revalidation Gate均已完成，当前下一优先级为Exp17。

---

# 8. Exp16：组件与部署语义已验证、性能采用已拒绝

## 8.1 永久保留的原Exp16裁决

Exp16不是简单的“Plugin失败”。以下能力已经由真实代码、产物和运行证据证明：

```text
IPluginV3 / Creator / Registry
ONNX GraphSurgeon custom domain与四输出图
仅支持真实需求的FP32 raw input ABI
显式TensorRT workspace，enqueue内无逐帧cudaMalloc/cudaFree
Plugin .so与Engine序列化
独立新进程dlopen(.so) → deserialize → enqueueV3
synthetic / frozen raw fixture
dual-output同一Engine内raw CPU decode与Plugin逐项零误差
```

因此Plugin工程和数学组件为`IMPLEMENTED + VERIFIED`。Compute Sanitizer因Jetson禁用GPU debugging
features而无法形成memcheck/racecheck PASS，也不得误表述为覆盖全部host/device race。

原150帧部署语义Gate仍永久为`REJECTED`：冻结Exp07+Exp15 control产生151个检测，Fresh Plugin Engine
产生153个检测，frame27和frame40各多一个刚跨过0.25阈值的person；旧比较器报告共有检测最大box差
138 source pixels。正确性优先停止条件触发，剩余两轮正式性能未运行，首轮速度只能作为失败现场，不能
宣称Plugin加速。

普通无Plugin control rebuild相对冻结Exp07 Engine也出现raw漂移；dual与control、dual与冻结Engine之间
同样不满足严格逐值等价。这证明跨独立Engine比较混入TensorRT rebuild/tactic selection变量，但不改变
原Exp16 `REJECTED`，也不能把Plugin自动判为`ACCEPTED`。

## 8.2 Exp07与fresh rebuild的证据边界

```text
冻结Serialized Exp07 Engine的重复执行稳定性
≠
同一ONNX、TensorRT版本和显式Builder参数fresh build后的raw bitwise一致性
```

Exp07无需重做。前者由Exp07/Exp09/Exp11的冻结Engine重复执行证据支持；后者已被Exp16普通control
rebuild反例否定。今后跨build比较必须记录Engine SHA256、Builder参数和build次数，并用检测级匹配、
模型级精度和build variance，而不是要求raw bitwise一致或依赖CSV行号。

## 8.3 Deployment Semantic Revalidation Gate（已完成，不新增实验编号）

该Gate只判断Plugin是否可能进入部署主线，不修改原Exp16 REJECT，不重写Plugin，不继续调CUDA Kernel，
也不重复Postprocess Gain Attribution Gate。

第一阶段只做frame27、frame40和138 px差异的forensic：

```text
candidate index与class
network-coordinate raw box与confidence
threshold crossing前后状态
inverse-letterbox geometry与source box
NMS输入、抑制关系、排序与最终输出
```

目标是区分“真实同候选框漂移”与“按detection_index/CSV行号错配导致的假大差”。跨Engine检测比较统一
改为`image + class + IoU cost`并使用Hungarian assignment，显式报告matched/unmatched，不再把CSV行号
当作检测身份。

第二阶段在同一219张test上比较：

```text
F0  Frozen Exp07 Engine + Exp15 CUB
B1  Fresh baseline rebuild #1 + Exp15 CUB
B2  Fresh baseline rebuild #2 + Exp15 CUB
P   Fresh Plugin Engine
```

至少两个普通baseline rebuild用于估计TensorRT build variance。冻结同一预处理、conf=0.25、NMS IoU、
匹配IoU、数据集和评估代码，报告P/R、mAP50、mAP50-95、固定阈值TP/FP/FN、tiny recall、small recall、
tiny+small recall、unmatched rate、matched bbox IoU分布和confidence delta分布；同时保留三个检测尺度及
临界threshold crossing审计。

只有当Plugin的模型级精度退化不差于普通baseline rebuild波动，并且paired/interleaved动态调频性能、
部署复杂度和维护成本满足事前采用条件时，Plugin主线状态才可改为`ACCEPTED`。否则保持组件
`IMPLEMENTED + VERIFIED`、主线`REJECTED`。任何结果都不得覆盖原Exp16正式失败目录、阈值和裁决。

## 8.4 2026-08-09 Revalidation真实结果

R1/R2确认frame27和frame40的差异均来自candidate 8222在`conf=0.25`附近发生threshold crossing；旧报告
的138 source pixels最大框差是额外检测导致CSV行号错配，并非同一候选框真实漂移。跨Engine比较已改为
确定性的`image + class + IoU≥0.50` Hungarian matching。

同一219张test、840个GT上的F0/B1/P/B2正式比较中，四者Recall均为0.87023810，tiny+small recall均为
0.79020979；Plugin的P/R、Gate-local AP、TP/FP/FN、unmatched rate、matched bbox IoU和confidence delta
均落在两个普通baseline rebuild相对冻结F0形成的build variance envelope内。因此系统级检测语义记为
`VERIFIED`。这里的AP因固定`conf=0.25`硬下限，仅用于Gate内公平比较，不替代Exp02/Exp07的标准全阈值mAP。

三轮25W动态调频paired/interleaved性能中，Plugin聚合wall FPS相对F0为−1.064788%，E2E mean为
+1.305272%；仅1/3轮FPS和1/3轮mean方向有利，未达到预冻结的稳定收益门槛。虽然三轮P95回归均未超过
5%，仍不足以抵偿额外Plugin ABI、`.so`加载和Engine维护复杂度。最终保持组件与语义
`IMPLEMENTED + VERIFIED`，性能采用和Runtime主线`REJECTED`；Exp15 CUB stable compaction继续为
`ACCEPTED`主线，不重跑挑选最好轮次，也不继续调整Plugin CUDA Kernel。

---

# 9. Exp17：量化诊断、敏感性与 Mixed Precision

Exp08 已完成256图校准集合及 tiny/small覆盖审计，不从“标签分布是否代表”重新开始。Exp17 首先审计
实际量化实现是 Explicit Q/DQ 还是 calibrator/cache路径，并保留已有负向结果：INT8更快、更小，但
tiny+small recall从0.7902降至0.4895，因此当前主线仍为FP16。

代码审计必须给出可追溯结论：若Exp08使用implicit calibrator/cache，先建立Explicit Q/DQ PTQ baseline；
若已经是Explicit Q/DQ，则直接进入后续审计。不得把校准cache存在简单等同于显式Q/DQ，也不得为了
获得更好结果删除Exp08 Full INT8 `REJECTED`证据。

## 9.1 Activation / Dynamic-range / Clipping Audit

新增重点：

```text
关键activation min/max与直方图
校准scale、zero point和dynamic range
饱和/clipping比例
FP32/FP16与INT8中间tensor误差
cosine/L2/max_abs等诊断指标
异常值对scale的影响
FP16/INT8在三个检测尺度上的score error与bbox error
relative L2与confidence threshold-crossing数量
```

不得仅从最终mAP倒推敏感层，也不得把标签覆盖审计等同于activation代表性。

## 9.2 优先敏感性分组

优先按检测尺度与功能分支，而不是逐178层暴力搜索：

```text
P3 / P4 / P5
classification branch
regression branch
DFL相关节点
完整Detect Head
必要时再扩展early/late backbone与neck
```

首轮按P3/P4/P5、classification、regression、DFL和完整Detect Head做粗粒度候选，不进行178层逐层暴力
扫描。每个候选执行Engine build、同一219-image test、tiny/small audit和GPU-only benchmark，记录P/R、
mAP50、mAP50-95、固定阈值TP/FP/FN、tiny/small/tiny+small recall、三个尺度的score/bbox error、relative
L2、threshold crossing、mean/P95与Engine size，并计算accuracy recovery per latency cost。目标是解释
tiny+small recall从0.7902降至0.4895的机制，而不只是寻找一个更好数字。

## 9.3 Explicit Q/DQ 与 Mixed Precision

若 Exp08 不是 Explicit Q/DQ，先建立可追溯Q/DQ PTQ baseline；若已经是，则直接进入模块级fallback。
从敏感性排名选择2～3个Mixed Precision方案，比较：

```text
FP16
Full INT8
Mixed-1
Mixed-2
```

以准确率—延迟 Pareto 决定是否采用。只有 Explicit PTQ、敏感性和 Mixed Precision仍不能恢复冻结的
小目标指标时，才评估QAT；不得一开始直接扩大到QAT。最终候选至少重复Build一次，并与至少一次普通
对照rebuild交叉比较，避免把TensorRT tactic variation误写成量化或fallback收益。

## 9.4 2026-08-10 真实结果与路线裁决

Exp08代码审计确认它使用`IInt8EntropyCalibrator2 + BuilderFlag.INT8 + int8_calibrator`，冻结ONNX中没有
Q/DQ。Exp17随后在Jetson以原256张train-only校准图建立Explicit Q/DQ baseline：95Q/183DQ、对称QInt8、
activation per-tensor、weight per-channel、FP32 bias与FP32 raw I/O、strongly typed TensorRT。219图
mAP50/mAP50-95/tiny+small recall为0.883443/0.528020/0.755245，三项精度门槛通过；但三轮paired
GPU-only相对FP16中位劣化12.82%，即精度恢复并未形成部署Pareto。

256图Detect Head activation审计中最大clipping ratio仅约2.49e-5，最高模拟量化relative-L2集中在P5/P3
classification而非DFL。219图同candidate-index raw审计进一步显示，旧implicit INT8在P3的score
relative-L2为0.771425，并有1138次从FP16的`>=0.25`掉到阈值下；Explicit QDQ对应值降到0.137478和93次，
这给出了tiny/small recall从0.489510恢复到0.755245的主要机制证据，但不构成单一scale算法的独占因果。

按证据构建P3-classification、all-classification、DFL和完整Detect Head四个fallback Engine。前三个完整
219图精度均通过门槛，tiny+small仍同为0.755245；两轮正反序GPU-only相对FP16中位延迟分别劣化26.71%、
40.57%、9.49%，完整Head筛选劣化57.99%。因此没有Mixed候选进入accuracy-latency Pareto，状态统一为
`IMPLEMENTED + VERIFIED + REJECTED`。不存在“最终候选”，所以不启动repeat build、动态端到端采用Gate或
QAT；冻结FP16 Engine与Exp15 CUB Runtime继续主线。简历可写显式量化/敏感性分析方法与负向决策能力，
不得写成“INT8/Mixed Precision加速落地”。

---

# 10. Exp18：CUDA Graph Decision Gate

只有Exp17完成后，在真实最终主线上重新用Nsight Systems证明Runtime存在足够明显的enqueue/launch
overhead时才实现CUDA Graph。不得复用Exp13旧时间线直接推导Exp18必做。
重点观察：

```text
enqueueV3 host duration
cudaLaunchKernel/API间隔
GPU bubble
短Kernel总量与平均时长
CPU launch interval相对GPU service time
```

TensorRT `enqueueV3()` 可用于 CUDA Graph capture，但当前链路存在：

```text
GPU compact
→ count D2H
→ CPU读取count
→ 决定payload bytes
→ variable D2H
```

这种中途CPU交互不适合整体捕获。若Gate通过，第一版只捕获固定地址、固定shape的device-side边界：

```text
CUDA preprocess → TensorRT enqueueV3 → GPU decode/filter → CUB compaction
```

H2D保持Graph外，Graph launch之后再执行count/payload D2H和CPU NMS；不得重新引入Exp14已拒绝的pinned
staging/Double Buffer复杂度。比较Normal vs Graph的文件吞吐、mean/P50/P95/P99、CPU launch overhead和
GPU idle。Engineering `VERIFIED` 与Mainline `ACCEPTED/REJECTED`继续分离；若没有明显enqueue-bound
证据，Exp18直接记为`SKIPPED_BY_EVIDENCE`，不为展示技术栈强行实现。

---

# 11. Exp19：最终联合 Benchmark

Exp19 不增加新技术，只比较baseline和已经`ACCEPTED`的最终路线。禁止自动重新加入Exp14 Double Buffer，
也不把未通过Gate的Plugin、Mixed Precision或CUDA Graph放进最终版本。

候选矩阵按实际结果收敛，例如：

```text
V0 Exp12同步基线
V1 Exp15 CUB主线
V2 Exp16 Plugin（若ACCEPTED）
V3 Exp17 Mixed Precision（若ACCEPTED）
V4 Exp18 Graph（若ACCEPTED）
V_Final 仅由ACCEPTED组件组成
```

文件视频用于测量最大吞吐。Camera受30 FPS输入节拍限制，必须重点报告`capture_wait_ms`、
`post_capture_processing_ms`、`frame_total`及P50/P95/P99，不能把约30 FPS吞吐无变化解释为优化无效或
成功。两种场景同时记录CPU/GPU利用率、功耗、温度、RSS和energy/frame。

正式部署主结果保持25W动态调频，代表默认部署行为；采用paired/interleaved顺序、至少3个独立进程。
固定时钟只作为低方差microbenchmark/代码差异诊断轨，不替代动态调频主结果，也不得混合两条轨道的数字。

只对最终`V_Final`重新执行54,000帧/约30分钟稳定性，检查性能漂移、RSS、功耗、温度、energy/frame、
NaN/Inf和Runtime错误；中间候选不重复长稳态。
所有版本记录Git commit、Engine/Plugin SHA256、功耗/时钟模式、TensorRT/CUDA版本、输入、warmup和迭代。

---

# 12. Exp20：项目收尾

Exp20 才统一完成：

```text
README最终结构
架构图与数据流图
最终Benchmark表
快速学习路线
项目讲解稿
简历要点
面试题与负向实验故事
公开仓库清理与License核查
```

不再在Exp20开发新的优化；Exp20交付完成后停止本项目当前主线开发。所有简历描述都必须回查能力—证据
矩阵，未达到`VERIFIED`的能力不得写成成果；`REJECTED`能力只能按“实现、测量、发现代价并拒绝”表述。

---

# 13. 性能实验统计与公平性规范

所有正式性能实验固定模型、Engine、输入、batch、预处理、阈值、NMS、功耗模式和计时范围，并采用：

```text
paired/interleaved顺序
至少3个独立进程
warmup与正式窗口分离
每轮起止温度/频率/时钟状态
mean + median + P50/P95/P99
CV + min/max + paired delta
失败轮不删除、不替换
```

文件视频用于最大吞吐；相机用于真实单帧延迟与jitter，并拆分`capture_wait_ms`、
`post_capture_processing_ms`和`frame_total`。动态调频是最终部署主轨；固定时钟是诊断轨。
Profiler只解释瓶颈，不与无Profiler正式性能数字混用。必须区分submit interval、GPU service time、queue
wait、submit-to-completion latency和pipeline wall throughput。

---

# 14. C++/CUDA Runtime QA 与工程技术栈

后续能力补强不只包含CUDA API，还包括：

```text
C++17 / RAII / move-only资源
CMake target与依赖边界
CUDA错误传播与异步错误检查
Stream/Event/buffer ownership
CTest与确定性fixture
Compute Sanitizer memcheck/initcheck/synccheck/racecheck（适用范围内）
ASan/UBSan（Host环境支持时）
Plugin ABI / namespace / version / symbol visibility
dlopen / ldd / RPATH / shared library部署
新进程生命周期与负向加载测试
artifact manifest / SHA256 / reproducible command
```

工具报告不是完整正确性的替代品。Device QA、Host QA、业务语义对照和端到端生命周期必须共同构成证据。

---

# 15. 必须亲自掌握的知识

## 15.1 Profiling 与性能模型

```text
NVTX、CUDA API、GPU Activity、Critical Path、GPU Bubble、Observer Effect
compute/memory/synchronization/input-rate bound
Amdahl定律、固定成本与变量成本
throughput、service time、queueing latency与tail latency
```

## 15.2 CUDA 内存与调度

```text
pageable/pinned、DMA、cudaHostAlloc/cudaHostRegister
cudaMemcpy/cudaMemcpyAsync
default/non-default stream、Event、dependency、synchronize
double buffer、slot ownership、retirement、race boundary
```

## 15.3 Kernel 与 CUB

```text
grid/block/thread/warp
coalescing、divergence、occupancy、waves/SM
SM/Memory throughput、cache行为
atomic顺序、prefix scan、stable compaction、temporary storage
```

## 15.4 TensorRT 与 Plugin

```text
Builder、Network、Engine、Runtime、ExecutionContext、enqueueV3
IPluginV3、Creator、Registry、Build/Runtime phase
shape、dtype、format、workspace、fields、namespace、version、lifecycle
GraphSurgeon node/tensor/custom domain/cleanup/toposort
```

## 15.5 Quantization 与 CUDA Graph

```text
PTQ、Q/DQ、scale、zero point、dynamic range、clipping
calibration、activation代表性、sensitivity、Mixed Precision、QAT
Graph capture/instantiate/replay、static address、CPU-dependent boundary
```

---

# 16. 最终项目故事

项目最终应形成五条有证据的故事：

1. 模型与部署正确性：模型消融→ONNX→TensorRT→FP32/FP16/INT8→一致性与采用决策；
2. Profiling与性能工程：Exp13测量→Exp14异步实现→观察overlap→尾延迟退化→拒绝；
3. CUDA算子：raw output→Atomic/CUB→NCU→局部与系统trade-off→CUB采用；
4. Plugin工程：IPluginV3/GraphSurgeon→独立进程与同Engine零误差→跨Engine混杂→系统Gate拒绝；
5. 量化恢复：INT8负向结果→activation/branch sensitivity→Mixed Precision→Pareto（完成后才能写）。

最有价值的当前闭环是：

```text
Exp13 Measure
→ Exp14 Optimize but Reject
→ Exp15 Change Direction and Accept
```

它证明会用API不等于优化成功，系统决策必须同时考虑正确性、吞吐、尾延迟和复杂度。

---

# 17. 明确停止与 Post-resume Extension

秋招前主线停止：

```text
新的YOLO结构/Attention/Loss/P2/Rep训练
Pruning/Distillation
DeepStream、多摄像头、RTSP和音视频扩张
GPU NMS
NVMM/EGLImage zero-copy
TVM/MLIR/Triton
LLM on Jetson
```

这些方向不是没有价值，而是当前ROI低或属于另一条能力树。Exp14 isolation audit是唯一明确保留的
optional/post-resume诊断项；新的YOLO结构、Attention/Loss、GPU NMS、NVMM zero-copy、DeepStream、
多摄像头、剪枝、蒸馏、TVM/MLIR/Triton不再扩展，也不阻塞Revalidation Gate与Exp17～Exp20。

## 17.1 可选 Exp14 isolation audit

若未来需要补强异步调度的因果解释，只使用预加载 Host Frame，排除 GStreamer、Camera、Video Decode、
CSV和图片输出，对比 pageable sync、pinned sync、pinned async single-slot、pinned async double-buffer。
必须分别记录 submit interval、GPU service time、queue wait、submit-to-completion latency和throughput。
无论结果如何，都不能直接推翻真实摄像头链路的 `REJECTED` 结论。

## 17.2 NVMM zero-copy（停止扩展）

当前相机路径仍经过 NVMM→BGR Host Frame→CUDA。NVMM/EGLImage/CUDA interop可能消除CPU中转，但与
GStreamer、EGL和NvBufSurface强耦合，集成与调试成本高，且当前秋招能力收益低于Plugin和量化诊断，
因此不纳入当前或Exp20后的既定开发范围。

---

# 18. Codex 后续执行规则

每个实验开始前必须先给出并等待审批：

```text
Current Dataflow
Current Bottleneck
Hypothesis
Single Main Variable
Files To Modify
Correctness Gate
Performance Gate
Stop Condition
Expected Evidence
```

执行顺序：

```text
实验前规划
→ 环境/Git/输入哈希
→ Smoke Test
→ 正确性 Gate
→ 正式 paired/interleaved实验
→ Profiling
→ 结果与失败现场
→ VERIFIED
→ 主线 ACCEPTED/REJECTED
→ 学习复盘
```

不得自动把 Pinned、Async、Plugin、Mixed Precision、CUDA Graph加入主线；不得覆盖旧结果、删除失败目录、
事后修改阈值、把Profiler数字混入正式性能、把Windows静态检查当作Jetson实验，或提前宣称未完成能力。

代码和证据保持模块化：

```text
runtime/  cuda/  plugins/  profiling/
tools/    tests/ docs/     results/
```

不要把Plugin、Runtime、CUDA测试和实验编排塞进单个 `main.cpp`。每次正式实验至少保留失败目录、run.log、
return code、配置、benchmark JSON/CSV、输入与产物SHA256、Git commit和正式总结；大型Engine、模型、
`.ncu-rep/.nsys-rep`及fixture只留目标机器，不进入普通Git。

---

# 19. 当前优先级与时间不足时的裁剪

当前推荐顺序：

```text
Priority 0  本V3 canonical文档合并
Priority 1  Postprocess Gain Attribution Gate（已完成）
Priority 2  Exp16 IPluginV3 + GraphSurgeon（组件已VERIFIED，原候选REJECTED）
Priority 3  Exp16 Deployment Semantic Revalidation Gate（已完成：语义VERIFIED、性能REJECTED）
Priority 4  Exp17 Explicit Q/DQ + Activation/Sensitivity + Mixed Precision（下一实验）
Priority 5  Exp18 CUDA Graph Decision Gate
Priority 6  Exp19 Final Benchmark
Priority 7  Exp20 Closeout并停止开发
```

如果秋招时间突然不足：

```text
Exp17最小粗粒度敏感性/Mixed Precision
→ Exp19
→ Exp20
```

Exp14 isolation audit和未通过Decision Gate的Exp18可裁剪；已完成的Postprocess Gain Attribution Gate不重复。

---

# 20. 给后续任务的总指令

> 当前项目已完成Exp00～Exp16及Postprocess Gain Attribution Gate。Exp13的计时边界可信；Exp14异步
> 工程`IMPLEMENTED + VERIFIED`但性能`REJECTED`；Exp15 CUB stable后处理`ACCEPTED`，且不能把19.19%
> 全部归因于D2H缩减；Exp16 Plugin工程、数学与独立进程闭环`IMPLEMENTED + VERIFIED`，原跨Engine部署
> 语义Gate永久`REJECTED`。其后续Revalidation已用candidate forensic、Hungarian检测匹配、219图模型级
> 指标和两个普通baseline rebuild证明Plugin检测语义落在build variance内，但三轮动态调频性能不满足采用
> 条件，因此语义`VERIFIED`、性能与主线`REJECTED`；不重写Plugin、不调CUDA Kernel。下一步Exp17先审计
> implicit calibrator/cache与explicit Q/DQ，必要时
> 建Explicit Q/DQ baseline，再做activation/clipping与P3/P4/P5、cls/reg/DFL、完整Head粗粒度敏感性和
> 2～3个Mixed Precision Pareto候选。Exp18只有在Exp17最终主线重新Nsight证明enqueue-bound时才捕获
> device-side preprocess→enqueueV3→GPU decode/filter→CUB，H2D与D2H保持Graph外；否则
> `SKIPPED_BY_EVIDENCE`。Exp19只比较baseline和ACCEPTED路线，Exp20统一收尾后停止开发。全程使用
> `IMPLEMENTED / VERIFIED / ACCEPTED / REJECTED`，未验证能力不得进入简历，旧失败和旧门槛不得改写。
