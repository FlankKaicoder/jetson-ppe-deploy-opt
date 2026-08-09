# Jetson PPE 项目后续深化路线与秋招能力补强方案

> 文档版本：V3，2026-08-08。
> 文档地位：本文件是仓库中后续技术路线、能力边界、优先级和验收原则的唯一 canonical 文档。
> 当前阶段：Exp00～Exp15 已完成；Exp16～Exp20 尚未执行。
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

截至 Exp15，已完成：

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

Exp14 isolation audit 降为 optional/post-resume，不阻塞 Exp16，也不改变当前主线 `REJECTED` 结论。

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
Exp15 的同日 E2E差值，仍需 Postprocess Gain Attribution Gate 拆解因果。

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
| IPluginV3 | 未完成 | 无Plugin `.so`/Engine/独立进程证据 | 不可写成果 |
| ONNX GraphSurgeon自定义图 | 未完成 | 无冻结修改后ONNX | 不可写成果 |
| Explicit Q/DQ / Mixed Precision | 未完成 | 尚无敏感性与Pareto结果 | 不可写成果 |
| Runtime CUDA Graph | 未完成 | trtexec使用不能替代自有Runtime集成 | 不可写成果 |
| GPU NMS / NVMM zero-copy | 未完成且非当前主线 | 无正式证据 | 不可写成果 |

此矩阵必须随实验更新。只有存在可追溯代码、日志、指标或哈希的能力，才允许进入最终简历成果。

---

# 6. 当前能力缺口与岗位映射

## 6.1 当前最高价值缺口

```text
TensorRT IPluginV3 与 Creator/Registry/Lifecycle
ONNX GraphSurgeon 自定义节点与图契约
Plugin ABI、workspace、独立进程加载和设备正确性 QA
量化 activation/dynamic-range/clipping 诊断
P3/P4/P5 与 cls/reg/DFL 模块敏感性
Explicit Q/DQ 与 Mixed Precision Pareto
由 Nsight 证据驱动的 CUDA Graph
最终统一条件下的系统 Benchmark
```

## 6.2 岗位映射

- 嵌入式 AI / Edge AI：当前已经具备投递基础，继续强化 GPU Runtime、正确性和系统性能解释；
- TensorRT / 模型部署：Exp16、Exp17 是最关键补强；
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
但不得把全部收益归因于D2H缩减，也不得声称已得到严格可加的因果百分比。下一优先级进入Exp16方案冻结。

---

# 8. Exp16：IPluginV3 + ONNX GraphSurgeon

Exp16 的目标是把 Exp15 已验证的 CUDA/CUB 算子接入 TensorRT 图，不是再发明一个新 CUDA 算子，也
不把“大幅加速”预设为成功条件。

## 8.1 开始前冻结 Postprocess ABI / Semantic Contract

正式方案必须先冻结：

```text
Plugin name / version / namespace
输入 tensor name / shape / dtype / layout
只支持当前实际需要的 FP32 raw input
class-max tie-break
confidence threshold 与边界包含规则
cxcywh → xyxy 数值语义
network coordinate / original coordinate 边界
candidate index与stable order
capacity=8400与overflow行为
NaN/Inf/非正宽高行为
输出 boxes_scores/classes/indices/count 的shape与dtype
CPU inverse-letterbox与NMS输入契约
FP32数值容差与最终检测digest
```

第一版不为展示技术栈强制增加没有实际需求的 FP16 Plugin IO。若未来真实图需要 FP16 IO，必须作为
独立扩展验证，不得用“Engine 是 FP16”推断 Plugin binding 就必然需要 half。

## 8.2 推荐输出

不向 TensorRT 暴露 opaque C++ struct，使用固定容量 Tensor：

```text
boxes_scores [1,8400,5] FP32
classes      [1,8400]   INT32
indices      [1,8400]   INT32
count        [1]        INT32
```

有效长度由 count 决定。Plugin 第一版保持 network-coordinate 输出；对少量候选的 inverse letterbox
和 class-aware NMS 继续留在 CPU，以降低动态几何输入和 GPU NMS复杂度。

## 8.3 TensorRT 10.3 实现边界

统一审计并使用当前环境真实接口：

```text
IPluginV3
IPluginV3OneCore / OneBuild / OneRuntime
IPluginCreatorV3One
PluginRegistry
Plugin Fields / getFieldsToSerialize
Build phase / Runtime phase
supportsFormatCombination或V3等效格式协商
workspace size / enqueueV3数据流
```

CUB temporary storage 必须来自显式 workspace 或经正式生命周期管理的资源；禁止在 `enqueue()` 内
逐帧 `cudaMalloc/cudaFree`。Plugin不得把可变地址、临时 Host指针或机器专用路径错误序列化进Engine。

## 8.4 GraphSurgeon 与产物链

```text
Exp06 ONNX
→ GraphSurgeon插入自定义domain/op节点并冻结custom opset约定
→ cleanup/toposort与结构审计
→ TensorRT Parser + Plugin Registry
→ Engine build/serialize
→ 独立新进程先加载Plugin .so
→ deserialize Engine
→ enqueueV3
```

记录原ONNX、修改后ONNX、Plugin `.so` 和Engine的大小/SHA256，以及生成命令、TensorRT/CUDA版本和
Git commit。Engine和大型模型产物仍不进入普通Git。

## 8.5 Device QA 与 Host QA

Device QA：

```text
固定与边界fixture
CPU Reference vs Exp15 CUDA vs Plugin
candidate count/index/class/order/confidence/box
capacity/overflow/NaN/Inf
Compute Sanitizer memcheck/initcheck/synccheck
适用时使用racecheck，并说明其覆盖边界
```

Host QA：

```text
RAII与异常路径
ASan/UBSan（环境支持时）
Creator/Registry/version/namespace错误路径
Plugin .so缺失或版本不匹配的可诊断失败
独立新进程加载 .so 后再 deserialize Engine
重复创建/销毁Runtime、Engine、Context和Plugin
ldd/RPATH/符号可见性与部署清单
```

Compute Sanitizer `racecheck` 等工具只覆盖其支持的设备访问模式，不得表述为覆盖全部 host/device race、
跨进程生命周期或业务级 buffer ownership；Host线程安全和资源所有权仍需独立测试与代码审计。

## 8.6 Exp16 Gate

Engineering Gate：Graph修改、Parser、Creator注册、Engine构建/序列化、独立进程反序列化和enqueue成功。

Correctness Gate：CPU、Exp15和Plugin的count/index/class/order完全一致，confidence/box满足冻结容差，
文件最终检测digest不变，无非法访问和生命周期错误。

Performance Gate：相对 Exp15 B 的同日 paired/interleaved测试 P95退化不超过5%。达到工程与正确性
要求可判 `VERIFIED`；只有满足采用条件且集成价值/维护成本合理，才判 `ACCEPTED`。Plugin接近Exp15 B
但没有加速时可以是“Engineering PASS / Mainline REJECTED”，不能因功能运行就自动进入主线。

---

# 9. Exp17：量化诊断、敏感性与 Mixed Precision

Exp08 已完成256图校准集合及 tiny/small覆盖审计，不从“标签分布是否代表”重新开始。Exp17 首先审计
实际量化实现是 Explicit Q/DQ 还是 calibrator/cache路径，并保留已有负向结果：INT8更快、更小，但
tiny+small recall从0.7902降至0.4895，因此当前主线仍为FP16。

## 9.1 Activation / Dynamic-range / Clipping Audit

新增重点：

```text
关键activation min/max与直方图
校准scale、zero point和dynamic range
饱和/clipping比例
FP32/FP16与INT8中间tensor误差
cosine/L2/max_abs等诊断指标
异常值对scale的影响
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

首轮控制在约6～8个候选。每个候选执行Engine build、219-image test、tiny/small audit和GPU-only
benchmark，记录mAP50、mAP50-95、tiny/small recall、mean/P95、Engine size，并计算 accuracy recovery
per latency cost。

## 9.3 Explicit Q/DQ 与 Mixed Precision

若 Exp08 不是 Explicit Q/DQ，先建立可追溯Q/DQ PTQ baseline；若已经是，则直接进入模块级fallback。
从敏感性排名选择2～3个方案，比较：

```text
FP16
Full INT8
Mixed-1
Mixed-2
```

以准确率—延迟 Pareto 决定是否采用。只有 Explicit PTQ、敏感性和 Mixed Precision仍不能恢复冻结的
小目标指标时，才评估QAT；不得一开始直接扩大到QAT。

---

# 10. Exp18：CUDA Graph Decision Gate

只有 Nsight Systems 证明 Runtime 存在足够明显的 enqueue/launch overhead 时才实现 CUDA Graph。
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

这种中途CPU交互不适合整体捕获。若 Gate 通过，优先捕获固定地址、固定shape的边界：

```text
H2D → CUDA preprocess → enqueueV3 → Plugin/GPU decode-filter
```

Graph launch之后再执行count/payload D2H和CPU NMS。比较Normal vs Graph的文件吞吐、mean/P50/P95/P99、
CPU launch overhead和GPU idle。Engineering `VERIFIED` 与Mainline `ACCEPTED/REJECTED`继续分离。

---

# 11. Exp19：最终联合 Benchmark

Exp19 不增加新技术，只组合已经 `ACCEPTED` 的能力。禁止自动重新加入 Exp14 Double Buffer，也不把
未通过Gate的Plugin、Mixed Precision或CUDA Graph放进最终版本。

候选矩阵按实际结果收敛，例如：

```text
V0 Exp12同步基线
V1 Exp15 CUB主线
V2 Exp16 Plugin（若ACCEPTED）
V3 Exp17 Mixed Precision（若ACCEPTED）
V4 Exp18 Graph（若ACCEPTED）
V_Final 仅由ACCEPTED组件组成
```

正式部署主结果保持25W动态调频，代表默认部署行为；采用 paired/interleaved顺序、至少3个独立进程，
报告mean/P50/P95/P99、wall FPS、CPU/GPU、功耗、温度、内存和精度。固定时钟只作为低方差
microbenchmark/代码差异诊断轨，不替代动态调频主结果，也不得混合两条轨道的数字。

最终主线重新执行54,000帧/约30分钟稳定性，检查性能漂移、RSS、功耗、温度、NaN/Inf和Runtime错误。
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

不再在Exp20开发新的优化。所有简历描述都必须回查能力—证据矩阵，未达到 `VERIFIED` 的能力不得写成
成果；`REJECTED` 能力只能按“实现、测量、发现代价并拒绝”表述。

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

文件视频用于最大吞吐，相机用于真实单帧延迟与jitter。动态调频是最终部署主轨；固定时钟是诊断轨。
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

项目最终应形成四条有证据的故事：

1. 模型与部署正确性：模型消融→ONNX→TensorRT→FP32/FP16/INT8→一致性与采用决策；
2. Profiling与性能工程：Exp13测量→Exp14异步实现→观察overlap→尾延迟退化→拒绝；
3. CUDA算子：raw output→Atomic/CUB→NCU→局部与系统trade-off→CUB采用；
4. 量化恢复：INT8负向结果→activation/branch sensitivity→Mixed Precision→Pareto（完成后才能写）。

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
DeepStream多路、RTSP和音视频扩张
GPU NMS
NVMM/EGLImage zero-copy
TVM/MLIR/Triton
LLM on Jetson
```

这些方向不是没有价值，而是当前 ROI 低或属于另一条能力树。Exp14 isolation audit、NVMM zero-copy、
GPU NMS、多摄像头、更多CUDA算子、TVM/MLIR等进入 optional/post-resume，不阻塞 Exp16～Exp20。

## 17.1 可选 Exp14 isolation audit

若未来需要补强异步调度的因果解释，只使用预加载 Host Frame，排除 GStreamer、Camera、Video Decode、
CSV和图片输出，对比 pageable sync、pinned sync、pinned async single-slot、pinned async double-buffer。
必须分别记录 submit interval、GPU service time、queue wait、submit-to-completion latency和throughput。
无论结果如何，都不能直接推翻真实摄像头链路的 `REJECTED` 结论。

## 17.2 可选 NVMM zero-copy

当前相机路径仍经过 NVMM→BGR Host Frame→CUDA。NVMM/EGLImage/CUDA interop可能消除CPU中转，但与
GStreamer、EGL和NvBufSurface强耦合，集成与调试成本高，且当前秋招能力收益低于Plugin和量化诊断，
因此只作为post-resume扩展。

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
Priority 2  Exp16 IPluginV3 + GraphSurgeon
Priority 3  Exp17 Activation/Sensitivity + Mixed Precision
Priority 4  Exp18 CUDA Graph Decision Gate
Priority 5  Exp19 Final Benchmark
Priority 6  Exp20 Closeout
```

如果秋招时间突然不足：

```text
Postprocess Gain Attribution Gate
→ Exp16
→ Exp17最小模块级敏感性/Mixed Precision
→ Exp19
→ Exp20
```

Exp14 isolation audit和未通过Decision Gate的Exp18可裁剪。

---

# 20. 给后续任务的总指令

> 当前项目已完成 Exp00～Exp15。Exp13 通过 Nsight Systems 定位同步与输入节拍边界；Exp14 完成
> Pinned/Stream/Event/Double Buffer并验证overlap，但因尾延迟严重退化而主线拒绝；Exp15 完成Atomic
> 与CUB GPU Decode/Filter/Compaction，CUB stable路径通过正确性、NCU和三轮Benchmark并成为当前FP16
> Runtime主线。下一步先完成不新增实验编号的Postprocess Gain Attribution Gate，再设计Exp16。Exp16
> 必须先冻结FP32 raw input的Postprocess ABI/semantic contract，使用TensorRT 10.3 IPluginV3与
> GraphSurgeon，显式管理CUB workspace，验证Device/Host QA及独立新进程加载 `.so` 后反序列化Engine。
> Exp17基于已有256图覆盖审计，转向activation/dynamic-range/clipping以及P3/P4/P5、cls/reg/DFL
> 敏感性和Mixed Precision。Exp18只有在Nsight证明enqueue/launch overhead足够明显时才实现CUDA Graph。
> Exp19只组合ACCEPTED能力，并以动态调频paired/interleaved结果作为部署主结论；Exp20统一收尾。
