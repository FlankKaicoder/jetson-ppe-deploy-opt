# Project Roadmap

各阶段的原理、实验复盘、设备重连 SOP 和下一实验预先规划统一维护在
`docs/项目全流程快速学习手册.md`；本文件只维护里程碑状态。

## M0：基线与结构消融（已完成）

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp00 | 项目初始化与范围冻结 | PASS |
| Exp01 | Jetson 环境审计与 C++/CUDA/TensorRT 编译验证 | PASS |
| Exp02 | PPE 数据集审计、YOLO11n 基线与独立评估 | PASS |
| Exp03 | YOLO11n-P2 小目标结构消融 | REJECT |
| Exp04 | 部署可重参数化结构消融 | REJECT |
| Exp05 | 轻量注意力与 Focal 分类损失消融 | REJECT |

`REJECT` 表示实验工程链路已完成，但候选方案未满足替换原始 YOLO11n 基线的
综合验收条件。负向实验与结果继续保留。

## M1：模型转换与 TensorRT

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp06 | PyTorch → ONNX 导出与一致性验证 | PASS |
| Exp07 | Jetson TensorRT FP32 / FP16 Engine | PASS |
| Exp08 | INT8 PTQ 与精度—性能比较 | REJECT |

## M2：C++ Runtime 与 CUDA 优化

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp09 | TensorRT C++ Runtime | PASS |
| Exp10 | CUDA 融合预处理 | PASS |
| Exp11 | 视频/摄像头端到端推理 | PASS |

## M3：板端综合验证

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp12 | 性能、功耗、温度与稳定性测试 | PASS |

## M4：GPU 推理性能工程深化

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp13 | Nsight Systems 端到端性能瓶颈画像 | PASS |
| Exp14 | Pinned Memory、CUDA Event 与 Double Buffer | REJECT |
| Exp15 | CUDA GPU 后处理与 Nsight Compute | PASS |
| Exp16 | TensorRT IPluginV3 与 ONNX GraphSurgeon | REJECT |
| Exp16 Gate | Deployment Semantic Revalidation（不新增实验编号） | REJECT |
| Exp17 | Explicit Q/DQ、INT8机制审计、粗粒度敏感性与 Mixed Precision | REJECT |
| Exp18 | CUDA Graph Decision Gate | REJECT |
| Exp19 | Baseline 与 ACCEPTED 最终路线综合 Benchmark | PASS |

## M5：项目发布

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp20 | README、学习路线、简历与面试材料 | PLANNED |

## 当前部署主线

```text
Exp02 YOLO11n baseline best.pt
→ Exp06 ONNX
→ Exp07 TensorRT FP32 / FP16
→ Exp08 INT8（候选 REJECT，保留 FP16）
→ Exp09 C++ Runtime
→ Exp10 CUDA 预处理
→ Exp11 摄像头
→ Exp12 Jetson 综合 Benchmark
→ Exp13 Profiling 基线
→ Exp14 异步流水线候选（REJECT，保留同步 FP16 Runtime）
→ Exp15 CUB stable compaction GPU 后处理（PASS，采用为 Runtime 主线）
→ Exp16 IPluginV3图内后处理（组件VERIFIED，全图候选REJECT）
→ Exp16 Deployment Semantic Revalidation Gate（语义PASS、性能REJECT，主线不采用）
→ Exp17 Explicit Q/DQ与Mixed Precision（工程/精度VERIFIED、性能REJECT，保留FP16）
→ Exp18 CUDA Graph（正确性/Profiling VERIFIED、端到端性能REJECT，主线不采用）
```

Exp13 已证明文件链路主要受阶段同步限制，相机链路主要受 30 FPS 输入节拍限制且仍存在
同步串行问题。Exp14 已证明 pinned memory、Event 与双缓冲能产生跨帧重叠，但重叠占比过小且
CPU staging、资源竞争和排队显著放大 P95；候选未满足冻结性能门槛，故 `REJECT`。Exp15 的 CUB
stable compaction 保持 CPU NMS 和冻结检测语义，把文件平均 D2H 压缩99.89%，文件 wall FPS
提升19.19%，同时满足文件/相机 P95门槛，故 `PASS` 并成为当前 Runtime 主线。Exp16已完成Plugin V3、
GraphSurgeon、显式workspace和独立进程部署闭环，组件同Engine数学零误差；但150帧正式语义Gate出现
151 vs 153检测及超限框差，因此全图候选 `REJECT`，没有执行完整三轮性能Gate，Exp15 B主线不变。
普通无Plugin rebuild相对冻结Exp07 Engine同样出现raw漂移，说明跨Engine比较混入rebuild/tactic selection
变量；这不等于Plugin候选自动通过，也不把Exp16改写成单纯“Plugin失败”。

不新增实验编号的 Postprocess Gain Attribution Gate 已完成：在P0/P1都使用pinned Host buffer且都传输
235,200 B后，P0→P1和P1→P2的P95三轮均改善，paired平均分别为−3.05%和−1.11%；FPS/mean方向混合，
因此不强行精确分摊Exp15的19.19%。P2仍因完整正确性、D2H/Host扫描缩减和原采用门槛保持主线地位。

Exp16 Deployment Semantic Revalidation Gate先对frame27、frame40和138 px报告差做candidate级forensic，
并以image+class+IoU/Hungarian替代CSV行号匹配；随后在同一219张test上比较Frozen Exp07、至少两个Fresh
baseline rebuild和Fresh Plugin Engine，报告模型级精度、小目标召回、固定阈值TP/FP/FN、unmatched rate、
bbox IoU与confidence delta，并估计普通TensorRT build variance。只有Plugin不差于正常rebuild波动且满足
性能/复杂度采用条件时才可`ACCEPTED`，否则保持`VERIFIED + REJECTED`。

该Gate已于2026-08-09完成。forensic确认frame27/40均为candidate 8222在0.25附近的threshold crossing，
旧138 px来自额外检测造成的CSV行号错配；B2普通rebuild与Fresh Plugin均让该候选过阈值，B1与F0均未过。
219图R3中P的TP/FP/FN为731/169/109，tiny+small recall为0.79020979，Gate-local
`mAP50@conf_floor_0.25`/`mAP50-95@conf_floor_0.25`为0.84351262/0.49550903，全部位于F0/B1/B2
build-variance envelope内，故部署语义Gate通过。随后R4动态调频三轮配对中，P的聚合wall FPS相对F0
为−1.0648%，E2E mean为+1.3053%，且仅1/3方向有利；虽然三轮P95均满足≤5%退化限制，仍未达到3%收益
和至少2/3同向条件，因此主线采用`REJECT`，Exp15 CUB继续作为Runtime主线。

Exp17已确认Exp08为implicit calibrator/cache，并建立95Q/183DQ的Explicit Q/DQ baseline。它将
tiny+small recall从旧implicit的0.489510恢复到0.755245，但paired GPU-only相对FP16中位劣化12.82%。
三个完整219图Mixed候选均通过精度门槛，但GPU性能劣化9.49%～40.57%，没有Pareto候选，故无“最终候选”
进入repeat build或部署Gate。Exp18仅在保持不变的FP16+Exp15 CUB真实最终主线上重新Nsight证明
enqueue/kernel-launch overhead明显时实现，只捕获device-side preprocess→enqueueV3→GPU
decode/filter→CUB，H2D和count/payload D2H保持Graph外；否则记`SKIPPED_BY_EVIDENCE`。

Exp18已完成。Decision Gate测得launch API median 1.577 ms、Graph候选GPU gap median 0.820 ms，允许实现
固定边界Graph；150帧候选轨迹和最终检测与Normal逐字节一致。节点级Nsight确认launch由67次/帧降至1次/帧，
但三组动态调频paired/interleaved的wall FPS/E2E mean/P95均仅1/3方向有利，聚合中位变化为
−0.780%/−1.081%/−1.092%，故端到端采用`REJECT`。当前主线仍为FP16+Exp15 CUB。

Exp19只比较baseline与已`ACCEPTED`路线。文件视频用于最大吞吐；30 FPS相机重点报告capture wait、
post-capture processing、frame total与P95/P99，同时记录CPU/GPU、功耗、温度、RSS和energy/frame。动态调频
为最终部署主轨，固定时钟只作诊断；54,000帧稳定性只对最终`V_Final`重跑。Exp20完成后停止扩展。

Exp19已完成：文件wall FPS/post-capture mean中位改善3.638%/3.352%，相机post-capture mean中位改善1.768%；
V_Final动态54,000帧长稳态全部强制门槛PASS。固定时钟诊断因sudo交互权限记`BLOCKED_PERMISSION`，未改变
动态主结果或设备governor。

全仓统一使用`IMPLEMENTED / VERIFIED / ACCEPTED / REJECTED`能力证据模型；实验表中的`PASS/REJECT`仍保留
历史裁决。未完成或未验证能力不得写入简历，旧负向结果、门槛和失败现场不得改写或删除。

不得用 RTX 3080 Ti 的验证速度代替 Jetson 性能结论，也不得提前把计划项表述为
已经完成。
