# 实验索引

## 学习总入口

`项目全流程快速学习手册.md` 汇总整个项目的知识主线、Exp00～Exp17 复盘、设备重连
SOP 和后续实验预先规划。后续每次实验都先在该文件末尾追加计划，实验完成后再追加
真实结果与学习复盘，不覆盖原计划。

| 实验 | 名称 | 状态 | 文档 |
|---|---|---|---|
| Exp00 | 项目初始化与范围冻结 | PASS | `experiments/exp00_project_initialization.md` |
| Exp01 | Jetson 环境与编译链审计 | PASS | `01_environment.md` |
| Exp02 | PPE 数据集审计与 YOLO11n 基线 | PASS | `02_YOLO11n基线训练与评估总结.md` |
| Exp03 | YOLO11n-P2 小目标结构消融 | REJECT | `03_YOLO11n-P2小目标结构消融总结.md` |
| Exp04 | 部署可重参数化结构消融 | REJECT | `04_YOLO11n部署可重参数化结构消融总结.md` |
| Exp05 | 轻量注意力与 Focal 损失消融 | REJECT | `05_YOLO11n轻量注意力与Focal损失消融总结.md` |
| Exp06 | PyTorch → ONNX 导出与一致性验证 | PASS | `06_YOLO11n基线ONNX导出与一致性验证总结.md` |
| Exp07 | Jetson TensorRT FP32 / FP16 | PASS | `07_YOLO11n基线TensorRT_FP32_FP16部署与验证总结.md` |
| Exp08 | INT8 PTQ 与精度—性能比较 | REJECT | `08_YOLO11n基线INT8_PTQ部署与验证总结.md` |
| Exp09 | TensorRT C++ Runtime | PASS | `09_TensorRT_CPP_Runtime部署与验证总结.md` |
| Exp10 | CUDA 融合预处理 | PASS | `10_CUDA融合预处理与验证总结.md` |
| Exp11 | 视频/摄像头端到端推理 | PASS | `11_视频摄像头端到端推理总结.md` |
| Exp12 | Jetson 性能、功耗、温度与稳定性测试 | PASS | `12_Jetson性能功耗温度稳定性总结.md` |
| Exp13 | Nsight Systems 端到端性能瓶颈画像 | PASS | `13_Nsight端到端性能瓶颈分析总结.md` |
| Exp14 | Pinned Memory、CUDA Event 与 Double Buffer | REJECT | `14_PinnedMemory_CUDAEvent_DoubleBuffer异步流水线总结.md` |
| Exp15 | CUDA GPU Decode/Filter/Compaction 与 Nsight Compute | PASS | `15_CUDA_GPU后处理与NsightCompute总结.md` |
| Exp16 | TensorRT IPluginV3 与 ONNX GraphSurgeon | REJECT | `16_TensorRT_IPluginV3与ONNX_GraphSurgeon总结.md` |
| Exp16 Gate | Deployment Semantic Revalidation（不新增实验编号） | REJECT | 见Exp16总结第10～11节 |
| Exp17 | Explicit Q/DQ、INT8机制审计、粗粒度敏感性与 Mixed Precision | REJECT | `17_ExplicitQDQ_INT8机制审计与MixedPrecision总结.md` |
| Exp18 | CUDA Graph Decision Gate | PLANNED | 待审批；可SKIPPED_BY_EVIDENCE |
| Exp19 | Baseline 与 ACCEPTED 最终路线综合 Benchmark | PLANNED | 待审批 |
| Exp20 | 项目收尾、简历与面试材料 | PLANNED | 待创建 |

Postprocess Gain Attribution Gate 已于2026-08-09完成，不新增 `Exp15.1` 编号。最终公平对照保持P0/P1
同为pinned且同为235,200 B D2H：GPU decode与CUB compaction对P95分别呈现约3.05%和1.11%的三轮
一致改善，但FPS/mean不足以稳定分摊；Exp15 CUB主线结论不变。详见Exp15总结第8节。

Exp16 已完成IPluginV3/GraphSurgeon/显式workspace与独立新进程部署闭环，同Engine Plugin数学零误差，
组件级标记为 `VERIFIED`；四输出全图候选在150帧正式Gate产生153而非151个检测并出现超限框差，故实验
总体 `REJECT`。正确性失败后未执行剩余正式性能轮次，Exp15 B继续作为Runtime主线。普通无Plugin rebuild
也相对冻结Exp07 raw漂移，因此原Exp16 REJECT永久保留，同时另设窄范围Deployment Semantic Revalidation
Gate判断未来是否采用；该Gate不得重写Plugin、调CUDA Kernel或覆盖原裁决。

Revalidation Gate先forensic frame27、frame40和138 px报告差，并以image+class+IoU/Hungarian做跨Engine
匹配；再在同一219张test比较Frozen Exp07+Exp15、至少两个Fresh baseline rebuild+Exp15与Fresh Plugin
Engine的P/R、mAP50、mAP50-95、固定阈值TP/FP/FN、tiny/small/tiny+small recall、unmatched rate、bbox
IoU及confidence delta。Plugin只有在模型级精度不差于普通rebuild波动且性能/复杂度满足采用条件时才可
`ACCEPTED`，否则保持`VERIFIED + REJECTED`。

Revalidation Gate已完成：R1/R2确认frame27/40的candidate 8222发生0.25阈值穿越，旧138 px是行号错配；
R3统一219图评估证明Fresh Plugin模型级语义处于F0/B1/B2普通build variance内。R4动态调频三轮配对
虽然P95全部通过，但Plugin相对F0聚合wall FPS为−1.0648%、E2E mean为+1.3053%，且仅1/3方向有利，
未满足采用门槛，故Gate为`REJECT`，组件能力保持`IMPLEMENTED + VERIFIED`，主线采用保持`REJECTED`。

## 状态定义

- `PLANNED`：方案或推荐顺序已经确定，尚未产生正式结果；
- `IN_PROGRESS`：正在开发或执行，尚未满足验收条件；
- `PASS`：实验完成并满足当前实验验收条件；
- `FAIL`：实验已执行但没有达到实验目的；
- `REJECT`：实验工程链路完成，但候选方案不进入最终主线；
- `SKIPPED`：依据预设规则跳过；
- `BLOCKED`：受到外部依赖或权限阻塞。

## 能力证据定义

- `IMPLEMENTED`：已有代码或工程链路，不表示已验证或已采用；
- `VERIFIED`：已有冻结输入下的正确性、生命周期或Profiling证据；
- `ACCEPTED`：已满足预冻结采用条件并进入当前主线；
- `REJECTED`：实现或验证可以成立，但候选不进入主线。

实验状态与能力证据是两条维度。例如Exp14为`IMPLEMENTED + VERIFIED + REJECTED`，Exp15 CUB stable
compaction为`VERIFIED + ACCEPTED`，Exp16 Plugin组件为`IMPLEMENTED + VERIFIED`而原全图主线候选为
`REJECTED`。未完成或未验证能力不得写入简历成果。

## Exp17 最终裁决

Exp08代码审计确认为implicit calibrator/cache。Exp17建立256图Explicit Q/DQ baseline并完成静态Q/DQ、
activation/clipping、P3/P4/P5 raw误差、粗粒度fallback、三个Mixed候选219图精度和两轮正反序GPU性能审计。
Explicit QDQ精度通过但GPU-only中位劣化12.82%；三个Mixed候选精度均通过但性能劣化9.49%～40.57%，
没有候选满足Pareto/采用门槛。状态为`REJECT`，能力为`IMPLEMENTED + VERIFIED + REJECTED`，FP16 Engine与
Exp15 CUB Runtime主线不变。没有最终候选，因此repeat build与动态端到端采用测试按证据停止。

## 后续冻结路线

- Exp18只在Exp17后最终主线上重新Nsight证明enqueue/kernel-launch overhead明显时实现device-side Graph；
  H2D和count/payload D2H保持Graph外，无证据则`SKIPPED_BY_EVIDENCE`；
- Exp19只比较baseline与`ACCEPTED`路线，动态调频为部署主轨，最终54,000帧只对`V_Final`重跑；
- Exp20完成README、架构图、结果表、简历和面试材料后停止开发，不扩张新技术方向。

## 当前模型决策

后续 TensorRT 和 Jetson 部署统一使用 Exp02 原始 YOLO11n baseline 生成的
Exp06 静态 FP32 ONNX。
Exp03～Exp05 作为负向或部署感知消融保留，不得表述为已经替换基线。
