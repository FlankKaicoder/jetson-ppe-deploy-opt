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
| Exp16 | TensorRT IPluginV3 与 ONNX GraphSurgeon | PLANNED |
| Exp17 | INT8 敏感性分析与 Mixed Precision | PLANNED |
| Exp18 | CUDA Graph 与最终 Runtime 优化 | PLANNED |
| Exp19 | 最终综合 Benchmark | PLANNED |

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
```

Exp13 已证明文件链路主要受阶段同步限制，相机链路主要受 30 FPS 输入节拍限制且仍存在
同步串行问题。Exp14 已证明 pinned memory、Event 与双缓冲能产生跨帧重叠，但重叠占比过小且
CPU staging、资源竞争和排队显著放大 P95；候选未满足冻结性能门槛，故 `REJECT`。Exp15 的 CUB
stable compaction 保持 CPU NMS 和冻结检测语义，把文件平均 D2H 压缩99.89%，文件 wall FPS
提升19.19%，同时满足文件/相机 P95门槛，故 `PASS` 并成为当前 Runtime 主线。Exp16 仍为待审批计划。

不得用 RTX 3080 Ti 的验证速度代替 Jetson 性能结论，也不得提前把计划项表述为
已经完成。
