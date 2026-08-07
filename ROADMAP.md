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
| Exp08 | INT8 PTQ 与精度—性能比较 | IN_PROGRESS |

## M2：C++ Runtime 与 CUDA 优化

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp09 | TensorRT C++ Runtime | PLANNED |
| Exp10 | CUDA 融合预处理 | PLANNED |
| Exp11 | 视频/摄像头端到端推理 | PLANNED |

## M3：板端验证与项目发布

| 实验 | 内容 | 状态 |
|---|---|---|
| Exp12 | 性能、功耗、温度与稳定性测试 | PLANNED |
| Exp13 | README、实验总结、简历与面试材料 | PLANNED |

## 当前部署主线

```text
Exp02 YOLO11n baseline best.pt
→ Exp06 ONNX
→ Exp07 TensorRT FP32 / FP16
→ Exp08 INT8
→ Exp09 C++ Runtime
→ Exp10 CUDA 预处理
→ Exp11 摄像头
→ Exp12 Jetson 综合 Benchmark
```

不得用 RTX 3080 Ti 的验证速度代替 Jetson 性能结论，也不得提前把计划项表述为
已经完成。
