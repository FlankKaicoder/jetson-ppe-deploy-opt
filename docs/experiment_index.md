# 实验索引

## 学习总入口

`项目全流程快速学习手册.md` 汇总整个项目的知识主线、Exp00～Exp10 复盘、设备重连
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
| Exp13 | 项目收尾、简历与面试材料 | PLANNED | 待创建 |

## 状态定义

- `PLANNED`：方案或推荐顺序已经确定，尚未产生正式结果；
- `IN_PROGRESS`：正在开发或执行，尚未满足验收条件；
- `PASS`：实验完成并满足当前实验验收条件；
- `FAIL`：实验已执行但没有达到实验目的；
- `REJECT`：实验工程链路完成，但候选方案不进入最终主线；
- `SKIPPED`：依据预设规则跳过；
- `BLOCKED`：受到外部依赖或权限阻塞。

## 当前模型决策

后续 TensorRT 和 Jetson 部署统一使用 Exp02 原始 YOLO11n baseline 生成的
Exp06 静态 FP32 ONNX。
Exp03～Exp05 作为负向或部署感知消融保留，不得表述为已经替换基线。
