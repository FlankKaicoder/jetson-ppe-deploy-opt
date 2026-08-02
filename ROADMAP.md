# Project Roadmap

## M1：模型与部署转换

| 实验 | 内容 |
|---|---|
| Exp00 | 仓库和实验管理体系初始化 |
| Exp01 | RTX 4090 与 Jetson 环境审计 |
| Exp02 | PPE 数据集审计和划分 |
| Exp03 | M0 基线 Smoke Test |
| Exp04 | M0 基线正式训练 |
| Exp05 | M1 P2 小目标模型 |
| Exp06 | 重参数化模块和单元测试 |
| Exp07 | M2 重参数化模型训练 |
| Exp08 | 训练态到部署态转换 |
| Exp09 | ONNX 导出和一致性验证 |

## M2：TensorRT 部署

| 实验 | 内容 |
|---|---|
| Exp10 | TensorRT FP32 和 FP16 |
| Exp11 | INT8 PTQ |
| Exp12 | TensorRT C++ 单图运行时 |
| Exp13 | C++ 视频推理和阶段计时 |

## M3：CUDA 优化与发布

| 实验 | 内容 |
|---|---|
| Exp14 | CUDA 融合预处理正确性 |
| Exp15 | Pinned Memory 和异步流水 |
| Exp16 | Jetson 综合 Benchmark |
| Exp17 | README、演示和简历总结 |
| Exp18 | 可选 TensorRT Plugin |

## 优先级

必须优先完成模型基线、P2 对照、重参数化转换、TensorRT FP16、
C++ Runtime、CUDA 融合预处理和 Jetson Benchmark。

高风险 Plugin 不得阻塞主线。
