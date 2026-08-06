# Jetson PPE Deploy Optimization

基于 Jetson Orin Nano Super 的 PPE 小目标检测、TensorRT 量化部署与
CUDA 推理优化项目。

## 当前状态

截至 2026-08-06，Exp00～Exp06 已完成。P2、部署可重参数化、轻量注意力和
Focal 分类损失均完成公平消融，但未满足替换基线的综合验收条件。后续部署主线
继续使用原始 YOLO11n baseline。Exp06 已完成 PyTorch → ONNX 导出与一致性
验证，下一阶段为 Exp07：Jetson TensorRT FP32 / FP16 Engine。

冻结基线：

| 项目 | 数值 |
|---|---:|
| Precision | 0.92161767 |
| Recall | 0.82040743 |
| mAP50 | 0.89270104 |
| mAP50-95 | 0.52047856 |
| tiny+small recall | 0.79020979 |
| 参数量 | 2,590,425 |

冻结 `best.pt` SHA256：

```text
79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
```

Exp06 冻结 ONNX SHA256：

```text
305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8
```

## 项目主线

```text
数据集审计
→ YOLO11n 基线与结构消融
→ PyTorch 模型冻结
→ ONNX 导出与一致性验证
→ TensorRT FP32 / FP16 / INT8
→ TensorRT C++ Runtime
→ CUDA 融合预处理
→ 视频与摄像头端到端推理
→ Jetson 性能、功耗、温度与稳定性测试
```

## 三端职责

- Windows：项目总控、代码和文档审查、分支合并、关键产物中转；
- AutoDL：RTX 3080 Ti 训练、评估、PyTorch 冻结、ONNX 导出与 ORT 验证；
- Jetson Orin Nano Super：TensorRT、CUDA、C++、摄像头和板端 Benchmark。

Windows 上的保存或静态检查不能替代 AutoDL/Jetson 上的真实实验。

## 实验管理

每个正式实验必须包含：

- 独立实验编号、分支和 Commit；
- 不覆盖的时间戳运行目录；
- 输入配置、环境、命令和返回码；
- Smoke Test、正式结果、异常和最终决策；
- 小型日志摘要、JSON/CSV、产物大小与 SHA256。

详细协作规范见 `AGENTS.md`，实验状态见 `docs/experiment_index.md`。

## 仓库内容边界

仓库保存源代码、配置、测试、实验文档和小型指标文件；不直接保存数据集、
模型权重、ONNX、TensorRT Engine、视频、完整训练输出或大型 Profiling 文件。

## License

项目许可证将在第三方依赖和代码复用范围核查完成后确定。
