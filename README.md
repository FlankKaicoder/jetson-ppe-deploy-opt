# Jetson PPE Deploy Optimization

基于 Jetson Orin Nano Super 的部署感知型 PPE 小目标检测、
TensorRT 量化部署与 CUDA 推理优化项目。

## 当前状态

项目处于初始化阶段。精度、延迟、吞吐、功耗和优化比例均以正式实验结果为准，
不提前填写未经验证的数据。

## 项目主线

1. PPE 数据集审计；
2. 轻量检测模型基线；
3. P2 小目标检测结构；
4. 部署可重参数化模块；
5. PyTorch → ONNX → TensorRT；
6. FP32、FP16、INT8 对照；
7. TensorRT C++ Runtime；
8. CUDA 融合预处理；
9. Jetson 性能、功耗与稳定性测试。

## 主要平台

- 训练端：NVIDIA RTX 4090
- 部署端：NVIDIA Jetson Orin Nano Super
- 系统：Ubuntu 22.04 / JetPack
- 技术栈：PyTorch、ONNX、TensorRT、CUDA、C++

## 实验管理

每个正式实验必须包含：

- 独立实验编号；
- 对应 Git 分支和 Commit；
- 不覆盖的运行目录；
- 完整执行命令；
- 配置和环境快照；
- 指标、异常和实验结论；
- `docs/experiments/` 下的总结文档。

## 仓库内容边界

仓库保存源代码、配置、测试、实验文档和小型指标文件。

仓库不直接保存完整数据集、模型权重、ONNX、TensorRT Engine、
大型视频和完整 Profiling 文件。

## License

项目许可证将在第三方依赖和代码复用范围核查完成后确定。
