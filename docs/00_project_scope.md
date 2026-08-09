# 项目范围

## 1. 项目名称

基于 Jetson Orin Nano Super 的部署感知型 PPE 小目标检测与
TensorRT/CUDA 推理优化。

## 2. 业务载体

计划检测工业和施工场景中的人员及防护装备：

- person
- helmet / hardhat
- safety vest

最终类别以数据集审计结果为准。

## 3. 核心目标

1. 建立可复现的检测模型基线；
2. 评价 P2 高分辨率结构对小目标的影响；
3. 实现部署可重参数化模块；
4. 验证训练态和部署态模型等价性；
5. 完成 PyTorch、ONNX、TensorRT 部署链；
6. 对比 FP32、FP16 和 INT8；
7. 实现 TensorRT C++ Runtime；
8. 实现 CUDA 融合预处理；
9. 分析延迟、吞吐、内存、功耗和温度；
10. 使用 Nsight Systems/Compute 定位 CPU/GPU、传输和同步瓶颈；
11. 实现并按预冻结门槛评估 Pinned Memory、异步流水、GPU 后处理和 CUDA Graph；功能可运行不等于采用；
12. 实现并验证 TensorRT IPluginV3 与 ONNX GraphSurgeon 扩展链路；
13. 分析 INT8 量化敏感模块并探索 Mixed Precision；
14. 形成可公开、可解释的 GitHub 工程。

## 4. 非核心范围

当前不以 Jetson BSP 深度定制、传感器驱动、自定义载板、
音视频同步、Web 平台或多路 DeepStream 系统为主线。

## 5. 成果边界

所有成果必须能够通过代码、实验记录和指标文件追溯。
未经完成和验证的功能不得表述为已经实现。

能力证据统一区分 `IMPLEMENTED / VERIFIED / ACCEPTED / REJECTED`。Exp03～Exp05、Exp08、Exp14和Exp16
等负向结论属于项目事实，不得为“全PASS”重训、放宽门槛或删除。后续工作遵守
`Measure → Identify → Optimize → Verify → Re-profile → Accept/Reject`。

Exp20完成后停止当前主线开发，不再扩展新的YOLO结构、Attention/Loss、GPU NMS、NVMM zero-copy、
DeepStream、多摄像头、剪枝、蒸馏、TVM/MLIR/Triton等方向。
