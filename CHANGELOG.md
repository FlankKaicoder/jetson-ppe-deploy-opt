# Changelog

## Unreleased

- 完成 Exp10 CUDA 融合预处理；5 种输入形状与 Jetson OpenCV 4.10 Reference 逐元素一致，
  `hd_wide` kernel-only 平均耗时由 CPU 2.28212 ms 降至 0.200761 ms（下降 91.20%）。
- 保留 Exp10 CMake CUDA 路径失败、非整数缩放误差及两次错误舍入假设的负向结果；最终按
  OpenCV 4.10 CV_8U 专用纵向 resize 路径复现定点语义。
- 完成 Exp09 TensorRT 10.3 C++ Runtime、CMake 构建、I/O/显存/Stream RAII 管理、
  Python/C++ 原始输出一致性和三独立进程生命周期验证；下一阶段为 Exp10 CUDA 融合预处理。
- 完成 Exp08 INT8 PTQ 构建、219 张 test 精度/尺度审计和同口径 GPU-only benchmark；
  性能与尺寸门槛通过，但 mAP50-95 和 tiny+small recall 门槛失败，候选 `REJECT`。
- 运行时部署主线保留 Exp07 FP16，下一阶段更新为 Exp09 TensorRT C++ Runtime。
- 启动 Exp08 INT8 PTQ，冻结 train-only 校准集选择规则和候选采用门槛；AutoDL
  256 张校准集分布审计通过。
- 新增 `docs/项目全流程快速学习手册.md`，汇总 Exp00～Exp07 学习主线并预先记录 Exp08 计划。
- 增加 AutoDL/Jetson 关机后重新开机的 SSH 重连、身份、Git、环境和产物哈希检查规范。
- 固化每次实验“开始前追加计划、结束后追加真实复盘”的持续学习记录流程。
- 纳入三端协同开发规范 `AGENTS.md`。
- 同步 README、Roadmap 和实验索引至 Exp06 完成后的真实状态。
- 补充 Exp00～Exp06 实验注册信息、冻结基线指标和 ONNX 哈希。
- 完成 Exp06 PyTorch → ONNX 导出、原始张量/NMS 检查和完整测试集一致性验证。
- 将 Exp06 ONNX 在 Jetson TensorRT 10.3 构建为 FP32/FP16 Engine。
- 完成 Exp07 单图、完整测试集和 GPU-only 诊断性能验证。
- 记录 FP16 约 2.94× GPU-only 加速及 Engine 哈希，明确不等同端到端性能。
- 将下一阶段更新为 Exp08 INT8 PTQ。

## 2026-08-02

- 初始化项目目录。
- 建立实验管理机制。
