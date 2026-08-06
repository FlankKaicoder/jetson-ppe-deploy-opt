# Changelog

## Unreleased

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
