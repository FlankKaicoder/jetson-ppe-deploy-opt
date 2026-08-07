# Exp08：YOLO11n 基线 INT8 PTQ 部署与验证总结

## 当前状态

```text
IN_PROGRESS
```

本文档只记录已经执行并具有日志、指标或哈希证据的结果。当前尚未构建 INT8 Engine，
尚未产生 INT8 精度或性能结论。

## 实验目标

1. 只从训练集生成确定性、可审计的 TensorRT INT8 校准数据；
2. 在目标 Jetson TensorRT 10.3 环境构建带 FP16 fallback 的 INT8 Engine；
3. 对比 PyTorch FP32、TensorRT FP32、FP16 和 INT8 的正确性与完整测试集精度；
4. 在同一 GPU-only 诊断口径下比较 Engine 大小、延迟和吞吐；
5. 根据事先冻结的精度、尺度召回和性能门槛决定是否采用 INT8。

## 冻结输入

| 输入 | SHA256 |
|---|---|
| Exp02 `best.pt` | `79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6` |
| Exp06 ONNX | `305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8` |
| Exp07 FP32 Engine | `01616a8144228db5edbf8948227e3bbaee43b22c495aba3c6c44212e43efe0f1` |
| Exp07 FP16 Engine | `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83` |

## 冻结校准方案

- 来源：Exp02 最终数据划分的 `train`，不使用 val/test；
- 数量：256 张；
- 基础随机种子：42；
- 候选数：1024；
- 选择规则：在确定性随机候选中，选择类别、尺寸、类别×尺寸联合分布及背景比例
  与完整训练集最大绝对比例偏差最小的候选；
- 代表性门槛：最大绝对比例偏差不超过 0.02，并覆盖全部类别、尺寸组和背景图；
- 尺寸定义：tiny `<0.0025`、small `[0.0025,0.01)`、medium `[0.01,0.04)`、
  large `>=0.04`，面积均为标注框相对图像面积；
- 输入预处理必须与 Exp07 保持一致：letterbox 640×640、padding=114、BGR→RGB、
  `/255`、HWC→CHW、FP32 输入。

## 预冻结候选采用门槛

完成工程链路不等于 INT8 候选可以替换 FP16。正式 INT8 结果产生前冻结以下门槛：

| 类别 | 门槛 |
|---|---|
| 构建与执行 | 返回码为 0；输出 `[1,7,8400]`；无 NaN/Inf |
| 完整 test mAP50-95 | 相对同运行时 FP16 绝对下降不超过 0.010 |
| 完整 test mAP50 | 相对同运行时 FP16 绝对下降不超过 0.015 |
| 固定阈值 tiny+small recall | 相对 FP16 绝对下降不超过 0.050 |
| GPU-only mean latency | 相对 FP16 至少降低 5% |
| Engine 大小 | 相对 FP16 至少降低 10% |

固定阈值尺度审计使用 `confidence=0.25`、`NMS IoU=0.70`、匹配 IoU=0.50。
若任一业务强制门槛失败，INT8 候选不替换 FP16；实验本身仍保留全部结果。

## 尚待执行

- [ ] AutoDL 校准集生成与分布审计；
- [ ] 校准归档传输及双端 SHA256；
- [ ] Jetson INT8 smoke build；
- [ ] Jetson INT8 正式 build 与校准缓存；
- [ ] 单图原始张量和 NMS 后一致性；
- [ ] 219 张独立 test 精度与尺度召回；
- [ ] 同口径 GPU-only benchmark；
- [ ] 最终决策与学习复盘。
