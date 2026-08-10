# Exp17 Explicit Q/DQ、INT8机制审计与 Mixed Precision 总结

## 1. 最终结论

Exp17 实验链路完成，主线候选裁决为 `REJECT`。Exp08 被确认是 implicit calibrator/cache；Exp17 成功建立
并验证 Explicit Q/DQ PTQ、activation/dynamic-range/clipping 审计、P3/P4/P5 raw 机制分析和粗粒度
Mixed Precision fallback，但 Full Explicit 与全部 Mixed 候选都没有满足 GPU 性能采用门槛。

- Explicit Q/DQ PTQ：`IMPLEMENTED + VERIFIED + REJECTED`；
- activation 与多尺度量化机制审计：`IMPLEMENTED + VERIFIED`；
- coarse Mixed Precision：`IMPLEMENTED + VERIFIED + REJECTED`；
- 当前部署主线：Exp07 FP16 Engine + Exp15 CUB stable compaction，保持 `ACCEPTED`。

## 2. 冻结输入与环境

Jetson Orin Nano Super，CUDA 12.6、TensorRT 10.3.0.30、25W/id 1动态调频。Exp17分支起点为
`main@729e2d9`。冻结Exp06 ONNX、Exp07 FP16 Engine、Exp08 implicit INT8 Engine SHA256分别为：

```text
305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8
88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83
5787fb3bae4dbd00909c1762efc9263566044bc4dc35a836c950312e85895f26
```

校准固定原256张train-only图；正式精度固定219张test、840 GT、640×640、batch 1。模型精度、GPU-only
诊断与端到端采用边界始终分开。

## 3. Explicit Q/DQ baseline

R0确认Exp08使用`IInt8EntropyCalibrator2`、`BuilderFlag.INT8`和`int8_calibrator`，冻结ONNX有353节点且
Q/DQ均为0。R1/R2建立Entropy Explicit Q/DQ：对称QInt8、activation per-tensor、weight per-channel、
FP32 bias、weighted-op输出不量化、FP32 raw I/O与strongly typed Engine。

正式QDQ ONNX含95Q/183DQ，全部scale正且有限、zero-point为0。ONNX/Engine SHA256为：

```text
5a28c30b0f92db1a94be7f290a781ff182df757fb71e36d749a5b64d1daf8325
43db95c68e9dd23d00b2c35e0cfe19a9d61ca75a1a92ffbf70245f530ceb66c9
```

| 后端 | mAP50 | mAP50-95 | tiny+small recall |
|---|---:|---:|---:|
| FP16 | 0.884374 | 0.521929 | 0.790210 |
| Explicit QDQ | 0.883443 | 0.528020 | 0.755245 |

精度Gate全部通过，但三轮paired GPU-only中QDQ mean约3.932～3.942 ms，FP16约3.484～3.645 ms；QDQ
三轮均更慢，中位劣化12.82%。Engine缩小47.67%不能抵消延迟失败。

## 4. 失败现场与最小修复

- Softmax未进入校准输出集合导致ORT range key缺失；将两个真实DFL Softmax纳入预登记集合。
- ORT默认INT32 bias DQ不被TensorRT 10.3接受；用正式`QuantizeBias=False`保持FP32 bias。
- 一次收集256图×176组中间输出被Linux以137杀死；保持全部256图和Entropy算法，改为8图一块、同一
  calibrator跨32块累积。
- Mixed runner首轮预创建artifact目录，与生成器非覆盖保护冲突；保留失败目录，仅改由生成器创建目录。
- 首轮raw `class_changes`包含全部低置信argmax而被放大；保留结果并新增FP16过阈值/两端并集口径重跑。

## 5. Activation 与三尺度机制

256图、20个Detect Head activation的最大clipping ratio约`2.49e-5`，严重饱和不是主要解释。最高模拟
relative-L2集中在P5 classification `0.064696`和P3 classification `0.062778`；DFL约
`0.014067/0.010889`，更符合低幅值rounding/离散化敏感。

| 后端 | P3/P4/P5 score relative-L2 | P3/P4/P5 threshold crossings |
|---|---|---|
| implicit INT8 | 0.771425 / 0.302026 / 0.308428 | 1143 / 612 / 572 |
| Explicit QDQ | 0.137478 / 0.146772 / 0.181561 | 142 / 287 / 306 |

旧implicit的P3穿越中1138次向下、5次向上，过阈值候选从FP16的1348降到215；Explicit仅93次向下、
49次向上，保留1304个。阈值相关类别变化只有0～3次，说明主要机制是score幅值压低而非类别交换。

## 6. Mixed Precision Pareto

构建并Smoke验证P3-classification、all-classification、DFL和完整Detect Head fallback，分别旁路4、12、2、
20对activation Q/DQ。三个代表候选完整219图结果：

| 候选 | mAP50 | mAP50-95 | tiny+small | GPU mean相对FP16 |
|---|---:|---:|---:|---:|
| P3 classification | 0.883948 | 0.528619 | 0.755245 | +26.71%（更慢） |
| Classification | 0.883436 | 0.529628 | 0.755245 | +40.57%（更慢） |
| DFL | 0.883105 | 0.526800 | 0.755245 | +9.49%（更慢） |

完整Detect Head在性能筛选中慢57.99%，未继续完整精度评估。三个完整候选虽通过精度Gate，却都未形成
Accuracy–Latency Pareto，最终决策为`NO_MIXED_CANDIDATE_ACCEPTED`。因为没有最终候选，repeat build和
动态端到端采用Gate不再启动；QAT也不进入本项目收尾范围。

## 7. 关键结果目录

```text
exp17_0_audit_20260809_221258
exp17_2_qdq_formal_20260809_224138
exp17_3_qdq_full_test_20260809_225245
exp17_4_qdq_benchmark_20260810_145400
exp17_5_qdq_scale_audit_20260810_145925
exp17_6_head_activation_formal_20260810_150714
exp17_7_raw_scale_formal_20260810_151504
exp17_8_mixed_build_20260810_152337
exp17_9_mixed_perf_screen_20260810_153903
exp17_10_mixed_full_test_20260810_154211
```

## 8. 下一步

Exp18只能在当前真实最终主线（FP16 + Exp15 CUB）重新Nsight后判断是否enqueue-bound。若launch/enqueue
overhead不明显，直接`SKIPPED_BY_EVIDENCE`；不得为展示CUDA Graph而实现，也不得复活Exp14双缓冲复杂度。
