# Final Postprocess Gain Attribution Evidence

本目录保存 Postprocess Gain Attribution Gate 的最终小型聚合证据：

- synthetic：3种GPU模式及fixed连续帧覆盖保护；
- fixture：冻结 raw fixture 上每路径1000次低方差microbenchmark；
- formal：P0/P1/P2各3个独立进程、每进程150帧的paired/interleaved结果；
- P0与P1均使用pinned Host buffer并传输235,200 B；P2平均传输263.84 B；
- 完整日志、逐帧CSV、构建目录和失败现场仅保留在Jetson。

最终正式 Jetson 目录：

```text
results/gpu_postprocess/postprocess_gain_gate_build_20260809_153046
results/gpu_postprocess/postprocess_gain_gate_build_20260809_154148
results/gpu_postprocess/postprocess_gain_gate_fixture_20260809_152550
results/gpu_postprocess/postprocess_gain_gate_formal_20260809_153201
```
