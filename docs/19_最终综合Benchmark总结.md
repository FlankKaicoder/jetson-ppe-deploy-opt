# Exp19 最终综合 Benchmark 总结

## 1. 最终裁决

Exp19状态为`PASS`。`V_Final`保持Exp07 FP16 Engine + Exp10 CUDA preprocess + Exp15 CUB stable compaction
+ CPU class-aware NMS；所有`REJECTED`组件均未进入最终版本。

## 2. 冻结环境与正确性

- Jetson Orin Nano Super，25W/id 1，动态`schedutil + nvhost_podgov`，起点`main@ea4de32`；
- Engine SHA256：`88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83`；
- 视频 SHA256：`f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665`；
- 二进制 SHA256：`d67b65de0e3c7487a59ca35626184917562056c6fa8dc53920d32315eb771ac4`；
- V0/V_Final的文件、相机Smoke及150帧语义Gate全部PASS；文件均为151检测和冻结digest。

## 3. 动态 paired/interleaved

| 场景 | wall FPS中位 | post-capture mean中位 | P95中位 | P99中位 | energy/frame中位 |
|---|---:|---:|---:|---:|---:|
| 文件 | +3.638%（2/3有利） | +3.352%（3/3有利） | +2.077% | −0.970% | +2.364% |
| 相机 | −0.008% | +1.768%（3/3有利） | +0.044% | −1.075% | −0.536% |

相机wall FPS由30 FPS输入节拍限制。V0 D2H为235,200 B/frame，文件V_Final为263.84 B/frame；D2H缩减
不能单独解释端到端收益。energy差异较小且方向不完全一致，不作单因果结论。

## 4. V_Final动态54,000帧稳定性

- 54,000帧、1801.995秒、wall 30.003 FPS、应用/监控返回0；
- frame total mean/P50/P95/P99：32.016/31.960/33.984/34.833 ms；
- capture mean/P95：18.399/20.369 ms；post-capture mean/P95：13.617/13.703 ms；
- VDD_IN mean/P95/max：8.171/8.200/8.580 W；energy/frame约0.2723 J；
- 最高温度57.031°C；首末10% P95变化+0.674%；
- 第60秒后RSS mean/max 372.612/375.113 MiB，增长−29.191 MiB，斜率0.208 MiB/min，SWAP增长0。

初始分析错误使用进程初始化前0.004 MiB样本，产生`rss_growth_limit` false positive。原始FAIL摘要已保留；
按Exp12冻结的第60秒稳态口径修复后，对同一证据重分析PASS，没有重跑或修改门槛。

## 5. 边界与下一步

文件方向冲突触发可选固定时钟诊断，但远端sudo需要交互密码，`jetson_clocks --store`未执行，记为
`BLOCKED_PERMISSION`；governor始终保持动态，不影响主结果。Exp20只做项目材料收尾，不再扩展新优化。
