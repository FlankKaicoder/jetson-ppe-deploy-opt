# Exp18 CUDA Graph Decision Gate 总结

## 1. 最终裁决

Exp18 状态为 `REJECT`，能力证据为 `IMPLEMENTED + VERIFIED + REJECTED`。

Decision Profile 证明当前 Exp07 FP16 Engine + Exp15 CUB 主线存在足够的 Host launch 与 GPU gap，因而允许实现首版 CUDA Graph；实现也通过冻结语义、独立进程和 Nsight 复验。但是三组动态调频 paired/interleaved 文件实验没有达到预冻结的端到端采用门槛，因此 CUDA Graph 不进入 Runtime 主线。主线继续保持 Exp10 CUDA preprocess + Exp07 FP16 Engine + Exp15 CUB stable compaction + CPU class-aware NMS。

## 2. 冻结输入与环境

- Jetson Orin Nano Super，aarch64，25W/id 1 动态调频；
- CUDA 12.6.68，TensorRT 10.3.0.30，Nsight Systems 2024.5.4；
- 起点：`main@a4a78aa`，分支：`exp/18-cuda-graph-decision`；
- FP16 Engine SHA256：`88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83`；
- 文件视频 SHA256：`f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665`；
- 150 帧最终检测：151 行，冻结 SHA256：`9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a`；
- batch=1、640×640、confidence=0.25、class-aware CPU NMS IoU=0.70。

## 3. Decision Gate

正式 Normal profile 逐帧区分 Host API、GPU Activity 与 Critical Path 上界：

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| launch API median | 1.577 ms/frame | ≥0.50 ms | PASS |
| launch/post-capture | 12.86% | ≥5% | PASS |
| Graph候选GPU gap median | 0.820 ms/frame，8.52% | ≥0.30 ms或≥5% | PASS |
| 预测E2E收益上界 | 5.58% | ≥3% | PASS |

`enqueueV3` NVTX Host range在profile中明显受观测器放大，未被当作GPU推理时长；预测上界只用于决定是否值得实现，不是采用收益。

## 4. 实现边界

Graph外：文件decode/相机capture、pageable H2D、count D2H与Host读取、按count的payload D2H、CPU candidate scan和CPU NMS。

Graph内：固定地址与固定shape的 `CUDA preprocess → TensorRT enqueueV3 → GPU decode/flag → CUB stable compaction`。

所有device buffer、Host count/candidate buffer、CUB temporary storage、Graph和GraphExec均在逐帧路径外创建；replay路径没有逐帧`cudaMalloc/cudaFree`。未引入Exp14 pinned staging、Double Buffer、多Stream跨帧队列、Exp16 Plugin或Exp17 QDQ。

## 5. 正确性与生命周期

- 5帧Graph smoke通过；
- Normal/Graph均完成多次独立150帧新进程运行，均为150帧、151检测、冻结检测digest；
- 显式candidate trace覆盖全部150帧、1,392个pre-NMS候选。Normal与Graph的candidate index、class、stable rank、confidence及source box CSV逐字节一致，SHA256均为`5b51e8505663ab411cb635065917761962ccf72115b8254dd544b125f9c29d99`；
- count/payload语义保持Exp15 CUB，平均D2H仍为263.84 B/frame；
- Graph capture前在同一TensorRT context与stream完成2次warmup，避免lazy initialization进入capture；
- Graph内分阶段CUDA Event在当前CUDA 12.6组合下读取返回`invalid argument`，故移除这些Event节点；Graph模式CSV中的相应分项0表示“不可用”，不表示耗时为零，分阶段GPU证据来自Nsight；
- Compute Sanitizer memcheck再次报告`GPU debugging features are disabled`并退出1，只能记为`UNSUPPORTED`，不能写成PASS或“覆盖全部race/非法访问”。

## 6. 三组动态调频 paired/interleaved 性能

冻结顺序为 `N1→G1 / G2→N2 / N3→G3`，每个进程150帧、2次warmup。正值表示Graph更好。

| Pair | Wall FPS变化 | E2E mean变化 | E2E P95变化 |
|---|---:|---:|---:|
| 1 | -11.890% | -14.999% | -1.405% |
| 2 | +2.982% | +2.431% | +10.982% |
| 3 | -0.780% | -1.081% | -1.092% |
| 中位数 | -0.780% | -1.081% | -1.092% |

Wall FPS、E2E mean和P95均只有1/3方向有利；P95三轮均未退化超过3%，但主要收益条件“Wall FPS或E2E mean至少改善3%，且至少2/3同向”失败，因此采用裁决为`REJECTED`。不得用Pair 2或更早单轮最佳值覆盖正式结果。

## 7. Re-profile 与相机边界

节点级Nsight复画像确认局部优化真实存在：Normal每帧67次launch，Graph每帧1次`cudaGraphLaunch`；launch API median由1.240 ms降至0.661 ms（-46.7%），Graph候选device gap由0.794 ms降至0.052 ms（-93.4%），device span median降低15.66%。这证明Graph减少了提交与bubble，但不足以在动态调频、文件decode和Host交互噪声中形成稳定E2E收益。

60帧Normal相机短profile为30.661 wall FPS，capture mean/P95为16.987/20.513 ms，E2E mean/P95为31.273/35.057 ms；相机仍为30 FPS input-rate-bound。由于文件采用Gate失败，不继续扩展相机Graph性能实验。

## 8. 失败、最小修复与学习结论

保留以下失败现场：首次构建因CUDA 12.6使用三参数`cudaGraphInstantiate`而失败，最小版本适配后通过；首次Graph replay因capture内分段Event计时返回`invalid argument`而失败，移除不可读Event节点后通过；两次Windows到远端bash的一次性命令因变量转义失败，均发生在实验应用或分析器执行前，后续改用正式runner和明确路径；Compute Sanitizer为平台不支持。

Exp18最重要的结论不是“CUDA Graph无效”，而是“CUDA Graph在API和GPU timeline上有效，但没有通过本项目的系统级采用Gate”。这继续验证：Host API Duration ≠ GPU Activity ≠ Critical Path；局部launch/gap下降也不自动等于Runtime主线收益。

下一步为Exp19，只比较baseline与已经`ACCEPTED`的最终路线；Exp18 Graph不得进入V_Final或简历成果。
