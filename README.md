# Jetson PPE Deploy Optimization

基于 Jetson Orin Nano Super 的 PPE 小目标检测、TensorRT 量化部署与
CUDA 推理优化项目。

## 当前状态

截至 2026-08-08，Exp00～Exp13 已完成。P2、部署可重参数化、轻量注意力和
Focal 分类损失均完成公平消融，但未满足替换基线的综合验收条件。后续部署主线
继续使用原始 YOLO11n baseline。Exp06 已完成 PyTorch → ONNX 导出与一致性
验证；Exp07 已在 Jetson 完成 TensorRT FP32 / FP16 Engine 构建、单图与完整
测试集一致性验证和 GPU-only 诊断 benchmark。Exp08 完成 train-only 校准、目标 Jetson
INT8 Engine 构建、219 张 test 精度/尺度审计和 GPU-only benchmark；INT8 虽降低延迟
25.44%、缩小 Engine 39.82%，但 mAP50-95 下降 0.01391、tiny+small recall 下降
0.30070，超过预冻结门槛，因此候选 `REJECT`，运行时主线继续使用 FP16。Exp09 已完成
TensorRT 10.3 C++ Runtime、Python/C++ 原始输出一致性和三独立进程生命周期验证；三份输出
与 Python TensorRT 参考逐字节一致。Exp10 已完成 CUDA 融合 letterbox、padding、BGR→RGB、
归一化和 NCHW 转换；5 种输入形状均与 Jetson OpenCV 4.10 Reference 逐元素一致。正式
`hd_wide` 计时中 CPU/kernel-only/含 pageable 传输总耗时分别为 2.28212/0.200761/1.88307 ms，
kernel-only 平均耗时下降 91.20%。Exp11 已完成文件视频三进程确定性和 IMX219
300 帧端到端功能验收。Exp12 已完成25W固定时钟三进程性能与54,000帧/约30分钟稳定性
验证：P95/P99为32.885/33.521 ms，最高温度57.968°C，VDD_IN mean 9.100 W，RSS斜率
0.00368 MiB/min。锁频相对未锁频没有加速且平均功耗增加14.76%，默认部署保持动态调频。
项目收尾已按用户批准顺延到 Exp20。Exp13 已用 Nsight Systems 完成同步 Runtime 的端到端
瓶颈画像：文件三轮 pipeline wall 吞吐均值61.583 FPS；相机三轮均值30.174 FPS、CV 0.028%。
文件模式 GPU idle 33.81%，相机模式 GPU idle 63.31%，两种场景 Kernel/Memcpy 重叠均为0；
文件归类为 synchronization-bound，相机归类为 input-rate-bound + synchronization-bound。
下一候选 Exp14 将验证 Pinned Memory、事件依赖和双缓冲，当前尚待用户审批。

快速理解整个项目、复习每次实验的假设/结果/失败经验，以及查看下一实验的预先规划：

```text
docs/项目全流程快速学习手册.md
```

AutoDL 和 Jetson 均允许在不用时关机；重新开机后必须先按学习手册和 `AGENTS.md`
执行 SSH 重连、机器身份、Git、环境和输入哈希检查，再继续实验。

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

Exp07 Engine（仅保存在 Jetson，不进入 Git）：

| 精度 | Engine SHA256 | 大小 | GPU-only mean |
|---|---|---:|---:|
| FP32 | `01616a8144228db5edbf8948227e3bbaee43b22c495aba3c6c44212e43efe0f1` | 14,880,428 bytes | 10.2415 ms |
| FP16 | `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83` | 8,951,540 bytes | 3.4797 ms |

Exp07 的诊断计时固定为 batch 1、640×640、500 ms warmup、200 次迭代、
CUDA Graph、spin wait 且关闭 H2D/D2H；未锁定 `jetson_clocks`，不是端到端
延迟。完整测试集 mAP50-95：

| 后端 | mAP50-95 |
|---|---:|
| Jetson PyTorch FP32 | 0.52189519 |
| TensorRT FP32 | 0.52160406 |
| TensorRT FP16 | 0.52192881 |

Exp11 已完成文件视频与 IMX219 端到端 C++ 推理。文件视频三个独立进程均处理
150 帧、151 个检测，检测 CSV SHA256 一致；IMX219 1920×1080@30 正式运行处理
300/300 帧，端到端 mean/P95 为 31.832/34.190 ms，有效处理率 31.415 FPS。
该数字是未锁频、带取帧/H2D/CUDA/TensorRT/D2H/NMS 的短时功能基线；功耗、温度、
资源占用、丢帧和长稳态结论留给 Exp12。

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
→ Nsight Systems 瓶颈画像
→ 内存、并发、GPU 后处理、Plugin 与混合精度优化
→ 最终综合 Benchmark 与项目收尾
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

详细协作规范见 `AGENTS.md`，实验状态见 `docs/experiment_index.md`，项目学习主线见
`docs/项目全流程快速学习手册.md`。

## 仓库内容边界

仓库保存源代码、配置、测试、实验文档和小型指标文件；不直接保存数据集、
模型权重、ONNX、TensorRT Engine、视频、完整训练输出或大型 Profiling 文件。

## License

项目许可证将在第三方依赖和代码复用范围核查完成后确定。
