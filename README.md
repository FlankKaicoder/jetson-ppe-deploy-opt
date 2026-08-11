# Jetson PPE 小目标检测与 TensorRT/CUDA 推理优化

这是一个面向 Jetson Orin Nano Super 的端到端部署优化项目：从 PPE 数据集审计、YOLO11n 训练与结构消融出发，完成 ONNX、TensorRT、C++/CUDA Runtime、IMX219 摄像头和 30 分钟稳定性验证，并用可复现证据决定每项优化是否进入最终主线。

> 项目状态：Exp00～Exp20 已完成，最终部署版本已冻结。后续仅做事实修正和必要维护，不继续扩展模型或部署技术栈。

## 项目亮点

- 建立 Windows、AutoDL、Jetson 三端串行协作和产物哈希链路，代码、小型结果与模型产物分通道管理。
- 使用固定数据划分和独立 test 集验证模型；保留 P2、重参数化、Attention/Focal、INT8 等负向实验，不为“全 PASS”回写门槛。
- 在 Jetson 上完成 TensorRT FP32/FP16、C++ Runtime、CUDA 融合预处理、GPU 后处理与 IMX219 实时推理。
- 使用 Nsight Systems / Compute 区分 Host API、GPU Activity 与 Critical Path，并坚持 `Measure → Identify → Optimize → Verify → Re-profile → Accept/Reject`。
- 对 Plugin、Explicit Q/DQ、Mixed Precision、CUDA Graph 等候选完成工程和正确性验证；性能不满足采用条件时保留为 `REJECTED`。
- 最终版本完成 54,000 帧、约 30 分钟相机稳定性测试，并记录延迟、吞吐、功耗、温度、RSS 和 energy/frame。

## 最终部署链路

```mermaid
flowchart LR
    A["文件 / IMX219"] --> B["CUDA fused preprocess"]
    B --> C["TensorRT FP16 enqueueV3"]
    C --> D["GPU decode / filter"]
    D --> E["CUB stable compaction"]
    E --> F["count + compact payload D2H"]
    F --> G["CPU inverse-letterbox + class-aware NMS"]
    G --> H["检测结果 / Benchmark"]
```

最终主线固定为：原始 YOLO11n baseline、静态 FP32 ONNX、Jetson 本机构建的 TensorRT FP16 Engine、TensorRT 10.3 C++ Runtime、CUDA 融合预处理和 Exp15 CUB 稳定压缩后处理。INT8、Plugin、Double Buffer 和 CUDA Graph 均未进入最终部署版本。

详细架构、三端职责和计时边界见 [系统架构与数据流](docs/20_系统架构与数据流.md)。

## 核心结果

### 模型与 Engine

| 指标 | 结果 | 计量范围 |
|---|---:|---|
| Precision / Recall | 0.921618 / 0.820407 | Exp02 独立 test |
| mAP50 / mAP50-95 | 0.892701 / 0.520479 | Exp02 独立 test |
| tiny+small recall | 0.790210 | Exp02 尺度审计 |
| TensorRT FP16 mAP50-95 | 0.521929 | Exp07 完整 219 图 |
| FP32 / FP16 Engine 大小 | 14.88 / 8.95 MB | Jetson 本机构建 |
| FP32 / FP16 GPU-only mean | 10.2415 / 3.4797 ms | `trtexec --noDataTransfers`，非端到端 |

### 部署优化与最终 Benchmark

| 项目 | 结果 | 计量范围 |
|---|---:|---|
| CUDA preprocess | 2.2821 → 0.2008 ms，−91.20% | CPU reference 对比 kernel-only |
| 后处理 D2H | 235,200 → 263.84 B/frame，−99.888% | Exp15/Exp19 文件输入 |
| 文件 wall FPS | 中位 +3.638%，2/3 有利 | Exp19 动态调频 paired/interleaved |
| 文件 post-capture mean | 中位 +3.352%，3/3 有利 | Exp19 动态调频 paired/interleaved |
| 相机 wall FPS | −0.008%，约 30 FPS | IMX219 input-rate-bound |
| 相机 post-capture mean | 中位 +1.768%，3/3 有利 | Exp19 动态调频 paired/interleaved |

Exp15 曾在其当日验收中得到文件 wall FPS `60.270 → 71.838`（+19.19%），但该数字与 Exp19 的最终配对编排不应拼接为累计收益；最终对外结果以 Exp19 为准。Kernel 更快不等于 Runtime 更快，D2H 缩减也不能单独解释全部收益。

### 最终稳定性

| 项目 | V_Final 结果 |
|---|---:|
| 帧数 / 时长 | 54,000 / 1,801.995 s |
| wall FPS | 30.003 |
| frame mean / P50 | 32.016 / 31.960 ms |
| P95 / P99 | 33.984 / 34.833 ms |
| VDD_IN mean / P95 / max | 8.171 / 8.200 / 8.580 W |
| energy/frame | 0.2723 J |
| 最高温度 | 57.031 °C |
| RSS slope | 0.208 MiB/min |
| Swap 增长 | 0 |

完整结果、范围说明和能力状态见 [最终结果与能力证据矩阵](docs/20_最终结果与能力证据矩阵.md)。

## 关键决策

| 候选 | 证据结论 | 主线决策 |
|---|---|---|
| P2、重参数化、Attention/Focal | 完成公平消融，综合指标未过门槛 | `REJECTED` |
| Full INT8 PTQ | 更快、更小，但 tiny+small recall `0.7902 → 0.4895` | `REJECTED` |
| Pinned / Async / Double Buffer | 重叠已验证，P95 明显退化 | `IMPLEMENTED + VERIFIED + REJECTED` |
| CUB stable compaction | 正确性、传输压缩和最终 E2E 收益成立 | `IMPLEMENTED + VERIFIED + ACCEPTED` |
| TensorRT IPluginV3 | 组件与语义通过；系统性能不满足采用门槛 | `IMPLEMENTED + VERIFIED + REJECTED` |
| Explicit Q/DQ / Mixed Precision | 精度机制得到解释，但 GPU 性能均慢于 FP16 | `IMPLEMENTED + VERIFIED + REJECTED` |
| CUDA Graph | launch `67 → 1`、GPU gap −93.4%，E2E 无稳定收益 | `IMPLEMENTED + VERIFIED + REJECTED` |

其中 `Host API Duration ≠ GPU Activity ≠ Critical Path`。优化必须在与目标部署一致的计时边界和正式 workload 下重新验证，不能用单个 kernel 或 API 指标替代系统结论。

## 能力证据口径

- `IMPLEMENTED`：已有代码或工程闭环，不代表正确或已进入主线。
- `VERIFIED`：已有冻结输入下的正确性、生命周期或 profiling 证据。
- `ACCEPTED`：通过预冻结条件并进入最终部署主线。
- `REJECTED`：实现或验证成立，但候选不进入主线。

单项能力可以同时是 `IMPLEMENTED + VERIFIED + REJECTED`。未完成、未验证或被证据拒绝的候选不得包装成简历中的性能成果。

## 实验地图

| 阶段 | 实验 | 状态 |
|---|---|---|
| 数据与模型 | Exp00～Exp05：范围、环境、数据、baseline、模型消融 | 完成；baseline `ACCEPTED` |
| 模型部署 | Exp06～Exp08：ONNX、TensorRT FP32/FP16、INT8 | FP16 `ACCEPTED`；INT8 `REJECTED` |
| Runtime | Exp09～Exp12：C++、CUDA preprocess、视频/相机、稳定性 | `PASS` |
| Profiling 与优化 | Exp13～Exp18：Nsight、异步、GPU 后处理、Plugin、Q/DQ、Graph | 仅 Exp15 后处理 `ACCEPTED` |
| 收尾 | Exp19～Exp20：最终 Benchmark、文档与求职材料 | `PASS` |

逐实验状态和证据入口见 [实验索引](docs/experiment_index.md)，时间顺序、真实命令和学习复盘见 [项目全流程快速学习手册](docs/项目全流程快速学习手册.md)。

## 仓库导航

```text
configs/     训练、导出和部署配置
cpp/         TensorRT C++ Runtime 与 CUDA 实现
docs/        项目范围、逐实验总结、架构和求职材料
results/     可提交的小型摘要、CSV、JSON 与哈希清单
scripts/     三端实验和审计脚本
tools/       数据、模型、ONNX 和结果处理工具
```

- [ROADMAP](ROADMAP.md)：冻结路线和实验状态。
- [项目范围](docs/00_project_scope.md)：目标、边界和验收原则。
- [系统架构与数据流](docs/20_系统架构与数据流.md)：三端架构、最终数据流和计时范围。
- [最终结果与能力证据矩阵](docs/20_最终结果与能力证据矩阵.md)：可引用结果与能力分层。
- [项目讲解与简历材料](docs/20_项目讲解与简历材料.md)：30 秒/2 分钟/5 分钟讲解、简历条目和 STAR 案例。
- [面试题库](docs/20_面试题库.md)：围绕模型、TensorRT、CUDA、profiling 和系统验证的问答。
- [协作规范](AGENTS.md)：三端 Git、实验、产物和安全规则。

## 三端分工与复现原则

- Windows：总控、文档、审查、分支合并与小型证据管理。
- AutoDL：训练、评估、PyTorch 冻结、ONNX 导出和 ORT 验证。
- Jetson：TensorRT、CUDA、C++、摄像头与板端 Benchmark。

仓库不提交数据集、模型权重、ONNX、TensorRT Engine、视频和大型 profiling 文件。跨机器产物通过独立通道传输，并用来源 Commit、大小和 SHA256 校验；TensorRT Engine 必须在目标 Jetson 上构建。

## 环境摘要

- AutoDL：RTX 3080 Ti、Python 3.12.3、PyTorch 2.8.0+cu128、Ultralytics 8.4.95。
- Jetson：Orin Nano Super、Ubuntu 22.04 / L4T R36.4.3、CUDA 12.6、TensorRT 10.3、OpenCV 4.10、GStreamer 1.20。

## License

项目许可证仍需在第三方依赖和代码复用范围完成核查后确定。在许可证明确前，请勿默认将仓库内容视为可自由再分发。
