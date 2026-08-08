# Exp14 Pinned Memory、CUDA Event 与 Double Buffer 异步流水线总结

## 1. 最终结论

Exp14 工程链路完成，但候选状态为 `REJECT`。

Variant A/B/C 均通过正确性验证，Variant C 也在文件和相机 Nsight 时间线中产生了可量化的
跨帧 Kernel/Memcpy 重叠；然而同日三轮无 profiler 对照显示：

- 文件 Variant C 相对同步 baseline 的 pipeline wall FPS 仅提升 **4.513%**，低于10%门槛；
- 文件 E2E P95 从14.732 ms升至38.198 ms，退化 **159.276%**；
- 相机吞吐仍受30 FPS输入节拍限制，Variant C E2E P95从34.212 ms升至93.571 ms，退化
  **173.506%**；
- 文件/相机时间线重叠分别只有2.961/0.860 ms，占正式窗口约0.149%/0.009%，不足以抵消
  pinned staging、采集资源竞争和双槽排队成本。

因此部署主线继续使用 Exp13 的同步 FP16 C++ Runtime。Exp15 仍为 `PLANNED`，需另行审批。

## 2. 实验边界与环境

| 项目 | 冻结值 |
|---|---|
| 执行设备 | Jetson Orin Nano Super / aarch64 |
| 系统 | L4T R36.4.3 |
| CUDA / TensorRT | 12.6.68 / 10.3.0.30 |
| OpenCV / GStreamer | 4.10 / 1.20.3 |
| Git 起点 | `main@453bc6e` |
| 实验分支 | `exp/14-async-double-buffer` |
| 功耗与时钟 | 25W/id 1，动态调频 |
| 输入/输出 | batch=1，`1×3×640×640` → `1×7×8400` |
| 检测配置 | confidence=0.25，class-aware NMS IoU=0.70 |
| 文件场景 | 150帧，每变体三轮无 profiler |
| 相机场景 | IMX219 sensor-id 0，1920×1080@30，300帧，每变体三轮 |

AutoDL 在本实验期间保持关机；没有训练、重新导出 ONNX 或重建 Engine。

### 2.1 冻结输入

| 产物 | 大小 | SHA256 |
|---|---:|---|
| ONNX | 10,566,605 bytes | `305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8` |
| FP16 Engine | 8,951,540 bytes | `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83` |
| 文件视频 | 874,518 bytes | `f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665` |

## 3. 变体与实现

| 变体 | Host memory | Buffer | 依赖与同步 |
|---|---|---:|---|
| baseline | pageable | 1 | 旧预处理/推理 API，各阶段立即同步 |
| A | pinned input/output | 1 | 保留串行调度，隔离 pinned memory 影响 |
| B | pinned input/output | 1 | upload/inference/download Stream + Event，终端回收 |
| C | pinned input/output | 2 | 三 Stream 双缓冲，跨帧提交与按序 retirement |

实现新增低层异步 CUDA 预处理入口和 TensorRT `enqueue_device_async`，旧同步 API 不变。同一个
TensorRT execution context 始终只在 inference stream 上串行 enqueue；slot 只有在 download 完成、
CPU decode/NMS/输出证据完成后才可重用。NVTX 覆盖 submit/complete、staging、H2D、preprocess、
TensorRT、D2H、wait、decode、NMS 和 output。

## 4. 正确性验收

- baseline/A/B/C 的5帧 Smoke Test 全部返回0，检测摘要完全一致；
- 文件12轮正式进程均为150帧、151检测，`detections.csv` SHA256 均为
  `9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a`；
- 相机12轮均处理300/300帧；
- 无 NaN/Inf、非法类别/置信度/框、帧序错位、slot 提前复用、CUDA 或 TensorRT 错误；
- Exp09、Exp10 既有目标在当前源码重新编译成功，baseline 三轮也实际覆盖旧同步 API。

正确性 Gate 通过不代表性能 Gate 通过。

## 5. 无 profiler 三轮结果

### 5.1 文件视频

| 变体 | FPS均值 | FPS CV | E2E mean ms | P95 ms | P99 ms | FPS变化 | P95退化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 69.595 | 3.582% | 13.267 | 14.732 | 14.875 | — | — |
| A | 66.033 | 4.714% | 16.833 | 18.929 | 19.071 | -5.119% | +28.482% |
| B | 66.499 | 6.168% | 21.417 | 24.157 | 25.016 | -4.449% | +63.974% |
| C | 72.736 | 3.170% | 33.465 | 38.198 | 39.626 | +4.513% | +159.276% |

### 5.2 IMX219 相机

| 变体 | FPS均值 | FPS CV | E2E mean ms | P95 ms | P99 ms | FPS变化 | P95退化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 30.141 | 0.087% | 31.836 | 34.212 | 35.044 | — | — |
| A | 30.155 | 0.063% | 50.000 | 52.449 | 54.110 | +0.048% | +53.308% |
| B | 30.174 | 0.008% | 57.260 | 59.162 | 60.449 | +0.109% | +72.931% |
| C | 30.197 | 0.084% | 90.225 | 93.571 | 94.685 | +0.188% | +173.506% |

冻结门槛为：文件 FPS至少提升10%或 E2E mean至少降低10%，且 P95退化不超过5%；相机不要求
吞吐提升，但 P95退化不超过5%。Variant C 两个场景均失败。

## 6. 阶段耗时与时间线归因

文件 baseline 的同步预处理均值为1.610 ms。A/B/C 的 pinned upload 均值分别为
0.890/1.026/0.916 ms，说明 pinned memory 对 DMA 阶段有效；但其 CPU staging 又增加
0.735/0.813/0.850 ms。B/C 文件 capture 均值升至7.513/7.241 ms，相机 B/C capture 均值约
25.561/25.576 ms，暴露出采集、CPU staging 和 GPU 工作的资源竞争。

| 正式 C Profile | Stream Wait | Kernel/Memcpy overlap | Upload/Inference overlap | GPU idle |
|---|---:|---:|---:|---:|
| 文件150帧 | 300 | 2.961472 ms | 2.921248 ms | 38.429% |
| 相机300帧 | 600 | 0.860192 ms | 0.828928 ms | 63.389% |

Exp13 同步基线的文件/相机 GPU idle 分别为33.810%/63.306%，且 overlap=0。Exp14 C 虽然把
重叠从0变为非0，但文件 idle 反而上升，相机 idle 几乎不变。双槽还使单帧完成等待后续 slot
retirement，造成 tail latency 大幅增加。由此可把结果归纳为：

```text
异步依赖正确 + 可观察到重叠
≠
重叠占比足够 + 系统吞吐/尾延迟满足验收
```

## 7. 失败现场与最小修复

1. 首次构建未显式指定 CUDA compiler，失败目录保留；新构建目录显式使用
   `/usr/local/cuda/bin/nvcc` 后通过。
2. 首次 Nsight stats 未带 `--force-export=true`，原 profile 目录保留；修正参数后新目录通过。
3. 首轮正式 baseline 被错误补全的检测摘要哈希拦截；权威输出没有变化，计划更正已先追加到学习
   手册，随后从新时间戳重跑全部三轮。
4. 历史 Exp09/Exp10 脚本权限和分支保护阻止在 Exp14 分支伪装正式重跑；改用当前构建目录重新编译
   旧目标，并用 baseline 正式轮覆盖旧接口集成回归。
5. 首次最终审计漏传时钟脚本参数且使用了错误 ONNX 路径，失败目录保留；只读恢复真实路径后用
   独立审计脚本在新目录完成验证。

没有删除失败结果、吞掉返回码、改写为 PASS 或降低性能门槛。

## 8. 关键输出

- 构建：`results/pipeline/exp14_0_build_20260808_181524`
- 文件聚合：`results/pipeline/exp14_file_compare_20260808_185151`
- 相机聚合：`results/pipeline/exp14_camera_compare_20260808_185152`
- 文件 Nsight：`results/pipeline/exp14_profile_C_file_20260808_183846`
- 相机 Nsight：`results/pipeline/exp14_profile_C_camera_20260808_183910`
- 最终设备审计：`results/pipeline/exp14_final_audit_20260808_185656`

大型 `.nsys-rep`、SQLite、逐帧 CSV 和图片只保存在 Jetson，不提交普通 Git。

## 9. 实验后状态与下一步

最终审计确认 CPU/GPU 未锁频，GPU min/max 306/918 MHz，空闲短采样最高约51.25°C，无残留
Exp14/Exp11/Nsight 进程，NVMe剩余约182 GB。该短测不替代 Exp12 的长期稳定性结论。

Exp15 推荐只围绕当前仍在 CPU 的 decode/filter/compaction 做独立方案设计，并用 Nsight Compute
回答 kernel occupancy、memory throughput、branch divergence 和 compaction 开销；在用户审批前不实现。
