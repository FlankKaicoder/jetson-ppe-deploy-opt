# Exp13 Nsight Systems 端到端性能瓶颈分析总结

## 1. 结论

Exp13 状态为 `PASS`。本实验没有修改冻结 FP16 Engine、pageable host memory、Buffer 数、
Stream 数、同步点或后处理算法，只加入可关闭的 NVTX 标记、Nsight 解析工具和更清晰的流水线
wall-time 口径。文件视频的首要瓶颈是阶段级同步，摄像头的首要瓶颈是 30 FPS 输入节拍，内部仍有
同步串行问题；没有证据支持先做 GPU NMS，也不能仅凭 Nsight Systems 宣称某 Kernel 为
memory-bound。

下一优先级由证据冻结为 Exp14：Pinned Memory → CUDA Event 依赖 → 双缓冲 → 跨帧拷贝/计算
重叠。Exp14 必须另行审批，Exp13 没有提前实施这些优化。

## 2. 环境与冻结输入

- Jetson Orin Nano Super，Ubuntu 22.04 / L4T R36.4.3，aarch64；
- CUDA 12.6、TensorRT 10.3、OpenCV 4.10、GStreamer 1.20；
- Nsight Systems 2024.5.4.34；25W/id 1，动态调频；
- 正式测量代码：`exp/13-nsight-profiling@5ad07aa`；
- FP16 Engine SHA256：`88dcc29…c0a83`，batch=1，输入 `1×3×640×640`；
- 文件视频 SHA256：`f00e116…a66665`；IMX219 sensor-id 0，1920×1080@30；
- `conf=0.25`，class-aware NMS IoU 0.70，warmup=2。

大体积 `.nsys-rep`、SQLite、Engine 和视频只保留在 Jetson；Git 只保存本总结、聚合 JSON、哈希
清单、代码和小型 CSV/JSON 工具。

## 3. 实现与测量边界

编译开关 `PPE_ENABLE_NVTX` 控制标记，关闭时仍构建原 `exp11_video_infer`，开启时构建
`exp13_profiled_video_infer`。每帧标记：`frame_total`、`capture`、`h2d`、
`preprocess_kernel`、`preprocess_sync`、`tensorrt_enqueue`、`d2h`、`inference_sync`、
`decode`、`nms`、`output`。

三种 FPS 不能混用：

- process wall FPS 包含进程启动、Engine 初始化和退出；
- pipeline wall FPS 覆盖稳态主循环，包括采集和输出，但排除一次性启动/退出；
- app effective FPS 沿用 Exp11 计时边界，不包含标注帧与 CSV 输出。

最终性能取三次无 profiler 运行；Nsight 运行只用于诊断，避免 observer effect 被当成收益或退化。

## 4. 正确性与 Smoke Test

- 环境审计、Nsight schema 检查和 30 帧 Smoke 全部 `PASS`；
- 原 Exp11 二进制、NVTX 关闭目标、NVTX 开启目标均完成 5 帧、5 检测；三份 detection CSV
  SHA256 都为 `5998153…da793`；
- 正式文件轮完成 150 帧、151 检测，三次无 profiler 与 Exp11 冻结参考的 detection CSV SHA256
  都为 `9f3f3345…e34e2d8a`；
- 正式相机轮和三次无 profiler 轮均完成 300 帧并返回0。场景内容会随时间变化，因此相机检测数和
  digest 不要求相等；帧数、返回码和输出合法性通过。

## 5. 三次无 Profiler 正式性能

| 场景 | Pipeline wall FPS mean / CV | App FPS mean | E2E mean | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| 文件 3×150 帧 | 61.583 / 3.002% | 66.138 | 15.130 ms | 16.729 ms | 16.937 ms |
| 相机 3×300 帧 | 30.174 / 0.028% | 31.423 | 31.824 ms | 34.715 ms | 35.783 ms |

文件模式可以超过 60 FPS；相机 pipeline wall 几乎严格稳定在 30.17 FPS，证明不能把摄像头场景
约 32 ms 的端到端时延解释为 Engine 计算时间。

## 6. Nsight 时间线

### 6.1 文件视频，150 帧

| 阶段 | Host range mean | 占 frame span |
|---|---:|---:|
| capture | 1.091 ms | 6.90% |
| H2D API range | 0.897 ms | 5.71% |
| preprocess kernel launch range | 0.062 ms | 0.39% |
| preprocess sync | 0.766 ms | 4.87% |
| TensorRT enqueue range | 6.089 ms | 38.73% |
| D2H API range | 4.053 ms | 25.78% |
| decode + NMS | 0.248 ms | 1.58% |
| output | 1.103 ms | 7.02% |

`frame_total` mean/P95/P99 为 15.717/17.623/20.385 ms。GPU busy/idle 为
66.19%/33.81%，Kernel 与 Memcpy 重叠为 0%。每帧 H2D 6,220,800 bytes，真实 GPU copy
均值 0.825 ms；每帧 D2H 235,200 bytes，真实 GPU copy 仅 0.0284 ms。

关键解释：D2H NVTX 主机范围 4.053 ms 不等于 D2H 传输耗时。它包住的
`cudaMemcpyAsync` 在当前 pageable、同 Stream、串行依赖下等待前序 TensorRT 工作完成；Nsight
CUDA API 汇总中 `cudaMemcpyAsync` 占 API 总时间 54.0%，而真实 D2H GPU activity 只有
0.0284 ms，形成同步阻塞的直接证据。

### 6.2 IMX219，300 帧

`frame_total` mean/P95/P99 为 33.228/36.363/37.571 ms；`capture` 均值 14.644 ms，
占正式 frame span 43.92%。GPU busy/idle 为 36.69%/63.31%，Kernel 与 Memcpy 重叠仍为 0%。
摄像头首先是 `input-rate-bound`，同时 GPU 大量空闲和零重叠说明同步流水线仍是次要瓶颈。

## 7. TensorRT 层级诊断

`trtexec --noDataTransfers --separateProfileRun --dumpLayerInfo --dumpProfile` 返回0。178 个层记录
的平均耗时之和为 5.1003 ms。最大单层是首层输入 Reformat CopyNode，0.1939 ms、占3.8%；之后
多个卷积/PWN 层各约 0.10 ms，没有一个单层支配整个 Engine。逐层 profiling 会改变执行特性，
该数字只做热点诊断，不替代 Exp07 GPU-only 或本实验端到端 Benchmark。

## 8. 瓶颈分类与决策

- 文件：`synchronization-bound`，同时 TensorRT 计算占比较高，但不是单层热点主导；
- 相机：`input-rate-bound + synchronization-bound`；
- CPU decode+NMS 仅约0.248 ms/帧，不是当前首要性能矛盾；
- H2D 体积和 pageable memory 值得优化，D2H 字节本身很小，主机 D2H 时间主要是等待；
- Nsight Systems 可证明时间线、空洞和同步，不能证明 Kernel 是 compute-bound 还是 memory-bound；
  后者留给 Nsight Compute。

因此 Exp14 应先消除不必要同步、引入 pinned staging 和双缓冲，使 H2D(N+1) 与 inference(N)
形成重叠；不能并发复用同一个 TensorRT execution context。GPU 后处理放在后续实验，因为当前
CPU decode/NMS 占比过低，先做它难以得到主要收益。

## 9. 失败现场与最小修复

所有失败目录均保留：首次 NVTX OFF 构建因公共头文件路径只在 ON 分支暴露而失败，修复为始终
暴露 wrapper；一次 PowerShell/Bash 变量转义生成字面量 `$plain/$nvtx/$out` 目录，后续改用明确
路径；首次 Nsight 编排因嵌套 grep 管道转义失败；分析器首次遇到稀疏 CSV 字段失败，改为字段并集；
系统没有 sqlite3 CLI，改用 Python 标准库而未安装依赖；增加 `pipeline_wall_seconds/FPS` 后重新
执行所有正式无 profiler 测试，旧测量不覆盖、不作为结论。首次 `trtexec` 使用了错误的 Engine
目录，失败现场保留，随后从已记录 app command 核对真实路径后重跑通过。

## 10. 已证实、尚未证实与学习要点

已证实：NVTX 可关闭且不改变检测语义；文件和相机正式结果可重复；当前流水线没有拷贝/计算
重叠；相机受输入节拍限制；文件模式存在 pageable copy 与阶段同步阻塞；Engine 没有单层绝对
热点；实验结束时 CPU/GPU 均未锁频，25W/id 1 不变。

尚未证实：Pinned Memory、双缓冲或多 Stream 的实际收益；Nsight Compute 的 SM/DRAM 指标；
NVMM 零拷贝；GPU NMS；高并发、多摄像头和长期优化版稳定性。上述内容不得写成已完成。

面试中应讲清：`cudaMemcpyAsync` 只有配合 pinned host memory 和正确依赖才具备真正异步重叠的
基础；NVTX host range 不等于 GPU activity；Profiler 数字不能和无 profiler Benchmark 混用；
30 FPS 摄像头会让更快的 Engine 看起来仍约 32 ms/帧；优化顺序应由关键路径和占比决定。
