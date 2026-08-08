# Exp12 Jetson 性能、功耗、温度与稳定性总结

## 1. 最终结论

状态：`PASS`

Jetson Orin Nano Super 在 25W 模式、CPU 1.344 GHz、GPU 918 MHz 固定时钟下，完成
3 次独立的 1,800 帧性能测试和 54,000 帧、约 30 分钟 IMX219 端到端稳定性测试。
所有运行均返回 0，帧数完整，无摄像头早停、NaN/Inf、非法检测、温度安全停止或资源监控失败。

长稳态端到端 P95/P99 为 32.885/33.521 ms，有效处理率 30.960 FPS；54,000 帧的实际
wall time 为 1,801.743 秒，即 29.971 帧/秒，能够跟随 30 FPS 摄像头节拍。最高 CPU/GPU/Tj
温度为 57.968°C，未触及 70°C passive trip；VDD_IN mean/P95/max 为
9.100/9.352/9.902 W；稳态 RSS 线性斜率为 0.00368 MiB/min，没有内存泄漏证据。

固定时钟没有带来性能收益：相对同一实验的未锁频 1,800 帧基线，三次锁频 mean 延迟平均
增加 0.984%，FPS 平均下降 0.974%，VDD_IN mean 平均增加 14.761%。因此锁频只用于公平
Benchmark，不作为项目默认运行配置；实验结束后恢复动态调频。

## 2. 环境、输入与计时边界

- 设备：Jetson Orin Nano Super，aarch64；
- 系统：Ubuntu 22.04 / L4T R36.4.3；
- CUDA 12.6.68，TensorRT 10.3.0.30，OpenCV 4.10，GStreamer 1.20；
- 功耗模式：25W/id 1；
- 正式运行代码：`exp/12-jetson-benchmark@1bf2c87d98eaba92598590d803cc97d60ddfb904`；
- FP16 Engine SHA256：`88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83`；
- Exp11 二进制 SHA256：`bf3717d8b4feb17617ea4e831dcc6fffdb69a13d52449c0727aa8697ada4a5c0`；
- 输入：IMX219 sensor-id 0，1920×1080@30，batch=1，imgsz=640；
- 后处理：`conf=0.25`、class-aware NMS IoU 0.70；
- 端到端范围：取帧 + pageable H2D + CUDA 融合预处理 + TensorRT + 输出 D2H + CPU NMS；
- warmup：Smoke 2 次，正式测试 20 次，不计入正式帧 CSV；
- 资源采样：tegrastats 与 `/proc/<pid>`，间隔 1 秒。

未锁频状态为 CPU `schedutil`、GPU `nvhost_podgov`；锁频后 6 个 CPU 的 min/max 均为
1.344 GHz，GPU min/max 均为 918 MHz。CPU 硬件上限 1.728 GHz，但 25W nvpmodel 将当前
policy 上限固定为 1.344 GHz，因此本实验没有越过功耗模式配置。

## 3. Smoke 与测量修正

首次监控 Smoke：

```text
results/benchmark/exp12_0_monitor_smoke_20260808_141647
```

监控器读取暂时不可用的 CV thermal zone 时，Python `pathlib` 收到非标准空读取并触发
`TypeError`；应用在启动前被监控器终止，失败 JSON 和返回码保留。用 `sleep 2` 最小复现定位到
热区读取后，修复为跳过不可读 zone，继续监控可用的 CPU/GPU/Tj/SOC 温区。

Smoke 同时发现首个 `/proc` RSS 样本早于 TensorRT/OpenCV 初始化，整机 SWAP 也不能归因于
目标进程。正式结果前已追加并冻结测量修正：稳定性 RSS 从第 60 秒后计算；SWAP 验收运行期间
是否增长，同时记录目标进程 VmSwap。没有修改延迟、FPS、温度、帧数或内存增长门槛。

修正后的正式 Smoke：

```text
results/benchmark/exp12_0_monitor_smoke_20260808_142521
```

结果为 300/300 帧、31.623 FPS、P95/P99 32.622/33.443 ms、最高 50.031°C，监控与解析
均 PASS。

## 4. 未锁频基线

目录：

```text
results/benchmark/exp12_1_unlocked_baseline_20260808_142533
```

| 指标 | 结果 |
|---|---:|
| 处理帧数 | 1,800 |
| 端到端 mean | 31.875 ms |
| P50 / P95 / P99 | 31.889 / 33.472 / 34.343 ms |
| 有效处理率 | 31.373 FPS |
| 最高 CPU/GPU/Tj | 52.875°C |
| VDD_IN mean / P95 / max | 8.124 / 8.160 / 8.200 W |
| CPU / GPU mean utilization | 20.552% / 35.213% |

该 run 作为同日、同程序、同摄像头、同 25W 模式的动态调频对照，不直接复用 Exp11 的短时结果。

## 5. 固定时钟三进程性能与重复性

运行目录：

```text
results/benchmark/exp12_2_locked_performance_20260808_143202
results/benchmark/exp12_2_locked_performance_20260808_143304
results/benchmark/exp12_2_locked_performance_20260808_143406
results/benchmark/exp12_2_locked_repeatability_20260808_143508
```

| Run | mean (ms) | P95 (ms) | P99 (ms) | FPS | 最高温度 | VDD_IN mean |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 32.194 | 32.731 | 33.344 | 31.061 | 55.687°C | 9.327 W |
| 2 | 32.203 | 32.725 | 33.044 | 31.053 | 56.781°C | 9.317 W |
| 3 | 32.166 | 32.970 | 33.442 | 31.088 | 57.375°C | 9.327 W |

三次都通过 P95 `<=40 ms`、P99 `<=50 ms`、FPS `>=29` 和温度 `<70°C` 门槛。
mean 延迟 CV 为 0.04885%，FPS CV 为 0.04886%，P95 相对跨度为 0.74888%，均显著低于
5%/5%/10% 的预冻结重复性门槛。

锁频三次平均 mean/FPS/VDD_IN 为 32.188 ms、31.067 FPS、9.323 W。相对未锁频基线：

- mean 延迟增加 0.9836%；
- FPS 下降 0.9740%；
- VDD_IN mean 增加 14.7606%。

锁频消除了 DVFS 变量并带来极高重复性，但当前同步、摄像头节拍受限链路没有因此变快；持续
高频反而增加功耗和温度。这是负向但有价值的部署决策证据。

## 6. 54,000 帧稳定性测试

目录：

```text
results/benchmark/exp12_3_locked_stability_20260808_143647
```

### 6.1 正确性与延迟

| 指标 | 结果 | 冻结门槛 | 判定 |
|---|---:|---:|---|
| 帧数 | 54,000 | 恰好 54,000 | PASS |
| 检测数 | 74,498 | 合法结构化输出 | PASS |
| 应用返回码 | 0 | 0 | PASS |
| 有效处理率 | 30.960 FPS | >=29 FPS | PASS |
| wall throughput | 29.971 FPS | 跟随 30 FPS 节拍 | PASS |
| mean / P50 | 32.300 / 32.318 ms | 报告 | PASS |
| P95 | 32.885 ms | <=40 ms | PASS |
| P99 | 33.521 ms | <=50 ms | PASS |
| 单帧最大值 | 66.598 ms | 报告，P99 为强制门槛 | PASS |

最初 10% 帧 P95 为 32.925 ms，最后 10% 为 32.885 ms，变化 -0.1234%，没有尾延迟随
运行时间恶化的证据。

### 6.2 功耗、温度与利用率

共获得 1,791 条有效 tegrastats 和 1,802 条进程样本：

| 指标 | mean | P95 | max |
|---|---:|---:|---:|
| VDD_IN | 9.100 W | 9.352 W | 9.902 W |
| CPU 温度 | 56.357°C | — | 57.250°C |
| GPU/Tj 温度 | 57.127°C | — | 57.968°C |
| 系统 RAM | 2,985.8 MB | — | 2,997 MB |

CPU/GPU 平均利用率分别为 13.878%/13.613%，目标进程 CPU 平均为 57.545%（以单核 100%
为基准）。最高温度比 70°C passive trip 低 12.032°C，未触发热安全停止，也没有热降频依据。

### 6.3 内存与交换

- 稳态 RSS mean/max：367.043/367.129 MiB；
- RSS 末值相对首值：-45.180 MiB，说明一次性缓存被释放；
- RSS 线性斜率：0.003680 MiB/min，低于 1 MiB/min 门槛；
- 整机 SWAP 始终为 1 MiB，增长 0；目标进程 VmSwap 为 0。

因此 30 分钟内没有持续内存增长或交换压力证据。

## 7. 证据文件与哈希

正式稳定性核心文件：

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `summary.json` | 1,776 B | `b19396eb0cd1f11203fa0beabf92a25ee00fdc89e0a631e668cc6e2826d25429` |
| `frames.csv` | 5,065,028 B | `433b24ca7808bb6da971028d6d8a19069ae17e33eb7764665f704f7161d164e0` |
| `detections.csv` | 6,402,110 B | `2e372ff170a2deb2632a89107a3519fd37792b92470ffc6d871d7fa15576f08a` |
| `tegrastats.log` | 520,826 B | `323502a70c98e5dddad461d37e80bee03c15776936b20a5ef2201832e26b9bff` |
| `process_samples.csv` | 137,743 B | `8b4365162b05c3fe5926f0f4527b77ab64947c6f964886a6e1fb36f3a17b6ff0` |
| `exp12_summary.json` | 5,607 B | `560f2c4a718946a3c66d6396d63500abfd4189d308f5f84e752573689a7a4d85` |

大型逐帧 CSV 和日志只保留在 Jetson，不进入普通 Git；仓库只提交汇总结果、哈希和文档。

## 8. 已证实、尚未证实与决策

已证实：在固定 25W、固定时钟、IMX219 1920×1080@30 和当前同步链路下，端到端尾延迟、
实时节拍、功耗、温度和内存能稳定维持 30 分钟；三次短测具有极高重复性；没有达到 thermal
passive trip、没有资源泄漏或应用错误。

尚未证实：数小时/全天稳定性、不同室温/风扇/机箱、多个摄像头、不同功耗模式、NVMM 零拷贝、
异步流水线、GPU NMS、系统其他高负载并发时的表现，以及上游 Argus 精确 drop counter。由于
OpenCV 未可靠暴露 PTS/drop counter，本实验只能证明固定帧数和 wall throughput 跟上节拍，
不能声称硬件层“绝对零丢帧”。

最终决策：Exp12 `PASS`。默认部署保持 25W 动态调频，不常驻 `jetson_clocks`；锁频只用于公平
Benchmark。下一实验为 Exp13 项目收尾、README、简历与面试材料。

实验结束时首次执行 `sudo jetson_clocks --restore`，发现 `/root/.jetsonclocks_conf.txt` 不存在；
该版本脚本不会在锁频时自动保存原状态，只有事先 `--store` 才能使用 `--restore`。没有伪造恢复
成功，而是重新应用 `sudo nvpmodel -m 1`。随后从 sysfs 验证 6 个 CPU 的 min/max 已恢复为
729.6/1,344 MHz，GPU min/max 已恢复为 306/918 MHz，CPU/GPU 锁频判定均为 `false`，功耗模式
仍为 25W/id 1。今后使用 `jetson_clocks` 前必须先把动态状态显式保存到实验专用配置文件。

## 9. 快速学习与面试回答

为什么锁频反而略慢？本链路由30 FPS摄像头节拍和串行同步共同约束，动态调频已经能及时升频；
常驻最高频率没有消除主要瓶颈，却增加静态/动态功耗和热负担。小于1%的差异也不能包装成加速。

怎样证明没有内存泄漏？不能比较进程刚 `exec` 时的几 KiB 与初始化后的几百 MiB。应排除模型、
CUDA Context 和解码器的一次性初始化，观察稳态 RSS 首尾差、最大值和随时间回归斜率。本实验
斜率仅 0.00368 MiB/min。

为什么平均 FPS 不够？平均值会隐藏偶发长尾、热降频和内存增长。本实验同时冻结 P95/P99、
首尾窗口退化、温区 trip、功耗分布、RSS 趋势、三进程 CV 和应用返回码，才能形成完整板端证据。
