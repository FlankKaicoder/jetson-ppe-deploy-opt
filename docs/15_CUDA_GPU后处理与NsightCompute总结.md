# Exp15 CUDA GPU Decode/Filter/Compaction 与 Nsight Compute 总结

## 1. 最终结论

Exp15 状态为 `PASS`。Variant A（Atomic）和 Variant B（CUB stable compaction）均完成 GPU
class-max、confidence filter、box decode、可变长候选压缩 D2H 和 CPU class-aware NMS，并通过
synthetic、固定视频、IMX219、Nsight Systems 与 Nsight Compute 验收。Variant B 满足全部冻结门槛，
采用为新的 FP16 C++ Runtime 后处理主线；YOLO11n baseline、Exp06 ONNX、Exp07 FP16 Engine 和
CPU NMS 语义不变。Atomic 实现作为低 Kernel 开销对照保留。

## 2. 环境、输入与公平边界

- Jetson Orin Nano Super，L4T R36.4.3，CUDA 12.6.68，TensorRT 10.3.0.30，25W/id 1动态调频；
- 分支起点：`main@0b17852638bf68125d5c75ce6331c54fc430b83a`；
- FP16 Engine：8,951,540 B，SHA256
  `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83`；
- 文件视频：874,518 B，SHA256
  `f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665`；
- batch=1、`1×3×640×640`、raw output `1×7×8400`、warmup=2、conf=0.25、CPU NMS IoU=0.70；
- baseline、A、B 分别进行文件150帧和相机300帧各三轮无 profiler，同日串行比较；
- Exp14 pinned/double-buffer 路径未叠加，Profiler 结果不混入正式性能数字。

## 3. 实现与正确性

`GpuCandidate` 为28 B，包含 candidate index、class、confidence 与四个 box 坐标。A 使用单 Kernel
和 atomic counter；B 生成 candidate/flag 后调用 CUB `DeviceSelect::Flagged` 保持候选索引顺序。
两者都先 D2H 4 B count，再复制 `count × 28 B` payload，capacity 固定8400且不静默截断。

synthetic 覆盖零/单/8400候选、阈值等于0.25、类别并列、非法数值、边界框和小容量 guard；A/B
相对 CPU reference 的 confidence/box 最大误差均为0，8400计数及越界保护通过。5帧 Smoke 中旧
Exp11、Exp15 baseline、A、B 的检测逐项一致。正式文件9/9轮均为150帧、151检测，SHA256均为
`9f3f33459f8d086a74249a57f21f158a73ca794a2229a9e1af40a03de34e2d8a`；相机9/9轮均处理300帧，
无 NaN/Inf、非法框、CUDA错误或帧序错误。

## 4. 三轮无 profiler 聚合

| 场景 | 变体 | Wall FPS | E2E mean (ms) | E2E P95 (ms) | D2H B/帧 |
|---|---|---:|---:|---:|---:|
| 文件 | baseline | 60.270 | 15.502 | 16.940 | 235,200.000 |
| 文件 | A Atomic | 70.201 | 13.310 | 14.668 | 263.840 |
| 文件 | B CUB | 71.838 | 12.939 | 14.403 | 263.840 |
| 相机 | baseline | 30.135 | 31.929 | 34.242 | 235,200.000 |
| 相机 | A Atomic | 30.159 | 31.830 | 34.400 | 278.276 |
| 相机 | B CUB | 30.136 | 31.848 | 34.794 | 279.956 |

相对同日 baseline，A 的文件 FPS +16.478%、mean -14.144%、P95 -13.412%、相机 P95 +0.462%；
B 的文件 FPS +19.194%、mean -16.534%、P95 -14.973%、相机 P95 +1.614%。A/B 文件 D2H 均减少
99.888%，全部满足“FPS或mean改善至少3%、文件/相机 P95退化不超过5%、D2H减少至少80%”门槛。
B 的整体文件性能更高且天然保持稳定次序，因此采用 B。

## 5. Nsight Systems 与 Nsight Compute

正式 B Nsight Systems 输出为 `exp15_profile_B_file_20260808_202631` 和
`exp15_profile_B_camera_20260808_202700`。文件正式窗口记录152次 decode/CUB/count/payload链路，
相机记录302次；文件 Kernel 平均约 decode 11.17 µs、CUB init 5.18 µs、select 21.17 µs，
相机分别约13.31/6.19/25.51 µs。时间线明确显示4 B count D2H、count同步、变长 payload D2H和
CPU NMS边界；同步仍存在，但传输与 CPU decode 工作量显著下降。

固定 raw fixture 为235,200 B，SHA256
`0e6aff4557d989ec62c26908988bcb5b15222de4a7d66f53ea73f36ce825abfe`。Nsight Compute 2024.3.1
正式报告如下：

| Kernel | Duration | SM % | Memory % | Achieved occupancy | Waves/SM | Branch efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Atomic decode/filter | 23.90 µs | 10.48 | 14.19 | 49.32% | 0.69 | 100.00% |
| CUB decode/flag | 21.12 µs | 12.54 | 20.31 | 51.91% | 0.69 | 99.45% |
| CUB init | 11.94 µs | 0.36 | 1.35 | 8.12% | 0.01 | 100.00% |
| CUB select | 35.01 µs | 9.80 | 25.71 | 64.17% | 0.69 | 100.00% |

Atomic 单 Kernel 更短；CUB 总 Kernel 时间更高，但省去 Atomic 输出在 CPU 侧恢复稳定顺序的成本，
因此真实 Runtime 中 B 更优。所有主要 Kernel 的 waves/SM不足1，SM和Memory throughput均远未饱和，
主要限制是8400候选形成的小 grid/launch固定成本，不是算力或 DRAM带宽打满。当前 `full` CSV 未暴露
独立 Warp Stall breakdown、atomic transaction 或具名 DRAM throughput 字段；文档明确记为
“NCU 2024.3.1 当前导出不可用”，未从其他指标推造数值。

## 6. 失败现场、产物和决策边界

首次 C++ synthetic 编译因 typed pointer 未显式转换为 `void**` 失败，最小修复后通过；首次远程正式
编排因 PowerShell 展开远端 shell变量而只触发 usage错误，改用独立编排脚本；首次非 root NCU 因
硬件计数器权限失败，目录 `exp15_ncu_atomic_20260808_203027` 保留为 `FAIL`，用户在交互终端以 sudo
运行后 Atomic/CUB 均生成报告和CSV。没有删除失败目录、隐藏返回码或修改系统驱动。

主要输出：构建 `exp15_0_build_20260808_201857`；文件/相机运行注册表
`exp15_formal_file_20260808_202128`、`exp15_formal_camera_20260808_202222`；聚合
`exp15_compare_20260808_202500`；NCU `exp15_ncu_atomic_20260808_205645`、
`exp15_ncu_cub_20260808_205742` 与 `exp15_ncu_compare_20260808_210000`。大型报告仅留Jetson。

## 7. 已证实、尚未证实与下一步

已证实：GPU decode/filter/compaction能在保持检测语义的同时显著压缩 D2H，并在当前同步 FP16
Runtime 中获得可复现的文件端收益；CUB稳定压缩的系统级收益优于更短的Atomic Kernel。尚未证实：
GPU NMS、TensorRT Plugin、GraphSurgeon、INT8 mixed precision 或 CUDA Graph 的收益，这些分别属于
Exp16～Exp18，不能写成当前成果。下一候选为 Exp16 IPluginV3 与 ONNX GraphSurgeon，必须先提交
方案并获得用户批准，不得直接实现。

面试时应能解释：为什么235 KB copy本身仅几十微秒但消除 CPU全量扫描后仍有系统收益；为什么
Kernel更短不等于 Runtime更快；Atomic的非确定输出如何影响后续稳定语义；CUB temporary storage与
stable compaction的代价；以及低 waves/SM为何说明“小工作量/固定开销”而非 GPU算力不足。

## 8. Postprocess Gain Attribution Gate（2026-08-09）

本 Gate 不新增 `Exp15.1` 编号，也不改变 Exp15 `PASS/ACCEPTED`。为拆分“GPU decode”与“压缩D2H”，
新增 P1 fixed路径：GPU写满8400×28 B Candidate，无效项使用严格的 `index=-1` sentinel；P0和P1最终
都使用pinned Host buffer并固定D2H 235,200 B，P2保持CUB count+可变长payload，平均263.84 B。三条
路径均保持150帧、151检测和冻结digest。

最终动态调频paired/interleaved结果如下：

| 路径 | Wall FPS | E2E mean | E2E P95 | D2H B/帧 |
|---|---:|---:|---:|---:|
| P0 pinned raw + CPU decode/NMS | 64.306 | 14.346 ms | 15.686 ms | 235,200.00 |
| P1 GPU decode + fixed D2H + CPU NMS | 66.544 | 13.795 ms | 15.208 ms | 235,200.00 |
| P2 GPU decode + CUB compact + CPU NMS | 65.037 | 14.164 ms | 15.038 ms | 263.84 |

P0→P1 的P95三轮均改善，paired平均为−3.05%；但FPS和mean三轮有正有负。P1→P2 的P95同样三轮均
改善，平均−1.11%；FPS和mean仍混合。P0→P2 的P95三轮均改善，平均−4.13%，但不能稳定分摊吞吐和
mean。冻结fixture的每路径1000次测试中，P0/P1/P2 total mean为0.2295/0.3634/0.1201 ms；P1固定
扫描8400项的局部成本明显，P2通过stable compaction消除大部分fixed scan和payload成本。

早期P0仍使用pageable `std::vector<float>`，与pinned P1比较会混入Host memory变量；这些目录作为
预修复证据保留，不用于最终归因。最终结论是：Exp15实际主线收益来自pageable raw路径、CPU全量decode
和完整raw传输被联合替换；在严格控制Host memory与字节后，GPU decode和compaction对P95各有小幅且
方向一致的贡献，但动态调频下FPS/mean不足以支持精确百分比分摊。P2继续作为主线，因为它保持正确性、
显著减少D2H与Host扫描，并已通过原Exp15采用门槛；不得把19.19%全部归因于235 KB copy。

Compute Sanitizer首次因命令未加入PATH失败，使用绝对路径后又因Jetson禁用GPU debugging features而
无法执行memcheck；测试程序自身通过，但该项明确记为“环境不支持”，不能表述为memcheck PASS。
