# Exp10：CUDA 融合预处理与验证总结

## 当前状态

```text
PASS
```

本文档先保留实验前冻结的边界和验收条件，再追加真实运行结果与决策。

## 实验目标

1. 以 Exp07 Python/OpenCV 预处理为语义基准，在 Jetson 实现 CUDA 融合预处理；
2. 单个 Kernel 完成 letterbox resize、padding、BGR→RGB、`/255` 和 HWC→CHW；
3. 覆盖方形、横向、纵向、缩小和放大输入；
4. 比较 CPU OpenCV Reference 与 CUDA 输出的逐元素误差、几何元数据和 padding；
5. 分开报告 CPU、CUDA kernel-only 和 H2D+kernel+D2H，避免错误性能结论；
6. 为后续 C++ Runtime 的 device-to-device 输入连接冻结接口。

## 冻结语义

CPU 黄金参考必须复现 Exp07：

```text
ratio = min(640 / height, 640 / width)
resized = round(original * ratio)
cv::resize(..., INTER_LINEAR)
left/right/top/bottom 使用 round(half_padding ± 0.1)
padding = BGR(114,114,114)
BGR → RGB
uint8 → FP32 / 255
HWC → NCHW [1,3,640,640]
```

CUDA Kernel 使用 OpenCV half-pixel 线性插值坐标，并复现 OpenCV 4.10 的 11-bit 系数和 CV_8U
专用纵向定点路径，再完成颜色、归一化和布局转换。输出为可直接绑定 Exp09 TensorRT Runtime
的连续 FP32 device tensor。

## 冻结输入与测试夹具

冻结探针图像：

```text
/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg
SHA256 = 39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e
```

该图和现有 test 图均为 640×640，无法覆盖 resize/padding。测试工具必须从冻结探针确定性生成
PNG 夹具，不使用随机增强：

| 名称 | H×W | 覆盖目的 |
|---|---:|---|
| square | 640×640 | 无 resize、无 padding |
| wide | 360×640 | 上下 padding |
| tall | 640×360 | 左右 padding |
| hd_wide | 720×1280 | 缩小与上下 padding |
| small_tall | 480×240 | 放大与左右 padding |

PNG 只保存在时间戳结果目录，不提交 Git；manifest 记录来源、生成规则、大小和 SHA256。

## 预冻结正确性门槛

每个测试夹具都必须满足：

| 检查 | 门槛 |
|---|---:|
| 输出形状 | `[1,3,640,640]` |
| NaN/Inf | 0 |
| resize 尺寸和四边 padding | 与 CPU Reference 完全一致 |
| padding 区域 | 与 `114/255` 完全一致 |
| max absolute error | `<= 2/255 + 1e-7` |
| mean absolute error | `<= 0.0005` |
| relative L2 error | `<= 0.001` |
| P99 absolute error | `<= 1/255 + 1e-7` |

误差门槛在 CUDA 正式结果产生前冻结；不得为通过实验而事后放宽。方形无 resize 夹具应逐字节等价。

## 预冻结执行与性能口径

- Smoke Test：`hd_wide`，warmup=2、iterations=5；
- 正式正确性：全部 5 个夹具；
- 正式性能：`hd_wide`，CPU/CUDA warmup=20、iterations=200；
- CPU：OpenCV resize + border + BGR/RGB + normalize + HWC/CHW；
- CUDA kernel-only：输入已在 device、输出留在 device；
- CUDA total：pageable H2D + kernel + FP32 output D2H + synchronize；
- 分别记录 mean/P50/P95/P99；不包含图像解码；`jetson_clocks` 未锁定时必须注明；
- kernel-only mean 至少比 CPU mean 低 30%，作为性能门槛；
- CUDA total 只做诊断，不作强制门槛，因为真实流水线会把输出直接留在 device 供 TensorRT 使用。

## 推荐实现

```text
cuda/CMakeLists.txt
cuda/include/cuda_preprocess.hpp
cuda/src/cuda_preprocess.cu
cuda/src/exp10_preprocess.cpp
tools/exp10_make_fixtures.py
tools/exp10_compare_preprocess.py
tools/exp10_0_cuda_preprocess_smoke.sh
tools/exp10_1_cuda_preprocess_formal.sh
```

运行目录固定为 `results/cuda_preprocess/exp10_*_YYYYMMDD_HHMMSS/`，不得覆盖历史结果。

## 风险与停止条件

- OpenCV 与 CUDA 的 half-pixel 插值或舍入语义不一致；
- width/height、x/y 或 BGR/RGB 颠倒；
- 奇数 padding 分配与 `±0.1` 规则不一致；
- 输出索引错误导致 HWC/CHW 混淆；
- 只测 640×640，漏掉真正的 resize 和 padding；
- 将 kernel-only 与包含传输的 CPU/端到端口径混比；
- CUDA 异步错误未在同步点暴露。

Smoke Test 失败、输出非有限、几何不一致、任一正确性门槛失败或 Kernel 性能门槛失败时，保留
目录与日志并停止正式采用；不删除负向结果。

## 实现与运行环境

- 执行设备：Jetson Orin Nano Super，aarch64；
- CUDA：12.6.68；OpenCV：4.10.0；CMake：3.22.1；G++：11.4.0；
- 正式结果对应代码 Commit：`e2ee8c2`；
- `jetson_clocks`：未锁定；
- 图像解码不计时，kernel-only 和 pageable H2D+kernel+D2H 分开统计；
- 正式输出目录：
  `results/cuda_preprocess/exp10_1_cuda_preprocess_formal_20260807_173638/`。

实现文件包括 CMake、CUDA 接口与 Kernel、C++ CPU Reference/计时程序、确定性夹具生成器、
逐元素比较器、正式汇总器以及 Smoke/正式运行脚本。原始 PNG、FP32 binary、完整日志和编译产物
留在 Jetson 时间戳目录，不进入 Git。

## Smoke Test

首次运行 `exp10_0_cuda_preprocess_smoke_20260807_171257` 在 CMake 配置阶段失败：非登录 SSH
环境的 `PATH` 没有暴露 CUDA compiler。失败目录和返回码保留；没有修改系统 CUDA，而是在项目
脚本中显式传入：

```text
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

修复后的 `exp10_0_cuda_preprocess_smoke_20260807_171417` 为 `PASS`。`hd_wide` 的 CPU 与 CUDA
共 1,228,800 个 FP32 元素全部一致，max/mean/relative-L2/P99/padding 误差均为 0；5 次小样本
计时为 CPU 2.78496 ms、kernel-only 0.283155 ms、total 2.63444 ms。Smoke 只证明最小链路可行，
不代替正式结果。

## 正式正确性结果

最终正式 run 的 5 个夹具全部 `PASS`：

| 夹具 | H×W | CPU mean (ms) | Kernel mean (ms) | Total mean (ms) | max / mean / rel-L2 / P99 / padding error |
|---|---:|---:|---:|---:|---:|
| square | 640×640 | 2.80762 | 0.262637 | 2.33711 | 全部 0 |
| wide | 360×640 | 2.49509 | 0.218240 | 1.69194 | 全部 0 |
| tall | 640×360 | 2.68307 | 0.223341 | 1.70414 | 全部 0 |
| hd_wide | 720×1280 | 2.28212 | 0.200761 | 1.88307 | 全部 0 |
| small_tall | 480×240 | 3.47549 | 0.250714 | 1.61603 | 全部 0 |

所有输出均为有限值和 `[1,3,640,640]`，resize/padding 几何与 CPU Reference 完全一致。
`square` 满足预冻结的逐字节等价要求；其余 4 个形状也达到了逐元素零误差，强于冻结门槛。

## 正式性能结果

正式性能夹具 `hd_wide` 使用 warmup=20、iterations=200：

| 口径 | mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| CPU 完整预处理 | 2.282124 ms | 2.340598 | 2.384411 | 2.445414 |
| CUDA kernel-only | 0.200761 ms | 0.135632 | 0.278627 | 0.281152 |
| CUDA pageable total | 1.883069 ms | 1.920672 | 1.971808 | 2.056235 |

kernel-only mean 相对 CPU 下降 `91.2029%`，通过“至少下降 30%”的预冻结门槛。约 11.37× 的
CPU/kernel 数值只适用于本次预处理计时边界；不能宣称为整模型、摄像头或端到端加速。total 包含
输入 H2D 和 4.9 MB FP32 输出 D2H，仅用于诊断；后续流水线应把输出留在 device 直接交给 TensorRT。

## 负向实验、定位与最小修复

1. `172229`：朴素 float 双线性只有 `small_tall` 失败，max/P99 仅 1 灰度级，但 relative L2
   为 0.00183528，超过 0.001 门槛；没有放宽门槛。
2. `172728`：改为常规 11-bit 合并定点累加后误差不变，double 坐标还使 kernel 变慢，证明
   “理论双线性等价”不等于“复现 OpenCV 优化实现”。
3. `173220`：全局半整数向下舍入破坏 `hd_wide`，且没有完全修复 `small_tall`；该假设被拒绝。
4. 对照 OpenCV 4.10 `resize.cpp` 后确认 CV_8U 专用纵向路径先将横向累加右移 4 位，再分别取
   两个纵向乘积高位，最后对剩余 2 位舍入。CUDA 复现该顺序，并为无缩放输入保留直拷快路，
   最终 5 个夹具均为零误差且性能门槛通过。

这些失败说明固定坐标和 11-bit 系数仍不足以保证逐像素一致；运算顺序、中间截断和平台优化路径
同样是数值语义的一部分。

## 已证实、尚未证实与决策

已证实：

- 单 CUDA Kernel 可以融合 letterbox、padding、BGR→RGB、`/255` 和 HWC→NCHW；
- 本实验 5 种确定性形状与 Jetson OpenCV 4.10 CPU Reference 逐元素一致；
- 冻结 `hd_wide` 口径下 kernel-only 性能门槛通过；
- CUDA 接口能够产出连续 FP32 device tensor。

尚未证实：

- CUDA 输出尚未在同一进程中直接绑定 Exp09 TensorRT Context；
- 尚未验证视频、IMX219 摄像头、GStreamer、NMS 和完整端到端延迟；
- 尚未锁定功耗模式或 `jetson_clocks`，没有功耗、温度、降频和长期稳定性结论。

最终决策：Exp10 `PASS`，保留 CUDA 融合预处理进入部署主线；下一步 Exp11 将其与 C++ TensorRT
Runtime、视频/摄像头采集和后处理连接，验证真实端到端数据流。

## 完成清单

- [x] Windows 规划审查；分支合并与三端收口在本总结提交后执行；
- [x] Jetson 创建 `exp/10-cuda-preprocess`；
- [x] 确定性夹具、CPU Reference、CUDA Kernel 与比较器；
- [x] CMake/NVCC 编译与 Smoke Test；
- [x] 5 夹具正式正确性；
- [x] CPU/kernel-only/total 正式性能；
- [x] 文档与学习复盘；三端 Git 收口在本分支合并后执行。
