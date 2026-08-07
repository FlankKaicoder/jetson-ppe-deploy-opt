# Exp10：CUDA 融合预处理与验证总结

## 当前状态

```text
IN_PROGRESS
```

本文档先冻结实验边界和验收条件。尚未执行的结果不得表述为完成。

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

CUDA Kernel 使用 OpenCV half-pixel 线性插值坐标，并在插值后按 uint8 语义舍入，再完成颜色、
归一化和布局转换。输出为可直接绑定 Exp09 TensorRT Runtime 的连续 FP32 device tensor。

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

- Smoke Test：`wide`，warmup=2、iterations=5；
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

## 尚待执行

- [ ] Windows 规划审查并合并；
- [ ] Jetson 创建 `exp/10-cuda-preprocess`；
- [ ] 确定性夹具、CPU Reference、CUDA Kernel 与比较器；
- [ ] CMake/NVCC 编译与 Smoke Test；
- [ ] 5 夹具正式正确性；
- [ ] CPU/kernel-only/total 正式性能；
- [ ] 文档、学习复盘与三端 Git 收口。
