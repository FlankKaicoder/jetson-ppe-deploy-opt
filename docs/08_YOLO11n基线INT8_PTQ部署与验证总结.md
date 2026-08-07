# Exp08：YOLO11n 基线 INT8 PTQ 部署与验证总结

## 当前状态

```text
REJECT
```

工程链路已经完成，但 INT8 候选没有通过预冻结精度与关键尺度召回门槛，不替换 FP16。

## 实验目标

1. 只从训练集生成确定性、可审计的 TensorRT INT8 校准数据；
2. 在目标 Jetson TensorRT 10.3 环境构建带 FP16 fallback 的 INT8 Engine；
3. 对比 PyTorch FP32、TensorRT FP32、FP16 和 INT8 的正确性与完整测试集精度；
4. 在同一 GPU-only 诊断口径下比较 Engine 大小、延迟和吞吐；
5. 根据事先冻结的精度、尺度召回和性能门槛决定是否采用 INT8。

## 冻结输入

| 输入 | SHA256 |
|---|---|
| Exp02 `best.pt` | `79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6` |
| Exp06 ONNX | `305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8` |
| Exp07 FP32 Engine | `01616a8144228db5edbf8948227e3bbaee43b22c495aba3c6c44212e43efe0f1` |
| Exp07 FP16 Engine | `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83` |

## 冻结校准方案

- 来源：Exp02 最终数据划分的 `train`，不使用 val/test；
- 数量：256 张；
- 基础随机种子：42；
- 候选数：1024；
- 选择规则：在确定性随机候选中，选择类别、尺寸、类别×尺寸联合分布及背景比例
  与完整训练集最大绝对比例偏差最小的候选；
- 代表性门槛：最大绝对比例偏差不超过 0.02，并覆盖全部类别、尺寸组和背景图；
- 尺寸定义：tiny `<0.0025`、small `[0.0025,0.01)`、medium `[0.01,0.04)`、
  large `>=0.04`，面积均为标注框相对图像面积；
- 输入预处理必须与 Exp07 保持一致：letterbox 640×640、padding=114、BGR→RGB、
  `/255`、HWC→CHW、FP32 输入。

## 预冻结候选采用门槛

完成工程链路不等于 INT8 候选可以替换 FP16。正式 INT8 结果产生前冻结以下门槛：

| 类别 | 门槛 |
|---|---|
| 构建与执行 | 返回码为 0；输出 `[1,7,8400]`；无 NaN/Inf |
| 完整 test mAP50-95 | 相对同运行时 FP16 绝对下降不超过 0.010 |
| 完整 test mAP50 | 相对同运行时 FP16 绝对下降不超过 0.015 |
| 固定阈值 tiny+small recall | 相对 FP16 绝对下降不超过 0.050 |
| GPU-only mean latency | 相对 FP16 至少降低 5% |
| Engine 大小 | 相对 FP16 至少降低 10% |

固定阈值尺度审计使用 `confidence=0.25`、`NMS IoU=0.70`、匹配 IoU=0.50。
若任一业务强制门槛失败，INT8 候选不替换 FP16；实验本身仍保留全部结果。

## 执行清单

- [x] AutoDL 校准集生成与分布审计；
- [x] 校准归档传输及三端 SHA256；
- [x] Jetson INT8 smoke build；
- [x] Jetson INT8 正式 build 与校准缓存；
- [x] 单图原始张量和 NMS 后诊断；
- [x] 219 张独立 test 精度与尺度召回；
- [x] 同口径 GPU-only benchmark；
- [x] 最终决策与学习复盘。

## Exp08.0 AutoDL 校准集准备结果

执行分支与提交：

```text
branch : exp/08-int8-calibration
commit : 703bc2d
run    : exp08_0_prepare_calibration_20260807_145356
status : PASS
```

结果目录：

```text
/root/autodl-tmp/jetson-ppe-deploy-opt/
results/int8/exp08_0_prepare_calibration_20260807_145356
```

大型归档目录（不进入 Git）：

```text
/root/autodl-tmp/jetson-ppe-artifacts/exp08/
exp08_0_prepare_calibration_20260807_145356/
construction_ppe3_train_calibration_256.tar.gz
```

| 项目 | 结果 |
|---|---:|
| 完整 train 图像 | 980 |
| 校准图像 | 256 |
| 完整 train 实例 | 3,922 |
| 校准集实例 | 996 |
| 背景图 | 2 |
| 基础 seed | 42 |
| 最终候选 seed | 639 |
| 候选数量 | 1,024 |
| 最大绝对分布偏差 | 0.0053134517 |
| 代表性门槛 | 0.0200000000 |
| 归档大小 | 32,384,828 bytes |

校准集类别实例为 person=401、helmet=306、safety_vest=289；尺寸实例为
tiny=28、small=139、medium=270、large=559，覆盖全部强制分组。归档 SHA256：

```text
a3056d1e1852bc55f10455a32e493cfcf4ebcaaad6558d1811a90b20e21bed72
```

校准 manifest SHA256：

```text
75b0c94f49aafc133402a43793dea40a7ca76131959b04043c87e69b44bd6d1d
```

### 预提交 Smoke Test 经验

最初使用 16 张/16 候选进行最小诊断时，脚本语法和归档生成正常，但样本没有覆盖背景图和
tiny 目标，因此代表性检查按预期返回 FAIL。没有放宽检查，而是使用已经预定的正式
256 张/1,024 候选配置重新验证，最终覆盖全部分组并通过 0.02 门槛。

这说明校准 smoke 不能只检查“是否能生成压缩包”，还必须检查样本是否覆盖模型将遇到的
激活分布；过小的 smoke 校准集不进入正式 INT8 构建。

## Exp08.1～08.2 Jetson INT8 构建与执行

校准归档经 AutoDL → Windows → Jetson 传输后，三端 SHA256 均为
`a3056d1e1852bc55f10455a32e493cfcf4ebcaaad6558d1811a90b20e21bed72`；Jetson 解压后的
manifest SHA256 为 `75b0c94f49aafc133402a43793dea40a7ca76131959b04043c87e69b44bd6d1d`，共 256 张。

正式构建结果：

| 项目 | 结果 |
|---|---:|
| run | `exp08_2_int8_formal_20260807_153244` |
| TensorRT | 10.3.0 |
| 校准图像 | 256 |
| 校准耗时 | 200.753 s |
| 总构建耗时 | 890.548 s |
| Engine 大小 | 5,387,244 bytes |
| Engine SHA256 | `5787fb3bae4dbd00909c1762efc9263566044bc4dc35a836c950312e85895f26` |
| calibration cache SHA256 | `f9196435eea6f65f1b530d93fa8bf4f048881bcc9a0485bf48676f32a9103465` |
| 输出 | `[1,7,8400]`，全部有限值 |

首次 16 张 smoke 的 Engine 实际已构建，但前台 SSH 输出通道超时关闭，最终 JSON 写入触发
BrokenPipe，runner 返回 1，失败目录 `exp08_1_int8_smoke_20260807_151314` 被保留。随后先独立
恢复执行验证，再让正式构建在远端完整运行，并缩短终端输出，避免把通信中断误判为构建失败。

单图原始张量诊断中 ORT 有 30 个检测、正式 INT8 有 15 个检测，relative L2 error 为
0.02636。这是强烈的量化误差信号，但不单独作为最终业务判据；最终结论由完整 test AP 和
固定阈值尺度审计决定。

## Exp08.3 完整 test 精度与尺度审计

正式 run：`exp08_3_int8_full_test_20260807_160357`，219 张、840 个实例、batch=1、
`rect=false`。同次运行结果：

| 后端 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| TensorRT FP16 | 0.90684248 | 0.81437407 | 0.88437355 | 0.52192881 |
| TensorRT INT8 | 0.86721510 | 0.81977294 | 0.87804586 | 0.50801483 |
| INT8 - FP16 | -0.03962739 | +0.00539888 | -0.00632768 | -0.01391398 |

mAP50 门槛通过，但 mAP50-95 的允许下降为 0.010，实际下降 0.01391398，失败。

固定 `conf=0.25`、NMS IoU=0.70、匹配 IoU=0.50 的尺度审计：

| 后端 | tiny TP/GT | small TP/GT | tiny+small recall |
|---|---:|---:|---:|
| FP16 | 24/29 | 89/114 | 0.79020979 |
| INT8 | 7/29 | 63/114 | 0.48951049 |
| INT8 - FP16 |  |  | -0.30069930 |

允许下降仅为 0.050，实际下降 0.30069930，关键业务门槛明显失败。第一次完整审计 run
`exp08_3_int8_full_test_20260807_155958` 还暴露了静态 batch=1 Engine 被一次性传入 219 张图的
形状错误；失败现场保留，审计器增加串行静态 Engine 模式后重跑得到上述正式结果。

## Exp08.4 GPU-only 诊断性能

run：`exp08_4_int8_benchmark_20260807_160526`。口径为 batch=1、640×640、500 ms warmup、
200 次、CUDA Graph、spin wait、关闭 H2D/D2H；`jetson_clocks` 未检查，因此不是端到端性能。

| 后端 | mean | P50 | P95 | P99 | throughput | Engine 大小 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 3.640914 ms | 3.486540 ms | 3.822961 ms | 3.826500 ms | 274.563 qps | 8,951,540 bytes |
| INT8 | 2.714804 ms | 2.714935 ms | 2.719243 ms | 2.720891 ms | 368.202 qps | 5,387,244 bytes |

INT8 相对 FP16 加速 1.3411×，平均延迟降低 25.4362%，Engine 缩小 39.8177%，两项性能门槛均通过。

## 最终决策

```text
工程链路：PASS
性能与尺寸：PASS
精度与关键尺度召回：FAIL
INT8 候选：REJECT
后续运行时主线：TensorRT FP16
```

Exp08 证明了“更快、更小”不能覆盖关键尺度精度退化。256 张校准集具备预定的标签分布代表性，
但标签分布代表性不等于激活范围一定足够；若未来要重新研究 INT8，应作为新实验比较更系统的
校准策略、敏感层高精度保留或 QAT，不得回写本轮门槛或把本轮 INT8 Engine 用于主线。

下一步进入 Exp09：基于已验证的 FP16 Engine 实现 TensorRT C++ Runtime。
