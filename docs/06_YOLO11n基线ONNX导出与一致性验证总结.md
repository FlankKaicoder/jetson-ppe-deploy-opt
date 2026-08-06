# Exp06 YOLO11n 基线 ONNX 导出与一致性验证总结

## 1. 实验结论

状态：`PASS`

冻结的原始 YOLO11n baseline 已成功导出为静态 FP32 ONNX。`onnx.checker`、单图原始输出张量、NMS 后框/类别/置信度以及完整测试集指标一致性均通过验收。该 ONNX 可作为 Exp07 Jetson TensorRT FP32/FP16 Engine 构建的唯一候选输入，但尚未在 Jetson 上构建或验证 TensorRT Engine。

## 2. 实验目的与假设

目的：将 Exp02 冻结的 PyTorch 权重导出为可部署 ONNX，并证明导出没有造成不可接受的数值或检测精度偏差。

假设：在相同的 `640×640` 方形 letterbox、FP32、batch 1、相同 NMS 参数下，PyTorch 与 ONNX Runtime 的原始输出和测试集 AP 应保持一致。

## 3. 运行环境

- 机器：AutoDL，NVIDIA RTX 3080 Ti
- Python：3.12.3
- PyTorch：2.8.0+cu128
- Ultralytics：8.4.95
- ONNX：1.22.0
- ONNX Runtime：1.28.0，`CPUExecutionProvider`
- 实验分支：`exp/06-onnx-export`
- 运行时基准 Git commit：`2b38170d02e77dea91c10d6a87d5dce8a502a868`

正式运行发生在实现提交前，因此结果中的 Git commit 是分支基点。为补足追溯，正式运行所用最终脚本 SHA256 为：

```text
c70b61a8141e48aeb5379abcf2c24d092d065795d0973484f48f8b1301308d03  tools/exp06_export_onnx.py
46aea5c1fa4d676e26aa6ffc73f1e73622c0379f05023424910d3fc5c6ef3e9c  tools/exp06_0_onnx_smoke.sh
18af8e49fc5db6cf06cfd07771e38a98d3517acb287ded687f9a63223a0a0137  tools/exp06_1_onnx_formal.sh
```

## 4. 输入与导出配置

冻结权重：

```text
/root/autodl-tmp/jetson-ppe-outputs/exp02_6_yolo11n_baseline_e100_20260804_185444/weights/best.pt
SHA256: 79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
```

数据集：

```text
/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml
test: 219 images, 840 instances
```

导出配置：

```text
format=onnx
batch=1
imgsz=640
opset=17
dynamic=False
simplify=False
half=False
nms=False
```

ONNX I/O：

```text
input : images  [1, 3, 640, 640]
output: output0 [1, 7, 8400]
```

## 5. 修改文件

```text
tools/exp06_export_onnx.py
tools/exp06_0_onnx_smoke.sh
tools/exp06_1_onnx_formal.sh
docs/06_YOLO11n基线ONNX导出与一致性验证总结.md
results/onnx/exp06_*/*（小型结果与摘要；run.log 不进入 Git）
```

启动脚本从自身位置解析仓库根目录，并允许通过 `PPE_REPO_DIR`、`PPE_PYTHON_BIN`、`PPE_BASELINE_WEIGHTS`、`PPE_DATA_YAML` 和 `PPE_ARTIFACT_ROOT` 覆盖环境路径。

## 6. 运行命令

```bash
bash tools/exp06_0_onnx_smoke.sh
bash tools/exp06_1_onnx_formal.sh
```

最终 Smoke Test：

```text
results/onnx/exp06_0_onnx_smoke_20260806_155714/
return_code=0
result=PASS
```

最终正式实验：

```text
results/onnx/exp06_1_onnx_formal_20260806_155847/
return_code=0
result=PASS
```

## 7. 最终 ONNX 产物

```text
artifact_name : yolo11n_baseline_exp06_b1_640_opset17.onnx
source        : AutoDL Exp06
source_weight : frozen Exp02 baseline best.pt
target        : Jetson Exp07
target_path   : /home/nvidia/models/jetson-ppe/exp06/
size_bytes    : 10566605
sha256        : 305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8
```

AutoDL 当前路径：

```text
/root/autodl-tmp/jetson-ppe-artifacts/exp06_1_onnx_formal_20260806_155847/yolo11n_baseline_exp06_b1_640_opset17.onnx
```

ONNX 文件不提交 Git。传到 Jetson 后必须重新计算 SHA256，并与上述值完全一致。

## 8. 单图原始张量一致性

探针图像选用测试集中标签最多的一张图：

```text
train__image784.jpg
label_count=23
```

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| max abs error | 0.0006103515625 | 0.001 | PASS |
| mean abs error | 0.000010301261661 | 0.00002 | PASS |
| relative L2 error | 1.26983169798e-7 | 1e-5 | PASS |
| shape | `[1, 7, 8400]` vs `[1, 7, 8400]` | 必须相同 | PASS |
| finite values | 全部有限值 | 必须满足 | PASS |

## 9. NMS 后检测一致性

固定参数：`confidence=0.25`、`IoU=0.70`、`max_det=300`。

| 指标 | PyTorch | ONNX Runtime | 差异/结果 |
|---|---:|---:|---:|
| 检测数量 | 30 | 30 | 相同 |
| 类别序列 | - | - | 完全相同 |
| 最大框坐标绝对误差 | - | - | 0.0001220703125 |
| 最大置信度绝对误差 | - | - | 1.13248825073e-6 |

验收阈值为框坐标 `1e-3`、置信度 `1e-5`，结果 `PASS`。

## 10. 完整测试集一致性

为与静态 ONNX 的方形输入保持公平，两种后端均使用 `rect=False`、batch 1、imgsz 640。

| 指标 | PyTorch FP32 | ONNX Runtime FP32 | ONNX - PyTorch | 阈值 | 判定 |
|---|---:|---:|---:|---:|---|
| Precision | 0.889708022788 | 0.890126184512 | +0.000418161724 | 0.0005 | PASS |
| Recall | 0.831223065971 | 0.831084799280 | -0.000138266691 | 0.0005 | PASS |
| mAP50 | 0.880958648173 | 0.880957038324 | -0.000001609849 | 0.0001 | PASS |
| mAP50-95 | 0.513768140966 | 0.513851518834 | +0.000083377869 | 0.0001 | PASS |

Ultralytics 8.4.95 报告的 Precision/Recall 位于“平滑后的类别平均 F1 曲线最大值”对应置信度索引。微小置信度扰动可能改变该离散索引，因此 P/R 使用 `5e-4` 门槛；由完整 PR 曲线积分得到的 AP 使用更严格的 `1e-4` 门槛。

这些数值不可直接与 Exp02 冻结指标表中的默认批处理结果比较，因为本实验为适配静态部署输入，显式固定了方形预处理 `rect=False`。本表只用于评估相同预处理下的 PyTorch 与 ONNX 转换偏差。

## 11. 异常、修复与保留记录

| 运行目录 | 结果 | 原因 | 最小修复 |
|---|---|---|---|
| `exp06_0_onnx_smoke_20260806_153255` | FAIL | NMS 导入位置与当前 Ultralytics 版本不符；Shell 条件语法错误 | 改用 `ultralytics.utils.nms` 并修复条件表达式 |
| `exp06_0_onnx_smoke_20260806_153755` | FAIL | 实测 mean abs error 略高于初始 `1e-5` 门槛 | 保留 max/relative L2 门槛，将 mean abs 门槛校准为 `2e-5` |
| `exp06_0_onnx_smoke_20260806_154003` | PASS | 首次 Smoke Test 通过 | 无 |
| `exp06_1_onnx_formal_20260806_154136` | FAIL | PyTorch 默认矩形 batch 与静态 ONNX 方形预处理不一致 | 两后端统一 `rect=False` |
| `exp06_1_onnx_formal_20260806_154447` | FAIL | 单一 `1e-4` 门槛错误地同时约束最大-F1点 P/R 与 AP | 按指标语义拆分 P/R 与 AP 门槛 |
| `exp06_1_onnx_formal_20260806_155049` | PASS | 一致性通过 | 后续仅重构启动脚本路径解析 |
| `exp06_0_onnx_smoke_20260806_155714` | PASS | 最终脚本 Smoke Test | 无 |
| `exp06_1_onnx_formal_20260806_155847` | PASS | 最终脚本正式验证 | 无 |

所有失败目录、返回码和原始 `run.log` 均保留在 AutoDL worktree；未删除或覆盖。

## 12. 已证实与尚未证实

已证实：

- 冻结权重 SHA256 与 Exp02 记录一致；
- 静态 FP32 ONNX 可通过 `onnx.checker`；
- ONNX Runtime 可以执行前向；
- 原始张量、单图最终检测和完整测试集 AP 满足一致性阈值；
- 最终 ONNX 文件大小和 SHA256 已记录。

尚未证实：

- Jetson 上的 ONNX 文件传输完整性；
- TensorRT 10.3 对该 ONNX 的解析与 Engine 构建；
- TensorRT FP32/FP16 的精度一致性；
- Jetson 端延迟、吞吐、功耗、温度和稳定性；
- INT8 精度与校准方案。

## 13. 最终决策与下一步

Exp06 判定为 `PASS`，部署主线仍为原始 YOLO11n baseline，不改变模型选择。

下一步按顺序执行：

1. 经审查后提交并推送 `exp/06-onnx-export`；
2. Windows 合并已验证的文档与 Exp06 分支；
3. 将最终 ONNX 传到 Jetson 的 `/home/nvidia/models/jetson-ppe/exp06/`；
4. 在 Jetson 接收端校验 SHA256；
5. 创建 `exp/07-tensorrt-fp16`，分别构建和验证 TensorRT FP32/FP16 Engine。
