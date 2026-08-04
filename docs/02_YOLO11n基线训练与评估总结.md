# Exp02：PPE 数据集审计、YOLO11n 基线训练与评估

## 1. 实验概述

Exp02 的目标是建立后续结构改造、ONNX/TensorRT 转换和 Jetson 部署优化所需的**可复现 YOLO11n 基线**。本实验不仅完成正式训练，还覆盖了训练链路 Smoke Test、独立测试集评估、误检漏检分析和目标尺寸审计。

本实验完成的主链路如下：

```text
数据集与标签检查
→ YOLO11n 预训练权重准备
→ 1 Epoch Smoke Test
→ 100 Epoch 正式训练
→ 独立 test 集评估
→ 固定阈值 TP/FP/FN 审计
→ tiny/small/medium/large 尺寸召回分析
→ 冻结基线模型与指标
```

最终状态：

```text
Exp02.5 训练 Smoke Test          : PASS
Exp02.6 正式基线训练             : PASS
Exp02.7 独立测试集评估           : PASS
Exp02.8 误检漏检与目标尺寸审计   : PASS
Exp02 基线冻结                   : YES
```

---

## 2. 实验环境

### 2.1 训练端

```text
平台             : AutoDL
GPU              : NVIDIA GeForce RTX 3080 Ti
GPU 显存         : 11913 MiB
Python           : 3.12.3
PyTorch          : 2.8.0+cu128
Torch CUDA       : 12.8
Ultralytics      : 8.4.95
```

### 2.2 Python 虚拟环境

```text
/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl
```

验证结果：

```text
python_executable = /root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python
cuda_available    = True
gpu_name          = NVIDIA GeForce RTX 3080 Ti
```

### 2.3 项目和输出目录

```text
项目目录：
/root/autodl-tmp/jetson-ppe-deploy-opt

训练输出根目录：
/root/autodl-tmp/jetson-ppe-outputs

预训练模型目录：
/root/autodl-tmp/models/ultralytics
```

---

## 3. 数据集配置

### 3.1 数据集 YAML

```text
/root/autodl-tmp/datasets/derived/
construction_ppe3_final_split_v1_20260804_175104/
construction_ppe3.yaml
```

### 3.2 类别定义

```text
0: person
1: helmet
2: safety_vest
```

### 3.3 数据划分

| 子集 | 图片数 | 背景图片数 | 实例数 |
|---|---:|---:|---:|
| train | 980 | 6 | 未在当前摘要单独统计 |
| val | 217 | 1 | 834 |
| test | 219 | 1 | 840 |

扫描结果：

```text
train: 980 images, 6 backgrounds, 0 corrupt
val  : 217 images, 1 backgrounds, 0 corrupt
test : 219 images, 1 backgrounds, 0 corrupt
```

### 3.4 重复标签问题

Smoke Test 扫描训练集时发现：

```text
train__image187.jpg: 1 duplicate labels removed
```

该问题不是图片损坏，而是某个标签文件中存在一行完全重复的标注。Ultralytics 在加载时自动去除了重复行，随后对源标签进行了去重并删除旧 cache。正式训练前检查结果为：

```text
duplicate_label_action=ALREADY_CLEAN
```

因此正式基线训练使用的是已清理数据。

---

## 4. Exp02.5：YOLO11n 一轮训练 Smoke Test

## 4.1 实验目的

Smoke Test 用于验证以下完整链路能否真实执行：

```text
数据 YAML
→ YOLO11n COCO 预训练权重
→ nc=80 覆盖为 nc=3
→ CUDA + AMP
→ 训练与验证
→ best.pt / last.pt
→ results.csv
```

本阶段只验证可运行性，不用于评价最终模型效果。

## 4.2 初期卡死现象

早期运行时出现多个进程长时间处于 sleeping 状态，GPU 和 CPU 都没有有效训练负载。进程包括：

```text
bash tools/exp02_5_yolo11n_train_smoke.sh
python - ...
```

日志中虽然能看到：

```text
result=PASS
```

但检查后发现这些 PASS 字符串只是被错误写入日志的 Python 源码，并不是实际运行结果。

## 4.3 根因

原 Shell 脚本错误组合了 heredoc、`python -`、管道和 `tee`：

```text
Python 源码被送入 tee
python - 没有获得预期标准输入
python 进程一直等待输入
```

因此造成“看似有日志、实际没有训练”的假象。

## 4.4 修复方式

将内联 Python 代码拆分为独立文件：

```text
tools/exp02_5_yolo11n_train_smoke.py
```

运行方式改为：

```text
python -u script.py ... 2>&1 | tee run.log
```

同时增加：

- 必需文件检查；
- Python 解释器检查；
- CUDA 可用性检查；
- 分阶段日志；
- `best.pt`、`last.pt`、`results.csv` 实体检查；
- `PIPESTATUS[0]` 返回码检查。

## 4.5 第二个阻塞问题：预训练模型缺失

检查发现：

```text
FAIL: /root/autodl-tmp/models/ultralytics/yolo11n.pt
```

下载并加载模型后通过验证：

```text
model_load=PASS
```

## 4.6 Smoke Test 配置

```text
model        : yolo11n.pt
epochs       : 1
imgsz        : 640
batch        : 8
workers      : 0
seed         : 42
device       : 0
AMP          : True
```

## 4.7 Smoke Test 结果

运行目录：

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp02_5_smoke_20260804_184432
```

整体指标：

```text
Precision    = 0.69473979
Recall       = 0.51650153
mAP50        = 0.62696914
mAP50-95     = 0.29115857
```

输出文件：

```text
weights/best.pt   : generated
weights/last.pt   : generated
results.csv       : generated
summary.txt       : generated
run.log           : generated
```

最终状态：

```text
exp02_5_training_smoke=PASS
python_return_code=0
```

## 4.8 首次运行耗时分析

总耗时：

```text
287.358 seconds
```

其中主要时间用于首次联网下载：

```text
Arial.ttf
Ultralytics AMP 检查所需的 yolo26n.pt
```

`yolo26n.pt` 只用于 AMP 环境检查，正式训练模型仍然是本地 `yolo11n.pt`。

---

## 5. Exp02.6：YOLO11n 100 Epoch 正式基线训练

## 5.1 训练配置

```text
model          : yolo11n.pt
epochs         : 100
imgsz          : 640
batch          : 16
workers        : 8
seed           : 42
deterministic  : True
optimizer      : AdamW
lr0            : 0.0015
lrf            : 0.01
momentum       : 0.9
weight_decay   : 0.0005
warmup_epochs  : 3.0
AMP            : True
cache          : False
patience       : 100
```

## 5.2 模型结构

类别覆盖前模型：

```text
YOLO11n
182 layers
2,590,425 parameters
6.4 GFLOPs
```

训练时执行：

```text
Overriding model.yaml nc=80 with nc=3
```

预训练权重成功迁移到三分类模型。

## 5.3 训练目录

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp02_6_yolo11n_baseline_e100_20260804_185444
```

## 5.4 训练耗时

```text
elapsed_seconds=717
约 11 分 57 秒
```

## 5.5 正式训练结果

最终重新加载 `best.pt` 的验证集结果：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.893 | 0.797 | 0.851 | 0.479 |
| person | 0.898 | 0.792 | 0.852 | 0.486 |
| helmet | 0.893 | 0.808 | 0.862 | 0.432 |
| safety_vest | 0.888 | 0.790 | 0.841 | 0.518 |

训练 CSV 中最大 validation mAP50：

```text
epoch       = 60
precision   = 0.85414
recall      = 0.80293
mAP50       = 0.85247
mAP50-95    = 0.47731
```

最大 validation mAP50-95：

```text
epoch       = 92
precision   = 0.89216
recall      = 0.79670
mAP50       = 0.85140
mAP50-95    = 0.47949
```

第 100 Epoch：

```text
precision   = 0.86362
recall      = 0.81863
mAP50       = 0.84633
mAP50-95    = 0.47626
```

最后一轮 Recall 略高，但综合 mAP 略低于最佳轮次，因此后续统一使用 `best.pt`，不使用 `last.pt`。

## 5.6 冻结基线权重

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp02_6_yolo11n_baseline_e100_20260804_185444/
weights/best.pt
```

SHA256：

```text
79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
```

`last.pt` SHA256：

```text
6efc109aed0e1aa10ceb1d90580fe3d872cc56d500ac8ddabc5953096b2ab651
```

正式训练状态：

```text
exp02_6_yolo11n_baseline=PASS
train_return_code=0
No abnormal messages detected
```

---

## 6. Exp02.7：独立测试集评估

## 6.1 评估原则

训练阶段使用 `train` 更新参数，使用 `val` 选择模型。Exp02.7 使用未参与训练和最佳轮次选择的 `test` 集，以获得更独立的基线指标。

评估设置：

```text
split     : test
images    : 219
instances : 840
imgsz     : 640
batch     : 16
workers   : 8
```

## 6.2 测试集整体结果

| 指标 | 数值 |
|---|---:|
| Precision | 0.92161767 |
| Recall | 0.82040743 |
| mAP50 | 0.89270104 |
| mAP50-95 | 0.52047856 |

## 6.3 分类别结果

| 类别 | Images | Instances | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 215 | 337 | 0.90684 | 0.79525 | 0.84739 | 0.88411 | 0.51720 |
| helmet | 165 | 259 | 0.95620 | 0.83784 | 0.89311 | 0.90335 | 0.48805 |
| safety_vest | 169 | 244 | 0.90181 | 0.82813 | 0.86340 | 0.89064 | 0.55619 |

## 6.4 速度统计

RTX 3080 Ti 批量验证统计：

```text
preprocess   = 1.39736733 ms/image
inference    = 3.08034440 ms/image
postprocess  = 1.54993258 ms/image
```

该速度只代表 RTX 3080 Ti 上 Ultralytics 验证流程的统计，不代表 Jetson 单帧端到端延迟。

## 6.5 结果分析

- `helmet` 的 mAP50 最高，为 0.90335；
- `helmet` 的 mAP50-95 为 0.48805，低于 person 和 safety_vest，说明严格 IoU 条件下定位质量仍有提升空间；
- test 指标高于 val，不自动等于数据泄漏，也可能是 test 集相对更容易；
- 后续所有模型必须保持相同数据划分，不能重新随机切分后比较。

最终状态：

```text
exp02_7_baseline_test=PASS
python_return_code=0
```

---

## 7. Exp02.8：误检、漏检与目标尺寸审计

## 7.1 实验目的

mAP 无法直接回答：

- 哪个类别误检最多；
- 哪个类别漏检最多；
- 小尺寸目标是否更难；
- 是否有必要增加 P2 高分辨率检测层。

因此在 test 集上使用固定阈值进行 class-aware GT 匹配审计。

## 7.2 审计配置

```text
confidence threshold : 0.25
NMS IoU              : 0.70
GT matching IoU      : 0.50
class-aware matching : enabled
imgsz                 : 640
```

尺寸定义为项目内部归一化面积规则：

```text
tiny   : area_ratio < 0.0025
small  : 0.0025 <= area_ratio < 0.01
medium : 0.01 <= area_ratio < 0.04
large  : area_ratio >= 0.04
```

该规则用于当前 PPE 数据集审计，不等同于 COCO 官方 AP_small 定义。

## 7.3 首次运行故障

首次运行报错：

```text
failed to read image:
/root/autodl-tmp/jetson-ppe-deploy-opt/image0.jpg
```

原因不是测试图片损坏。批量 `model.predict()` 返回的 `result.path` 在当前调用方式下可能是内部占位名称：

```text
image0.jpg
image1.jpg
...
```

原脚本错误执行：

```python
image_path = Path(result.path).resolve()
```

修复为：

```python
for image_path, result in zip(images, results):
```

由数据集扫描得到的真实图片路径和推理结果按输入顺序对应。

## 7.4 整体审计结果

```text
GT          = 840
TP          = 731
FP          = 169
FN          = 109
Precision   = 0.81222222
Recall      = 0.87023810
```

固定阈值 Precision/Recall 用于错误审计，不替代 Ultralytics 的 mAP 指标。

## 7.5 分类别结果

| 类别 | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| person | 337 | 288 | 86 | 49 | 0.77005348 | 0.85459941 |
| helmet | 259 | 232 | 26 | 27 | 0.89922481 | 0.89575290 |
| safety_vest | 244 | 211 | 57 | 33 | 0.78731343 | 0.86475410 |

关键现象：

```text
person FP=86，为三个类别中误检最多
helmet 的固定阈值 Precision/Recall 最好
```

## 7.6 不同尺寸目标结果

| 尺寸 | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| tiny | 29 | 24 | 5 | 0.82758621 |
| small | 114 | 89 | 25 | 0.78070175 |
| medium | 222 | 194 | 28 | 0.87387387 |
| large | 475 | 424 | 51 | 0.89263158 |

合并 tiny 与 small：

```text
GT     = 143
TP     = 113
Recall = 113 / 143
       = 0.79020979
```

合并 medium 与 large：

```text
GT     = 697
TP     = 618
Recall = 618 / 697
       = 0.88665710
```

差距：

```text
0.88665710 - 0.79020979
= 0.09644731
约 9.65 个百分点
```

说明当前基线对 tiny/small 目标的召回明显低于中大型目标，为 Exp03 P2 消融实验提供了数据依据。

## 7.7 典型困难样本

错误得分最高的样本包括：

```text
train__image1181.jpg : GT=15, TP=5, FP=5, FN=10
train__image784.jpg  : GT=23, TP=21, FP=9, FN=2
test__image450.jpg   : GT=8,  TP=7, FP=13, FN=1
test__image779.jpg   : GT=5,  TP=0, FP=0, FN=5
```

这些样本反映了两类主要问题：

1. 密集场景中同时出现漏检和重复/错误预测；
2. 个别图像上模型完全未输出有效检测。

需要注意，文件名中的 `train__`、`val__` 是原始文件名前缀，文件实际位于派生数据集的 `images/test/` 下；仅凭文件名不能断定数据泄漏。

## 7.8 输出目录

```text
results/evaluation/
exp02_8_baseline_error_size_audit_fixed_20260804_192312
```

输出文件：

```text
summary.txt
summary.json
per_class.csv
per_size.csv
per_image.csv
worst_samples.csv
worst_visuals/
run.log
```

最终状态：

```text
exp02_8_baseline_error_size_audit=PASS
python_return_code=0
No abnormal messages detected
```

---

## 8. Exp02 最终基线

最终冻结指标：

```text
Precision   = 0.92161767
Recall      = 0.82040743
mAP50       = 0.89270104
mAP50-95    = 0.52047856
```

冻结权重：

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp02_6_yolo11n_baseline_e100_20260804_185444/
weights/best.pt
```

SHA256：

```text
79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
```

后续任何模型结构实验都必须至少与该基线比较：

- validation mAP；
- independent test mAP；
- tiny/small 固定阈值召回；
- 参数量和 GFLOPs；
- 推理速度与显存占用；
- 后续 Jetson/TensorRT 部署代价。

---

## 9. 已证实、未证实与适用范围

### 9.1 已证实

- YOLO11n 三分类训练链路可以稳定运行；
- 正式训练、验证、测试和错误审计均已完成；
- 当前数据集不存在扫描层面的损坏图片；
- small 目标是四个尺寸组中召回率最低的一组；
- tiny+small 合并召回率明显低于 medium+large；
- person 是固定阈值审计中误检最多的类别。

### 9.2 尚未证实

- 当前基线在 Jetson 上的真实 TensorRT 延迟和吞吐；
- FP16、INT8 对精度和性能的影响；
- 摄像头实时流中的端到端表现；
- 长时间稳定运行时的温度、功耗和 P95 延迟。

### 9.3 结论适用范围

当前指标适用于：

```text
固定数据划分
imgsz=640
Ultralytics 8.4.95
PyTorch 2.8.0+cu128
RTX 3080 Ti 训练/验证环境
```

不能直接将 RTX 验证速度表述为 Jetson 部署性能。

---

## 10. 实验结论

Exp02 成功建立了一个可追溯、可复现的 YOLO11n PPE 三分类基线。模型在独立测试集上达到：

```text
mAP50     = 0.89270
mAP50-95  = 0.52048
```

同时错误审计显示 tiny/small 目标召回率比中大型目标低约 9.65 个百分点，因此有必要通过结构消融验证小目标优化方案。但任何方案都必须以同一数据划分、相同训练设置和部署成本为约束，不能只看单一 mAP 指标。

最终决策：

```text
Exp02 baseline quality          : ACCEPTED
Exp02 baseline reproducibility  : ACCEPTED
Exp02 baseline frozen           : YES
Next experiment                 : Exp03 P2 small-object ablation
```
