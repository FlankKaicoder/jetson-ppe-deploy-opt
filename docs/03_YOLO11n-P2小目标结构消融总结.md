# Exp03：自定义 YOLO11n-P2 小目标检测结构消融实验

## 1. 实验概述

Exp02 的固定阈值尺寸审计表明，YOLO11n 对 tiny/small 目标的召回率低于中大型目标：

```text
tiny recall         = 0.82759
small recall        = 0.78070
medium recall       = 0.87387
large recall        = 0.89263
tiny+small recall   = 0.79021
medium+large recall = 0.88666
```

因此提出假设：

> 在 YOLO11n 中增加 stride=4 的 P2 高分辨率检测层，可能改善 PPE 场景中小目标的检测能力。

Exp03 完成了从结构调查、自定义 YAML、前向探针、Smoke Test、100 Epoch 正式训练、独立测试到尺寸审计的完整消融流程。

实验链路：

```text
检查本地 P2 配置
→ 发现不存在官方 YOLO11-P2 YAML
→ 基于本地 YOLO11 配置构建自定义 P2 网络
→ 前向传播与 stride 探针
→ 1 Epoch Smoke Test
→ 100 Epoch 正式训练
→ independent test 评估
→ 固定阈值尺寸审计
→ 与 YOLO11n 基线对比
→ 根据精度—性能权衡停止 P2 主路线
```

最终状态：

```text
Exp03.0 自定义 P2 结构构建与探针   : PASS
Exp03.1 一轮训练 Smoke Test       : PASS
Exp03.2 100 Epoch 正式训练        : PASS
Exp03.3 独立 test 评估            : PASS
Exp03.4 尺寸和错误审计            : PASS
P2 小目标改进假设                 : FAIL
P2 最终主路线决策                 : STOP
```

---

## 2. 公平对照原则

YOLO11n 基线和 YOLO11n-P2 保持相同：

```text
数据集划分
输入尺寸
训练轮数
Batch
Workers
随机种子
优化器
学习率
权重衰减
验证集和测试集
```

正式训练固定设置：

```text
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
```

主要实验变量：

```text
YOLO11n     : P3/8 + P4/16 + P5/32
YOLO11n-P2  : P2/4 + P3/8 + P4/16 + P5/32
```

---

## 3. Exp03.0：P2 配置调查与自定义结构实现

## 3.1 本地配置调查

当前环境：

```text
Ultralytics 8.4.95
```

本地能找到：

```text
yolo26-p2.yaml
yolov8-p2.yaml
yolov8-ghost-p2.yaml
```

但找不到：

```text
yolo11n-p2.yaml
```

首次直接执行：

```python
YOLO("yolo11n-p2.yaml", task="detect")
```

报错：

```text
FileNotFoundError: 'yolo11n-p2.yaml' does not exist
```

该失败不是 Python、CUDA 或 Ultralytics 环境异常，而是当前安装版本确实未提供该配置。

## 3.2 为什么不能直接改用 yolo26-p2.yaml

改用 YOLO26-P2 会同时改变：

- 模型代际；
- 检测头实现；
- 训练机制；
- 参数和损失相关设置。

这样无法与 YOLO11n 构成单变量 P2 消融，因此没有采用。

## 3.3 自定义配置

基于当前环境的原始配置：

```text
ultralytics/cfg/models/11/yolo11.yaml
```

生成：

```text
configs/models/yolo11n_p2.yaml
configs/models/upstream_yolo11_ultralytics_8.4.95.yaml
configs/models/yolo11n_p2_manifest.json
```

自定义结构保留 YOLO11 Backbone，在 Neck/Head 中增加 P2 高分辨率路径，并通过自底向上路径恢复 P3、P4、P5，最终形成四尺度检测。

## 3.4 结构探针

探针结果：

```text
YOLO11n_p2 summary
layers                 = 217
parameters             = 2,740,032
trainable_parameters   = 2,740,016
GFLOPs                  = 11.4
stride                  = [4.0, 8.0, 16.0, 32.0]
detect_class            = Detect
detect_nc               = 80
detect_nl               = 4
detect_from             = [19, 22, 25, 28]
network_scale           = n
```

`detect_nc=80` 是构建空网络时的默认 COCO 类别数，训练读取 PPE YAML 后会覆盖为 3。

前向输出：

```text
output[0] shape = [1, 84, 34000]
```

解释：

```text
84 = 4 个解码后的边框参数 + 80 个类别分数
```

候选位置数量：

```text
P2 160×160 = 25600
P3  80×80  =  6400
P4  40×40  =  1600
P5  20×20  =   400
总计        = 34000
```

基线三尺度候选位置：

```text
80×80 + 40×40 + 20×20 = 8400
```

P2 模型候选位置约为基线的：

```text
34000 / 8400 ≈ 4.05 倍
```

最终状态：

```text
exp03_0_yolo11n_p2_probe=PASS
python_return_code=0
```

---

## 4. Exp03.1：一轮训练 Smoke Test

## 4.1 实验目的

验证：

```text
自定义 P2 YAML
→ 加载 YOLO11n COCO 权重
→ nc=80 覆盖为 nc=3
→ 四尺度前向与反向传播
→ CUDA + AMP
→ validation
→ best.pt / last.pt / results.csv
```

## 4.2 权重迁移

```text
Transferred 525/593 items from pretrained weights
```

说明：

- Backbone 和部分原 Neck 权重成功继承；
- 新增 P2 路径及部分重构层没有完全对应的预训练权重；
- 新增层需要从随机初始化开始学习。

## 4.3 Smoke Test 配置

```text
epochs    : 1
imgsz     : 640
batch     : 8
workers   : 0
seed      : 42
optimizer : AdamW
AMP       : True
```

## 4.4 Smoke Test 结构与资源

训练时三分类模型：

```text
YOLO11n_p2
217 layers
2,667,084 parameters
10.4 GFLOPs
```

显存：

```text
GPU_mem = 2.67G
```

相比 Exp02.5 基线 Smoke Test：

```text
YOLO11n GPU_mem     ≈ 1.31G
YOLO11n-P2 GPU_mem  ≈ 2.67G
```

## 4.5 一轮结果

```text
Precision    = 0.09480682
Recall       = 0.52421977
mAP50        = 0.28189219
mAP50-95     = 0.09377955
```

一轮结果明显低于基线，但 Smoke Test 只用于验证链路。新增 P2 路径从随机初始化开始，不能用一轮指标判断最终效果。

输出目录：

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp03_1_yolo11n_p2_smoke_20260804_193813
```

最终状态：

```text
trained_detect_nc=3
detect_nl=4
stride=[4.0, 8.0, 16.0, 32.0]
exp03_1_yolo11n_p2_training_smoke=PASS
python_return_code=0
```

---

## 5. Exp03.2：YOLO11n-P2 100 Epoch 正式训练

## 5.1 训练目录

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp03_2_yolo11n_p2_e100_20260804_194556
```

## 5.2 配置与权重哈希

自定义模型配置 SHA256：

```text
89160613ee686bba1d2cecaefda8e48b86b6b315ea5c001612876b5461764fd6
```

YOLO11n 预训练权重 SHA256：

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

## 5.3 正式训练耗时

```text
elapsed_seconds=832.505
约 13 分 53 秒
```

基线训练耗时：

```text
717 秒
```

P2 训练耗时增加：

```text
832.505 - 717 = 115.505 秒
约 +16.1%
```

## 5.4 验证集最佳结果

最大 validation mAP50：

```text
epoch       = 77
precision   = 0.84922
recall      = 0.81026
mAP50       = 0.85348
mAP50-95    = 0.47547
```

最大 validation mAP50-95：

```text
epoch       = 56
precision   = 0.83330
recall      = 0.78538
mAP50       = 0.83762
mAP50-95    = 0.47953
```

`best.pt` 重新验证：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.834 | 0.785 | 0.837 | 0.479 |
| person | 0.846 | 0.777 | 0.826 | 0.466 |
| helmet | 0.869 | 0.790 | 0.862 | 0.458 |
| safety_vest | 0.787 | 0.789 | 0.823 | 0.513 |

## 5.5 固定权重

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp03_2_yolo11n_p2_e100_20260804_194556/
weights/best.pt
```

SHA256：

```text
bf6e386751ab4aa325307cecb0efbf7cc3788a49e8553e46538316a78a238b8c
```

`last.pt` SHA256：

```text
117425c49b952fe371db06cd9a20d1d69e2b21c29bb142cd8f9cf013cff7b8ce
```

最终状态：

```text
exp03_2_yolo11n_p2_formal_training=PASS
python_return_code=0
```

---

## 6. 验证集初步对比

| 项目 | YOLO11n 基线 | YOLO11n-P2 | 变化 |
|---|---:|---:|---:|
| 最大 val mAP50 | 0.85247 | 0.85348 | +0.00101 |
| 最大 val mAP50-95 | 0.47949 | 0.47953 | +0.00004 |

验证集上两者几乎相同，P2 没有显示出明确整体收益。

但是最终判断不能只看 val，需要在独立 test 集和小目标尺寸分组中验证。

---

## 7. Exp03.3：独立测试集评估

## 7.1 整体结果

| 指标 | YOLO11n 基线 | YOLO11n-P2 | P2 变化 |
|---|---:|---:|---:|
| Precision | 0.92161767 | 0.89402613 | -0.02759154 |
| Recall | 0.82040743 | 0.80662734 | -0.01378009 |
| mAP50 | 0.89270104 | 0.88405588 | -0.00864516 |
| mAP50-95 | 0.52047856 | 0.51273409 | -0.00774447 |

结论：

```text
P2 在独立 test 集上的整体 Precision、Recall、mAP50 和 mAP50-95 均下降。
```

## 7.2 分类别对比

| 类别 | 基线 mAP50 | P2 mAP50 | 变化 | 基线 mAP50-95 | P2 mAP50-95 | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.88411 | 0.87484 | -0.00927 | 0.51720 | 0.50611 | -0.01109 |
| helmet | 0.90335 | 0.88500 | -0.01835 | 0.48805 | 0.47676 | -0.01129 |
| safety_vest | 0.89064 | 0.89232 | +0.00168 | 0.55619 | 0.55533 | -0.00086 |

原本最希望通过 P2 改善的 `helmet` 并未获益，mAP50 和 mAP50-95 都下降。

## 7.3 RTX 验证速度对比

| 模型 | preprocess | inference | postprocess |
|---|---:|---:|---:|
| YOLO11n | 1.397 ms | 3.080 ms | 1.550 ms |
| YOLO11n-P2 | 1.137 ms | 4.016 ms | 1.396 ms |

推理阶段变化：

```text
4.016 / 3.080 - 1 ≈ 30.4%
```

P2 在 RTX 3080 Ti 批量验证中的推理统计约增加 30.4%。该数据不是 Jetson 端到端速度，但足以表明 P2 计算开销更高。

最终状态：

```text
exp03_3_run=PASS
python_return_code=0
No abnormal messages detected
```

---

## 8. Exp03.4：P2 误检、漏检与目标尺寸审计

## 8.1 首次运行 OOM

使用以下设置：

```text
imgsz=640
batch=16
conf=0.25
NMS IoU=0.70
GT match IoU=0.50
```

`model.predict()` 前向阶段发生：

```text
torch.OutOfMemoryError: CUDA out of memory
GPU total capacity     : 11.63 GiB
process memory in use  : 10.51 GiB
additional allocation : 1.78 GiB
```

原因是 P2 四尺度检测头在 batch=16 下产生更大的高分辨率中间张量。Exp03.3 的 `val()` 能运行，不代表审计脚本的 `predict(stream=True)` 在同一 batch 下也一定能运行，两者内部缓存和处理路径不同。

## 8.2 是否影响公平性

将审计 Batch 从 16 改为 4：

```text
batch: 16 → 4
```

保持不变：

```text
imgsz=640
conf=0.25
NMS IoU=0.70
GT match IoU=0.50
尺寸分组规则
```

Batch 只影响显存占用和吞吐，不改变逐图预测框、阈值、NMS 和 GT 匹配规则，因此仍可用于精度审计对比。

## 8.3 P2 整体审计结果

```text
GT          = 840
TP          = 718
FP          = 187
FN          = 122
Precision   = 0.79337017
Recall      = 0.85476190
```

## 8.4 分类别结果

| 类别 | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| person | 337 | 284 | 87 | 53 | 0.76549865 | 0.84272997 |
| helmet | 259 | 220 | 40 | 39 | 0.84615385 | 0.84942085 |
| safety_vest | 244 | 214 | 60 | 30 | 0.78102190 | 0.87704918 |

## 8.5 尺寸结果

| 尺寸 | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| tiny | 29 | 18 | 11 | 0.62068966 |
| small | 114 | 86 | 28 | 0.75438596 |
| medium | 222 | 192 | 30 | 0.86486486 |
| large | 475 | 422 | 53 | 0.88842105 |

最终状态：

```text
exp03_4_run=PASS
python_return_code=0
No abnormal messages detected
```

---

## 9. 基线与 P2 固定阈值对比

## 9.1 整体 TP/FP/FN

| 指标 | YOLO11n 基线 | YOLO11n-P2 | 变化 |
|---|---:|---:|---:|
| TP | 731 | 718 | -13 |
| FP | 169 | 187 | +18 |
| FN | 109 | 122 | +13 |
| Precision | 0.81222222 | 0.79337017 | -0.01885205 |
| Recall | 0.87023810 | 0.85476190 | -0.01547620 |

P2 同时造成：

```text
正确检测减少
误检增加
漏检增加
固定阈值 Precision 下降
固定阈值 Recall 下降
```

## 9.2 分类别 Recall

| 类别 | 基线 Recall | P2 Recall | 变化 |
|---|---:|---:|---:|
| person | 0.85459941 | 0.84272997 | -0.01186944 |
| helmet | 0.89575290 | 0.84942085 | -0.04633205 |
| safety_vest | 0.86475410 | 0.87704918 | +0.01229508 |

`safety_vest` Recall 略有提高，但：

- FP 从 57 增加到 60；
- Precision 从 0.78731 降到 0.78102；
- 无法抵消 person 和 helmet 的损失。

尤其 `helmet` Recall 下降约 4.63 个百分点，与 P2 的预期目标相反。

## 9.3 分尺寸 Recall

| 尺寸 | 基线 Recall | P2 Recall | 变化 |
|---|---:|---:|---:|
| tiny | 0.82758621 | 0.62068966 | -0.20689655 |
| small | 0.78070175 | 0.75438596 | -0.02631579 |
| medium | 0.87387387 | 0.86486486 | -0.00900901 |
| large | 0.89263158 | 0.88842105 | -0.00421053 |

P2 对所有尺寸组的 Recall 都没有提升，其中 tiny 下降最明显。

## 9.4 tiny + small 合并结果

基线：

```text
GT     = 143
TP     = 113
Recall = 113 / 143
       = 0.79020979
```

P2：

```text
GT     = 143
TP     = 104
Recall = 104 / 143
       = 0.72727273
```

变化：

```text
0.72727273 - 0.79020979
= -0.06293706
```

即 tiny+small 合并召回率下降约：

```text
6.29 个百分点
```

因此 P2 没有解决 Exp02 发现的小目标召回问题。

---

## 10. 模型复杂度与资源代价

| 项目 | YOLO11n 基线 | YOLO11n-P2 | 变化 |
|---|---:|---:|---:|
| 三分类训练模型参数量 | 约 2.590M | 2.667M | 约 +3.0% |
| GFLOPs | 6.4 | 10.4 | +62.5% |
| 训练耗时 | 717 s | 832.5 s | +16.1% |
| RTX inference 统计 | 3.080 ms | 4.016 ms | +30.4% |
| 审计 batch=16 | 可运行 | CUDA OOM | P2 更高显存压力 |

说明 P2 的主要代价不是参数量，而是高分辨率特征图和检测头中间张量带来的计算量、显存和带宽压力。

---

## 11. 为什么 P2 没有带来预期收益

当前实验能证实的是“P2 在该配置和该数据集上无效”，不能仅凭现有结果唯一确定根因。可能因素包括：

1. 新增 P2 路径没有完整预训练权重，需要从随机初始化学习；
2. 数据集中的 tiny 样本只有 29 个，统计量较小；
3. small 目标虽然有 114 个，但现有训练规模和增强未必足以让新增高分辨率分支充分收敛；
4. 高分辨率分支可能引入更多背景响应和重复候选，导致 FP 增加；
5. 当前自定义 Neck/Head 融合方式未必是该数据集的最优 P2 结构；
6. 小目标问题可能更多来自数据质量、遮挡、密集重叠、类别边界或输入分辨率，而不是缺少 stride=4 检测层。

这些属于基于结果的合理分析，不应表述为已经被单独实验完全证明的根因。

---

## 12. 实验结论

自定义 YOLO11n-P2 成功完成结构构建、四尺度前向传播、预训练权重迁移、正式训练、独立测试和固定阈值尺寸审计，因此从工程实现角度是成功的。

但作为模型改进路线，其结果为负：

```text
理论计算量约增加 62.5%
RTX 推理统计约增加 30.4%
batch=16 审计发生 CUDA OOM
独立 test Precision 下降
独立 test Recall 下降
独立 test mAP50 下降
独立 test mAP50-95 下降
tiny Recall 大幅下降
small Recall 下降
tiny+small 合并 Recall 下降约 6.29 个百分点
helmet Recall 和 mAP 均下降
```

因此不能在项目或简历中表述为：

```text
加入 P2 后提升了小目标检测精度
```

正确表述应为：

> 针对 PPE 小目标漏检问题，设计并实现 YOLO11n-P2 四尺度检测结构，在同数据划分和同训练参数下完成消融。结果显示 P2 将理论计算量由 6.4 GFLOPs 增加至 10.4 GFLOPs，但独立测试集 mAP50-95 下降约 0.77 个百分点，tiny+small 固定阈值召回率下降约 6.29 个百分点，因此基于精度—性能权衡终止该路线。

---

## 13. 项目价值

Exp03 是一次有效的负向消融，体现了完整工程决策过程：

```text
从基线错误审计提出假设
→ 实现自定义网络结构
→ 检查张量尺寸和 stride
→ 验证预训练权重迁移
→ 公平训练
→ 独立测试
→ 小目标分组审计
→ 量化计算与显存代价
→ 用数据决定停止路线
```

这比“随意添加模块并只展示最好结果”更有可信度，也能在面试中说明如何评价模型改造是否值得进入部署主线。

---

## 14. 已证实、未证实与决策

### 14.1 已证实

- 当前 Ultralytics 8.4.95 不提供可直接使用的 `yolo11n-p2.yaml`；
- 自定义 YOLO11n-P2 四尺度结构可以正确构建和训练；
- 检测 stride 为 `[4, 8, 16, 32]`；
- P2 在当前数据和训练配置下没有提升整体精度；
- P2 没有提升 tiny/small 召回率；
- P2 明显增加计算量、推理时间和显存压力。

### 14.2 尚未证实

- 其他 P2 融合结构是否一定无效；
- 增加训练轮数或更换预训练方式是否能改善结果；
- 更大输入分辨率是否比 P2 更有效；
- 数据重采样、难例挖掘或专门小目标增强是否更有效；
- P2 在 Jetson TensorRT 上的精确端到端延迟和功耗。

### 14.3 当前决策

```text
P2 architecture implementation : PASS
P2 structure probe             : PASS
P2 training pipeline           : PASS
P2 independent evaluation      : PASS
P2 small-object improvement    : FAIL
P2 deployment cost             : UNACCEPTABLE
P2 final-route decision        : STOP
```

下一步进入部署可重参数化结构路线，目标从“增加高分辨率检测层”转为：

```text
训练阶段使用多分支结构
→ 部署前融合为单分支卷积
→ 验证转换前后数值等价性
→ 尽量不增加 TensorRT 推理图复杂度
```
