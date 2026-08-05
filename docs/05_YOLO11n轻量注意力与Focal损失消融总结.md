# Exp05：YOLO11n 轻量注意力与 Focal 分类损失消融实验

## 1. 实验结论

Exp05 回到原始 YOLO11n 基线，分别验证 P3/8 单点 Residual CBAM-Lite 与仅作用于分类项的 Focal Loss。两个方案的代码、训练、权重保存重载和独立 test 均通过工程检查，但都未满足 tiny+small 召回验收线，因此不替换基线，也不执行 Attention + Focal 组合训练。

```text
Exp05.0 结构、损失与数据审计                    PASS
Exp05.1 P3 Residual CBAM-Lite 工程链路           PASS
Exp05.1 候选模型替换决策                         REJECT
Exp05.2 Focal Classification Loss 工程链路       PASS
Exp05.2 候选模型替换决策                         REJECT
Exp05.3 Attention + Focal                        SKIPPED
Exp05.4 独立 test、尺寸审计与复杂度比较           PASS
最终部署主线                                     保留原始 YOLO11n baseline
```

## 2. 公平对照与验收边界

正式训练均从同一份 COCO 预训练 `yolo11n.pt` 开始，保持数据划分、100 epochs、640 输入、batch 16、seed 42、AdamW、学习率、数据增强、AMP 和验证流程一致。唯一实验变量分别为：

- Exp05.1：在 Neck 的 P3/8 输出（模型第 16 层，64 通道，80×80）后加入一个 Residual CBAM-Lite；
- Exp05.2：检测框和 DFL 损失不变，仅将分类 BCE 替换为 Focal BCE，`gamma=1.5`、`alpha=0.25`。

基线与最低验收线：

```text
test mAP50       = 0.89270104
test mAP50-95    = 0.52047856
tiny+small recall= 113/143 = 0.79020979

候选最低要求：
test mAP50-95    > 0.52047856
tiny+small recall>= 0.79020979
```

推荐进入部署主线还要求 mAP50-95 至少提高 0.003，或 tiny+small recall 至少提高 0.02，同时 mAP50 不明显下降、参数量增幅不超过约 5%。

## 3. Exp05.0 审计与实现验证

环境：RTX 3080 Ti 12 GB、Python 3.12.3、PyTorch 2.8.0+cu128、Ultralytics 8.4.95。

审计确认：

- Detect 使用第 16、19、22 层输出，P3/8 对应第 16 层；
- 当前分类损失为逐元素 `BCEWithLogitsLoss(reduction=none)`；
- Residual CBAM-Lite 零初始化残差缩放时与原网络严格恒等；
- 保存重载最大绝对误差为 0；
- 注意力新增 4,259 个参数，梯度传播正常；
- Focal 对易负样本的相对权重约为 BCE 的 `9.22e-05`，对困难负样本约为 `0.61998`；
- train/val/test 分别为 980/217/219 张图，训练集包含 tiny 128、small 539、medium 1067、large 2188 个实例。

## 4. 独立 test 对比

| 模型 | 参数量 | 参数变化 | mAP50 | 相对基线 | mAP50-95 | 相对基线 | test P | test R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline | 2,590,425 | 0 | 0.89270104 | 0 | 0.52047856 | 0 | 0.92161767 | 0.82040743 |
| P3 Residual CBAM-Lite | 2,594,684 | +4,259（+0.1644%） | 0.89846973 | +0.00576869 | 0.52056023 | +0.00008167 | 0.89793076 | 0.84341607 |
| Focal Classification Loss | 2,590,425 | 0 | 0.87988608 | -0.01281496 | 0.52693756 | +0.00645900 | 0.88855592 | 0.81769571 |

Attention 的 mAP50-95 改善只有 0.00008，低于推荐增益；Focal 的 mAP50-95 提升 0.00646，但 mAP50 下降 0.01281。最终决策还必须结合固定阈值尺寸审计。

## 5. 固定阈值尺寸召回审计

所有模型使用相同口径：test split、`imgsz=640`、`conf=0.25`、class-aware NMS IoU 0.70、匹配 IoU 0.50。

| 模型 | tiny | small | tiny+small | 相对基线 | medium | large |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline | 24/29 = 0.82758621 | 89/114 = 0.78070175 | 113/143 = 0.79020979 | 0 | 0.87387387 | 0.89263158 |
| P3 Residual CBAM-Lite | 18/29 = 0.62068966 | 90/114 = 0.78947368 | 108/143 = 0.75524476 | -0.03496503 | 0.89639640 | 0.89684211 |
| Focal Classification Loss | 15/29 = 0.51724138 | 86/114 = 0.75438596 | 101/143 = 0.70629371 | -0.08391608 | 0.85135135 | 0.87578947 |

注意力模型提高了整体 test recall，但 tiny 召回显著下降；Focal 对 tiny 和 small 均产生明显退化。两个候选都低于 `0.79020979` 的强制验收线。

## 6. 分支决策

### Exp05.1

工程实现成功，参数增幅仅 0.1644%，但 tiny+small recall 下降 0.03497，且 mAP50-95 几乎没有改善。因此候选模型判定为 `REJECT`。

### Exp05.2

Focal 只影响训练损失，部署计算图和参数量不变；虽然 mAP50-95 提升 0.00646，但 mAP50 与 tiny+small recall 同时下降，tiny+small 降幅达到 0.08392。因此候选模型判定为 `REJECT`。

### Exp05.3

预先规定只有 Exp05.1 或 Exp05.2 至少一个有效时才组合。两项均未通过强制尺寸召回线，所以组合实验标记为 `SKIPPED_BY_RULE`，不消耗额外训练资源。

## 7. 最终结论

Exp05 没有产生可替换基线的模型。后续 ONNX、TensorRT 和 Jetson 部署主线继续使用原始 YOLO11n baseline：

```text
test mAP50        0.89270104
test mAP50-95     0.52047856
tiny+small recall 0.79020979
parameters        2,590,425
```

本实验说明，在当前数据分布下，整体 AP 或总体 recall 的局部改善不能代表 PPE 小目标能力改善；tiny+small 固定阈值审计必须继续作为模型替换的硬门槛。
