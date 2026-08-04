# Exp04：YOLO11n 部署可重参数化结构消融实验

## 1. 实验概述

Exp03 的 YOLO11n-P2 消融结果表明，直接增加高分辨率检测层不仅没有改善 tiny/small 目标，反而增加了计算量、显存压力和推理延迟。因此 Exp04 不再继续扩大检测图，而是转向一条更符合端侧部署目标的路线：

> 训练阶段使用多分支卷积增强特征学习，部署前将多分支解析融合为单个 3×3 卷积，使 TensorRT/Jetson 推理图中不保留额外分支。

本实验围绕 YOLO11n Neck 中的两个下采样卷积设计并实现部署可重参数化模块，完整验证了：

```text
独立模块数学融合
→ YOLO11n 结构集成
→ COCO 预训练权重迁移
→ 1 Epoch Smoke Test
→ 100 Epoch 正式训练
→ 训练态多分支转部署态单分支
→ 保存与重新加载
→ validation 精度一致性
→ independent test 公平对比
→ 固定阈值 TP/FP/FN 审计
→ tiny/small 尺寸审计
→ 阈值校准与低置信度恢复诊断
→ 最终路线决策
```

最终状态：

```text
Exp04.0  独立 RepConvBlock 融合探针              : PASS
Exp04.1  YOLO11n 结构集成与整网数值验证           : PASS
Exp04.2  一轮训练 Smoke Test                     : PASS
Exp04.3  100 Epoch 正式训练                      : PASS
Exp04.4  训练态到部署态转换与验证集一致性          : PASS
Exp04.5a 独立 test 三模型公平评估                 : PASS
Exp04.5b 固定阈值错误与尺寸审计                   : PASS
Exp04.5c 验证集阈值校准与 test 复评               : PASS
Exp04.5d 低置信度 tiny 恢复能力诊断               : PASS
工程实现与部署转换目标                            : PASS
整体 AP 保持                                      : PASS
PPE tiny 目标保持                                 : FAIL
最终模型替换决策                                  : REJECT
Exp04 路线角色                                    : 部署感知结构消融
```

---

## 2. 实验背景与研究问题

### 2.1 Exp02 基线暴露的问题

Exp02 的固定阈值审计结果为：

```text
tiny recall         = 0.82758621
small recall        = 0.78070175
medium recall       = 0.87387387
large recall        = 0.89263158
tiny+small recall   = 0.79020979
medium+large recall = 0.88665710
```

说明 YOLO11n 对 tiny/small 目标的召回明显低于中大型目标。

### 2.2 Exp03 的结论

Exp03 尝试通过 P2/4 高分辨率检测层改善小目标，但结果显示：

```text
GFLOPs               6.4 → 10.4
独立 test mAP50-95   下降
tiny+small Recall    下降
RTX 推理时间          增加
显存压力              增加
```

因此停止 P2 主路线。

### 2.3 Exp04 的核心问题

Exp04 不再直接增加部署图复杂度，而是研究：

1. 能否在训练时增加额外卷积分支；
2. 能否在部署前严格融合为单个 3×3 卷积；
3. 融合前后数值和验证指标是否一致；
4. 整体 AP 是否能够保持；
5. tiny/small 检测能力是否至少不低于 YOLO11n 基线；
6. 该结构是否值得进入后续 ONNX/TensorRT/Jetson 主路线。

---

## 3. 公平对照原则

Exp04 正式训练与 Exp02 YOLO11n 基线保持一致：

```text
数据集划分
输入尺寸
训练轮数
Batch
Workers
随机种子
确定性设置
优化器
学习率
权重衰减
Warmup
AMP
validation 和 independent test
```

正式训练设置：

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
cache          : False
patience       : 100
```

正式训练仍从原始 COCO 预训练权重开始：

```text
/root/autodl-tmp/models/ultralytics/yolo11n.pt
```

SHA256：

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

没有从已经完成 PPE 训练的 Exp02 `best.pt` 继续微调，避免给 Exp04 额外训练优势。

主要实验变量只有：

```text
YOLO11n 基线：普通 Conv-BN-SiLU 下采样层
YOLO11n-Rep：训练时 3×3 Conv-BN + 1×1 Conv-BN，部署时融合为单 3×3 Conv
```

---

## 4. 实验环境与目录

### 4.1 训练端

```text
平台        : AutoDL
GPU         : NVIDIA GeForce RTX 3080 Ti
显存        : 11913 MiB
Python      : 3.12.3
PyTorch     : 2.8.0+cu128
Ultralytics : 8.4.95
```

### 4.2 Python 虚拟环境

```text
/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl
```

### 4.3 项目目录

```text
/root/autodl-tmp/jetson-ppe-deploy-opt
```

### 4.4 数据集 YAML

```text
/root/autodl-tmp/datasets/derived/
construction_ppe3_final_split_v1_20260804_175104/
construction_ppe3.yaml
```

类别：

```text
0: person
1: helmet
2: safety_vest
```

数据划分：

| 子集 | 图片数 | 实例数 |
|---|---:|---:|
| train | 980 | 当前 Exp04 摘要未单独重新统计 |
| val | 217 | 834 |
| test | 219 | 840 |

---

## 5. 重参数化模块设计

## 5.1 训练态结构

在训练阶段，模块包含：

```text
3×3 Conv-BN
      +
1×1 Conv-BN
      +
可选 Identity-BN
      ↓
     SiLU
```

其中：

- 当 `in_channels == out_channels` 且 `stride == 1` 时，存在 Identity-BN 分支；
- 当发生通道投影或 `stride == 2` 时，不使用 Identity 分支；
- Exp04 正式集成的第 17、20 层均为 `stride=2`，因此实际训练结构为双卷积分支。

## 5.2 部署态结构

部署前执行解析融合后，结构变为：

```text
单个 3×3 Conv（带 bias）
      ↓
     SiLU
```

部署图中不再保留：

```text
1×1 Conv
BatchNorm
分支求和
Identity 分支
```

## 5.3 Conv-BN 融合公式

对于卷积权重 `W`、卷积偏置 `b`、BN 参数 `γ`、`β`、运行均值 `μ`、运行方差 `σ²` 和 `eps`：

```text
scale = γ / sqrt(σ² + eps)

W_fused = W × scale

b_fused = β + (b - μ) × scale
```

原 Conv 通常没有 bias，此时视为：

```text
b = 0
```

## 5.4 1×1 到 3×3 的转换

1×1 卷积核补零到 3×3 中心：

```text
0 0 0
0 W 0
0 0 0
```

## 5.5 Identity 到 3×3 的转换

Identity 分支转换为通道对角位置为 1 的 3×3 delta kernel：

```text
每个通道中心位置 = 1
其他位置          = 0
```

随后再与 Identity-BN 融合。

## 5.6 多分支最终融合

```text
W_deploy = W_3x3 + pad(W_1x1) + W_identity
b_deploy = b_3x3 + b_1x1 + b_identity
```

因此训练态多分支与部署态单卷积在实数数学意义上等价；实际浮点运行会存在极小舍入误差。

---

## 6. Exp04.0：独立 RepConvBlock 融合探针

## 6.1 实验目的

在接入 YOLO11n 前，先独立验证：

```text
Conv-BN 融合公式
1×1 Kernel 补零
Identity-BN 融合
多分支权重和偏置求和
switch_to_deploy() 幂等性
转换后 Conv/BN 数量
```

## 6.2 测试场景

### 场景一：同通道、stride=1、包含 Identity

```text
case=same_channels_identity
output_shape=[2, 16, 32, 32]
conv_before=2
conv_after=1
bn_before=3
bn_after=0
params_before=2656
params_after=2320
max_abs_error=3.33786010742e-06
mean_abs_error=1.35837581183e-07
relative_l2_error=2.86915621928e-07
result=PASS
```

### 场景二：通道投影、stride=1

```text
case=channel_projection
output_shape=[2, 24, 32, 32]
conv_before=2
conv_after=1
bn_before=2
bn_after=0
params_before=3936
params_after=3480
max_abs_error=2.62260437012e-06
mean_abs_error=1.02207458497e-07
relative_l2_error=3.10388458047e-07
result=PASS
```

### 场景三：通道投影、stride=2

```text
case=stride2_projection
output_shape=[2, 24, 16, 16]
conv_before=2
conv_after=1
bn_before=2
bn_after=0
params_before=3936
params_after=3480
max_abs_error=2.14576721191e-06
mean_abs_error=1.02053725470e-07
relative_l2_error=3.02908318872e-07
result=PASS
```

最终：

```text
overall=PASS
No abnormal messages detected.
```

## 6.3 Exp04.0 结论

独立模块级别已经证实：

- 三种主要结构都能正确融合；
- Conv 数量由 2 变为 1；
- BatchNorm 数量变为 0；
- 最大绝对误差处于 `10^-6` 量级；
- 重复执行部署转换不会破坏结构。

但此时尚未证明：

- YOLO11n 图结构可以正常替换；
- Ultralytics 训练器可以训练自定义模块；
- EMA 和检查点可以保存自定义类；
- 整网输出和检测指标一致。

---

## 7. Exp04.1：YOLO11n 结构集成与数值验证

## 7.1 插入位置调查

自动搜索 YOLO11n Head/Neck 中满足以下条件的顶层 Conv：

```text
3×3 kernel
stride=2
padding=1
dilation=1
groups=1
输入通道 = 输出通道
位于 Neck/Head 区域
```

得到：

```text
target_indices=[17, 20]
```

张量尺寸：

```text
layer 17:
[1, 64, 80, 80]
→ [1, 64, 40, 40]

layer 20:
[1, 128, 40, 40]
→ [1, 128, 20, 20]
```

这两个层正是 Neck 中从高分辨率特征向更低分辨率路径进行下采样的卷积。

## 7.2 初始化策略

为保证从 COCO/基线模型替换后初始输出完全不变：

```text
原始 3×3 Conv-BN
→ 复制到 Rep 的 3×3 分支

新增 1×1 Conv-BN
→ BN gamma = 0
→ BN beta  = 0
→ 初始贡献为 0
```

因此初始状态满足：

```text
普通 Conv 输出 = Rep 训练态输出
```

## 7.3 初次整网探针

结构结果：

```text
baseline_parameter_count     = 2,590,425
rep_training_parameter_count = 2,611,289
rep_deploy_parameter_count   = 2,590,233
rep_training_block_count     = 2
converted_block_count        = 2
all_blocks_deployed          = True
```

基线到训练态 Rep：

```text
max_abs_error     = 0
mean_abs_error    = 0
relative_l2_error = 0
```

说明：

- 插入位置正确；
- 原始权重迁移正确；
- 新分支零贡献初始化正确；
- Detect Head 输入尺寸没有改变；
- 整网初始输出逐元素完全一致。

但默认 CUDA/TF32 下训练态到部署态出现：

```text
max_abs_error     = 0.105926513672
mean_abs_error    = 0.000120162843447
relative_l2_error = 2.01863325109e-05
```

旧阈值为 `1e-4`，因此初次结果为 FAIL。

## 7.4 Exp04.1a：逐 Tensor 误差定位

误差分布：

```text
P3 feats[0]             max error = 0
P4 feats[1]             max error = 0.00333333
P5 feats[2]             max error = 0.00607681
raw boxes               max error = 0.00597191
scores                  max error = 0.00875282
最终 decoded output     max error = 0.10592651
```

结果与结构路径一致：

- P3 不经过第 17、20 层，因此误差为 0；
- P4 经过第 17 层，出现极小误差；
- P5 继续经过第 20 层，误差继续传播；
- 最终 DFL/边框解码将前面微小误差放大到坐标空间。

最终 `0.1059` 对 640 输入约为：

```text
0.1059 / 640 ≈ 0.0001655
约为图像边长的 0.0166%
```

## 7.5 Exp04.1b：关闭 TF32 的严格数值复测

运行设置：

```text
cudnn_benchmark      = False
cudnn_deterministic  = True
cudnn_allow_tf32     = False
matmul_allow_tf32    = False
float32 precision    = highest
```

### CUDA FP32、TF32 关闭

```text
max_abs_error     = 0.000274658203125
mean_abs_error    = 2.30718801523e-07
relative_l2_error = 4.63498849939e-08
```

### CPU FP32

```text
max_abs_error     = 0.0001220703125
mean_abs_error    = 1.31281348278e-07
relative_l2_error = 3.05781442564e-08
```

与默认 CUDA/TF32 相比，最大误差从约 `1.06e-1` 降至 `2.75e-4`，下降约 386 倍。

说明最初较大的最终坐标差异主要来自：

```text
多分支卷积与求和的浮点累加顺序
vs
融合后单卷积的浮点累加顺序
+ cuDNN/TF32 数值路径差异
```

而不是融合公式错误。

## 7.6 最终数值验收标准

整网检测输出不能简单沿用独立模块 `1e-5` 阈值，因此最终采用：

```text
baseline → training max error <= 1e-5
training → deploy max error   <= 5e-4
mean absolute error           <= 1e-6
relative L2 error             <= 1e-6
```

该阈值仍比实际相对 L2 误差宽松约两个数量级，并非无依据地放宽。

最终 Exp04.1：

```text
结构集成                    : PASS
权重迁移                    : PASS
训练态初始化等价            : PASS
部署转换                    : PASS
严格 FP32 数值等价          : PASS
```

---

## 8. Exp04.2：一轮训练 Smoke Test

## 8.1 实验目的

验证完整训练链：

```text
COCO yolo11n.pt
→ 构建三分类 YOLO11n
→ 替换第 17、20 层
→ 真实反向传播
→ EMA
→ best.pt / last.pt
→ 自定义模块重新加载
→ 新增分支参数更新
→ 训练后部署转换
```

## 8.2 Smoke Test 配置

```text
epochs    : 1
imgsz     : 640
batch     : 8
workers   : 0
seed      : 42
device    : 0
optimizer : AdamW
AMP       : True
```

## 8.3 结构与检查点结果

```text
base_candidate_indices  = [17, 20]
base_parameter_count    = 2,624,080
trainer_rep_count       = 2
ema_rep_count           = 2
checkpoint_rep_count    = 2
checkpoint_training_form= True
learned_1x1_branches    = True
```

第 17 层：

```text
branch_1x1_gamma_abs_max       = 0.0041046142578125
branch_1x1_gamma_nonzero_count = 64
```

第 20 层：

```text
branch_1x1_gamma_abs_max       = 0.004119873046875
branch_1x1_gamma_nonzero_count = 128
```

全部 BN gamma 从零变为非零，证明新增分支确实参与了反向传播和优化器更新。

## 8.4 一轮训练指标

```text
Precision = 0.58929
Recall    = 0.39847
mAP50     = 0.46367
mAP50-95  = 0.19395
```

这些指标只用于证明训练链正常，不用于模型优劣判断。

## 8.5 训练后转换误差

```text
deploy_parameter_count = 2,590,233
max_abs_error           = 9.1552734375e-05
mean_abs_error          = 1.15911507906e-07
relative_l2_error       = 5.74849414103e-08
```

最终：

```text
overall=PASS
```

---

## 9. Exp04.3：100 Epoch 正式训练

## 9.1 训练目录

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp04_3_yolo11n_rep_e100_20260804_223547
```

## 9.2 训练耗时

```text
elapsed_seconds = 711.882
约 11 分 52 秒
```

Exp02 基线训练约 717 秒，因此两者正式训练耗时基本相当。

## 9.3 正式训练结构

```text
pretrained_parameter_count = 2,624,080
trainer_parameter_count    = 2,611,289
trainer_rep_count          = 2
ema_rep_count              = 2
checkpoint_rep_count       = 2
checkpoint_parameter_count = 2,611,289
checkpoint_training_form   = True
learned_1x1_branches       = True
```

`pretrained_parameter_count=2,624,080` 对应原始 COCO 80 类模型；训练时覆盖为 PPE 3 类后，参数量发生变化。

## 9.4 新分支学习结果

第 17 层：

```text
gamma_abs_max       = 0.052886962890625
gamma_abs_mean      = 0.010207124054431915
gamma_nonzero_count = 64
```

第 20 层：

```text
gamma_abs_max       = 0.08477783203125
gamma_abs_mean      = 0.012402204796671867
gamma_nonzero_count = 128
```

与 Smoke Test 相比，gamma 绝对值明显增大，说明 1×1 分支在正式训练中形成了非零贡献。

## 9.5 第 100 Epoch

```text
Precision   = 0.88445
Recall      = 0.77923
mAP50       = 0.83386
mAP50-95    = 0.47244
train box   = 0.99329
train cls   = 0.49779
train dfl   = 1.29830
val box     = 1.52764
val cls     = 0.75879
val dfl     = 1.82962
```

## 9.6 最大 validation mAP50

```text
epoch       = 64
Precision   = 0.85379
Recall      = 0.81538
mAP50       = 0.84834
mAP50-95    = 0.47721
```

## 9.7 最大 validation mAP50-95

```text
epoch       = 65
Precision   = 0.88962
Recall      = 0.77968
mAP50       = 0.84771
mAP50-95    = 0.48506
```

与 Exp02 基线 validation 最佳值对比：

| 指标 | YOLO11n 基线 | YOLO11n-Rep | 变化 |
|---|---:|---:|---:|
| 最大 mAP50 | 0.85247 | 0.84834 | -0.00413 |
| 最大 mAP50-95 | 0.47949 | 0.48506 | +0.00557 |

validation 上表现为：

```text
mAP50 略低
mAP50-95 略高
整体基本相当
```

## 9.8 正式权重

训练态 `best.pt`：

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp04_3_yolo11n_rep_e100_20260804_223547/
weights/best.pt
```

SHA256：

```text
5dcc487e122670fb0b2b37a2a7d74ac2c7401bb19638065c1b27c16b1bf75938
```

`last.pt` SHA256：

```text
6bcf17bdf4618e4c1d9f9780fb3cf098f1c7c4fdc1c08c8fbee959fd806c0e36
```

## 9.9 训练后部署转换复测

```text
converted_block_count = 2
all_blocks_deployed   = True
deploy_parameter_count= 2,590,233
max_abs_error         = 6.103515625e-05
mean_abs_error        = 8.33107671521e-08
relative_l2_error     = 3.16038436975e-08
```

最终：

```text
overall=PASS
```

---

## 10. Exp04.4：训练态到部署态转换与验证

## 10.1 转换目标

```text
训练态 best.pt
→ 两个 RepConvBlock 分支融合
→ 保存独立部署态检查点
→ 重新加载
→ 严格 FP32 前向对比
→ validation 指标对比
```

原训练态权重不被覆盖。

## 10.2 部署态权重

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp04_4_yolo11n_rep_deploy_20260804_225458/
weights/best_deploy_fp32.pt
```

SHA256：

```text
9694391a75cc6127a6452b9d5e28930f76eb38194f67f0dfdbde7e61b30fc7ae
```

文件大小：

```text
10,763,710 bytes
```

该文件显式保存为 FP32，因此比 Ultralytics strip 后常见的 FP16 训练检查点更大；文件大小增加不等于计算量增加。

## 10.3 结构变化

```text
training_rep_count       = 2
training_form            = True
training_parameter_count = 2,611,289

converted_block_count    = 2
deploy_rep_count         = 2
deploy_form              = True
deploy_parameter_count   = 2,590,233

reloaded_rep_count       = 2
reloaded_deploy_form     = True
reloaded_parameter_count = 2,590,233
```

训练态到部署态减少：

```text
2,611,289 - 2,590,233 = 21,056 个参数
```

部署态比普通 PPE YOLO11n 基线少 192 个参数：

```text
第 17 层 64 通道 BN：gamma+beta 128 参数
融合后 Conv bias：64 参数
减少 64

第 20 层 128 通道 BN：gamma+beta 256 参数
融合后 Conv bias：128 参数
减少 128

总计减少 192
```

## 10.4 前向数值一致性

```text
training_vs_deploy_max_abs_error     = 0.0001220703125
training_vs_deploy_mean_abs_error    = 8.12939618137e-08
training_vs_deploy_relative_l2_error = 2.10414404751e-08
deploy_vs_reloaded_max_abs_error     = 0
```

说明：

- 训练态到部署态转换满足验收阈值；
- 部署态保存后重新加载逐元素完全一致。

## 10.5 validation 指标一致性

| 指标 | 训练态 | 部署态 | 差值 |
|---|---:|---:|---:|
| Precision | 0.890738481864 | 0.890738479424 | -2.44e-09 |
| Recall | 0.781173318280 | 0.781173277291 | -4.10e-08 |
| mAP50 | 0.848307134857 | 0.848307134857 | 0 |
| mAP50-95 | 0.485888493164 | 0.485888493164 | 0 |

最大指标绝对差异：

```text
4.09889525654e-08
```

因此已证实：

> Rep 训练态多分支模型与部署态单卷积模型在 validation 指标上等价。

## 10.6 PyTorch 验证速度记录

```text
training inference = 2.4821 ms/image
deploy inference   = 1.0164 ms/image
```

该结果说明部署态结构有积极的速度信号，但不作为最终加速结论，因为：

- 验证执行顺序不同；
- CUDA kernel cache 和内存状态不同；
- Ultralytics 可能在运行时对普通 Conv-BN 再融合；
- 当前并非统一 TensorRT Engine Benchmark；
- 尚未在 Jetson 上测量 P50/P95、吞吐、温度和功耗。

真正的部署速度结论留到 ONNX/TensorRT/Jetson 阶段。

---

## 11. Exp04.5a：独立 test 集三模型公平评估

## 11.1 对比模型

```text
1. Exp02 YOLO11n 基线
2. Exp04 YOLO11n-Rep 训练态
3. Exp04 YOLO11n-Rep 部署态
```

所有模型在同一程序、相同环境、相同 test 集下重新评估。

## 11.2 整体指标

| 模型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n baseline | 0.92161767 | 0.82040743 | 0.89270104 | 0.52047856 |
| Rep training | 0.89763789 | 0.83732690 | 0.89113548 | 0.52323236 |
| Rep deploy | 0.89763765 | 0.83732952 | 0.89113652 | 0.52323539 |

Rep 部署态相对基线：

```text
Precision   -0.02398002
Recall      +0.01692209
mAP50       -0.00156451
mAP50-95    +0.00275683
```

换算为百分点：

```text
Precision   -2.40 pp
Recall      +1.69 pp
mAP50       -0.16 pp
mAP50-95    +0.28 pp
```

因此 Rep 模型不是整体全面提升，而是呈现：

```text
Precision 下降
Recall 上升
mAP50 基本持平
mAP50-95 略升
```

## 11.3 分类别指标

### YOLO11n 基线

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.90683909 | 0.79525223 | 0.88411403 | 0.51719728 |
| helmet | 0.95620015 | 0.83783784 | 0.90335270 | 0.48805186 |
| safety_vest | 0.90181377 | 0.82813224 | 0.89063639 | 0.55618653 |

### Rep 部署态

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| person | 0.89060949 | 0.83679525 | 0.88924671 | 0.51628032 |
| helmet | 0.93255770 | 0.85421080 | 0.89690092 | 0.49698334 |
| safety_vest | 0.86974575 | 0.82098251 | 0.88726194 | 0.55644251 |

主要现象：

- person Recall 明显提高，但 Precision 下降；
- helmet Recall 和 mAP50-95 略升，但 Precision、mAP50 下降；
- safety_vest mAP50-95 基本不变，Precision 和 Recall 略降。

## 11.4 训练态与部署态 test 等价性

```text
最大四项指标差异 = 3.03271149626e-06
```

因此：

```text
training_deploy_metric_equivalence=PASS
```

已经证明部署转换没有造成有意义的 test 指标变化。

## 11.5 PyTorch test 速度记录

```text
baseline inference     = 1.9142 ms/image
rep training inference = 1.1195 ms/image
rep deploy inference   = 1.0690 ms/image
```

同样只作为日志记录，不作为最终 TensorRT/Jetson 加速结论。

---

## 12. Exp04.5b：固定阈值错误与目标尺寸审计

## 12.1 审计设置

```text
imgsz             = 640
confidence        = 0.25
NMS IoU           = 0.70
GT matching IoU   = 0.50
class-aware match = True
batch             = 4
image_count       = 219
```

新脚本首先精确复现 Exp02 基线，结果为：

```text
overall              = PASS
class_person         = PASS
class_helmet         = PASS
class_safety_vest    = PASS
size_tiny            = PASS
size_small           = PASS
size_medium          = PASS
size_large           = PASS
```

因此 Exp04.5b 与 Exp02 审计逻辑一致，比较结果可信。

## 12.2 固定阈值整体结果

| 指标 | Baseline | Rep Deploy | 变化 |
|---|---:|---:|---:|
| GT | 840 | 840 | 0 |
| TP | 731 | 721 | -10 |
| FP | 169 | 142 | -27 |
| FN | 109 | 119 | +10 |
| Precision | 0.81222222 | 0.83545771 | +0.02323548 |
| Recall | 0.87023810 | 0.85833333 | -0.01190476 |
| F1 | 0.84022989 | 0.84674105 | +0.00651116 |

Rep 模型在 `conf=0.25` 下：

```text
误检减少 27 个
漏检增加 10 个
Precision 提高约 2.32 pp
Recall 下降约 1.19 pp
F1 提高约 0.65 pp
```

## 12.3 分类别固定阈值结果

### Baseline

| 类别 | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 337 | 288 | 86 | 49 | 0.77005348 | 0.85459941 | 0.81012658 |
| helmet | 259 | 232 | 26 | 27 | 0.89922481 | 0.89575290 | 0.89748549 |
| safety_vest | 244 | 211 | 57 | 33 | 0.78731343 | 0.86475410 | 0.82421875 |

### Rep Deploy

| 类别 | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 337 | 289 | 63 | 48 | 0.82102273 | 0.85756677 | 0.83889695 |
| helmet | 259 | 222 | 32 | 37 | 0.87401575 | 0.85714286 | 0.86549708 |
| safety_vest | 244 | 210 | 47 | 34 | 0.81712062 | 0.86065574 | 0.83832335 |

主要现象：

- person TP +1、FP -23，改善明显；
- helmet TP -10、FN +10，是整体漏检增加的主要来源；
- safety_vest FP -10，但 TP -1。

## 12.4 不同尺寸目标结果

| 尺寸 | Baseline TP/GT | Baseline Recall | Rep TP/GT | Rep Recall | Recall 变化 |
|---|---:|---:|---:|---:|---:|
| tiny | 24/29 | 0.82758621 | 16/29 | 0.55172414 | -0.27586207 |
| small | 89/114 | 0.78070175 | 90/114 | 0.78947368 | +0.00877193 |
| medium | 194/222 | 0.87387387 | 193/222 | 0.86936937 | -0.00450450 |
| large | 424/475 | 0.89263158 | 422/475 | 0.88842105 | -0.00421053 |
| tiny+small | 113/143 | 0.79020979 | 106/143 | 0.74125874 | -0.04895105 |
| medium+large | 618/697 | 0.88665710 | 615/697 | 0.88235294 | -0.00430416 |

最需要关注的是：

```text
tiny TP        24 → 16
tiny Recall    0.8276 → 0.5517
下降约 27.59 个百分点

tiny+small TP  113 → 106
tiny+small Recall 下降约 4.90 个百分点
```

这与项目的小目标改进目标冲突。

---

## 13. Exp04.5c：验证集阈值校准

## 13.1 为什么需要阈值校准

Exp04.5a 显示 Rep 的整体 AP 基本持平，但 Exp04.5b 在 `conf=0.25` 下 tiny Recall 大幅下降。可能存在两种情况：

```text
A. 正确 tiny 框仍存在，只是置信度低于 0.25
B. 正确 tiny 框本身没有形成，属于真实检测/定位退化
```

为了避免直接在 test 集调参，Exp04.5c 使用：

```text
val 选择阈值
→ test 验证
```

## 13.2 固定 conf=0.25

结果与 Exp04.5b 完全复现：

```text
baseline_fixed_test_reproduced=PASS
rep_fixed_test_reproduced=PASS
```

## 13.3 validation 最佳 F1 阈值

```text
baseline threshold = 0.45
baseline val F1    = 0.83834586

Rep threshold      = 0.43
Rep val F1         = 0.83793970
```

## 13.4 使用 validation 最佳 F1 阈值评估 test

| 指标 | Baseline@0.45 | Rep@0.43 | 变化 |
|---|---:|---:|---:|
| TP | 701 | 695 | -6 |
| FP | 94 | 80 | -14 |
| FN | 139 | 145 | +6 |
| Precision | 0.88176101 | 0.89677419 | +0.01501319 |
| Recall | 0.83452381 | 0.82738095 | -0.00714286 |
| F1 | 0.85749235 | 0.86068111 | +0.00318876 |
| tiny Recall | 0.68965517 | 0.55172414 | -0.13793103 |
| small Recall | 0.74561404 | 0.73684211 | -0.00877193 |
| tiny+small Recall | 0.73426573 | 0.69930070 | -0.03496503 |

阈值校准后：

- Rep 的整体 F1 仍略高；
- FP 仍更少；
- tiny Recall 仍低 13.79 个百分点；
- tiny+small Recall 仍低约 3.50 个百分点。

说明问题不能由简单的全局最佳 F1 阈值消除。

## 13.5 相近 Precision 工作点

在 validation 上，使 Rep Precision 接近 baseline `conf=0.25` 的阈值为：

```text
Rep threshold = 0.26
```

test 结果：

```text
TP        = 720
FP        = 138
FN        = 120
Precision = 0.83916084
Recall    = 0.85714286
F1        = 0.84805654

tiny TP   = 16
tiny Recall = 0.55172414
tiny+small TP = 106
tiny+small Recall = 0.74125874
```

相对 baseline `conf=0.25`：

```text
TP  -11
FP  -31
FN  +11
F1  +0.00783
tiny Recall      -27.59 pp
tiny+small Recall -4.90 pp
```

因此在相近误检控制水平下，tiny 退化依然存在。

---

## 14. Exp04.5d：低置信度 tiny 恢复能力诊断

## 14.1 诊断目的

使用 Exp04.5c 已缓存的 `conf=0.001` 低阈值预测结果，检查：

```text
随着阈值不断降低，Rep 是否能重新找回 baseline 已检测出的 tiny 目标
```

## 14.2 Baseline 低阈值结果

| 阈值 | TP | FP | F1 | tiny TP | tiny Recall | tiny+small TP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 799 | 7828 | 0.168797 | 25 | 0.862069 | 132 |
| 0.010 | 774 | 1500 | 0.497110 | 25 | 0.862069 | 124 |
| 0.050 | 758 | 501 | 0.722249 | 25 | 0.862069 | 118 |
| 0.100 | 750 | 332 | 0.780437 | 25 | 0.862069 | 118 |
| 0.150 | 741 | 242 | 0.812946 | 25 | 0.862069 | 115 |
| 0.200 | 736 | 197 | 0.830231 | 24 | 0.827586 | 113 |
| 0.250 | 731 | 169 | 0.840230 | 24 | 0.827586 | 113 |

Baseline 的最大 tiny TP：

```text
25 / 29
```

## 14.3 Rep Deploy 低阈值结果

| 阈值 | TP | FP | F1 | tiny TP | tiny Recall | tiny+small TP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 794 | 7525 | 0.173381 | 19 | 0.655172 | 126 |
| 0.010 | 769 | 1514 | 0.492475 | 19 | 0.655172 | 120 |
| 0.050 | 752 | 539 | 0.705772 | 18 | 0.620690 | 116 |
| 0.100 | 744 | 312 | 0.784810 | 17 | 0.586207 | 113 |
| 0.150 | 736 | 223 | 0.818232 | 17 | 0.586207 | 109 |
| 0.200 | 729 | 172 | 0.837450 | 16 | 0.551724 | 107 |
| 0.250 | 721 | 142 | 0.846741 | 16 | 0.551724 | 106 |

Rep 的最大 tiny TP：

```text
19 / 29
```

对应阈值约为：

```text
0.04
```

此时 FP：

```text
617
```

## 14.4 决定性结论

Baseline 在 `conf=0.25` 时：

```text
tiny TP = 24
```

Rep 即使将阈值降低到 `0.001`，最多只能达到：

```text
tiny TP = 19
```

因此：

```text
reaches_baseline_fixed_tiny_tp=NO
```

说明至少 5 个 baseline 能正确检测的 tiny 实例，在 Rep 模型中没有形成类别一致且 IoU≥0.5 的有效候选框。问题不是简单的置信度整体偏低。

## 14.5 tiny+small 恢复代价

Rep 在 `conf=0.10` 时可以恢复：

```text
tiny+small TP = 113
```

与 baseline `conf=0.25` 相同，但：

```text
Rep FP      = 312
Baseline FP = 169
FP 增加     = 143
```

因此即使恢复 tiny+small 总数，也需要付出明显不可接受的误检代价，并且 tiny 本身仍未恢复。

最终诊断：

```text
route_decision=
STOP_AS_FINAL_MODEL:
tiny targets are not recovered even at low confidence
```

---

## 15. 参数量、结构与初步性能对比

| 项目 | YOLO11n 基线 | Rep 训练态 | Rep 部署态 |
|---|---:|---:|---:|
| PPE 三分类参数量 | 2,590,425 | 2,611,289 | 2,590,233 |
| 相对基线参数变化 | 0 | +20,864 | -192 |
| RepConvBlock 数量 | 0 | 2 | 2 |
| 每个 RepBlock 卷积分支 | 普通单 Conv-BN | 3×3 + 1×1 | 单 3×3 Conv |
| RepBlock 中 BN | 普通 BN | 两个 BN | 0 |
| 分支求和 | 无 | 有 | 无 |
| 训练时间 | 约 717 s | 711.882 s | 不适用 |
| PyTorch test inference 记录 | 1.914 ms | 1.119 ms | 1.069 ms |

注意：

- 训练态参数比基线增加约 0.81%；
- 部署态参数几乎与基线相同；
- 部署态没有额外多分支推理图；
- PyTorch 日志速度不是最终 TensorRT/Jetson Benchmark。

---

## 16. 为什么整体 AP 基本持平，但 tiny 目标退化

当前实验可以证实结果现象，但不能仅凭现有数据唯一确定内部根因。合理解释包括：

1. 第 17、20 层位于 Neck 下采样路径，多分支训练改变了 P4/P5 特征分布；
2. tiny 目标主要依赖高分辨率 P3 特征，但 P3、P4、P5 在后续融合和 Detect Head 中相互影响；
3. Rep 训练可能优化了整体类别分离和中大型目标，导致整体 FP 减少；
4. 极少数 tiny 样本可能对训练随机性和特征分布变化非常敏感；
5. tiny 仅有 29 个实例，8 个 TP 的变化会造成很大的百分比波动；
6. 结构改造可能使部分 tiny 框的定位 IoU 低于 0.5，而不只是置信度降低；
7. 当前只训练一次固定 seed，不能证明所有 seed 都有相同程度的 tiny 下降。

这些属于基于结果的合理分析，不应写成已经被独立实验完全证明的根因。

已经被 Exp04.5d 明确排除的是：

> tiny 退化仅仅由 `conf=0.25` 过高造成。

因为即使将阈值降到 `0.001`，Rep 的最大 tiny TP 仍只有 19，无法达到 baseline 的 24。

---

## 17. Exp04 最终结论

## 17.1 工程实现结论

从工程与部署转换角度，Exp04 成功完成：

```text
独立模块融合
YOLO11n 结构替换
预训练权重迁移
零贡献初始化
自定义 Trainer
EMA 保存
自定义检查点保存与加载
100 Epoch 正式训练
训练态到部署态解析融合
部署检查点保存与重新加载
严格 FP32 数值验证
validation 指标等价
independent test 指标等价
```

因此：

```text
reparameterization implementation = PASS
training pipeline                  = PASS
deployment conversion              = PASS
numerical equivalence              = PASS
checkpoint portability             = PASS
```

## 17.2 模型精度结论

独立 test AP：

```text
mAP50      0.89270 → 0.89114   基本持平
mAP50-95   0.52048 → 0.52324   略升约 0.28 pp
```

固定 `conf=0.25`：

```text
FP          169 → 142
F1          0.84023 → 0.84674
```

说明 Rep 结构具有更少误检和略高固定阈值 F1 的积极结果。

但项目重点指标 tiny 目标明显退化：

```text
tiny TP       24 → 16
tiny Recall   0.82759 → 0.55172
```

低阈值最多只能恢复到：

```text
tiny TP = 19
```

因此：

```text
small-object business objective = FAIL
tiny preservation               = FAIL
final model replacement         = REJECT
```

## 17.3 最终路线决策

```text
exp04_engineering_result = PASS
exp04_accuracy_result    = PARTIAL
exp04_tiny_result        = FAIL

final_accuracy_model     = Exp02 YOLO11n baseline
rep_model_role           = deployment-aware ablation
route_decision           = STOP_AS_FINAL_MODEL
```

Exp04 不继续进行重复训练或更多阈值搜索，避免在 tiny 样本仅 29 个的 test 集上过度调参。

---

## 18. 最终保留权重

### 18.1 项目最终精度基线

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp02_6_yolo11n_baseline_e100_20260804_185444/
weights/best.pt
```

SHA256：

```text
79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
```

用途：

```text
后续 ONNX/TensorRT/Jetson 主部署模型
```

### 18.2 Exp04 训练态重参数化权重

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp04_3_yolo11n_rep_e100_20260804_223547/
weights/best.pt
```

SHA256：

```text
5dcc487e122670fb0b2b37a2a7d74ac2c7401bb19638065c1b27c16b1bf75938
```

用途：

```text
保留多分支训练结构
复现训练态结果
重新执行部署融合
```

### 18.3 Exp04 部署态权重

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp04_4_yolo11n_rep_deploy_20260804_225458/
weights/best_deploy_fp32.pt
```

SHA256：

```text
9694391a75cc6127a6452b9d5e28930f76eb38194f67f0dfdbde7e61b30fc7ae
```

用途：

```text
部署可重参数化结构消融
ONNX/TensorRT 计算图对照
训练态与部署态转换验证
```

该权重不作为最终精度主模型。

---

## 19. 主要代码与实验文件

### 19.1 模型实现

```text
models/blocks/reparam_block.py
models/reparam_yolo.py
models/reparam_trainer.py
```

### 19.2 主要实验脚本

```text
tools/exp04_0_reparam_block_probe.py
tools/exp04_0_reparam_block_probe.sh

tools/exp04_1_yolo11n_reparam_probe.py
tools/exp04_1_yolo11n_reparam_probe.sh
tools/exp04_1b_strict_numerical_replay.sh
tools/exp04_1c_finalize_acceptance.sh

tools/exp04_2_yolo11n_rep_smoke.py
tools/exp04_2_yolo11n_rep_smoke.sh

tools/exp04_3_yolo11n_rep_train.py
tools/exp04_3_yolo11n_rep_train.sh

tools/exp04_4_convert_and_validate.py
tools/exp04_4_convert_and_validate.sh

tools/exp04_5a_reparam_independent_test.py
tools/exp04_5a_reparam_independent_test.sh

tools/exp04_5b_reparam_error_size_audit.py
tools/exp04_5b_reparam_error_size_audit.sh

tools/exp04_5c_threshold_calibration.py
tools/exp04_5c_threshold_calibration.sh
```

Exp04.5d 使用 Exp04.5c 已生成的：

```text
threshold_sweep.csv
```

进行离线低置信度恢复分析，没有重新推理。

### 19.3 关键结果目录

```text
results/model_design/exp04_0_reparam_block_probe_*
results/model_design/exp04_1_yolo11n_reparam_probe_20260804_215212
results/model_design/exp04_1b_strict_numerical_replay_20260804_220217
results/model_design/exp04_1c_reparam_acceptance_*

results/training/exp04_2_yolo11n_rep_smoke_20260804_222225
results/training/exp04_3_yolo11n_rep_e100_20260804_223547

results/model_conversion/exp04_4_yolo11n_rep_deploy_20260804_225458

results/evaluation/exp04_5a_reparam_independent_test_20260804_225914
results/evaluation/exp04_5b_reparam_error_size_audit_*
results/evaluation/exp04_5c_threshold_calibration_20260804_231846
```

---

## 20. 已证实、未证实与适用范围

### 20.1 已证实

- RepConvBlock 的 3×3、1×1、Identity 分支可以正确融合；
- YOLO11n Neck 第 17、20 层可以替换为部署可重参数化模块；
- 原始 Conv-BN 权重可以迁移到 Rep 3×3 分支；
- 新增 1×1 分支可以零贡献初始化；
- 自定义模块可以参加反向传播并由优化器更新；
- 自定义模块能够被 EMA、`best.pt` 和 `last.pt` 正确保留；
- 多分支训练态可以融合为单 3×3 Conv 部署态；
- 部署态保存与重新加载结果完全一致；
- 训练态和部署态 validation/test 指标近似完全一致；
- Rep 模型整体 mAP50 基本持平、mAP50-95 略升；
- Rep 模型固定阈值 FP 更少、F1 略高；
- Rep 模型 tiny Recall 明显下降；
- tiny 下降不能通过简单降低置信度阈值恢复。

### 20.2 尚未证实

- 不同随机种子下 tiny 退化是否完全一致；
- 其他 Rep 插入位置是否能避免 tiny 退化；
- 更换为三分支 stride=1 Rep 模块是否更有效；
- 更长训练、更强 tiny 数据增强或重采样是否能改善结果；
- Rep 部署态在 Jetson TensorRT 上的真实 P50/P95 延迟；
- Rep 部署态在 FP16/INT8 下的精度和性能；
- 实时摄像头场景中的端到端误检、漏检和稳定性。

### 20.3 结论适用范围

当前结论适用于：

```text
当前 PPE 三分类数据集
当前 train/val/test 固定划分
YOLO11n
Neck 第 17、20 层 stride=2 双分支 Rep 替换
当前 100 Epoch 训练配置
seed=42
imgsz=640
```

不能据此泛化为：

```text
所有 Rep 结构都不适合小目标检测
所有数据集上 Rep 都会降低 tiny Recall
所有 TensorRT 部署中 Rep 都一定更快
```

---

## 21. 项目与面试价值

Exp04 的价值不在于得到一个全面优于基线的模型，而在于完成了一条可信的部署感知模型改造链：

```text
提出训练时多分支、部署时单分支假设
→ 推导融合公式
→ 实现独立模块
→ 做模块级数值探针
→ 集成真实 YOLO 图结构
→ 处理预训练权重迁移
→ 自定义 Trainer 和 EMA
→ 完成正式训练
→ 生成独立部署检查点
→ 做 validation/test 等价性验证
→ 做固定阈值错误审计
→ 做 tiny/small 分组审计
→ 做阈值校准和低置信度恢复分析
→ 用业务目标决定不替换最终模型
```

这体现了：

- 不是只修改 YAML，而是实现了可部署的自定义模块；
- 理解 Conv-BN 和多分支解析融合；
- 能处理 Ultralytics 自定义训练、EMA、检查点序列化；
- 能区分数值等价、AP 等价和业务指标等价；
- 不只看整体 mAP，还检查固定工作点 TP/FP/FN 和 tiny/small；
- 能根据精度—部署—业务目标共同做路线决策。

---

## 22. 推荐项目表述

不能表述为：

```text
加入重参数化模块后提升了 PPE 小目标检测精度。
```

推荐表述：

> 设计并实现 YOLO11n Neck 部署可重参数化模块，在训练阶段使用 3×3/1×1 多分支卷积，部署前解析融合为单 3×3 卷积；完成自定义 Trainer、EMA、检查点保存加载和整网数值一致性验证，训练态与部署态验证指标最大差异低于 4.1×10^-8。独立测试中整体 mAP50-95 提升约 0.28 个百分点、固定阈值 FP 减少 27 个，但 tiny 目标召回由 82.76% 降至 55.17%，且低置信度扫描无法恢复，因此将其作为部署感知结构消融保留，未替换最终 YOLO11n 精度基线。

简历精简版本：

> 实现 YOLO11n 多分支训练/单卷积部署的结构重参数化模块，完成 Conv-BN/Identity 解析融合、自定义训练与检查点转换；部署前后 mAP 差异低于 3.1×10^-6。通过固定阈值和尺寸审计发现整体 F1 略升但 tiny Recall 明显下降，基于业务目标终止最终替换，保留为部署优化消融。

---

## 23. 后续主线

Exp04 至此结束。后续主线继续使用 Exp02 YOLO11n 基线：

```text
Exp02 YOLO11n best.pt
→ PyTorch 到 ONNX 导出
→ ONNX 数值一致性
→ TensorRT FP32/FP16
→ INT8 校准与精度评估
→ C++ TensorRT Runtime
→ CUDA/GPU 预处理
→ Jetson 端 P50/P95、吞吐、功耗和温度测试
```

Exp04 Rep 部署模型仅作为可选结构对照：

```text
检查 ONNX/TensorRT 图是否保持单卷积
比较 Engine 层数和算子图
验证是否确实没有额外部署分支
```

最终精度主模型仍为 Exp02 YOLO11n 基线。
