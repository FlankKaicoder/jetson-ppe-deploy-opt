# AGENTS.md

# Jetson PPE 项目三端协同开发规范

> 本文件是 Codex、ChatGPT 及人工开发者在本仓库中的最高优先级项目协作说明。
> 任何自动化修改、命令执行、实验运行、Git 操作和结果总结，都必须先阅读并遵守本文件。
> 若本文件与某次临时口头指令冲突，以用户最新明确指令为准；若用户没有明确授权，不得自行扩大操作范围。

---

# 1. 项目定位

项目名称：

```text
基于 Jetson Orin Nano Super 的 PPE 小目标检测与 TensorRT/CUDA 推理优化
```

项目目标不是单纯训练一个检测模型，而是完成以下完整工程链路：

```text
数据集审计
→ YOLO11n 基线训练
→ 模型结构消融
→ PyTorch 模型冻结
→ ONNX 导出与一致性验证
→ TensorRT FP32 / FP16 / INT8
→ TensorRT C++ Runtime
→ CUDA 融合预处理
→ Jetson 端性能、功耗、温度与稳定性验证
→ README、实验总结、简历与面试材料
```

当前项目强调：

- 可复现；
- 可追溯；
- 不覆盖历史实验；
- 负向实验也要保留；
- 结论必须有代码、日志、指标或哈希支撑；
- 模型效果和部署性能必须分开评价；
- 未完成、未验证的功能不得表述为已完成。

---

# 2. 当前已知开发环境

本项目存在三个独立工作环境，它们共享同一个 GitHub 仓库，但用途不同。

## 2.1 Windows 本地环境

用途：

```text
项目总控
代码阅读
文档维护
Git 分支审查
合并 main
关键产物中转
```

Windows 本地仓库路径由当前 Codex 工作区决定，本文件不假设固定盘符。

Windows 不具备完整的 AutoDL 训练环境和 Jetson 部署环境，因此：

- 不能把“Windows 上代码能保存”当作“实验已经通过”；
- 训练代码必须在 AutoDL 验证；
- TensorRT、CUDA、C++、摄像头和板端性能必须在 Jetson 验证。

---

## 2.2 AutoDL 训练服务器

当前仓库：

```text
/root/autodl-tmp/jetson-ppe-deploy-opt
```

当前 Python 虚拟环境：

```text
/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl
```

当前训练输出根目录：

```text
/root/autodl-tmp/jetson-ppe-outputs
```

当前数据集 YAML：

```text
/root/autodl-tmp/datasets/derived/
construction_ppe3_final_split_v1_20260804_175104/
construction_ppe3.yaml
```

当前主要环境：

```text
GPU          : NVIDIA RTX 3080 Ti
Python       : 3.12.3
PyTorch      : 2.8.0+cu128
Ultralytics  : 8.4.95
ONNX         : 1.22.0
ONNX Runtime : 1.28.0
```

AutoDL 主要用于：

```text
数据集处理
训练
验证
测试集评估
误检漏检审计
模型结构实验
PyTorch 权重冻结
ONNX 导出
ONNX Runtime 一致性验证
量化前模型准备
```

---

## 2.3 Jetson Orin Nano Super

当前仓库：

```text
/home/nvidia/projects/jetson-ppe-deploy-opt
```

当前主要环境：

```text
系统        : Ubuntu 22.04 / L4T R36.4.3
架构        : aarch64
CUDA        : 12.6
cuDNN       : 9.3
TensorRT    : 10.3
OpenCV      : 4.10
GStreamer   : 1.20
```

Jetson 主要用于：

```text
TensorRT Engine 构建
FP32 / FP16 / INT8 推理
TensorRT C++ Runtime
CUDA Kernel
CUDA 融合预处理
GStreamer
IMX219 摄像头
板端延迟、吞吐、功耗、温度与稳定性测试
```

---

# 3. 三端协同的核心原则

## 3.1 GitHub 只同步代码和小型实验记录

以下内容通过 GitHub 同步：

```text
.py
.cpp
.cu
.h
.hpp
.sh
.yaml
.yml
.json
.csv
.md
CMakeLists.txt
小型 txt 日志摘要
```

以下内容不得直接提交到普通 Git：

```text
.pt
.pth
.onnx
.engine
.plan
.cache
数据集
完整训练输出目录
大体积图片
视频
大型 run.log
编译产物
临时下载文件
```

大型产物通过：

```text
scp
rsync
Windows 中转目录
对象存储
AutoDL 本地目录
Jetson 本地目录
```

进行传输。

---

## 3.2 代码和模型产物走不同通道

推荐数据流：

```text
代码、配置、文档
AutoDL / Jetson
        ↓ git push
GitHub
        ↓ git fetch / pull
Windows 审查并合并
```

模型产物流：

```text
AutoDL best.pt
    ↓ 导出
AutoDL ONNX
    ↓ scp / rsync
Windows 归档或直接传输
    ↓
Jetson 构建 TensorRT Engine
```

重要限制：

- TensorRT Engine 不跨机器复用；
- Engine 必须在目标 Jetson 的 TensorRT/CUDA/硬件环境中重新构建；
- Git 中只记录 Engine 构建配置、版本、哈希和结果，不提交 Engine 文件。

---

## 3.3 每台机器只做自己适合的工作

### Windows 负责

- 文档；
- 代码审查；
- 分支比较；
- 合并；
- GitHub 管理；
- 关键产物中转；
- 项目总览。

### AutoDL 负责

- 训练；
- 精度评估；
- ONNX 导出；
- ONNX Runtime 验证；
- 训练端脚本调试。

### Jetson 负责

- TensorRT；
- CUDA；
- C++；
- GStreamer；
- 摄像头；
- 板端性能测试。

禁止为了“方便”在错误环境中宣称实验完成。

---

# 4. Codex 启动后的强制检查

Codex 每次进入任意工作区后，必须先执行只读检查，不得立刻修改文件。

## 4.1 识别当前机器

依次检查：

```bash
hostname
whoami
pwd
uname -m
```

根据路径和架构判断当前环境：

```text
Windows 工作区
AutoDL x86_64 训练服务器
Jetson aarch64 部署设备
```

不得在未识别环境前运行训练、编译或系统操作。

---

## 4.2 检查 Git 状态

必须执行：

```bash
git rev-parse --show-toplevel
git branch --show-current
git status --short
git log -5 --oneline --decorate
git remote -v
```

必须先报告：

- 当前仓库路径；
- 当前分支；
- 是否存在未提交修改；
- 本地是否领先或落后远端；
- 最近提交。

若存在用户未说明的未提交修改：

- 不得覆盖；
- 不得自动 reset；
- 不得 stash；
- 不得提交；
- 必须先向用户说明。

---

## 4.3 阅读项目上下文

每项任务开始前，至少阅读：

```text
AGENTS.md
README.md
ROADMAP.md
docs/00_project_scope.md
docs/experiment_index.md
与当前实验直接相关的 Markdown 总结
```

例如进行 ONNX 实验前，应阅读：

```text
docs/02_YOLO11n基线训练与评估总结.md
docs/03_YOLO11n-P2小目标结构消融总结.md
docs/04_YOLO11n部署可重参数化结构消融总结.md
docs/05_YOLO11n轻量注意力与Focal损失消融总结.md
```

不得只根据文件名猜测当前项目状态。

---

# 5. 当前实验状态

当前已经完成：

```text
Exp00 项目初始化与范围冻结
Exp01 Jetson 环境审计与 C++/CUDA/TensorRT 编译验证
Exp02 PPE 数据集审计与 YOLO11n 基线
Exp03 YOLO11n-P2 小目标结构消融
Exp04 部署可重参数化结构消融
Exp05 轻量注意力与 Focal 分类损失消融
Exp06 PyTorch → ONNX 导出与一致性验证
Exp07 Jetson TensorRT FP32 / FP16 Engine 构建与验证
Exp08 INT8 PTQ 构建、精度—性能比较与候选决策
Exp09 TensorRT C++ Runtime 与三进程生命周期验证
Exp10 CUDA 融合预处理与五形状正确性/性能验证
Exp11 视频/摄像头端到端推理
Exp12 性能、功耗、温度与稳定性测试
Exp13 Nsight Systems 端到端性能瓶颈画像与优化基线
Exp14 Pinned Memory、CUDA Event 与 Double Buffer 异步流水线消融
```

当前模型决策：

```text
后续部署主线继续使用原始 YOLO11n baseline
```

冻结基线权重：

```text
/root/autodl-tmp/jetson-ppe-outputs/
exp02_6_yolo11n_baseline_e100_20260804_185444/
weights/best.pt
```

基线 SHA256：

```text
79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6
```

基线独立 test 指标：

```text
Precision          = 0.92161767
Recall             = 0.82040743
mAP50              = 0.89270104
mAP50-95           = 0.52047856
tiny+small recall  = 0.79020979
```

Exp03、Exp04、Exp05 均具有工程学习价值，但没有满足替换原始基线的综合验收条件。

Exp06、Exp07 已通过各自验收。Exp08 工程链路已完成，但 INT8 因 mAP50-95 和
tiny+small recall 超过预冻结退化门槛而 REJECT；运行时主线保留 FP16。
Exp09 已通过 C++ Runtime 编译、原始输出一致性和三独立进程生命周期验收。
Exp10 已通过五种输入形状逐元素一致性和 CUDA kernel-only 性能验收。
Exp11 已通过文件视频三进程确定性和 IMX219 300 帧端到端功能验收。
Exp12 已通过固定时钟三进程重复性和 54,000 帧/约 30 分钟稳定性验收；默认部署恢复动态调频。
Exp13 已通过 NVTX 正确性回归、文件/相机 Nsight Systems 时间线、三轮无分析器重复性和
TensorRT 逐层诊断验收。文件模式归类为 synchronization-bound，相机模式归类为
input-rate-bound + synchronization-bound；CPU decode/NMS 不是当前首要瓶颈。
Exp14 已完成 Pinned Memory、CUDA Event、单缓冲异步和双缓冲三 Stream 工程链路、文件/相机
三轮无分析器对照与正式 Nsight 验证。Variant C 文件吞吐仅提升4.51%，P95退化159.28%；相机
P95退化173.51%，虽观察到跨帧重叠但未满足冻结门槛，因此候选 `REJECT`，运行时主线继续使用
Exp13 同步 FP16 Runtime。
用户已批准将项目收尾顺延到 Exp20。当前下一候选为 Exp15 CUDA GPU Decode/Filter/Compaction
与 Nsight Compute，必须先提交方案供用户审批，不得提前实现。

Codex 不得擅自重新选择模型主线。

---

# 6. 下一阶段推荐顺序

后续部署阶段建议按以下顺序推进：

```text
Exp06  PyTorch → ONNX 导出与一致性验证
Exp07  Jetson TensorRT FP32 / FP16 Engine
Exp08  INT8 PTQ 与精度—性能比较
Exp09  TensorRT C++ Runtime
Exp10  CUDA 融合预处理
Exp11  视频/摄像头端到端推理
Exp12  性能、功耗、温度与稳定性测试
Exp13  Nsight Systems 端到端性能瓶颈画像与优化基线
Exp14  Pinned Memory、Async 与 Double Buffer 流水线
Exp15  CUDA GPU Decode/Filter/Compaction 与 Nsight Compute
Exp16  TensorRT IPluginV3 与 ONNX GraphSurgeon
Exp17  INT8 敏感性分析与 Mixed Precision
Exp18  CUDA Graph 与最终 Runtime 优化
Exp19  最终综合 Benchmark
Exp20  README、学习路线、简历与面试材料
```

若用户另行指定编号，以用户最新指令为准。

---

# 7. Git 分支策略

## 7.1 main 分支

`main` 必须保持：

- 已验证；
- 可复现；
- 不包含半成品；
- 不包含大型产物；
- 不包含本机专用绝对路径泄漏；
- 不包含未解释的临时修改。

禁止直接在 `main` 上开发。

---

## 7.2 一项实验一个分支

推荐命名：

```text
exp/06-onnx-export
exp/07-tensorrt-fp16
exp/08-int8-ptq
exp/09-trt-cpp-runtime
exp/10-cuda-preprocess
```

修复分支：

```text
fix/exp06-onnx-output-mismatch
```

文档分支：

```text
docs/deployment-workflow
```

---

## 7.3 禁止多个环境同时修改同一分支

错误方式：

```text
AutoDL 和 Jetson 同时在 exp/06-onnx-export 上修改并推送
```

正确方式：

```text
AutoDL：exp/06-onnx-export
Jetson：exp/07-tensorrt-fp16
Windows：审查并合并
```

如果同一个实验确实需要跨环境，优先采用串行流程：

```text
AutoDL 完成并 push
→ Windows 审查并 merge
→ Jetson pull main
→ Jetson 创建下一分支
```

---

## 7.4 创建分支前

必须执行：

```bash
git switch main
git pull --ff-only
git status --short
git switch -c exp/XX-name
```

若 `git pull --ff-only` 失败，不得自动 rebase、merge 或 reset，必须先说明原因。

---

# 8. Git 提交规范

## 8.1 禁止默认执行的操作

未经用户明确授权，不得执行：

```text
git commit
git push
git merge
git rebase
git reset --hard
git clean -fd
git checkout -- .
git restore .
git stash
force push
删除远端分支
```

Codex 可以准备修改、运行测试、展示 diff，但提交和推送需要用户明确同意。

---

## 8.2 禁止直接使用 git add .

提交前必须先执行：

```bash
git status --short
```

然后按明确文件列表添加：

```bash
git add \
  tools/exp06_export_onnx.py \
  tools/exp06_export_onnx.sh \
  docs/06_onnx_export.md
```

不得把整个目录不加检查地加入 Git。

---

## 8.3 提交信息

推荐格式：

```text
feat: add baseline ONNX export and consistency audit
fix: correct ONNX output tensor matching
docs: summarize Exp06 ONNX export results
test: add TensorRT runtime smoke test
perf: add CUDA fused preprocess benchmark
```

一次提交只做一类明确工作。

---

## 8.4 提交前检查

必须检查：

```bash
git diff --check
git status --short
git diff --stat
git diff
```

需要确认：

- 没有大型二进制；
- 没有模型权重；
- 没有数据集；
- 没有绝对路径误提交；
- 没有密码、Token、私钥；
- 没有覆盖历史实验；
- 文档中的结果与真实日志一致。

---

# 9. Windows、AutoDL、Jetson 的具体协作流程

## 9.1 Windows 本地仓库

Windows 是默认的总控和审查端。

主要工作：

```text
读取所有分支
查看 diff
维护 README
维护实验索引
合并经过验证的实验
维护 main
保存关键模型产物备份
```

典型流程：

```bash
git fetch --all --prune
git switch main
git pull --ff-only
git diff main..origin/exp/06-onnx-export
```

审核通过后再合并。

Windows 不应承担：

- AutoDL 训练结果真实性验证；
- Jetson TensorRT 运行验证；
- Jetson CUDA 性能结论。

---

## 9.2 AutoDL 训练工作流

AutoDL 任务开始前：

```bash
cd /root/autodl-tmp/jetson-ppe-deploy-opt
git switch main
git pull --ff-only
git switch -c exp/XX-name
```

必须确认虚拟环境：

```bash
/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python --version
```

运行 Python 时优先使用绝对解释器：

```bash
/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python
```

正式实验前必须：

```text
依赖检查
输入文件检查
1 次 Smoke Test
输出文件检查
返回码检查
```

不得直接启动长时间训练。

---

## 9.3 Jetson 部署工作流

Jetson 任务开始前：

```bash
cd /home/nvidia/projects/jetson-ppe-deploy-opt
git switch main
git pull --ff-only
git switch -c exp/XX-name
```

编译前必须记录：

```bash
uname -a
nvcc --version
dpkg-query -W nvinfer-bin libnvinfer10 2>/dev/null || true
cmake --version
g++ --version
```

TensorRT Engine 构建前必须记录：

```text
ONNX 文件路径
ONNX SHA256
Git commit
TensorRT 版本
输入尺寸
Batch
精度模式
Workspace 配置
构建命令
```

Engine 不得提交 Git。

---

# 10. 大型模型产物传输规范

## 10.1 产物清单

每个跨机器产物必须配套记录：

```text
文件名
文件类型
来源机器
来源实验
来源 Git commit
生成命令
文件大小
SHA256
目标机器
目标路径
```

示例：

```text
artifact_name : yolo11n_baseline_exp06.onnx
source        : AutoDL
source_commit : abcdef1
source_weight : best.pt
sha256        : ...
target        : Jetson
target_path   : /home/nvidia/models/jetson-ppe/exp06/
```

---

## 10.2 推荐目录

AutoDL：

```text
/root/autodl-tmp/jetson-ppe-artifacts/
```

Windows：

```text
用户自行指定的 jetson-ppe-artifacts 目录
```

Jetson：

```text
/home/nvidia/models/jetson-ppe/
```

---

## 10.3 传输后必须校验

发送端：

```bash
sha256sum model.onnx
```

接收端：

```bash
sha256sum model.onnx
```

两个哈希必须一致。

不得只根据文件名认为传输成功。

---

# 11. 实验目录和非覆盖规则

所有实验结果必须使用时间戳目录。

推荐格式：

```text
results/<category>/expXX_name_YYYYMMDD_HHMMSS/
```

例如：

```text
results/onnx/exp06_baseline_export_20260806_143000/
results/tensorrt/exp07_fp16_build_20260807_101500/
results/benchmark/exp10_cuda_preprocess_20260809_210000/
```

禁止：

```text
results/latest/
覆盖旧 summary.txt
复用同一个固定输出目录
删除失败实验目录后重跑
```

失败实验也应保留：

```text
run.log
return_code
failure_summary.txt
```

---

# 12. 实验脚本要求

每个正式实验脚本必须具备：

- `set -uo pipefail` 或等效错误处理；
- 明确的项目目录；
- 明确的 Python 解释器；
- 输入文件存在性检查；
- 时间戳输出目录；
- 日志路径；
- 返回码捕获；
- 关键输出文件检查；
- 最终 PASS/FAIL；
- 不覆盖历史目录。

不应把超长 Python 程序直接嵌套在复杂 Shell heredoc 管道中。

优先：

```text
独立 .py
+
独立 .sh
```

避免再次出现 Python 标准输入被管道错误占用的问题。

---

# 13. Smoke Test 规则

正式实验前必须先做最小 Smoke Test。

示例：

## 训练

```text
1 epoch
较小 batch
workers=0
验证 best.pt / last.pt / results.csv
```

## ONNX

```text
单张固定输入
导出一次
onnx.checker
ONNX Runtime 前向
输出 Tensor 数量和形状检查
```

## TensorRT

```text
构建一个 Engine
单张输入
执行一次 enqueue
检查返回码
检查输出有限值
```

## CUDA

```text
小尺寸输入
CPU Reference 对比
最大绝对误差
CUDA 错误检查
```

Smoke Test 通过不代表正式实验结果有效，但 Smoke Test 失败时禁止直接继续正式实验。

---

# 14. 正确性与一致性验证

## 14.1 ONNX 导出

至少验证：

```text
PyTorch 输出
vs
ONNX Runtime 输出
```

记录：

```text
输出 Tensor 名称
输出 Tensor 形状
max_abs_error
mean_abs_error
relative_l2_error
```

同时验证最终检测结果：

```text
框
类别
置信度
mAP
```

---

## 14.2 TensorRT

至少验证：

```text
ONNX Runtime
vs
TensorRT FP32
vs
TensorRT FP16
vs
TensorRT INT8
```

不得只根据 Engine 成功构建就判断部署成功。

---

## 14.3 CUDA 预处理

至少验证：

```text
CPU OpenCV Reference
vs
CUDA Preprocess
```

比较：

```text
Resize
Letterbox
Padding
BGR/RGB
Normalize
HWC/CHW
FP32/FP16
```

---

# 15. 公平实验规则

模型或部署优化对比必须固定：

```text
数据集划分
输入尺寸
Batch
随机种子
评估集
置信度阈值
NMS 配置
匹配 IoU
预处理
计时范围
功耗模式
```

模型端指标和部署端指标分开报告：

```text
模型端：
Precision
Recall
mAP50
mAP50-95
tiny/small recall

部署端：
build time
engine size
inference latency
end-to-end latency
P50/P95/P99
FPS
memory
power
temperature
```

不能用 RTX 3080 Ti 的验证速度代替 Jetson 性能。

---

# 16. 性能测试规则

性能测试前必须明确：

```text
是否 warmup
warmup 次数
正式迭代次数
是否包含预处理
是否包含 H2D/D2H
是否包含后处理
Batch
输入尺寸
功耗模式
jetson_clocks 状态
```

若正式测试需要 `jetson_clocks`，锁频前必须先显式保存动态状态，例如：

```bash
sudo jetson_clocks --store /tmp/expXX_jetsonclocks_before.conf
sudo jetson_clocks
```

测试结束后使用同一个明确文件恢复并验证 CPU/GPU min/max；不得假设无参数锁频会自动生成
`${HOME}/.jetsonclocks_conf.txt`。若保存文件不存在，不得把 `--restore` 失败写成恢复成功。

Jetson 端建议记录：

```text
平均延迟
P50
P95
P99
吞吐率
CPU 占用
GPU 占用
内存
功耗
温度
是否降频
```

不得只报告一个 FPS。

---

# 17. 日志与结果总结

每个实验至少产生：

```text
run.log
summary.txt 或 summary.md
summary.json
关键 CSV
输入配置快照
SHA256 清单
```

总结必须包含：

1. 实验目的；
2. 实验假设；
3. 环境；
4. 输入文件；
5. 修改文件；
6. 运行命令；
7. 返回码；
8. 输出目录；
9. 关键指标；
10. 异常和修复；
11. 已证实；
12. 尚未证实；
13. 最终决策；
14. 下一步。

不得根据记忆补写不存在的指标。

---

# 18. Codex 修改代码时的行为要求

Codex 必须遵循：

```text
先读
→ 再计划
→ 小改动
→ 运行最小测试
→ 展示结果
→ 再继续
```

一次不要同时大范围修改：

```text
训练代码
部署代码
Git 结构
文档
系统环境
```

如果任务较大，应拆分为多个可验证步骤。

---

# 19. Codex 默认禁止的危险操作

未经用户明确授权，不得执行：

```bash
sudo apt upgrade
sudo apt full-upgrade
pip install -U 大量核心依赖
conda update
删除虚拟环境
删除数据集
删除实验输出
rm -rf 未明确路径
chmod -R 777
修改系统 CUDA
修改 TensorRT
修改驱动
修改 L4T
修改 JetPack
格式化磁盘
重新烧录系统
```

需要安装依赖时：

1. 先说明原因；
2. 说明安装位置；
3. 优先安装到项目虚拟环境；
4. 给出可回滚方案；
5. 获得用户确认后执行。

---

# 20. 密钥和隐私

禁止读取、显示或提交：

```text
SSH 私钥
GitHub Token
AutoDL 密钥
浏览器 Cookie
环境中的 Access Token
```

`.env`、私钥和凭据文件必须加入 `.gitignore`。

日志中若出现 Token，必须先脱敏再写入文档。

---

# 21. 出错处理

出现错误时，不得立即进行大范围重装或重构。

必须按顺序：

```text
保留现场
记录命令
记录返回码
读取最后日志
定位最小失败点
提出最小修复
运行最小验证
```

禁止为了“快速通过”：

- 放宽所有检查；
- 吞掉异常；
- 使用 `|| true` 隐藏真实失败；
- 把失败结果写成 PASS；
- 删除失败日志。

---

# 22. 分支冲突处理

若发现远端分支已被其他环境更新：

```bash
git fetch origin
git status
git log --oneline --graph --decorate --all -20
```

不得自动：

```text
rebase
force push
reset --hard
```

先向用户说明：

- 本地提交；
- 远端提交；
- 分叉点；
- 冲突文件；
- 推荐处理方式。

---

# 23. 文档和代码的一致性

修改实验代码后，应同步检查：

```text
README
ROADMAP
docs/experiment_index.md
CHANGELOG
实验总结
experiment_registry.csv
```

但不得为了“看起来完整”提前更新尚未完成的结果。

状态应明确写成：

```text
PLANNED
IN_PROGRESS
PASS
FAIL
REJECT
SKIPPED
BLOCKED
```

---

# 24. Windows Codex 推荐首条指令

```text
请先阅读 AGENTS.md、README.md、ROADMAP.md 和 docs/experiment_index.md。

不要修改任何文件，也不要执行提交或推送。

请检查：
1. 当前仓库路径；
2. 当前分支；
3. git status；
4. 最近 5 个提交；
5. 已完成实验；
6. 当前部署主线；
7. Windows、AutoDL、Jetson 三端职责；
8. Exp13 已证实的瓶颈和后续 Exp14 推荐工作流；
9. `docs/项目全流程快速学习手册.md` 中的当前计划和待确认事项。

最后只输出检查结果和建议，不执行修改。
```

---

# 25. AutoDL Codex 推荐首条指令

```text
当前工作环境应为 AutoDL 训练服务器。

请先阅读 AGENTS.md、`docs/项目全流程快速学习手册.md`、Exp06～Exp12 总结，
检查当前分支与 git 状态。

不要修改代码，不要提交，不要推送。

请确认：
1. Python 虚拟环境；
2. 冻结 baseline best.pt；
3. baseline SHA256；
4. 数据集 YAML；
5. Exp06 ONNX 文件及 SHA256 是否与冻结记录一致；
6. Exp08 INT8 为何被 REJECT，且不得改写为运行时主线；
7. Exp14 是否需要 AutoDL 侧提供新的输入产物（默认不需要）。

先给出执行计划和风险，不要直接开始。
```

---

# 26. Jetson Codex 推荐首条指令

```text
当前工作环境应为 Jetson Orin Nano Super。

请先阅读 AGENTS.md、`docs/项目全流程快速学习手册.md`、docs/01_environment.md、
Exp06～Exp12 总结和当前部署相关文档。

不要修改代码，不要提交，不要推送。

请确认：
1. Jetson 系统与架构；
2. CUDA、TensorRT、OpenCV、GStreamer 版本；
3. 当前仓库分支和状态；
4. 是否已经获得经过校验的 ONNX；
5. TensorRT Engine 是否需要在本机重新构建；
6. FP16 主线 Engine、哈希和验证结果是否可用；
7. Exp08 INT8 REJECT 结论是否与记录一致；
8. Exp13 结论是否一致，以及 Exp14 流水线优化的最小执行计划（尚待用户审批）。

先输出检查结果，不执行构建。
```

---

# 27. 当前阶段的推荐协作链

```text
ChatGPT
负责实验设计、验收条件和结果解释
        ↓
AutoDL Codex
维护冻结模型、ONNX 与数据侧参考结果
        ↓
GitHub 实验分支
同步代码、配置和小型结果
        ↓
Windows Codex
审查 diff、文档和 Git 历史
        ↓
main
合并已验证内容
        ↓
Jetson Codex
以 Exp13 时间线证据为基线，等待用户审批 Exp14
        ↓
Windows Codex
审查时间线证据、瓶颈分类和后续优化假设
        ↓
ChatGPT
审查 Exp14 方案，用户审批后才进入实现
```

---

# 28. 最终执行原则

任何任务都必须满足：

```text
明确机器
明确分支
明确输入
明确输出
明确验收条件
明确是否允许提交
```

Codex 不应替用户做未经授权的技术路线决策。

Codex 的目标不是“尽快改完”，而是：

```text
在正确机器上
以最小风险
完成可复现修改
保留真实日志
给出可审查结果
```
# 29. tips
每次对话开头都以nn你好开头然后进行对话分析

---

# 30. 设备重启、SSH 重连与持续学习手册

## 30.1 设备关机是正常状态

AutoDL 是按需租用服务器，Jetson 长期开机会持续发热，因此两台设备在不用时均可关机。
Codex 不得假设上一次 SSH 会话、后台进程或设备状态在重新开机后仍然有效。

用户说明设备已经开机或要求继续远程实验后，必须先重新连接，并至少执行：

```bash
hostname
whoami
pwd
uname -m
git rev-parse --show-toplevel
git branch --show-current
git status --short
git log -5 --oneline --decorate
git remote -v
```

随后根据任务检查：

```text
AutoDL：项目虚拟环境、GPU、输入权重/ONNX、数据集路径和 SHA256
Jetson：CUDA/TensorRT 环境、输入 ONNX/Engine、功耗模式和 SHA256
```

重连后不得直接续跑上次命令。必须先确认：

1. 连接目标仍是预期机器；
2. 仓库路径、分支和工作区状态没有变化；
3. 三端 Git 提交关系满足当前串行流程；
4. 输入产物仍存在且哈希一致；
5. 上次实验是否已经正常结束，还是留下了需要保留的失败现场。

若 SSH 不可达，应区分超时、拒绝连接和认证失败并报告用户；不得反复盲目重试，
也不得擅自修改设备网络、SSH 配置或系统服务。连接地址或端口变化时由用户提供新信息。
仓库不得记录密码、SSH 私钥或其他凭据。

## 30.2 项目全流程快速学习手册

项目的统一学习入口为：

```text
docs/项目全流程快速学习手册.md
```

每次实验开始前，必须先在该文件末尾追加“实验前规划”，至少包含：

```text
状态与日期
核心问题和实验假设
执行机器与三端分工
冻结输入和哈希
控制变量
执行步骤与 Smoke Test
验收阈值
风险和停止条件
本实验预期学会的知识
```

每次实验结束后，必须继续在文件末尾追加“实验结果与学习复盘”，至少包含：

```text
真实命令、环境、Commit 和输出目录
真实指标、产物大小和 SHA256
失败、定位过程与最小修复
已证实、尚未证实和最终决策
与基线或上一阶段的公平比较
用户必须能讲清楚的原理
面试问答和下一实验的衔接
```

该手册采用只追加的实验日志方式：不得覆盖旧计划来伪装事后预测，不得删除负向实验，
不得填写未运行的结果。若计划变化，应追加变更说明。实验状态必须使用本文件定义的
`PLANNED`、`IN_PROGRESS`、`PASS`、`FAIL`、`REJECT`、`SKIPPED` 或 `BLOCKED`。

手册内容必须与 README、ROADMAP、`docs/experiment_index.md`、实验总结和
`results/experiment_registry.csv` 交叉核对。每次设备重连和新实验开始时，
除第 4 节规定的上下文外，还必须阅读手册中“当前状态”和最新追加记录。

项目结束时，以该手册为依据整理快速学习路线、项目讲解稿、简历要点和面试题；
不得把尚未完成或尚未验证的能力写成项目成果。
