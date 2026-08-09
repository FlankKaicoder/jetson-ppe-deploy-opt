# Exp16 TensorRT IPluginV3 与 ONNX GraphSurgeon 总结

## 1. 最终结论

Exp16 总体状态为 `REJECT`。本实验完成了 TensorRT 10.3 IPluginV3、Creator/Registry、ONNX
GraphSurgeon 自定义节点插入、显式 TensorRT workspace、Engine 序列化，以及完全独立新进程先加载
Plugin `.so` 再反序列化和 `enqueueV3()` 的工程闭环。Plugin decode/filter/CUB 组件在 synthetic、冻结
raw fixture 和同一 dual-output Engine 内均达到逐项零误差，因此组件级能力记为 `VERIFIED`。

四输出全图候选没有通过正式150帧跨 Engine 语义 Gate：Exp15 B control 为151个最终检测，Plugin候选
为153个，且共有检测中存在超过冻结2 pixel门槛的框差。正式编排按预设条件在第一轮停止，未执行剩余两轮
性能测试。因此 Plugin候选不进入部署主线，不得宣称正式加速；Exp15 B CUB stable compaction继续保持
`ACCEPTED`。

## 2. 环境、起点与冻结输入

- Jetson Orin Nano Super，aarch64，L4T R36.4.3，CUDA 12.6.68，TensorRT 10.3.0.30；
- 分支：`exp/16-tensorrt-plugin-v3`，实验起点 `main@7c60a170cbb82f295934f5a8c58b22f85aa21e83`；
- AutoDL保持关机，本实验没有训练、重新导出基线ONNX或修改系统CUDA/TensorRT；
- Exp06 ONNX SHA256：
  `305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8`；
- Exp07 FP16 Engine SHA256：
  `88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83`；
- 文件视频 SHA256：
  `f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665`；
- 冻结 raw fixture：235,200 B，SHA256
  `0e6aff4557d989ec62c26908988bcb5b15222de4a7d66f53ea73f36ce825abfe`；
- 正式部署性能仍使用25W/id 1动态调频；本实验没有锁定 `jetson_clocks`。

## 3. Plugin ABI 与实现

Plugin name/version/namespace 为 `PpeYoloDecodeCompact` / `1` /
`com.flankkaicoder.ppe`。GraphSurgeon 使用相同 custom domain，正式图只暴露四个固定容量输出：

| Tensor | Shape | Type | 语义 |
|---|---|---|---|
| `boxes_scores` | `[1,8400,5]` | FP32 | `x1,y1,x2,y2,confidence` |
| `classes` | `[1,8400]` | INT32 | class id |
| `indices` | `[1,8400]` | INT32 | 原始candidate index |
| `count` | `[1]` | INT32 | 有效前缀长度 |

输入仅支持当前真实需要的 LINEAR FP32 `output0 [1,7,8400]`，不为展示技术栈增加无需求的FP16 Plugin
IO、动态batch、GPU NMS或额外geometry tensor。class tie保留较小class id，confidence采用
`>=0.25`，CUB `DeviceSelect::Flagged`保持index稳定顺序。Host先复制并验证4 B count，再仅复制有效
candidate前缀并执行inverse-letterbox和现有class-aware CPU NMS。

Plugin的candidate、flag和CUB temporary storage全部来自 `getWorkspaceSize()` 对应的TensorRT显式
workspace；`enqueue()` 内没有逐帧 `cudaMalloc/cudaFree`。序列化只保存稳定标量契约，不保存Host/
Device地址或本机路径。

## 4. 构建、GraphSurgeon 与独立进程闭环

隔离的用户级GraphSurgeon目录为 `/home/nvidia/.local/jetson-ppe-exp16-py`，包含NumPy 2.2.6、ONNX
1.17.0、ONNX GraphSurgeon 0.5.8和protobuf 7.35.1；没有修改系统Python。正式四输出ONNX SHA256为
`cfea7b13bb11cb11f8026f0f327d8939167a6b991cdb6e2ec8b038a014736d04`。

修正后候选目录为 `results/plugin/exp16_8_rebuild_20260809_170133`：四输出FP16 Engine为
8,590,132 B、SHA256
`c0b1cca81c18da176e04a38607f85d95f54720f7984bceeb51fa47d7f29c56cf`；Plugin `.so` SHA256为
`b5d402ffab879758c16289c7c385ddab7eeaa0c270a76f65042eb35a5e8477f1`。Engine按Exp07的FP16、
noTF32、1024 MiB workspace、builder optimization level 3和detailed profiling参数在目标Jetson本机
构建。

独立进程程序不会复用构建进程的Registry状态，而是启动后先 `dlopen()` 指定 `.so`，核对Creator
name/version/namespace，再deserialize Engine、创建context、绑定四个输出并调用 `enqueueV3()`。
三次独立进程均成功，输出hash相同；缺失或错误Plugin库、错误namespace/version和非法count均进入明确
错误路径。

## 5. Device QA、Host QA 与正确性证据

固定raw QA目录 `exp16_4_fixture_smoke_20260809_164713` 得到10个候选，index/class/order完全一致，
box/confidence最大误差均为0。synthetic目录 `exp16_5_synthetic_20260809_164952` 覆盖zero、single、
all-8400和boundary/invalid，count分别为0/1/8400/3，所有受测字段零误差，capacity guard有效。

应用构建目录 `exp16_6_app_build_20260809_165659` 同时构建旧Exp11、Exp15和Exp16目标，旧接口编译回归
通过。修正后的5帧Plugin Smoke目录 `exp16_9_rebuild_file_smoke_20260809_170957` 返回0，每帧candidate
count为10，D2H为284 B；检测数、类别、index和顺序与Exp15 B一致，但跨Engine浮点CSV digest不同，
因此按冻结停止条件没有直接进入正式性能实验。

Compute Sanitizer在本Jetson上报告GPU debugging features disabled，故memcheck/initcheck/synccheck/
racecheck记为“环境不支持”，不是PASS，也不表述为覆盖全部Host/Device race。Host侧的RAII、错误路径、
Creator/Registry、独立进程加载和重复生命周期已验证；这些证据不能替代未执行的系统级race检查。

## 6. Dual-output 与无Plugin control诊断

dual诊断目录为 `exp16_10_dual_diagnostic_20260809_172429`。dual ONNX SHA256为
`237d063da0fef1eadd116b69da7bab76c1ea3b3946b64864760c1b1acd1d678b`，Engine为8,712,260 B、
SHA256 `faf828285ba7c657421888e8b8ad9409291a5dc74b8e8e5dcb943a67f42a5f18`。同一次
`enqueueV3()`同时读取上游raw和Plugin输出，得到207个阈值前候选；CPU decode(raw)与Plugin的
count/index/class/order完全一致，box和confidence最大误差均为0。这证明当前Plugin数学本身没有引入
观测到的差异。

dual raw与冻结Exp07 raw的max/mean/relative-L2误差为1.09375/0.0169231/0.00019495。经用户批准又
从原始Exp06 ONNX、相同TensorRT和Exp07显式参数构建无Plugin control，目录
`exp16_11_control_rebuild_20260809_175317`。control Engine为8,902,516 B、SHA256
`57554549bfeeceff17aebe902cb2894d4e5d2617d35dbdfd37f0eb8ff8584fd2`；其raw与冻结Exp07 raw、dual raw
的比较也均超过严格逐值阈值。由此确认普通rebuild/tactic漂移是跨Engine比较中的真实混杂变量，但不能
因此忽略最终部署语义。

## 7. 修订合同与正式停止结果

用户批准后，在正式运行前冻结修订合同：同Engine raw→Plugin继续要求零误差；跨独立Engine改为每个
150帧进程都产生151检测，frame/detection/class/顺序完全一致，源图box最大差不超过2 pixel、confidence
最大差不超过0.005。正式性能顺序冻结为三个独立进程对：`control→plugin / plugin→control /
control→plugin`，每轮Plugin P95退化不得超过5%；任一轮语义失败立即停止。

正式总控目录 `exp16_formal_compare_20260809_183103` 在第一轮停止：

| 路径 | 帧 | 检测 | detections SHA256 | Wall FPS | E2E mean/P95 | D2H B/帧 |
|---|---:|---:|---|---:|---:|---:|
| Exp15 B control | 150 | 151 | `9f3f3345...e34e2d8a` | 64.195 | 14.339/14.846 ms | 263.840 |
| Plugin候选 | 150 | 153 | `2109807a...0e538c8` | 72.024 | 12.625/14.704 ms | 264.213 |

Plugin在frame 27和40各新增一个person检测，confidence为0.250179231和0.250912547，刚越过0.25阈值；
按键对齐后的151个共有行中confidence最大差0.00347364，但box最大差达到138 source pixels。新增检测和
box差均违反正式语义Gate。两个应用及基础validator返回0，semantic comparator和总控返回1；第二、
三轮未启动。表中速度只作为失败现场，不能宣称Plugin加速或P95 Gate通过。

## 8. 失败现场与最小修复

- 第一套虚拟环境因系统缺少`python3.10-venv`失败，现场保留；随后使用项目外用户级target目录，不安装
  系统包；
- 隔离NumPy 2.2.6与系统OpenCV的NumPy 1.x ABI不兼容，误用于跨Engine比较时导入失败；随后使用未修改
  的系统Python完成比较，不升级系统NumPy/OpenCV；
- 首个Engine参数与Exp07不完全一致，修正后在新时间戳目录重建，旧候选不删除；
- 初版语义比较器按CSV行zip，新增检测导致后续错位噪声；改为按检测键对齐并显式报告extra/missing key，
  不改变门槛或正式FAIL；
- Jetson `tegrastats`不支持`--count`，首轮边界采样文件保留真实错误；脚本改为受控1秒`timeout`并记录
  返回码124，独立Smoke约48.7～48.9°C，但不回填为首轮温度；
- 多次远端临时筛选命令因PowerShell/SSH引号层失败，均未修改结果；最终使用无复杂引号的明确frame查询。
- pre-commit首次误把仓库根目录当作CMake入口，`exp16_precommit_build_20260809_185220`配置返回1并保留；
  确认项目采用`plugins/`和`app/`两个独立入口后，在新目录
  `exp16_precommit_build_20260809_185437`重新配置。Plugin库、四个Plugin工具、Exp11旧应用、Exp15应用
  和Exp16应用的Release构建返回码全部为0，冻结Engine和正式Plugin `.so`目录未被覆盖。

## 9. 能力—证据矩阵与下一步

| 能力 | 状态 | 证据/限制 |
|---|---|---|
| IPluginV3、Creator/Registry、GraphSurgeon | IMPLEMENTED | 代码、四输出ONNX与本机构建 |
| 显式workspace、禁止enqueue动态分配 | VERIFIED | fixture/synthetic/Engine运行与代码审计 |
| 独立新进程加载`.so`后deserialize | VERIFIED | 三独立进程、相同输出hash |
| Plugin decode/filter/CUB数学 | VERIFIED | raw fixture、synthetic、dual同Engine零误差 |
| Compute Sanitizer覆盖 | —（未达到VERIFIED） | 平台禁用GPU debugging features |
| 四输出Plugin替换Exp15 B | REJECTED | 150帧151 vs 153，box最大差138 px |
| Plugin正式性能收益 | —（未达到VERIFIED） | 正确性失败后停止，只有一对诊断数字 |
| 简历中的已采用Plugin优化 | REJECTED | 候选未进入部署主线 |

本实验能够作为“实现并验证TensorRT Plugin工程闭环、用同Engine诊断隔离算子数学、最终因系统语义漂移
拒绝部署候选”的学习证据，但不能写成已采用或已加速的项目成果。实验结束当时计划直接进入Exp17；
2026-08-09的当前路线已由下述第10节重校准为先审批窄范围Deployment Semantic Revalidation Gate，再决定
是否进入Exp17。Exp08已经完成的数据分布和tiny/small覆盖不重复从头做。

## 10. Deployment Semantic Revalidation Gate（待审批，不改写原 REJECT）

本节是2026-08-09基于真实结果追加的窄范围后续计划，不是对原Exp16结果的重判。原150帧正式Gate、
`REJECT`、失败目录、检测门槛和“没有正式三轮性能结论”永久保留；Exp15 B继续是当前`ACCEPTED`主线。
本Gate不新增Exp16.1编号，不重写Plugin，也不继续优化CUDA Kernel。

### 10.1 Forensic Gate

先只审计frame27、frame40和旧比较器报告的最大138 source pixels差异，逐级保存并对照：

```text
candidate index / class
network-coordinate raw box / confidence
0.25 threshold crossing
inverse-letterbox geometry / source box
NMS输入、排序、抑制关系与最终检测
```

目标是确认138 px属于真实同候选框漂移，还是`detection_index`/CSV行号对齐导致不同候选被误配的假大差。
跨Engine比较必须改为`image + class + IoU cost`的Hungarian assignment，显式报告matched、unmatched、
extra和missing；CSV行号只作输出顺序信息，不再作为检测身份。

### 10.2 219张模型级比较与 Build Variance

在同一冻结219张test、同一预处理、conf=0.25、class-aware NMS IoU和匹配IoU下比较：

```text
F0  Frozen Exp07 Engine + Exp15 CUB
B1  Fresh rebuilt baseline #1 + Exp15 CUB
B2  Fresh rebuilt baseline #2 + Exp15 CUB
P   Fresh Plugin Engine
```

至少两个普通baseline rebuild用于估计TensorRT build/tactic variance。每条路径报告Precision、Recall、
mAP50、mAP50-95、固定阈值TP/FP/FN、tiny recall、small recall、tiny+small recall、unmatched rate、
matched bbox IoU分布、confidence delta分布和threshold-crossing数量。任何跨Engine差异都必须与B1/B2之间
及其相对F0的正常build波动对照，不能只比较Plugin与冻结Engine。

### 10.3 采用条件

只有Plugin的模型级精度不差于正常baseline rebuild波动，并且随后动态调频paired/interleaved性能、
部署复杂度和维护成本满足预先审批的采用条件时，Plugin主线状态才可改为`ACCEPTED`。否则继续保持：

```text
Plugin engineering/math : IMPLEMENTED + VERIFIED
Original Exp16 result   : REJECTED
Mainline adoption       : REJECTED
Current runtime         : Exp15 B CUB stable compaction
```

Gate执行前必须先追加学习手册实验前规划并获得人工批准；本轮文档重校准本身不启动Gate。
