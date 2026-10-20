# ScanQA foresight 实现

## 数据流

```text
32 帧 RGB ── SigLIP ── semantic projector ──┐
32 帧离线 Wan 特征 ── generative projector ─┤
                                           ↓
                              token-gated 融合 + 3D 编码
                                           ↓
Qwen 输入：[视觉 / 问题 / assistant 前缀] [32 queries] [答案]
                                           ↓
                       ┌───────────────────┴────────────────┐
                       ↓                                    ↓
                 Z [B,32,2048]                     答案预测 → QA loss
                       ↓ LayerNorm + Linear
                 C [B,32,4096]
                       ↓
                  冻结 Wan ← 带噪视频 latent、timestep
                       ↓
              velocity 预测 → video flow loss
```

queries 是连续可学习参数 `[32,2048]`，不增加词表，不与 32 帧逐一对应。同一组 queries 随上下文产生不同 Z。queries 位于第一个答案标签前，因果计算保证 Z 不能读取答案；后续答案位置可读取 queries。标签 `-100` 控制文本损失，因果遮罩控制信息可见性，两者作用不同。

推理时只在完整 prompt 后插入一次 queries，再自回归生成答案。新增视频教师不加载；原 VEGA 的离线 Wan 特征仍是输入依赖。

## 张量尺寸

| 阶段 | 单样本尺寸 |
|---|---|
| SigLIP RGB | `[32,3,384,384]` |
| SigLIP patch 特征 | `[32,729,1152]` |
| 语义投影与空间下采样 | `[32,196,2048]` |
| 离线 Wan 特征 | `[32,3072,14,14]` |
| 生成式特征投影 | `[32,196,2048]` |
| 融合、3D 编码、网格换行后 | `[6720,2048]` |
| Qwen query 位置输出 Z | `[1,32,2048]` |
| Wan 条件 C | `[1,32,4096]` |
| 教师 RGB | `[1,3,5,256,320]` |
| VAE latent | `[1,48,2,16,20]` |
| Wan 视频 tokens | `[1,160,3072]` |
| velocity 预测 | `[1,48,2,16,20]` |

Wan 按原接口将条件补零到 512 个位置，再投影到内部 3072 维。RGB 五帧由时序 VAE 联合编码；第一帧保留一个 latent 时间位置，后四帧压缩成第二个位置。

## 数据与选帧

训练 400 条问答来自 20 个场景，每场景 20 条；验证 100 条来自 5 个独立场景。一条 JSONL 是原问答对象加 `foresight` 字段。原问答包含 `id`、`video`、`conversations` 和 `metadata`；其中 `human/gpt` 是问题/标准答案的角色标签。

`foresight` 记录 32 帧观测路径、离线特征路径、锚点、四帧目标、五帧完整顺序、原始帧编号及位姿路径。它不存 queries、Z、噪声或 velocity。

选帧先在观测中尝试靠近扫描中部的锚点，再取保存序列中紧随其后的四帧。相邻原始编号差 10；目标与 32 帧 RGB 观测不重叠，相对锚点最大位移不超过 0.5m、最大旋转不超过 30°。每场景选一段并固定，同场景问答复用该片段。

这是静态扫描的留出视角预测。其他输入视角可能位于目标之后；原场景几何信息也仍然可用，不能据此声称严格未来预测或完整几何留出。

## 视频损失

五帧 RGB 缩放/居中裁剪至 256×320，归一化到 `[-1,1]`。冻结 VAE 在 `no_grad` 下生成 clean latent `y`；首个 latent 时间片保留干净值，只在第二个时间片加噪并计算损失。

```text
u ~ Uniform(0,1)
sigma = 5u / (1 + 4u)
x_sigma = (1-sigma)y + sigma*epsilon
velocity_target = epsilon-y
timestep = 1000*sigma；锚点位置为 0
L_video = 目标 latent 区域的 velocity 均方误差
L_total = L_QA + 0.1*L_video
```

每个样本前向预测一次 velocity，不运行多步采样，不解码 RGB。Wan 参数冻结，但保留对 C 的梯度；梯度经过 C、投影、Z 传回 Qwen、queries 及相关可训练视觉模块。Z 是前向中间结果，不是被单独持久优化的参数。

## 默认配置

| 项目 | 配置 |
|---|---|
| 语言骨干 | Qwen3.5-2B language-only，hidden 2048 |
| 语义视觉 | SigLIP SO400M patch14 / 384 |
| 视频教师 | Wan2.2 TI2V-5B，30 blocks，hidden 3072 |
| 原视觉输入 | 32 帧，离线特征 3072×14×14 |
| Queries | 32×2048；65,536 参数 |
| 条件投影 | LayerNorm + Linear 2048→4096；8,396,800 参数 |
| 新增参数合计 | 8,462,336 |
| 可训练范围 | 语言、语义视觉、桥接/融合等原可训练部分及新增模块 |
| 冻结范围 | 视频教师 DiT、VAE |
| 学习率 | 一般参数 1e-5；视觉 2e-6；foresight 1e-4 |
| Batch | micro=1，global=8，累积=8/GPU 数 |
| 训练长度 | 当前小规模数据 1 epoch，约 50 steps |
| 优化器 | AdamW，betas .9/.999，eps 1e-8，weight decay 0，裁剪 1 |
| 调度 | warmup .03，cosine |
| 内存 | BF16，Qwen/Wan activation checkpointing，ZeRO-3 |
| 保存 | 每 50 steps，末步保存完整 checkpoint；最多保留 2 个 |
| 上下文 | 32768，包括新增 queries |

每个训练进程持有完整冻结教师；教师不参与 ZeRO 分片，也不进入模型 checkpoint。VAE 使用 FP32 参数与原生 BF16 autocast，DiT 使用 BF16，time MLP 为 FP32。离线视觉特征的提取协议与教师低分辨率视频监督是两条独立路径。

## 当前边界

两步 GPU 试跑验证了前后向和保存，不是效果实验。尚需补齐验证特征，完成 E00/E01/E03 对照。没有动作头、机器人训练或 VSI-Bench 适配。预测视角未显式以相机位姿作为条件；监督是否改善空间关系表示需要实验确认。
