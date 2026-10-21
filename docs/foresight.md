# Foresight 场景理解训练

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

## 当前数据实例与选帧

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

## 当前示例的默认配置

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

## 实验状态与边界

两步 GPU 试跑验证了前后向和保存，不是效果实验。尚需补齐验证特征，完成 baseline / queries / video 对照。没有动作头、机器人训练或 VSI-Bench 适配。预测视角未显式以相机位姿作为条件；监督是否改善空间关系表示需要实验确认。

## 通用训练接口

唯一训练入口为 `scripts/3d/train/train_scene_understanding.py`，默认配置 `configs/3d/foresight.yaml`。`--mode baseline|queries|video` 控制方法，不依赖 `metadata.dataset` 必须为 ScanQA。默认的 ScanQA 路径只是现成的数据实例。

训练清单可以是 JSON 数组、JSONL 或 datasets YAML。YAML 的相对 `json_path` 以仓库根目录为基准，可混合多个文本生成数据集，使用 `sampling_strategy: all` 或 first/end/random 抽样。数据源内的样本结构为：

```json
{
  "id": "sample_0",
  "video": "scannet/sceneXXXX_00",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe the relation between the table and the chair."},
    {"from": "gpt", "value": "The chair is beside the table."}
  ],
  "metadata": {"dataset": "custom_spatial_task"},
  "foresight": {
    "video": "scannet/sceneXXXX_00",
    "target_mode": "heldout_view_clip_prediction",
    "observed_frames": ["data/.../observed_0.jpg", "data/.../observed_1.jpg"],
    "anchor_frame": "data/.../observed_0.jpg",
    "teacher_clip_frames": ["data/.../observed_0.jpg", "data/.../target_0.jpg", "data/.../target_1.jpg", "data/.../target_2.jpg", "data/.../target_3.jpg"],
    "heldout_target_frames": ["data/.../target_0.jpg", "data/.../target_1.jpg", "data/.../target_2.jpg", "data/.../target_3.jpg"]
  }
}
```

示例省略了其余观测，`observed_frames` 实际数量必须等于配置 `data.frame_count`，顺序必须与 loader 实际选帧及缓存一致。`video` 要能在场景元数据中找到；仅修改字符串或 JSON 外形不会生成图像、深度、位姿或离线特征。

`baseline` / `queries` 不要求 `foresight` 字段，不要求原始清单与配帧清单成对存在。`video` 要求每个训练样本都有该字段。支持同一场景的不同样本选择不同监督片段，不再强制固定帧编号间隔；`teacher_clip_frames` 的顺序由数据准备方负责，训练不会随机重排。锚点必须属于观测、目标必须与观测不重叠。保留兼容原清单的 `heldout_view_clip_prediction_not_strict_future` 协议名，不据此升级为严格未来预测。

配置 `data.validation` 可省略，场景是否分离由 `require_disjoint_validation_scenes` 决定；提供验证清单不代表训练时自动评估。训练前检查要求训练特征齐全，报告缺失验证特征但不阻止仅训练。

当前共享范围是 ScanNet/EmbodiedScan 场景资产上的单轮文本生成任务，如问答和按提示生成描述。对象 proposal 定位、DPO 和连续动作输出尚未与 foresight 分支连接，不能仅换标注就视为支持。

## 准备其他文本任务的留出片段

```bash
python scripts/3d/preprocessing/prepare_foresight_data.py \
  --project /absolute/path/to/repo \
  --train-file data/processed/task_train.json \
  --validation-file data/processed/task_val.json \
  --feature-id wan22_ti2v_offline_uniform32_1280x704_t300_b20_s42_v1 \
  --output /absolute/path/to/new_manifests
```

该工具沿用 ScanNet 数字帧名、位姿和既有观测缓存选取片段；`--train-scenes`、`--val-scenes`、`--examples-per-scene` 可限制规模，默认使用符合要求的所有场景/样本。`--frames`、`--teacher-frames`、相机运动阈值可调；`--frame-id-stride` 可选，不指定时不强制帧编号间隔。已有数据沿用原固定清单，不在此次重构中重新选帧。

对其他资产布局或非数字文件名，可自行准备上述显式配帧列表，再适配底层场景读取；运行时配帧校验不要求数字文件名。


