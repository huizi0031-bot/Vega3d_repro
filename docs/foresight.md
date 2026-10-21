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

## 原版标注与选帧

默认读取 `data/processed/scanqa_train_llava_style.json` 和 `scanqa_val_llava_style.json`。这里沿用原版 VEGA 支持的 `id`、`video`、`conversations`、`metadata` 等字段；`human/gpt` 是问题与标注答案的角色名。无需新增问答字段或生成 `train_paired.jsonl`。

加载层保留 `VideoProcessor` 的 uniform32 观测，并与离线特征中的 `frame_files` 精确对齐。`ScenePairSampler` 按场景自动选择视频监督：优先尝试扫描中部的观测帧作为锚点，目标取保存序列中紧随其后的四张 RGB；目标不与观测重叠，最大相对位移为 0.5m、最大旋转为 30°。阈值位于配置的 `data.max_translation`、`data.max_rotation`。不强制原始编号间隔，实际间隔由已有抽帧决定。

同场景问答复用片段。启动器在 CPU 上检查全部训练场景，保存 `foresight_record/scene_pairs.json`，所有 rank 读取同一快照。该文件只记录选帧，不复制问答；不属于原始数据格式。恢复会核对快照及资源指纹。直接使用底层 Dataset 而未提供快照时，每个进程也会确定性选帧并在内存中复用。

找不到合格片段或缺少训练资产时明确报错，不静默丢样本，也不退回另一条问答。旧样本若带显式 `foresight` 字段，继续验证并使用原配帧，支持同场景多个不同片段。

这是静态扫描的留出视角预测。其他观测可能位于目标之后；原场景几何信息仍可用，不能声称严格未来预测或完整几何留出。历史两步 GPU 试跑使用 400 条训练问答、20 场景和 100 条验证问答、5 场景；全量实验尚未完成。

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
| 训练长度 | 默认全量 1 epoch，max_steps=-1，按实际样本数计算步数 |
| 优化器 | AdamW，betas .9/.999，eps 1e-8，weight decay 0，裁剪 1 |
| 调度 | warmup .03，cosine |
| 内存 | BF16，Qwen/Wan activation checkpointing，ZeRO-3 |
| 保存 | 每 50 steps，末步保存完整 checkpoint；最多保留 2 个 |
| 上下文 | 32768，包括新增 queries |

每个训练进程持有完整冻结教师；教师不参与 ZeRO 分片，也不进入模型 checkpoint。VAE 使用 FP32 参数与原生 BF16 autocast，DiT 使用 BF16，time MLP 为 FP32。离线视觉特征的提取协议与教师低分辨率视频监督是两条独立路径。

## 实验状态与边界

两步 GPU 试跑验证了前后向和保存，不是效果实验。尚需补齐验证特征，完成 baseline / queries / video 对照。没有动作头、机器人训练或 VSI-Bench 适配。预测视角未显式以相机位姿作为条件；监督是否改善空间关系表示需要实验确认。

## 通用训练接口

唯一入口为 `scripts/3d/train/train_scene_understanding.py`，默认配置 `configs/3d/foresight.yaml`。`--mode baseline|queries|video` 控制方法，数据集名称不限定为 ScanQA。三组使用同一份原版标注。

支持 JSON 数组、JSONL 或 datasets YAML。YAML 的相对 `json_path` 以仓库根目录为基准，完整训练使用 `sampling_strategy: all`。原版样本无需附加视频监督字段，例如：

```json
{
  "id": "sample_0",
  "video": "scannet/sceneXXXX_00",
  "conversations": [
    {"from": "human", "value": "<image> Describe the relation between the table and the chair."},
    {"from": "gpt", "value": "The chair is beside the table."}
  ],
  "metadata": {"dataset": "custom_spatial_task"}
}
```

`video` 必须对应完整的 ScanNet/EmbodiedScan 场景资产。仅修改 JSON 外形不会生成图像、深度、位姿、几何参数或离线特征。当前共享范围是单轮文本生成，定位 proposal、DPO、量化/LoRA 和连续动作输出尚未接入。

```bash
python scripts/3d/train/train_scene_understanding.py --mode video --check-data --audit-file results/data_audit.json
python scripts/3d/train/train_scene_understanding.py --mode video --check-loader
python scripts/3d/train/train_scene_understanding.py --mode video --gpus 2,3 --run-name foresight --execute
```

`--data-path` 和 `--validation-path` 覆盖标注路径。`data.validation` 可省略；`require_disjoint_validation_scenes` 控制场景分离。指定验证标注不表示训练时自动评估。

预检查按场景汇总 RGB、深度、位姿、几何参数和缓存错误，并检查缓存头中的协议、张量形状、帧顺序；不扫描整个特征张量。训练资源有错误则退出码为 1，拒绝开训；验证错误单独报告。`--check-loader` 进一步读取一条真实样本，运行 collator 并检查张量，不训练。

运行保存配置、标注与源码哈希、选帧及资产指纹。大特征文件使用缓存头校验及大小/修改时间指纹，不逐字节哈希；RGB/深度保存文件状态，位姿另计算内容哈希。修改数据、元数据、配置或选帧后应新建实验，不直接恢复旧实验。

`prepare_foresight_data.py` 保留为可选的历史子集与配帧清单导出工具，其场景筛选行为不用于默认全量训练。
