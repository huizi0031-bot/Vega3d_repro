# ScanQA foresight 第一版实现与配置记录

日期：2026-10-20。工程：`VEGA-3D-qwen3.5-wan2.2`。

**状态：实现训练链路；本次不启动实验训练。** CPU 小模型接口检查与真实大模型 GPU 验证必须分开记录。本文的默认值是第一轮工程设置，不是优化过的论文超参数。完整机器可读配置在 `configs/foresight/scanqa_pilot.yaml`，验证结果在 `docs/foresight_verification_20261020.md`。

## 1. 本方案实际计算什么

```text
原有 32 帧 RGB / 深度 / 3D 信息 + 问题
    ├─ SigLIP → 原语义 projector ────────────────────┐
    └─ 原有离线 Wan 特征 → generative projector ────┤
                                                    ↓
                              原 VEGA 融合 + 3D 位置编码
                                                    ↓
Qwen 输入：[系统/视觉/问题/assistant 前缀] [32 个可学习 queries] [标准答案，仅训练]
                                                    ↓
                              Qwen3.5 语言骨架，因果前向
                                 ├─ 答案位置 → LM head → L_QA
                                 └─ query 位置最后层输出 Z：[B,32,2048]
                                                    ↓
                              LayerNorm + Linear(2048,4096)
                                                    ↓
                              Wan 条件 C：[B,32,4096]
                                                    ↓
冻结 Wan DiT ← 带噪视频 latent + token-wise timestep
      ↓
预测 velocity → 与 noise − clean latent 比较 → L_video
      ↓
梯度穿过冻结 Wan → 条件投影 → Z → queries / 可训练 Qwen / 原视觉桥接等
```

queries 是连续可学习参数 `[32,2048]`，不是新词表 ID，也不要求模型自回归输出 32 个离散词。每个样本使用同一组参数，但经由各自上下文计算出不同的 Z。槽位标签全部为 `IGNORE_INDEX=-100`，不对 queries 计算文本交叉熵。

训练时在第一个有效答案标签前插入 queries；Qwen 原生 assistant 前缀（包括关闭 thinking 时模板产生的前缀）仍在它们前面。因果掩码保证 Z 看不到随后标准答案。推理时在相同的完整 generation prompt 后插入 queries，然后生成答案。连续 queries 不会重新插入到每一个自回归解码步。

QA 通过同一次 Qwen 前向读取 queries 的上下文状态，不另加 action expert，也不让 Wan 再提取 Z 的特征。**本次实现的是理解任务分支，尚未实现机器人动作头。**

## 2. 视频目标、VAE 与 flow loss

数据使用已准备的 400 条训练 QA / 20 个训练场景，以及官方验证集中的 100 条 QA / 5 个独立场景。一个训练场景对应一个五帧片段，故 400 条 QA 不等于 400 个独立视频目标。原始标注、共享数据和离线特征不修改。

1. QA 输入保持原有 32 帧及顺序。新增 loader 再次检查实际采样结果与配对 manifest 完全一致。
2. teacher 的五帧是：一个已在 QA 输入中的 anchor，随后四个不在 32 帧 RGB 输入中的帧。只送往 teacher，不进入 SigLIP、QA video_dict 或文本上下文。
3. 每帧以 RGB 读取，bicubic 居中裁剪/缩放至 **H=256,W=320**，归一化到 `[-1,1]`，形成 `[B,3,5,256,320]`。
4. 使用 Wan2.2 原生 48 通道时序 VAE **联合编码整个片段**，不是五张图分别编码。VAE stride `(4,16,16)`，预期 latent `[B,48,2,16,20]`。
5. 首个 latent 时间片保留 clean 值，timestep 设为 0；只对后续 latent 加噪和计算损失。由于 VAE 时序压缩，四帧目标对应后一个 latent 时间片，不是一帧一个 latent。
6. 在 native DiT patch `(1,2,2)` 后，本设置有 `2×8×10=160` 个视频 tokens，其中首 80 个为 anchor tokens。

明确记录本版采样分布：

```text
u ~ Uniform(0,1)
sigma = 5u / (1 + 4u)          # shift=5，数值裁剪至 [1e-5,1-1e-5]
x_sigma = (1-sigma) z + sigma epsilon
target_velocity = epsilon - z
t = 1000 sigma               # anchor tokens 的 t=0
L_video = 目标 latent 区域上预测 velocity 的均方误差
L = L_QA + 0.1 L_video
```

这里沿用 Wan 的 latent、velocity 参数化和 TI2V 的逐 token 时间/首帧条件接口；**uniform-u 分布、shift 和权重是本方案显式选择，不宣称复现 Wan 未公开的完整预训练配方。** 不运行多步采样，不解码未来 RGB，不计算像素重建损失，不加载 UMT5。投影后的 32 个连续向量直接走 Wan 原生 text-context 接口，Wan 内部仍按其原始 `text_len=512` 补零。

Wan DiT 与 VAE 都冻结。VAE 编码使用 `no_grad`；DiT 前向保留对条件 C 的梯度，不能用旧特征提取器的 `inference_mode` 或 `.detach()` 包住此路径。启用 DiT block 的非重入 activation checkpointing，减少反向传播所需激活。

## 3. 参数与训练设置

| 项目 | 默认设置 |
|---|---|
| 初始化语言权重 | `data/models/Qwen3.5-2B-Language`，语言骨架，不是完整 Qwen VL |
| 语义视觉 | 原 SigLIP SO400M / patch14 / 384 |
| 原 Wan 输入特征 | 32 个独立视角，1280×704，t=300，block=20，3072×14×14，现有离线缓存 |
| 原融合/位置编码 | token_gated；avg-discrete-sin3d；原 projector 和视觉路径保留 |
| foresight queries | 32×2048 = 65,536 个参数 |
| 条件投影 | LayerNorm(2048) + Linear(2048,4096)，8,396,800 个参数 |
| 新增总参数 | 8,462,336；不包括冻结 teacher |
| teacher | Wan2.2-TI2V-5B；3072 hidden，30 blocks，24 heads，48 latent channels |
| teacher 精度 | DiT bf16，time embedding/projection fp32；VAE fp32 参数 + 原生 bf16 autocast |
| 可训练部分 | 原视觉塔/桥接/融合/语言等按 VEGA tunable parts；新增 queries/投影开启梯度 |
| 学习率 | 原语言/一般参数 1e-5；视觉塔 2e-6；queries 和条件投影 1e-4 |
| batch | 每 GPU 1；global 8；累积步数 = 8 / GPU 数，必须整除 |
| 轮次 | 1 epoch；max_steps=-1；可在显式启动时覆盖 |
| 优化 | AdamW torch；betas=(0.9,0.999)，eps=1e-8，max_grad_norm=1；weight decay 0；warmup 0.03；cosine；seed/data_seed=42 |
| 上下文 | 32768；queries 也计入预算，超出即报错，不静默裁掉 queries 或答案 |
| 内存 | bf16；Qwen gradient checkpointing；Wan block checkpointing；ZeRO-3；torch_compile=false |
| 数据加载/采样 | workers=0；不启用原多任务 megabatch 丢弃采样；不 drop last |
| 记录/保存 | 每 optimizer step 日志，分别记 QA/video loss；每 50 步 checkpoint，最多保留 2 个；成功结束另保存完整 checkpoint 和 final model |

teacher 作为每个进程单独持有的运行时对象，**不交给 ZeRO 分片、不进入优化器、不打包进训练 checkpoint**。因此每张训练 GPU 仍需容纳完整 frozen Wan；bf16 DiT 约 10 GB，加 VAE、激活和 VEGA 后实际显存需由用户启动时验证。本次没有据此承诺某种显卡/卡数一定能跑。

本配置从语言预训练权重初始化原视觉桥接；三个实验必须用同一初始化。若改用已经训练过的完整 VEGA checkpoint，应同时修改三组实验的 `model.initial_checkpoint`，使用能被现有工厂识别的 `qwen35` 路径，并记录来源，不能给新方案单独使用更强初始化。

## 4. 三组配置与启动方式

| 实验 | queries | 额外视频损失 | 数据 |
|---|---|---|---|
| E00 | 无 | 无 | 原格式的相同 400 条 QA |
| E01 | 有 | 无 | 同一批 QA；条件投影只有零权重占位，不从视频学习 |
| E03 | 有 | 0.1×flow loss | 对应的 paired QA；QA 输入仍是同一 32 帧 |

服务器项目根目录：
`/data/zhdu/vega3d_repro/code/VEGA-3D-qwen3.5-wan2.2/`

使用 Qwen3.5 环境的 Python。以下预览和数据检查**不启动训练、不占用 GPU、不会生成训练目录**：

```bash
cd /data/zhdu/vega3d_repro/code/VEGA-3D-qwen3.5-wan2.2
PY=/data/zhdu/vega3d_repro/envs/vega3d-qwen35/bin/python
$PY scripts/3d/train/train_scanqa_foresight.py --experiment E03 --check-data
$PY scripts/3d/train/train_scanqa_foresight.py --experiment E03 --check-loader
$PY scripts/3d/train/train_scanqa_foresight.py --experiment E00
```

**只有用户决定开始时**，才显式加 `--execute`。下面的显卡编号仅作示例，不代表当前空闲卡或已获运行决定：

```bash
# 用户决定启动后，先选实际显卡；不传 --execute 永远只预览。
$PY scripts/3d/train/train_scanqa_foresight.py --experiment E03 --gpus 0,1 \
  --run-name qwen35-foresight-e03-first --max-steps 2 --execute

# 对照组分别使用 E00 / E01 和不同 run-name，保持其他设置相同。
# 中断恢复要求显式给相同 run-name、实验和配置，并保留 checkpoint-*。
$PY scripts/3d/train/train_scanqa_foresight.py --experiment E03 --gpus 0,1 \
  --run-name qwen35-foresight-e03-first --max-steps 2 --resume --execute
```

第一条真实启动建议先由用户决定进行极短的验证，再决定更长运行；本次交付未执行这些命令。不要把两步训练的 checkpoint 直接当成完整实验结果。

启动时输出目录必须不存在，恢复必须有 checkpoint；不会覆盖一个已有实验，也没有自动后台任务/自动重试。默认输出到 `/data/zhdu/vega3d_repro/checkpoints/<run-name>/`。`foresight_record/` 内自动保存：

- 完整解析配置、实验编号、命令行、GPU 编号、数据 SHA-256 和检查结果；
- Git commit、工作区状态、源码 SHA-256、依赖版本；
- 实际可训练参数与设备信息（训练入口初始化时记录）；
- 逐步日志和退出状态；恢复操作另存记录。

`--check-loader` 会在隐藏 GPU 的情况下，读取一条真实训练记录并通过原数据处理器与 collator；不加载语言/视频大模型，不更新参数。`max_steps` 可在恢复时显式延长，其他配置、源码、数据和 GPU 进程数必须保持一致。若要改 loss、queries、分辨率或初始化，另起 run-name。源码/config/docs 进入版本管理；数据、模型权重、运行目录、日志不上传。

## 5. 推理与检查边界

保存的 config 和权重包含 queries 与条件投影。加载 `LlavaQwen35ForCausalLM` 后，现有 `generate` 会自动在 prompt 后插入 queries，不需要额外给出目标视频或 foresight token ID。训练 teacher 只在有标签、训练模式且视频权重大于零时惰性加载，推理不会因为 config 保留权重 0.1 而加载它。

但原 VEGA 的 Wan 输入特征路径仍保留。因此“部署不用额外监督 Wan”**不等于**“整个系统再也不依赖 Wan 特征”。验证仍需按原协议获得验证场景的离线特征；数据预检查会单独报告缺失场景，不自动启动提取，也不会拿训练场景缓存代替。

独立评测入口 `scripts/3d/eval/eval_scanqa_foresight.py` 使用真实 Qwen3.5 chat template，默认确定性 greedy、单 beam、最多 512 个新 tokens，输出兼容原 ScanQA 指标脚本的 JSONL。没有 Ray、旧 Qwen 模板或隐式调用五任务 smoke。以下示例也默认只预览；实际模型路径在用户训练完成后才存在：

```bash
$PY scripts/3d/eval/eval_scanqa_foresight.py \
  --model-path /data/zhdu/vega3d_repro/checkpoints/qwen35-foresight-e03-first \
  --answer-file /data/zhdu/vega3d_repro/experiments/scanqa_foresight_pilot_20261020_v1/e03_predictions.jsonl
# 缓存齐备、用户决定运行时，再显式加 --gpu 0 --execute。
```

该入口保留训练时的 prompt 空白与 assistant 前缀，且先检查验证缓存是否完整。模型与数据协议一起存入 checkpoint config；不会用旧评测的固定 vocab 大小覆盖模型配置。已有结果文件拒绝覆盖。

CPU 检查使用真实微型 Qwen3.5 和微型 native Wan DiT 验证梯度、因果边界、首帧 mask、CE 等价、checkpoint 重算和保存/重载/自回归插入次数，不执行优化器步。它不能替代真实预训练 Wan + VAE 的 GPU 数值与显存检查，也不能证明 ScanQA 有提升。

## 6. 当前研究限制

- 片段是留出视角，其他 QA 输入可能在目标之后。名称保持 `heldout_view_clip_prediction_not_strict_future`，不能宣传为严格未来因果预测。
- 原完整场景几何仍用于 3D 位置表示；“目标 RGB 未输入”不代表完整 3D 信息也被留出。对照组使用相同几何协议。
- 暂未把未来相机位姿作为条件，因此目标视角存在歧义；视频损失是否学到有利于 QA 的空间表示，需要实验验证。
- 256×320 低于原 Wan 工作分辨率，是降低首轮工程验证成本的选择，尚未验证最优性与迁移质量。
- 没有增加动作数据、动作头、策略部署或论文指标结果。VSI-Bench 场景重叠与泛化评估仍需后续单独设计。

## 7. 代码位置

| 路径 | 职责 |
|---|---|
| `llava/model/foresight.py` | 连续 query 参数、插入/掩码/位置、仅有效答案 CE |
| `llava/model/language_model/llava_qwen.py` | Qwen 主前向、取 Z、组合损失、保存与推理 |
| `llava/model/multimodal_generative_encoder/wan22_foresight.py` | 独立 frozen teacher、VAE、flow 加噪与目标/损失 |
| `.../wan22/modules/model.py` | native DiT 中默认关闭的 block checkpointing 开关 |
| `llava/train/foresight_data.py` | 配对校验与 teacher-only RGB 读取 |
| `llava/train/train_3d.py` | 参数注册、数据/collator 接线、训练参数状态记录 |
| `llava/train/llava_trainer.py` | 新模块学习率组与分项 loss 日志 |
| `configs/foresight/scanqa_pilot.yaml` | 三组完整实验设置 |
| `scripts/3d/train/train_scanqa_foresight.py` | 默认预览、只读预检查、显式启动与实验记录 |
| `scripts/3d/eval/eval_scanqa_foresight.py` | 原生 Qwen3.5 模板的独立 ScanQA 推理，默认预览 |
| `tests/test_foresight.py` | 无 optimizer step 的 CPU 架构检查 |

旧 smoke 工具继续服务旧流程；新入口只复用已有的 checkpoint / RNG 兼容入口，不依赖五任务 smoke 执行器。
