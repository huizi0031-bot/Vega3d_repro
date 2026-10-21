# VEGA-3D · Qwen3.5 / Wan2.2

在 VEGA-3D 上使用 Qwen3.5-2B 语言骨干与 Wan2.2-TI2V-5B 生成式特征，保留 SigLIP、特征融合、3D 位置编码及原物体定位头。foresight 是共享的可选训练方法，数据集通过清单配置，不为每个任务复制训练入口。

## 目录

```text
llava/
  model/                         # 模型、融合、3D 编码与 foresight
  train/                         # 数据集、collator、Trainer
  eval/                          # 任务推理和指标
  video_utils.py                 # 场景资产及离线特征读取
scripts/
  3d/preprocessing/              # 各数据集转换和 foresight 配帧
  3d/train/                      # 统一训练入口、参数与数据检查
  3d/eval/                       # 场景问答生成及原版任务入口
  visualization/                 # Wan 特征可视化
  zero3.json
configs/3d/                      # 基线和 foresight 方法配置
wan_offline_tools/               # 生成式特征提取、训练兼容入口
environment/qwen35/              # 完整依赖、离线安装、环境检查
tests/                           # CPU 测试
docs/                            # 方法和实验记录
data/                            # 数据、权重、特征；不进 Git
```

## 已做的准备

| 内容 | 状态 |
|---|---|
| 语言模型 | Qwen3.5 原生语言骨干、tokenizer、聊天模板；视觉塔仍是 SigLIP |
| 生成式特征 | Wan2.2 离线特征为 3072 维，与语义特征融合并加入 3D 位置编码 |
| 本地权重 | Qwen3.5-2B-Language、SigLIP SO400M、Wan2.2 DiT / VAE 已准备 |
| 方法 | 32 个 queries、条件投影、冻结 Wan / VAE、flow loss、文本损失、保存与恢复 |
| 数据实例 | ScanQA 400 条训练问答 / 20 场景，100 条验证问答 / 5 独立场景 |
| 观测与监督 | 32 帧观测；教师片段为观测锚点加 4 帧留出视角；清单已固定 |
| 离线特征 | 20 个训练场景齐备；5 个验证场景待补 |
| Python 环境 | 离线包准备完成，独立环境重建和 CPU 检查通过，见 [环境记录](docs/environment_validation.md) |
| GPU 记录 | 重构前在 8709 双卡完成 2 个优化器步骤并保存；未完成完整训练或效果评测 |
| 原版其他任务 | 数据转换、混合清单、推理和指标代码保留；新骨干适配待验证 |

## 方法

```text
RGB → SigLIP ───────────────┐
Wan2.2 离线特征 ───────────┴→ 融合 + 3D 编码 → Qwen3.5 → 文本回答 → 文本损失
                                              └→ foresight Z → 条件投影 ─┐
留出视角片段 → 冻结 VAE → 加噪 latent ─────────────────────────────────┴→ 冻结 Wan
                                                                         ↓
                                                              velocity → flow loss
```

默认 Z 为 `32×2048`，条件为 `32×4096`，教师视频为 5 帧 `256×320`。损失为 `L_text + 0.1 L_video`。教师参数冻结，但保留对条件输入的梯度。推理不加载新增视频教师；原有离线 Wan 观测特征仍需提供。

当前目标是静态场景的留出视角预测，不能称为严格未来动态预测。当前 foresight 分支支持单轮文本生成，定位 proposal 分支、DPO 和机器人动作头尚未接通。场景资产仍使用 ScanNet / EmbodiedScan 的布局；数据集名称不受限，新数据仍需满足该读取接口。方法、张量与数据契约见 [foresight](docs/foresight.md)。

## Python 环境

Python 3.10.18、Torch 2.6.0 / CUDA 12.4、Transformers 5.3.0、DeepSpeed 0.16.9、Accelerate 1.12.0、Diffusers 0.30.3，使用 SDPA。`requirements.txt` 引用 `environment/qwen35/requirements.lock.txt` 的完整依赖清单。

8709 原实验环境：

```bash
source /data/zhdu/vega3d_repro/envs/vega3d-qwen35/bin/activate
```

公司服务器复制代码、数据、权重和离线包后，按实际位置替换路径：

```bash
bash environment/qwen35/install.sh \
  --prefix /path/to/envs/vega3d-qwen35 \
  --bundle /path/to/environment_bundle/qwen35 \
  --test --check-data
source /path/to/envs/vega3d-qwen35/reproduction/activate.sh
```

离线包位于 8709 的 `/data/zhdu/vega3d_repro/environment_bundle/qwen35/`。安装说明见 [environment](environment/qwen35/README.md)。仅检查 Python 环境可省略 `--check-data`；数据检查使用默认方法配置。CUDA 驱动与运行时编译工具链需由目标机器提供。

## 数据与配置

```text
data/
  models/{Qwen3.5-2B-Language,siglip-so400m-patch14-384,Wan2.2-TI2V-5B}/
  embodiedscan/                  # 场景元数据
  metadata/                     # 框、采样等信息
  processed/                    # 原任务标注
  scanqa_foresight/              # 现有实验清单，换任务可另设目录
    train_paired.jsonl
    train_original.json
    val_original.json
    scene_pairs.json
  scannet/
    posed_images/<scene>/       # RGB、深度、位姿、内参
    <feature_id>/<scene>/wan22_ti2v.pt
```

现有 `data/scanqa_foresight` 是软链接，目标为 `/data/zhdu/vega3d_repro/experiments/scanqa_foresight_pilot_20261020_v1/data`。交接时需复制真实目标，不能删除整个 `experiments`。缓存 ID 和帧列表保持一致，不直接改名。

`configs/3d/foresight.yaml` 默认指向这些已准备的数据。换任务修改 `data.train`、可选的 `data.validation`，其余模型和优化参数按需要调整。训练数据支持 JSON、JSONL 和多数据集 YAML。`conversations` 中的 `gpt` 是标准答案角色名，不表示答案由 GPT 生成。

## 训练

所有命令在仓库根目录执行。统一入口为 `scripts/3d/train/train_scene_understanding.py`。

| `--mode` | 设置 |
|---|---|
| `baseline` | 原模型，不加 queries 和视频监督 |
| `queries` | 加 queries，只用文本损失 |
| `video` | 加 queries，同时使用文本损失与视频 flow loss |

三种模式可用同一份带配帧的数据，关闭视频监督时忽略配帧字段。无需为每个任务新建启动器。

```bash
# CPU 数据和真实样本检查
python scripts/3d/train/train_scene_understanding.py --mode video --check-data
python scripts/3d/train/train_scene_understanding.py --mode video --check-loader

# 预览双卡两步命令，不启动训练
python scripts/3d/train/train_scene_understanding.py --mode video --gpus 2,3 --run-name foresight-test --max-steps 2

# 用户决定训练时，另加 --execute；正式训练不传 --max-steps 2
# 换同格式任务：--data-path /path/to/train.jsonl
# 对照：--mode baseline 或 --mode queries
```

批量默认为每卡 1、全局 8，两卡累积 4 次；训练 1 epoch。通用多任务基线使用 `--config configs/3d/qwen35_wan22.yaml`，默认读取原 `multi.yaml` 并保留定位头。foresight 与定位头尚未接通，入口拒绝不支持的组合。检查范围、恢复规则见 [训练说明](scripts/3d/train/README.md)。

## 数据准备与评估

`prepare_foresight_data.py` 从指定的训练 / 验证问答、观测缓存及 ScanNet 帧中准备留出片段，路径、场景数和采样参数由命令指定，详见 [方法与数据](docs/foresight.md)。已有清单可继续使用，无需重新生成。

验证特征检查也由配置的数据清单驱动：

```bash
python wan_offline_tools/extract_scene_features.py --split validation --check-only
# 提取需用户选空卡后另加 --gpu 0 --execute
```

当前缺 `scene0011_00`、`scene0064_00`、`scene0084_00`、`scene0249_00`、`scene0377_00`；缺失时检查退出码为 1。

文本回答生成共用入口，计分仍按任务分别执行：

```bash
python scripts/3d/eval/eval_scene_qa.py --model-path checkpoints/scene-video --question-file data/scanqa_foresight/val_original.json --answer-file results/answers.jsonl
# 实际推理另加 --gpu 0 --execute
python llava/eval/eval_scanqa.py --input-file results/answers.jsonl
```

ScanQA 计分另需原始多参考答案标注；METEOR 需要 Java。其他任务的原版转换和评估保留于 `scripts/3d/preprocessing/` 与 `llava/eval/`，其聊天模板、Ray 调度和定位分支尚需适配，见 [评估说明](scripts/3d/eval/README.md)。

离线特征可用 `scripts/visualization/wan22_features.py` 绘制，见 [可视化说明](scripts/visualization/README.md)。相似度图不能单独证明几何对应正确。

重构前试跑数值见 [GPU 记录](docs/foresight_gpu_smoke_20261020.md)。没有完整训练、三组对照或验证指标，不据此判断方法效果。来源与许可证见 [THIRD_PARTY](THIRD_PARTY.md)。
