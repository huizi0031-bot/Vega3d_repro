# VEGA-3D · Qwen3.5 / Wan2.2

本项目在 VEGA-3D 中加入 foresight queries，用冻结视频生成模型的监督训练 VLM 表示。当前要验证：在相同数据和训练设置下，视频生成先验能否提升空间理解能力。理解实验完成后，再考虑扩展到动作任务。

三组实验共用同一训练入口和数据清单：

| 模式 | Queries | 视频辅助监督 | 比较目的 |
|---|---|---|---|
| `baseline` | 无 | 无 | 原模型基线 |
| `queries` | 有 | 无 | 检查增加 queries 本身的作用 |
| `video` | 有 | 有 | 检查视频监督是否带来额外提升 |

三组均保留原有 Wan 观测特征和 3D 编码，实验只改变 queries 和视频辅助监督。

## 实现与进展

语言骨干为 Qwen3.5-2B，语义视觉塔保留 SigLIP SO400M，生成式观测特征来自 Wan2.2-TI2V-5B。RGB 特征与离线 Wan 特征融合，结合深度和相机参数生成 3D 位置编码后输入 VLM。

```text
场景观测 → SigLIP / 离线 Wan 特征 → 融合与 3D 编码 → Qwen（问题 + queries）→ 答案
                                                        ↓
                                                   Z → 条件投影
                                                        ↓
监督片段 → 冻结 VAE → 加噪 latent ───────────────────→ 冻结 Wan → velocity
```

32 个 queries 在 Qwen 中得到 `Z [B,32,2048]`，投影为 `[B,32,4096]` 的 Wan 条件。教师输入是 1 帧观测锚点和 4 帧留出视角，分辨率为 `256×320`。总损失为 `L_text + 0.1 × L_video`。Wan / VAE 参数冻结，视频损失通过条件分支回传到投影、queries 和相关可训练模块。queries 位于答案前，不能读取答案内容。

推理保留 queries，不加载视频监督教师；原有离线 Wan 观测特征仍需提供。当前监督是静态扫描的留出视角预测。

已完成模型接入、三种训练模式、数据检查、保存与恢复接口，以及 CPU 回归检查。foresight 链路已在双卡上完成两个优化器步骤并正常保存；通用入口已通过三种模式的真实样本加载。尚未完成完整训练、三组对照和验证集评测，目前没有理解能力提升的结果。

默认配置使用原版 VEGA 的完整 ScanQA 标注：训练 26,515 条问答、562 个场景；验证 4,675 条问答、71 个场景。已有双卡试跑使用 400 条训练问答和 100 条验证问答；全量训练尚未执行。全量资源是否齐备，以数据检查报告为准。

## 1. 获取代码与资源

以下以 `/data/zhdu/vega3d_repro` 为工作目录，路径可调整。已有仓库时直接进入代码目录。

```bash
PROJECT_ROOT=/data/zhdu/vega3d_repro
REPO_ROOT="$PROJECT_ROOT/code/VEGA-3D-qwen3.5-wan2.2"
mkdir -p "$PROJECT_ROOT/code"
git clone -b experiment/qwen35-wan22 https://github.com/huizi0031-bot/Vega3d_repro.git "$REPO_ROOT"
cd "$REPO_ROOT"
```

Git 仅包含代码和环境规范。数据、模型权重、场景特征和离线安装包需要单独复制。当前配置要求以下路径：

```text
data/
  models/
    Qwen3.5-2B-Language/           # 已导出的语言权重和 tokenizer
    siglip-so400m-patch14-384/     # 语义视觉权重
    Wan2.2-TI2V-5B/               # DiT、VAE 等原模型文件
  processed/
    scanqa_train_llava_style.json # 原版 VEGA 训练标注
    scanqa_val_llava_style.json   # 原版 VEGA 验证标注
  embodiedscan/
    embodiedscan_infos_train.pkl
    embodiedscan_infos_val.pkl
    embodiedscan_infos_test.pkl
  metadata/
    scannet_train_gt_box.json
    scannet_val_pred_box.json
  scannet/
    posed_images/<scene>/        # RGB .jpg、深度 .png、位姿 .txt、内参
    wan22_ti2v_offline_uniform32_1280x704_t300_b20_s42_v1/
      <scene>/wan22_ti2v.pt
```

特征提取还需要 `llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt`，该缓存不在 Git 中，应一并复制。已有观测特征时，训练不调用特征提取工具。

迁移软链接时复制其实际文件，或在新机器重新建立链接。场景元数据中的完整帧列表要保留；实际读取的 RGB、深度、位姿和教师目标图像必须齐全，观测帧顺序必须与缓存一致。原版任务标注保持不变，不需要生成 `train_paired.jsonl`。这里的原版标注指 VEGA 已支持的 `*_llava_style.json`，不是未经原版预处理的 ScanQA 发布文件。

## 2. 安装 Python 环境

环境为 Linux x86_64、Python 3.10.18、Torch 2.6.0 / CUDA 12.4、Transformers 5.3.0、DeepSpeed 0.16.9，使用 SDPA。环境与数据集名称无关。

将离线包放到 `$PROJECT_ROOT/environment_bundle/qwen35/`，其中应有 `python-runtime.tar.gz`、`manifest.json` 和 `wheelhouse/`。在仓库根目录安装到一个尚不存在的目录：

```bash
bash environment/qwen35/install.sh \
  --prefix "$PROJECT_ROOT/envs/vega3d-qwen35" \
  --bundle "$PROJECT_ROOT/environment_bundle/qwen35"

source "$PROJECT_ROOT/envs/vega3d-qwen35/reproduction/activate.sh"
python -m pip check
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

每个新终端都需要激活环境。已有原实验环境时，使用其 `bin/activate`，不重复安装。安装包不包含 NVIDIA 驱动和系统编译工具；DeepSpeed 运行时编译需要可用的 C++ / CUDA 工具链。完整说明见 [环境说明](environment/qwen35/README.md)。

## 3. 检查数据与配置

以下命令在仓库根目录执行，使用 [foresight.yaml](configs/3d/foresight.yaml)：

```bash
python scripts/3d/train/train_scene_understanding.py \
  --mode video --check-data --audit-file results/data_audit.json
python scripts/3d/train/train_scene_understanding.py --mode video --check-loader
```

`--check-data` 读取完整标注，逐场景检查观测 RGB、深度、位姿和场景几何参数，核对离线特征的场景、协议、尺寸及帧顺序，并为 video 模式自动选择监督片段。缺失缓存、损坏文件或找不到合格片段都会列入报告；存在训练错误时退出码为 1，不启动训练，也不删样本。验证资源错误单独报告，不阻止仅训练。检查不扫描离线特征张量的全部数值；实际读取时仍会检查有限值。

自动选帧保留原版 uniform32 观测。锚点优先选择靠近扫描中部的观测帧，目标取保存序列中紧随其后的四张图像；目标不能与观测重叠，相对锚点最大位移为 0.5m、最大旋转为 30°。不强制数字编号间隔；间隔由已有抽帧数据决定。同一场景的问答复用一段监督片段。

`--check-loader` 在同样的全量检查通过后，额外读取一条真实样本。video 模式的教师张量应为 `[1,3,5,256,320]`，输出包含 `cuda_initialized: false`、`optimizer_steps: 0`。该命令不加载视频教师、不进行优化。

配置中的 `data.train`、`data.validation` 支持原版 JSON、JSONL 和 datasets YAML。全量训练使用 `sampling_strategy: all`，不设样本数或场景数上限。切换同类文本任务时替换标注路径；底层仍要求 ScanNet / EmbodiedScan 场景资产。foresight 尚未接入定位 proposal 分支和动作头。

```bash
python scripts/3d/train/train_scene_understanding.py \
  --mode video --data-path /path/to/train.json \
  --validation-path /path/to/val.json --check-data
```

## 4. 全量训练

下例使用物理 GPU `2,3`，按实际空闲卡调整。先预览命令：

```bash
python scripts/3d/train/train_scene_understanding.py \
  --mode video --gpus 2,3 --run-name foresight-smoke --max-steps 2
```

确认参数后加 `--execute`，执行两个优化器步骤：

```bash
python scripts/3d/train/train_scene_understanding.py \
  --mode video --gpus 2,3 --run-name foresight-smoke --max-steps 2 --execute
```

数据检查通过后，可先进行上述两步试跑；全量训练使用新名称重新开始。以下命令读取完整 ScanQA 标注，使用相同初始权重和公共参数，每组训练 1 epoch，按顺序执行：

```bash
python scripts/3d/train/train_scene_understanding.py --mode baseline --gpus 2,3 --run-name baseline --execute
python scripts/3d/train/train_scene_understanding.py --mode queries  --gpus 2,3 --run-name queries  --execute
python scripts/3d/train/train_scene_understanding.py --mode video    --gpus 2,3 --run-name foresight --execute
```

默认参数如下，调整实验时修改配置，并保持三组的公共设置一致：

| 参数 | 当前值 |
|---|---|
| 每卡 batch / 全局 batch | 1 / 8；两卡时累积 4 次 |
| 训练长度 | 1 epoch，`max_steps: -1`；步数由完整标注条数和全局 batch 决定 |
| 学习率 | 常规模块 `1e-5`；视觉塔 `2e-6`；foresight `1e-4` |
| 优化器 / 调度 | AdamW；cosine；warmup ratio `0.03` |
| 精度与显存设置 | BF16、SDPA、梯度检查点、ZeRO-3 |
| 上下文 / 随机种子 | 32768 / 42 |
| 保存间隔 | 每 50 步，训练结束保存完整 checkpoint |

全局 batch 必须能被 `GPU 数 × 每卡 batch` 整除。每个进程单独持有冻结 Wan 教师，教师不由 ZeRO-3 分片。

输出位于 `checkpoints/<run-name>/`，配置、数据和源码哈希、包版本、日志、退出状态保存在其中的 `foresight_record/`。video 模式另存 `scene_pairs.json`，只记录每场景的选帧结果，不复制或修改问答标注；所有训练进程读取同一份记录。同名目录不会直接覆盖。中断恢复在原命令后加 `--resume`，要求有周期 checkpoint，且配置、数据、源码、进程数和训练参数保持一致。恢复时还会核对选帧记录及已检查场景资产的指纹。两步试跑设置不同于完整训练，正式训练使用新名称从相同初始权重开始。

## 5. 特征检查与评估

训练和验证都需要原 VEGA 分支的离线 Wan 观测特征。按完整标注检查缺失场景，已有缓存不重复提取。提取是独立的 GPU 操作：

```bash
python wan_offline_tools/extract_scene_features.py --split train --check-only
python wan_offline_tools/extract_scene_features.py --split validation --check-only
python wan_offline_tools/extract_scene_features.py --split train --gpu 2 --execute
python wan_offline_tools/extract_scene_features.py --split validation --gpu 2 --execute
```

检查发现缺失时退出码为 1。提取会核对指定 GPU 是否空闲，沿用已有缓存的协议；先确认 RGB-D、相机文件、Wan 权重和空文本条件缓存已复制齐全。

以完整 foresight 训练的输出为例，在一张空闲卡上生成回答并计分：

```bash
python scripts/3d/eval/eval_scene_qa.py \
  --model-path checkpoints/foresight \
  --question-file data/processed/scanqa_val_llava_style.json \
  --answer-file results/foresight.jsonl --gpu 2 --execute

python llava/eval/eval_scanqa.py --input-file results/foresight.jsonl
```

ScanQA 计分另需 `data/processed/scanqa_val_llava_style.json` 中的多参考答案，METEOR 需要 Java。baseline 和 queries 使用各自模型目录及新的输出文件，验证数据与解码设置一致；已有回答文件不会覆盖。当前训练不自动执行验证。

比较三组的独立验证指标，重点看 `video` 相比 `queries` 是否有额外收益。训练损失下降或两步试跑成功，不能替代理解能力的效果结果。

## 代码入口

| 位置 | 内容 |
|---|---|
| `llava/model/`、`llava/train/` | 模型、foresight、数据加载与训练 |
| `scripts/3d/train/` | 统一训练入口、参数与检查 |
| `scripts/3d/preprocessing/` | 原版数据转换、可选清单导出与模型资源导出 |
| `scripts/3d/eval/`、`llava/eval/` | 回答生成和任务计分 |
| `scripts/visualization/` | Wan 观测特征可视化 |
| `wan_offline_tools/` | 特征提取、模型准备和训练兼容接口 |
| `environment/qwen35/` | 环境规范、离线安装与检查 |
| `configs/3d/`、`docs/foresight.md` | 配置与方法细节 |

`qwen35_wan22.yaml` 保留原多任务基线配置，默认关闭 foresight、启用定位头，与本次三组理解实验使用的 `foresight.yaml` 分开。其他任务的原版转换和评估代码保留，新骨干上的适配与验证尚未全部完成。

来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
