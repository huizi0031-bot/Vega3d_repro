# ScanQA Foresight — Qwen3.5 + Wan2.2

基于 VEGA-3D 的空间问答实验。语言骨干为 Qwen3.5-2B，保留 SigLIP、生成式特征融合和 3D 位置编码，在语言序列中加入 32 个 foresight queries，通过冻结 Wan2.2 的视频 flow loss 辅助训练。

当前只保留 ScanQA 的训练、推理、特征补齐和测试入口。实验目标是验证跨视角预测监督能否改善空间问答；尚无完整训练或效果评测结论。

## 实验状态

| 项目 | 进展 |
|---|---|
| 小规模数据 | 训练 400 条 QA / 20 个场景；验证 100 条 QA / 5 个独立场景 |
| E03 工程试跑 | 2026-10-20 在 8709 双卡完成 2 个 optimizer steps，正常退出 |
| 梯度与保存 | QA / video 两项 loss 有限，模型及完整 checkpoint 已保存 |
| 显存 | 每卡采样峰值约 46.4 GiB，详见试跑记录 |
| 训练场景特征 | 20 个场景齐备 |
| 验证场景特征 | 5 个场景待补齐，未运行验证集推理 |
| E00 / E01 对照 | 尚未训练 |
| 完整训练与效果 | 尚未完成 |

架构与损失见 [方法说明](docs/foresight_scanqa.md)，实际数值见 [两步试跑记录](docs/foresight_gpu_smoke_20261020.md)。代码整理与 CPU 检查见 [整理记录](docs/repository_maintenance_20261020.md)。

## 环境

当前验证环境为 Python 3.10.18、Torch 2.6.0、Transformers 5.3.0、DeepSpeed 0.16.9、Accelerate 1.12.0、Diffusers 0.30.3。使用 SDPA，不要求安装 flash-attn。

8709 已有环境：

```bash
source /data/zhdu/vega3d_repro/activate.sh
```

新环境按以下版本安装。此安装流程尚未在全新公司环境执行；安装后需先通过下述 CPU 检查。

```bash
conda create -n scanqa-foresight python=3.10 -y
conda activate scanqa-foresight
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-scanqa.txt
python -m pip install torch-scatter==2.1.2+pt26cu124 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install -e . --no-deps
```

`requirements-qwen35.txt` 保留模型核心依赖版本；`requirements-scanqa.txt` 补充数据加载与训练需要的依赖。Torch 和 torch-scatter 的 CUDA 版本需要匹配。

## 数据和权重

Git 只保存代码，不包含数据、权重、离线特征及训练输出。准备以下目录；已有资源可用软链接接入。

```text
data/
├── models/
│   ├── Qwen3.5-2B-Language/           # 导出的语言权重、tokenizer
│   ├── siglip-so400m-patch14-384/     # 语义视觉权重
│   └── Wan2.2-TI2V-5B/               # DiT 三个分片及 Wan2.2_VAE.pth
├── scanqa_foresight/
│   ├── train_original.json
│   ├── train_paired.jsonl
│   ├── val_original.json
│   └── scene_pairs.json
├── embodiedscan/
│   └── embodiedscan_infos_{train,val,test}.pkl
├── metadata/
│   ├── scannet_train_gt_box.json
│   └── scannet_val_pred_box.json
└── scannet/
    ├── posed_images/<scene>/          # RGB、深度、位姿、intrinsic.txt
    └── wan22_ti2v_offline_uniform32_1280x704_t300_b20_s42_v1/
        └── <scene>/wan22_ti2v.pt
```

8709 的数据清单仍存放在 `/data/zhdu/vega3d_repro/experiments/scanqa_foresight_pilot_20261020_v1/data`，仓库内 `data/scanqa_foresight` 指向该目录。数据根目录和输出位置可在 [配置](configs/foresight/scanqa_pilot.yaml) 中修改，相对路径按仓库根目录解析。

迁移资源时检查软链接目标，不能只复制链接。保留完整的场景帧列表元数据；删除未采样的元数据条目会改变 uniform32 选帧，导致特征不对齐。补验证特征时还需要已导出的 `llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt`；已有缓存可直接复制，不需重新加载 UMT5。

## 检查和训练

所有命令在仓库根目录运行。默认仅输出配置；训练必须带 `--execute`。

```bash
# 检查权重文件、训练观测、配对清单；同时列出缺失的验证特征。
python scripts/3d/train/train_scanqa_foresight.py --experiment E03 --check-data

# CPU 读取一条真实样本，检查 tokenizer、视觉数据和五帧监督张量。
python scripts/3d/train/train_scanqa_foresight.py --experiment E03 --check-loader

# CPU 模型测试。
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests QWEN35_TEST_TOKENIZER=data/models/Qwen3.5-2B-Language python -m unittest discover -s tests -p 'test_*.py' -v

# 两步试跑。
python scripts/3d/train/train_scanqa_foresight.py --experiment E03 --gpus 0,1 --max-steps 2 --run-name qwen35-e03-smoke --execute

# 在当前 400 条 QA 上训练 1 epoch，约 50 个 optimizer steps。
python scripts/3d/train/train_scanqa_foresight.py --experiment E03 --gpus 0,1 --run-name qwen35-e03-pilot --execute
```

| 实验 | Queries | 视频 loss | QA 数据 |
|---|---|---|---|
| E00 | 无 | 无 | 原问答 |
| E01 | 32 个 | 无 | 相同问答 |
| E03 | 32 个 | 权重 0.1 | 相同问答 + 配对视频片段 |

默认 micro batch=1、global batch=8；累积步数自动按 GPU 数计算。对照组使用不同 run-name，保持其余配置一致。当前数据加载器只支持本配置的单轮 ScanQA 问答，不直接接受 VSI-Train-10k 或机器人动作数据。

输出默认写入 `checkpoints/<run-name>/`。`foresight_record/` 保存配置、源码/数据校验值、依赖版本、训练日志与退出码。输出目录不能重用；中断恢复需相同配置、源码、数据和 GPU 进程数，并增加 `--resume`。两步试跑改变了训练日程，正式实验应从相同初始化新建运行。

## 验证集

先查看待补的离线特征；该工具默认只检查，不提取。

```bash
python wan_offline_tools/extract_scanqa_validation.py --check-only
# 有空闲卡时，指定 GPU 执行特征提取。
python wan_offline_tools/extract_scanqa_validation.py --gpu 0 --execute
```

五个缺失场景：`scene0011_00`、`scene0064_00`、`scene0084_00`、`scene0249_00`、`scene0377_00`。检查模式发现缺失时返回 1，属预期结果。

```bash
# 先检查模型与特征，再加 --gpu 0 --execute 执行推理。
python scripts/3d/eval/eval_scanqa_foresight.py   --model-path checkpoints/qwen35-e03-pilot   --answer-file results/e03_scanqa.jsonl

# 推理完成后计算原 ScanQA 指标，METEOR 需要 Java。
python llava/eval/eval_scanqa.py --input-file results/e03_scanqa.jsonl
```

指标脚本另需 `data/processed/scanqa_val_llava_style.json`，其中应保留每条问题的多参考答案 `metadata.answers`。标准答案只用于计分；推理 prompt 不包含答案。推理保留原 VEGA 离线特征输入，不加载新增的视频监督教师。

## 代码位置

- `llava/model/foresight.py`：query 参数、插入位置、答案损失。
- `llava/model/language_model/llava_qwen.py`：Qwen 前向、Z、损失组合和生成。
- `llava/model/multimodal_generative_encoder/wan22_foresight.py`：冻结视频教师、VAE 和 flow loss。
- `llava/train/foresight_data.py`：观测/目标隔离检查和五帧读取。
- `wan_offline_tools/train_entry.py`：训练兼容层和 checkpoint 保存。
- `repro/prepare_scanqa_foresight_pilot.py`：从处理后的 ScanQA 数据生成小规模清单。

## 来源

当前实现保留 VEGA-3D 的特征融合和 3D 编码，使用 Qwen3.5、SigLIP 及 Wan2.2。代码来源、引用和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
