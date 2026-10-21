# 场景理解训练

入口为 `train_scene_understanding.py`，默认配置 `configs/3d/foresight.yaml`。默认直接读取原版 VEGA 的完整 ScanQA 标注，不需要预先生成 paired 问答。

| 模式 | Queries | 视频监督 | 数据输入 |
|---|---|---|---|
| `baseline` | 无 | 无 | 原版标注 |
| `queries` | 有 | 无 | 原版标注 |
| `video` | 有 | 有 | 原版标注，加载层自动选择监督片段 |

```bash
python scripts/3d/train/train_scene_understanding.py --mode video --check-data --audit-file results/data_audit.json
python scripts/3d/train/train_scene_understanding.py --mode video --check-loader
python scripts/3d/train/train_scene_understanding.py --mode video --gpus 2,3 --run-name foresight-smoke --max-steps 2 --execute
```

完整训练使用新的输出目录，不指定 `--max-steps`；配置默认为 1 epoch、`max_steps: -1`，没有样本数或场景数限制：

```bash
python scripts/3d/train/train_scene_understanding.py --mode baseline --gpus 2,3 --run-name baseline --execute
python scripts/3d/train/train_scene_understanding.py --mode queries --gpus 2,3 --run-name queries --execute
python scripts/3d/train/train_scene_understanding.py --mode video --gpus 2,3 --run-name foresight --execute
```

`--data-path` / `--validation-path` 可覆盖训练、验证路径，支持 JSON、JSONL 或 datasets YAML。YAML 的相对路径以仓库根目录为基准；全量使用 `sampling_strategy: all`。默认训练文件为 `data/processed/scanqa_train_llava_style.json`，验证文件为 `data/processed/scanqa_val_llava_style.json`。官方原始任务标注仍需经过原版 VEGA 的转换流程。

video 模式按原版 uniform32 观测选择锚点及四帧留出画面，保持观测与离线特征顺序不变。运动阈值配置为 `data.max_translation`（米）、`data.max_rotation`（度）。每场景固定一段，在 `checkpoints/<run-name>/foresight_record/scene_pairs.json` 保存，供所有 rank 和恢复训练共同读取。旧清单已有的 `foresight` 字段仍可使用，不自动替换其显式配帧。

检查会汇总训练资源或配帧错误并以退出码 1 阻止开训，不静默删样本。验证错误单独记录。`--check-loader` 额外在 CPU 上读取真实样本；不执行模型前向或训练。完整检查读取缓存头验证协议、维度和帧顺序，实际 loader 读取张量时再检查有限值。

原多任务基线使用 `--config configs/3d/qwen35_wan22.yaml`，默认关闭 foresight 并保留定位头。foresight 当前支持同一场景资产上的单轮文本生成；定位 proposal、DPO、量化/LoRA 和动作头尚未接入。

运行记录包含配置、标注与源码哈希、资产指纹、包版本、日志及退出状态。恢复使用原命令加 `--resume`，配置、参数、进程数、标注、选帧和已检查资产必须一致。

安装、资源准备和评估见 [项目说明](../../../README.md)，机制与数据接口见 [foresight](../../../docs/foresight.md)。
