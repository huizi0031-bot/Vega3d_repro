# 场景理解训练

统一入口为 `train_scene_understanding.py`。`train_qwen35_wan22.sh` 是同一入口的 shell 包装。数据读取、训练参数、检查和记录不按数据集复制。

| 模式 | Queries | 视频监督 | 需要 foresight 配帧 |
|---|---|---|---|
| `baseline` | 无 | 无 | 否 |
| `queries` | 有 | 无 | 否 |
| `video` | 有 | 有 | 是，每个训练样本都需要 |

```bash
python scripts/3d/train/train_scene_understanding.py --mode video --check-data
python scripts/3d/train/train_scene_understanding.py --mode video --check-loader
python scripts/3d/train/train_scene_understanding.py --mode video --gpus 2,3 --run-name foresight-test --max-steps 2
# 确认预览后，在上一命令末尾加 --execute 才训练
```

默认配置为 `configs/3d/foresight.yaml`，暂时指向已准备的 ScanQA 清单。它是可替换的数据路径，不是数据集限制。三种模式使用同一份清单；不开视频监督时，忽略其中的 `foresight` 字段。配置和数据格式见 [foresight](../../../docs/foresight.md)。

其他文本生成任务修改 `data.train`，或使用 `--data-path /path/train.jsonl`。支持单个 JSON、JSONL 或 `datasets` YAML；YAML 中每条数据源可设置 `all`、`first:N`、`end:N`、`random:N` 或百分比采样。相对路径以仓库根目录为基准。`data.validation` 可选，训练过程不自动运行验证；`require_disjoint_validation_scenes` 控制是否要求场景分离。

原版多任务基线仍可使用同一入口：

```bash
python scripts/3d/train/train_scene_understanding.py --config configs/3d/qwen35_wan22.yaml --data-path scripts/3d/train/multi.yaml --gpus 2,3
```

`qwen35_wan22.yaml` 默认 `baseline`，保留原定位头。foresight 目前接入单轮文本生成任务，定位 proposal 分支、DPO、量化/LoRA 和动作头尚未接入；入口会拒绝不支持的组合。场景资产仍沿用 ScanNet/EmbodiedScan 与离线 Wan 特征布局，任意新视频数据需要先适配该读取层。

`--check-data` 检查标注、训练场景缓存是否存在、模型文件、配帧结构和图像完整性；不读取全部缓存张量。`--check-loader` 额外通过真实 loader/collator 读取一个样本，核对其实际采样、缓存与教师输入。它不表示所有场景的缓存内容都已逐个验证。两种检查不会启动训练。

每次执行保存配置、输入文件哈希、源代码哈希、包版本、训练日志及退出状态。恢复使用相同 `--run-name --resume`，并要求配置、命令参数、数据、源码和进程数一致。旧 ScanQA 启动器的 checkpoint 可作初始化资产另行处理，不能直接当作新入口的严格恢复记录。
