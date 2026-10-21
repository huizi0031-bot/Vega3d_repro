

| 文件 | 用途 |
|---|---|
| `train_entry.py` | 启动 train_3d，适配 checkpoint/RNG 接口与末步保存 |
| `qwen35_memory_compat.py` | 仅在答案监督位置投影 logits，降低训练内存 |
| `extract_scene_features.py` | 按数据清单检查或补齐场景离线特征（train / validation） |
| `wan22_features.py` | 共享的 Wan2.2 特征提取实现，保留现有缓存协议 |
| `export_qwen35_language.py` | 从完整 Qwen3.5 权重导出语言部分 |
| `prepare_qwen35_wan22.py` | 准备本地模型链接及空文本条件缓存 |

训练与评测统一从仓库 [README](../README.md) 中的入口启动。训练前无需运行任何源码补丁工具。

`extract_scene_features.py` 默认读取 `configs/3d/foresight.yaml`，可用 `--split train|validation` 或 `--data-path` 选择数据。已有缓存提供一致的提取协议，没有参考缓存时须显式提供 `--reference-cache`。默认只检查，指定空闲 `--gpu` 和 `--execute` 才提取。
