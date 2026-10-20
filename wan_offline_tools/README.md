# ScanQA 运行工具

| 文件 | 用途 |
|---|---|
| `train_entry.py` | 启动 train_3d，适配 checkpoint/RNG 接口与末步保存 |
| `qwen35_memory_compat.py` | 仅在答案监督位置投影 logits，降低训练内存 |
| `extract_scanqa_validation.py` | 检查或补齐 5 个验证场景的离线特征 |
| `wan22_features.py` | 共享的 Wan2.2 特征提取实现，保留现有缓存协议 |
| `export_qwen35_language.py` | 从完整 Qwen3.5 权重导出语言部分 |
| `prepare_qwen35_wan22.py` | 准备本地模型链接及空文本条件缓存 |

训练与评测统一从仓库 [README](../README.md) 中的入口启动。训练前无需运行任何源码补丁工具。
