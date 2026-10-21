# 任务评估

Qwen3.5 文本回答统一由 `eval_scene_qa.py` 生成，适用于相同单轮场景问答格式，是否有 foresight queries 由保存的模型配置决定。显式提供 `--question-file` 和 `--answer-file`，默认预览，指定 `--gpu` 与 `--execute` 才加载 GPU 模型。推理不加载新增视频教师；原有离线 Wan 特征仍需要。

```bash
python scripts/3d/eval/eval_scene_qa.py --model-path checkpoints/scene-video --question-file data/scanqa_foresight/val_original.json --answer-file results/answers.jsonl
```

生成与计分分开。ScanQA 使用 `llava/eval/eval_scanqa.py`，其他任务使用其对应指标。当前五个验证场景特征仍缺失，尚无验证指标。

同目录保留原版任务 shell 入口与 `llava/eval/model_*.py`，方便后续适配。原版推理涉及 Ray、旧聊天模板、原 checkpoint 路径和任务特定物体框/指标，不代表已经完成 Qwen3.5 验证。定位任务需独立适配 proposal 分支，不能使用当前文本回答入口替代定位评估。
