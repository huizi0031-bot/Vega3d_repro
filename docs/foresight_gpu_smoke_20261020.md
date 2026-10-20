# ScanQA E03 两卡工程试跑记录

日期：2026-10-20。主机：8709。物理 GPU：0、7；每张约 48 GiB。

完整完成 2 个 optimizer steps，退出码 0。全局 batch=8，两卡各 micro batch=1，梯度累积=4；共处理约 16 条训练问答。
保留 32 帧、原特征融合与 3D 编码、32 个 foresight queries、5 帧 256×320 的冻结 Wan teacher，以及 QA+0.1×video flow 损失。

| step | total loss | QA loss | video loss | grad norm（裁剪前） | learning rate |
|---|---|---|---|---|---|
| 1 | 9.043998718261719 | 8.92611312866211 | 1.178863763809204 | 1002.2069558434732 | 0.0 |
| 2 | 8.968022346496582 | 8.902021408081055 | 0.6600175499916077 | 999.4441810411603 | 1e-05 |

GPU 显存采样峰值：0号 47520 MiB（46.41 GiB）, 7号 47360 MiB（46.25 GiB）。采样间隔约 3 秒，可能漏掉更短的峰值。

启动修复：纯数字 hostname 8709 被解析成 0.0.34.5，导致 elastic rendezvous 超时。该单机入口现使用 static rendezvous、127.0.0.1 和本次独立端口；不改训练超参数。原始失败记录保留。

最终模型：`/data/zhdu/vega3d_repro/checkpoints/qwen35-foresight-e03-smoke-gpu07-static-20261020-184755`
完整 checkpoint：`/data/zhdu/vega3d_repro/checkpoints/qwen35-foresight-e03-smoke-gpu07-static-20261020-184755/checkpoint-2`
日志与配置：`/data/zhdu/vega3d_repro/experiments/scanqa_gpu_trials/qwen35-foresight-e03-smoke-gpu07-static-20261020-184755`

检查了 global_step=2、两份优化器状态文件、两份 RNG 状态、最终模型中的 queries/条件投影形状及有限值。GPU 完整断点续训未测试。

这次验证的是训练、反向传播、参数更新和保存的工程链路；不是完整训练或 ScanQA 效果评测。该双卡 48 GiB 配置的结果不能证明 8706 的 24 GiB 卡可以直接运行。

补充：这次发现末尾保存钩子会在 Trainer 已保存同一步后重复保存。后续版本已改为成功保存后记录步数，仅在末步尚未保存时补存；不改变 checkpoint 内容或训练计算。本次实际试跑仍使用记录中原版本。
