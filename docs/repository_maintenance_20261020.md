# 仓库整理记录 — 2026-10-20

## 范围

当前工作区收敛到 Qwen3.5 / SigLIP / Wan2.2 的 ScanQA foresight 实验。删除旧五任务脚本、其他语言和生成模型适配器、服务与可视化工具、旧版 README 和个人服务器同步说明。保留当前模型运行需要的特征融合、3D 编码、数据处理、ScanQA 指标、测试及第三方许可证。

训练入口为 `scripts/3d/train/train_scanqa_foresight.py`，推理入口为 `scripts/3d/eval/eval_scanqa_foresight.py`。旧源码补丁工具替换为 `wan_offline_tools/train_entry.py`，不再在启动时修改源文件。离线特征提取实现保留在 `wan_offline_tools/wan22_features.py`。

默认数据清单改为 `data/scanqa_foresight`，输出改为仓库内 `checkpoints/`；相对路径按仓库根目录解析。现有数据目录通过链接接入。历史 checkpoint 和实验记录保留原位置。未改变 32 帧、32 queries、教师分辨率、损失或优化器配置。

## 检查

- 27 项 CPU 回归通过，包括 query 因果边界、梯度、微型 Wan flow、推理模型加载、保存重载和微型 Trainer 断点恢复。
- 实际 E03 数据预检查和一条真实样本加载通过；CUDA 未初始化，数据检查未执行优化器更新。
- 单卡 batch：RGB `[32,3,384,384]`，离线特征 `[1,32,3072,14,14]`，教师 RGB `[1,3,5,256,320]`。
- 新环境安装尚未从零验证；本次没有启动预训练大模型 GPU 训练或验证集推理。
- 5 个验证场景的离线特征仍待补齐。

此前两步 GPU 试跑使用清理前源码，结果见 `foresight_gpu_smoke_20261020.md`。不能将该结果写成清理后版本的 GPU 验证。

## 备份

8709 备份目录：`/data/zhdu/vega3d_repro/maintenance/scanqa_cleanup_20261020-205357/`。

`repository.bundle` 保存原 Git 历史；`working_tree.zip`、`working_tree.patch`、`files.json` 保存整理前工作区与校验值。移出的旧目录和本地开发备份保存在该目录内。数据、权重、Python 环境及已有 checkpoint 不在删除范围。
