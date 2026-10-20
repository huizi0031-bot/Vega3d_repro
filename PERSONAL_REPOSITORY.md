# 个人仓库与服务器目录

本仓库是 `huizi0031-bot/Vega3d_repro`，保存当前 Qwen3.5 语言骨干与 Wan2.2 的 VEGA-3D 复现实验代码。用户确认采用公开仓库。

## 服务器位置

- 运行根目录：`/data/zhdu/vega3d_repro`。
- 当前 Git 工作目录：`/data/zhdu/vega3d_repro/code/VEGA-3D-qwen3.5-wan2.2`。
- 个人 GitHub 远程名：`github`；原 `origin` 保留，用于保留项目来源信息。
- 当前本地分支保持原名；首次推送到 GitHub `main` 并建立跟踪。

## 跟踪范围

跟踪代码、配置、测试、工具脚本、说明和第三方许可证。数据、模型权重、Python 环境、离线特征缓存、checkpoint、日志和运行目录保留在服务器。原数据链接不会随 Git 上传，克隆后需按 `README_QWEN35_WAN22.md` 配置。

底层实现基于 [VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D)，保留上游历史、引用与许可证。原有评测依赖资源也保留。当前仓库快照不代表完成了 foresight 新分支或验证了其效果。

## 后续同步

在上述 Git 工作目录内检查改动，提交明确需要保存的文件，再推送：

```bash
git status
git add <需要保存的文件>
git commit -m "说明本次修改"
git push
```

保存文件不会自动同步 GitHub，提交后仍需推送。服务器使用只对本仓库授权的专用部署密钥；私钥位于用户 SSH 目录，不纳入本仓库。

## ScanQA 起步

数据准备脚本：`repro/prepare_scanqa_foresight_pilot.py`。说明：`docs/scanqa_pilot.md`。已在服务器生成小规模数据清单，但新增模型、数据加载和视频损失仍需后续实现。数据清单本身不提交，避免把数据集内容混入代码仓库。
