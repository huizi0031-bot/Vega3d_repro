# Qwen3.5 / Wan2.2 Python 环境

本目录保存训练环境的依赖清单、离线安装脚本和检查工具，适用于共享当前模型与读取接口的任务。ScanQA 是已经使用过的数据实例，环境不限定数据集名称。

平台为 Linux x86_64，Python 3.10.18，Torch 2.6.0 / CUDA 12.4，Transformers 5.3.0，DeepSpeed 0.16.9，Accelerate 1.12.0，Diffusers 0.30.3。当前使用 SDPA。

## 安装

Git 中保存环境规范，运行时和 wheel 单独复制。离线资源目录应包含：

```text
qwen35/
  python-runtime.tar.gz
  manifest.json
  observed-environment.json
  wheelhouse/
```

在仓库根目录执行，下面两个路径替换为实际绝对路径。目标环境目录必须尚不存在。

```bash
bash environment/qwen35/install.sh \
  --prefix /path/to/envs/vega3d-qwen35 \
  --bundle /path/to/environment_bundle/qwen35

source /path/to/envs/vega3d-qwen35/reproduction/activate.sh
python -m pip check
```

安装脚本核对资源清单与 SHA256，安装 Python 和依赖，再将当前仓库安装为 editable package；随后自动检查 156 项依赖版本、导入路径、CPU 算子和关键模块。检查报告位于新环境的 `reproduction/verification.json`。

每个新终端重新激活环境。激活脚本设置 `PYTHONNOUSERSITE=1` 并清除外部 `PYTHONPATH`，避免继承其他环境的包。已有原实验环境使用其 `bin/activate`。

## 数据与 GPU 检查

模型和数据复制完成后，在仓库根目录运行：

```bash
python scripts/3d/train/train_scene_understanding.py --mode video --check-data
python scripts/3d/train/train_scene_understanding.py --mode video --check-loader
```

这两项使用训练配置的资源路径，不占 GPU、不执行优化器更新。当前仓库未保留 `tests/`，安装示例不使用 `--test`。换任务时调整训练配置，数据检查不固定为 ScanQA。

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

GPU 试跑按根目录 [训练说明](../../README.md) 执行。离线包不包含 NVIDIA 驱动、系统 C++ / CUDA 编译工具、模型和数据。DeepSpeed 运行时编译仍需要目标机器的工具链；CPU 检查通过后还要验证实际 GPU 前后向。

## 文件用途

| 文件 | 用途 |
|---|---|
| `requirements.lock.txt` | 完整依赖版本，根目录 `requirements.txt` 引用它 |
| `requirements-offline.lock.txt` | 离线安装包版本与校验值 |
| `manifest.json` | Python 版本、安装层次与资源校验值 |
| `observed-environment.json` | 导出时的原环境记录 |
| `install.sh` | 安装到全新目录 |
| `verify.py` | 环境和项目入口检查 |
| `verification.json` | 环境独立恢复时保存的历史检查结果 |
| `export_installed.py`、`prepare_manifest.py` | 重新制作离线包与清单 |

原实验环境继承了基础环境和用户目录中的包，离线包按实际导入优先级记录生效版本。大部分 wheel 从已安装文件重新封装，保留二进制扩展；Decord 使用同版本官方兼容 wheel。安装按记录的层次恢复覆盖关系。

仅执行 `pip install -r requirements.txt` 无法约束上游包的 CUDA 构建及原安装覆盖顺序，完整复现使用上述离线安装流程。`torch-scatter==2.1.2+pt26cu124` 与 Torch / CUDA 版本配套，不单独替换。
