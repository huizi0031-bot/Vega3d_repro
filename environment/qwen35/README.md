# Qwen3.5 实验环境

本目录记录实验所用的 Python 环境。Python 为 3.10.18，平台为 Linux x86_64；Torch 为 2.6.0 / CUDA 12.4，Transformers 为 5.3.0。当前模型使用 SDPA。

## 文件

| 文件 | 内容 |
|---|---|
| `requirements.lock.txt` | 实际生效依赖的完整版本清单 |
| `requirements-offline.lock.txt` | 相同依赖及离线 wheel 的 SHA256 |
| `manifest.json` | Python、安装层次、安装包校验值及排除项 |
| `observed-environment.json` | 原环境的依赖快照及 `pip check` 结果 |
| `install.sh` | 从离线资源安装到全新目录 |
| `verify.py` | 检查版本、导入路径、依赖一致性、CPU 算子和项目入口 |
| `export_installed.py` | 从实验环境导出运行时与已安装的依赖 |
| `prepare_manifest.py` | 生成清单和校验值 |

## 安装

环境规范随 Git 保存，Python 运行时和 wheel 文件单独交接。离线目录必须包含 `python-runtime.tar.gz`、`manifest.json` 和完整的 `wheelhouse/`。8709 上的位置为：

```text
/data/zhdu/vega3d_repro/environment_bundle/qwen35/
```

将仓库和离线目录复制到目标机器后，在仓库根目录执行以下命令，替换为实际绝对路径。目标环境目录必须不存在，脚本不会覆盖已有环境。

```bash
bash environment/qwen35/install.sh \
  --prefix /path/to/envs/vega3d-qwen35 \
  --bundle /path/to/environment_bundle/qwen35 \
  --test

source /path/to/envs/vega3d-qwen35/reproduction/activate.sh
python -m pip check
```

已接入模型和数据时，可在安装命令中再加 `--check-data`，通过统一入口读取一条真实 video 模式样本。`--test` 和 `--check-data` 均为 CPU 检查，不启动训练。未复制 tokenizer 时，对应测试会跳过；复制后可重新检查。完整报告保存在新环境的 `reproduction/verification.json`。

激活脚本会设置 `PYTHONNOUSERSITE=1` 并清除 `PYTHONPATH`，防止从用户目录或其他环境加载同名包。新终端需重新激活。运行 `python` 前可用 `which python` 确认路径。

不要只执行 `pip install -r requirements.lock.txt` 来复现本环境。版本清单无法约束下载源中的 CUDA 构建，也无法恢复原有包之间的覆盖顺序；离线安装脚本会核对文件校验值并按记录的顺序安装。尤其是 `torch-scatter==2.1.2+pt26cu124`，必须与本次 Torch / CUDA 组合匹配。

## 与原环境的关系

原 `vega3d-qwen35` 是继承基础环境的 venv，还会加载用户目录中的包。本次记录按实际导入优先级选择生效版本；大部分离线 wheel 从已安装文件重新封装，保留二进制扩展，不重新编译；Decord 的例外见下文。它们的校验值对应本次导出的文件，与上游发布 wheel 的校验值不同。


原工作环境未作修改。Python 包和运行时可离线交接；显卡驱动、系统编译工具、数据、模型权重及离线场景特征不在此目录内。DeepSpeed 运行时编译算子仍取决于目标机器的 CUDA 工具链。CPU 检查不能代替目标机器上的 GPU 试跑。

## 检查记录

8709 上已从离线资源独立安装，156 项版本核对、`pip check`、27 项 CPU 测试及一条真实 E03 样本加载通过。GPU 未用于此次检查。详见 [环境与代码检查](../../docs/environment_validation.md)。
