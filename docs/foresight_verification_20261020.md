# Foresight 实现验收记录 — 2026-10-20

配置、架构及使用方式见 [foresight_scanqa.md](foresight_scanqa.md)。本记录区分 CPU 接口验收与尚未运行的大模型实验。

## 已执行

### 1. 新增 CPU 检查：12 / 12 通过

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
QWEN35_TEST_TOKENIZER=data/models/Qwen3.5-2B-Language \
/data/zhdu/vega3d_repro/envs/vega3d-qwen35/bin/python tests/test_foresight.py
```

使用真实微型 Qwen3.5 与真实微型 native Wan DiT，无预训练大模型加载，无 optimizer step。

- 三组 E00/E01/E03 命令行均能由现有训练参数解析器接受；不实例化 GPU 训练参数。
- 从微型语言 checkpoint 经原模型工厂加载新 queries，维度正确且初始化有限、非零。
- 使用本机准备好的真实 tokenizer，生成 prompt 与训练第一个答案 token 前的输入完全相同。
- 新模块学习率 1e-4 / 原参数 1e-5 分组正确，无参数重复；QA/video 分项日志工作，global_step 保持 0。
- RGB clip 与 collator 保持独立 teacher 字段；目标与输入重叠、采样顺序不一致时报错。
- 修改标准答案不会改变 Z；修改问题上下文会改变 Z；QA 损失可传回 queries。
- 只对 video loss 反向：queries、条件投影和 Qwen 均得到有限非零梯度；native Wan 参数冻结且梯度为空。
- 同时开启 Qwen gradient checkpointing 与 Wan block checkpointing，上述视频梯度仍成立。
- 开关 Wan block checkpointing 时，固定噪声/时间的 loss 和 context 梯度相同。
- 保存并重载后结果一致；带 KV cache 的两步生成仅在 prefill 插入一次 queries；teacher 未加载。
- 有效答案位置的 CE 与完整 causal CE 等价。
- 首帧 latent 保持 clean，timestep=0，损失 mask=0；后续区域的加噪、velocity 与时间正确。
- 左/右 padding、query 标签忽略与上下文溢出检查通过。

部分检查合并在同一个测试函数，测试函数总数为 12。最终执行摘要：`Ran 12 tests ... OK`。

### 2. 旧架构 CPU 回归：7 / 7 通过

执行 `tests/test_qwen35_wan22.py` 中以下七项：CPU UMT5 导入、真实训练模型工厂、真实 tokenizer 掩码、LM 前后向与重载生成、SigLIP/融合/3D/grounding 梯度、语言导出权重映射、原 Wan 单帧 DiT hook/空间 packing。

明确**没有执行**其中的 `test_real_trainer_step_and_complete_checkpoint_resume`，因为该测试会执行优化器步骤。本次不代替用户启动训练。

### 3. 数据与实际 loader

```bash
/data/zhdu/vega3d_repro/envs/vega3d-qwen35/bin/python \
  scripts/3d/train/train_scanqa_foresight.py --experiment E03 --check-loader
```

结果为 `PREVIEW_ONLY`。读取 400 条训练 QA（20 个场景）、100 条验证 QA（5 个不同场景）、20 个独立 teacher clips；训练离线缓存存在，teacher JPEG 可读取，原始与 paired QA 顺序/内容一致。

实际第一条样本经原 VideoProcessor / dataset / collator 后：

| 项目 | 实测 |
|---|---|
| 文本 tokens（视觉展开前） | 80 |
| 有效答案 tokens | 6 |
| SigLIP 图像输入 | `[32,3,384,384]` |
| 原 Wan 输入特征 | `[1,32,3072,14,14]` |
| world coordinates | `[1,32,384,384,3]` |
| 单独 teacher RGB | `[1,3,5,256,320]` |
| CUDA 初始化 | `False` |
| optimizer steps | `0` |

读取后原始 annotation 未被修改。修复了原 `_get_item` 直接修改存储 conversation 的问题，避免反复读取时重复添加空间提示。

数据 SHA-256：

```text
train_original.json  fc1c437bbc429fa8cea7bba2dfc5feddee352e1fbcef0dbb2e0729a5052e115b
train_paired.jsonl   7286ca36c437f244d7d0583522cefbc7a067e889dd70952e1d6f1a66e895e0a5
val_original.json    71c7297e057307261fc4f9017ca7ceccd2f9164efa5cfdc051a1c6aae9fc29bb
```

### 4. 静态检查与环境

新增/修改 Python 源码语法检查与 `git diff --check` 通过。检查结束时不存在 `checkpoints/qwen35-foresight-*` 训练目录。未执行 Wan 缓存提取或后台任务。

| 依赖 | 实际版本 |
|---|---|
| Python | 3.10.18 |
| Torch | 2.6.0 |
| Transformers | 5.3.0 |
| Diffusers | 0.30.3 |
| Accelerate | 1.12.0 |
| DeepSpeed | 0.16.9 |
| Safetensors | 0.8.0 |
| Tokenizers | 0.22.2 |
| NumPy | 1.26.1 |

发现 Qwen3.5 环境仍引用已不存在的 `envs/vega3d-qwen3` 路径。本次只修复了该环境的解释器符号链接、`pyvenv.cfg` home 和一个 cuDNN 符号链接，改指向当前 `envs/vega3d` 中已存在的文件。未安装/升级软件包，未修改基础环境文件或旧项目。

变更备份/记录：

```text
/data/zhdu/vega3d_repro/tmp/qwen35_environment_links_20261020_104148/
/data/zhdu/vega3d_repro/tmp/qwen35_dependency_links_20261020_104425.json
/data/zhdu/vega3d_repro/tmp/foresight_before_20261020_104007/
```

保留原 tokenizer 的参数与文件。当前 Transformers 在加载它时输出 regex 警告；本次没有改变 tokenization 规则，训练/推理前缀相等检查通过。没有安装可选 FLA/causal-conv 快速内核，CPU 使用原有 Torch fallback。native Wan 的 CUDA autocast 在 CPU 检查时被自动禁用，属该检查的预期行为。

原始检查日志存放在服务器 `vega3d_repro/tmp/`：`foresight_cpu_checks_20261020.log`、`foresight_regression_20261020.log`、`foresight_real_loader_20261020.log`。这些运行日志不上传 GitHub。

## 尚未执行及下一步边界

1. 未启动 E00 / E01 / E03 实验训练，没有训练好的 foresight 参数，没有任何性能提升结果。
2. 未用真实 5B Wan DiT + 真实 VAE 做 GPU 前后向；其混合精度数值、显存峰值和速度仍需验证。
3. ZeRO-3 分布式参数、优化器 checkpoint 恢复，以及真实大模型 checkpoint 端到端生成尚未在 GPU 运行。query 插入使用标准模块 forward，供 ZeRO hooks 管理；CPU 检查不能替代分布式验收。
4. 未执行正式 ScanQA 指标评分或验证特征提取。五个验证场景缓存均缺失：`scene0011_00`、`scene0064_00`、`scene0084_00`、`scene0249_00`、`scene0377_00`。
5. 片段协议仍是留出视角预测，原完整 3D 几何仍保留；不能将其当成严格未来动态预测或动作能力证据。

**是否开始真实 GPU 验证与训练，由用户决定。**
