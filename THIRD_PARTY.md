# 来源与许可证

本项目基于以下工作，保留场景理解、数据转换与任务评估的相关代码及其版权声明和许可证。当前语言骨干为 Qwen3.5，生成式视觉编码为 Wan2.2。

| 来源 | 使用部分 |
|---|---|
| [VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 语义/生成式特征融合、3D 编码、ScanQA 数据与训练框架 |
| [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM)、[LLaVA](https://github.com/haotian-liu/LLaVA) | 上游多模态骨架与数据处理 |
| [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) | 语言预训练权重和原生 Transformers 实现 |
| [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) | 语义视觉编码 |
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) | TI2V-5B DiT、VAE、特征提取和 flow 参数化 |
| [InternVLA-A1.5](https://internrobotics.github.io/internvla-a15.github.io/) | latent foresight 与冻结视频模型监督的相关方法 |

仓库许可证见 [LICENSE](LICENSE)。Wan 代码与固定上游版本见 [UPSTREAM.json](llava/model/multimodal_generative_encoder/wan22/UPSTREAM.json)，许可证见 [LICENSE.txt](llava/model/multimodal_generative_encoder/wan22/LICENSE.txt)。模型权重及 ScanNet/EmbodiedScan 等数据按各自条款使用，未随 Git 发布。

```bibtex
@inproceedings{wu2026vega,
  title={Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding},
  author={Xianjin Wu and Dingkang Liang and Tianrui Feng and Kui Xia and Yumeng Zhang and Xiaofan Li and Xiao Tan and Xiang Bai},
  booktitle={ECCV},
  year={2026}
}
```
