# VEGA-3D: Qwen3.5 language backbone + Wan2.2

This independent project is based on the current VEGA-3D-qwen3 working tree.
It does not modify the existing Qwen3 project, its environment, or its running job.

## Architecture

- Language: **Qwen/Qwen3.5-2B**, language backbone only. The official checkpoint is
  multimodal; its pretrained text weights are explicitly exported and renamed to
  the text-only HF layout. Its native vision encoder, merger and MTP are excluded.
- Semantic vision: the same **SigLIP SO400M / patch14 / 384** as the Qwen3 version.
- Generative vision: frozen **Wan-AI/Wan2.2-TI2V-5B**, used in text-to-video mode.
  The adapter uses its official 48-channel Wan2.2 VAE, a fixed empty UMT5 condition,
  timestep 300, block index 20, and pooled `14 x 14` intermediate features.
- Wan features now have **3072 channels**, rather than 1536. Their existing
  two-layer GELU projector maps them to the language hidden size (2048).
- Keep the original semantic projector, token-gated fusion, 3D position encoding,
  object proposal processing and InfoNCE grounding-head design.
- Keep the five-task dataset YAML, original task sampler (including dropping
  each task group's last megabatch), losses and metric implementations.

Wan2.2-TI2V-5B uses **1280 x 704** input for its VAE/patch stride; 1280 x 720 was
the old Wan2.1 setting. Scene views are still sampled uniformly and processed
independently, as in the existing Qwen3 implementation. This is a representation
probe on real images, not normal video generation. Same block/timestep numbers
are a starting setting; their optimality for Wan2.2 has not been established.

## Files and environment

Server project: `/data/zhdu/vega3d_repro/code/VEGA-3D-qwen3.5-wan2.2`.
New Python: `/data/zhdu/vega3d_repro/envs/vega3d-qwen35/bin/python`.
The environment reads unchanged general packages from the Qwen3 environment and
overlays the pinned packages in `requirements-qwen35.txt`; it does not upgrade the
old environment. A matching PyG `torch-scatter` wheel is installed in the new
environment. Torch 2.6 is needed for the native HF5 safe checkpoint restore path.

This variant uses Torch SDPA for Qwen and Wan, rather than relying on a
flash-attn binary built for the old Torch. Initialization of newly created visual-bridge parameters is explicit; incomplete
language-backbone loading raises an error. This changes the attention kernel, not
the attention architecture. Qwen's optional FLA/causal-conv kernels are not
required for the CPU checks; full training throughput has not been benchmarked.

The official Wan2.2 sources are vendored under
`llava/model/multimodal_generative_encoder/wan22/` with license and pinned commit.
Package imports are made lazy and incompatible optional flash imports are caught;
the DiT and VAE architectures remain upstream implementations. The UMT5 constructor uses a lazy CUDA device default so CPU prompt export does not initialize CUDA while importing the module.

## Prepare weights (CPU)

No large new model weights are present in the original project. Download the
official checkpoints to `/data/zhdu/vega3d_repro/models/Qwen3.5-2B` and
`/data/zhdu/vega3d_repro/models/Wan2.2-TI2V-5B`, then run:

```bash
export PATH=/data/zhdu/vega3d_repro/envs/vega3d-qwen35/bin:$PATH
cd /data/zhdu/vega3d_repro/code/VEGA-3D-qwen3.5-wan2.2
python wan_offline_tools/prepare_qwen35_wan22.py
```

Alternatively add `--download` to download from Hugging Face, with optional
`--endpoint` for an endpoint you configure. The server must be able to reach it. Download revisions are pinned in the preparation tool.
Preparation exports `Qwen3.5-2B-Language` using safetensors (no GPU), sets local
model links, and exports a separate versioned Wan2.2 empty-prompt embedding.
It refuses to overwrite a populated language export or conflicting model link.

## Prepared assets in this deployment

Both original model snapshots have been downloaded and all 38,774,397,520 bytes
verified against the official pinned-revision file manifests. The language-only
export contains 320 tensors / 3,763,655,360 weight bytes. The Wan2.2 empty-prompt
cache is finite and has shape `[1, 4096]`; it was generated on CPU.

Assets are visible under `/data/zhdu/vega3d_repro/models/` and stored under
`/home/xjwu/vega3d_models/` through directory links. Wan2.2's UMT5 weights reuse
an existing byte-identical, SHA-256-verified file from the Wan2.1 package through
a read-only-by-convention symlink; the original file was not modified.
The download, CPU compatibility fix and final preparation records are under
`wan_full_smoke_tools/maintenance/model_download_20261015/`.
This preparation does not claim a pretrained GPU smoke test or full training run.

## Launchers

Use `train_qwen35_wan22.sh` for the small end-to-end smoke / train / eval modes,
and `train_qwen35_wan22_full.sh` for full-data extraction and training. Both are
under `scripts/3d/train/`; `--help` documents each interface.

```bash
# Configuration preview only; no GPU work.
bash scripts/3d/train/train_qwen35_wan22_full.sh --gpus 0,1,4,5 --dry-run

# After weights are prepared and GPUs are available:
bash scripts/3d/train/train_qwen35_wan22_full.sh --stage all --gpus 0,1,4,5
```

Full training retains 32 frames, global batch 128, 1 epoch, context 32768,
language LR 1e-5, vision LR 2e-6, warmup .03 and cosine scheduling.
Save every 500 optimizer steps and at successful training end; retain the latest
complete optimizer/RNG checkpoint and final model. Logs record each optimizer step.
Outputs go to `/data/zhdu/vega3d_repro/checkpoints/` with `qwen35-` run names.
Full training does not automatically run the five full validation benchmarks.

Wan2.2 caches have independent `wan22_ti2v_offline_uniform...` identifiers and
`wan22_ti2v.pt` payloads; old Wan2.1 caches are not reused. Runtime records,
weights, generated manifests and backups stay Git-ignored. Code/config/docs remain
versionable. No commit or push is performed by these scripts.

## Verification limits

Run `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python tests/test_qwen35_wan22.py`.
Tests use actual tiny Qwen3.5 and SigLIP modules to check forward/backward,
generation cache, native save/reload, fusion/grounding gradient paths, exact
pretrained language weight-key conversion and a tiny official Wan DiT hook.
The Wan contract test stubs VAE latents and uses a CPU SDPA backend; it does not
verify pretrained 5B GPU extraction. Eight CPU integration tests passed, including the actual training model factory,
real tokenizer masking, CPU-only UMT5 import and full optimizer/RNG checkpoint resume. The saved model
retains its trained SigLIP tensors when reloaded. Complete pretrained GPU smoke and full-data
training remain separate verification steps and are not claimed by CPU tests.

## Primary sources

- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/docs/transformers/v5.3.0/model_doc/qwen3_5
- https://github.com/Wan-Video/Wan2.2
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
