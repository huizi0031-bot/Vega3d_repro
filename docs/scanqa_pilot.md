# ScanQA foresight pilot — 2026-10-20

## Scope
Data-only preparation. Existing project code, source annotations, model weights,
and shared images are unchanged. No model or GPU training is launched.

The original 32 QA observations are preserved exactly using input-cache metadata.
Four previously unused RGB frames after an observed anchor form the prediction
target. The target is a heldout-view clip, not strict future prediction: other QA
observations may occur later in time. This preserves ScanQA's scene-level QA
evidence and avoids silently changing the task to partial-observation QA.

## Files
- train_original.json: unchanged-format ScanQA subset for matched baselines.
- val_original.json: official-validation QA subset, in separate scenes.
- train_paired.jsonl: original annotation plus explicit foresight metadata.
- scene_pairs.json: shared scene-level frame plans, avoiding ambiguity.
- scanqa_only.yaml: baseline-only configuration pointing to train_original.json.
- summary.json: counts, source hashes, checks and limitations.

The current training loader ignores the added foresight metadata; this file is
NOT enough to activate the new objective. A separate loader and model branch
must explicitly read the target frames and compute the video prediction loss.
Validation should retain the original QA observation protocol. Its offline Wan
feature cache is a separate prerequisite; do not silently use training caches.

## Proposed next implementation
1. E00: same ScanQA rows, original VEGA architecture and observation frames.
2. E01: append trainable foresight query slots before answers; task loss only.
3. E03: add projected query outputs as the frozen Wan conditional context;
   jointly encode the five-frame teacher clip with Wan's native temporal VAE;
   apply its native conditional flow/denoising objective in addition to QA loss.
4. Sample noise/time using the actual Wan implementation. Do not re-use the
   detached single-frame feature-extraction forward for this loss. Preserve
   gradients to the conditional projection and query-producing network.
5. Check gradient flow, query use, both losses, memory and clean reload on a
   short run before deciding full-data scale and optimization settings.

The observed first frame in the teacher clip is an anchor. Whether it is supplied
as an explicit image condition or its loss is masked is an implementation choice
to record and test; this data manifest does not implement either convention.
Spatial resolution, query count, trainable parameters and loss weights remain
unfixed. Existing 1280x704 single-image extraction settings are not automatically
an affordable or optimal video-training resolution.

Heuristic camera thresholds (0.5 m, 30 degrees) only select a manageable debugging
subset. They are not action tolerances, a proposed contribution, or optimized
hyperparameters. Inspect sample images before interpreting learning curves.
