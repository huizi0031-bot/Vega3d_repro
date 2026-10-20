#!/usr/bin/env python3
"""Prepare a small or full five-task dataset and extract unprojected Wan-T2V features.

Uses this repository's VideoProcessor and Wan encoder, with the same configured number of
uniformly sampled, resized, SigLIP-normalized input frames used in training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASETS = [
    "sqa3d_train_llava_style.json", "scanqa_train_llava_style.json",
    "scan2cap_train_llava_style.json", "scanrefer_vg_train_llava_style.json",
    "multi3drefer_train_llava_style.json",
]


def select_samples(datasets, scene_count=2, samples_per_scene=2):
    scene_sets = [{r["video"] for r in rows if isinstance(r.get("video"), str)}
                  for rows in datasets]
    common = set.intersection(*scene_sets)
    if len(common) < scene_count:
        raise ValueError(f"Need {scene_count} scenes shared by all five datasets; found {len(common)}")
    scenes = sorted(common)[:scene_count]
    selected = []
    counts = []
    for rows in datasets:
        subset = []
        for scene in scenes:
            matches = [r for r in rows if r.get("video") == scene]
            subset.extend(matches[:samples_per_scene])
        selected.extend(subset)
        counts.append(len(subset))
    return scenes, selected, counts


def scene_seed(scene, base_seed):
    return (base_seed + int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16)) % (2**31)


def load_full_data(data_path):
    import yaml
    path = Path(data_path)
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or not spec.get("datasets"):
        raise ValueError("Full training requires the original datasets YAML")
    sources, datasets = [], []
    for item in spec["datasets"]:
        if item.get("sampling_strategy", "all") != "all":
            raise ValueError("Full training requires sampling_strategy=all")
        name = item["json_path"]
        rows = json.loads(Path(name).read_text())
        if not rows or any(not isinstance(row.get("video"), str) for row in rows):
            raise ValueError("Empty data or missing video: " + name)
        sources.append({"json_path": name, "samples": len(rows),
                        "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()})
        datasets.append(rows)
    if len(datasets) != 5:
        raise ValueError("Expected all five original training datasets")
    scenes = sorted({row["video"] for rows in datasets for row in rows})
    return sources, datasets, scenes


def prepare_plan(args):
    full = getattr(args, "all_train", False)
    if full:
        sources, rows, scenes = load_full_data(args.data_path)
        selected = None
        counts = [len(dataset) for dataset in rows]
    else:
        rows = [json.loads((Path("data/processed") / name).read_text()) for name in DATASETS]
        scenes, selected, counts = select_samples(rows, args.scenes)
    feature_id = (f"wan22_ti2v_offline_uniform{args.frames}_{args.size.replace('*', 'x')}"
                  f"_t{args.timestep}_b{args.block}_s{args.seed}_v1")
    root = Path("data/scannet") / feature_id
    plan = {
        "format_version": 1, "feature_id": feature_id, "scenes": scenes,
        "frames": args.frames, "size": args.size, "timestep": args.timestep,
        "block": args.block, "seed": args.seed, "shift": 5.0,
        "frame_chunk_size": args.chunk_frames, "strategy": "resize",
        "sampling": "uniform", "feature_dim": 3072, "spatial": [14, 14],
        "input_dtype": "bfloat16", "image_processor": "SigLipImageProcessor-384",
        "checkpoint": args.checkpoint, "prompt_embedding": args.prompt_embedding,
        "sample_counts": dict(zip(DATASETS, counts)),
    }
    if full:
        plan["sample_counts"] = {source["json_path"]: source["samples"] for source in sources}
        plan["scope"] = "full training scene union; original YAML sampling=all"
        plan["training_data"] = str(args.data_path)
        plan["training_sources"] = sources
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError(f"Existing manifest has different settings: {manifest}; use a new seed/configuration")
    if getattr(args, "check_only", False):
        if not manifest.is_file():
            raise FileNotFoundError("Extract full features first: " + str(manifest))
    else:
        root.mkdir(parents=True, exist_ok=True)
        generated = Path("wan_offline_tools/generated")
        generated.mkdir(parents=True, exist_ok=True)
        def atomic_json(path, value):
            temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
            temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
            temp.replace(path)
        atomic_json(manifest, plan)
        if full:
            atomic_json(generated / "qwen35_wan_full_manifest.json", {"manifest": str(manifest)})
        else:
            atomic_json(generated / "qwen35_wan_offline_smoke.json", selected)
            atomic_json(generated / "qwen35_wan_offline_manifest.json", {"manifest": str(manifest)})
    print("[Qwen3.5] Scenes:", len(scenes), flush=True)
    print("[Qwen3.5] Samples per dataset:", plan["sample_counts"], flush=True)
    print("[Qwen3.5] Total samples:", sum(counts), flush=True)
    print("[Qwen3.5] Feature ID:", feature_id, flush=True)
    return plan, root


def extract(args, plan, root):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from llava.video_utils import VideoProcessor
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    from llava.model.multimodal_generative_encoder.builder import build_generative_vision_tower
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Extraction expects one visible GPU; pass --gpu with a single physical GPU index")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This BF16 extraction configuration requires BF16 support")
    device = torch.device("cuda:0")
    print("Extraction GPU:", args.gpu, torch.cuda.get_device_name(0), flush=True)
    for required in (Path(args.checkpoint), Path(args.prompt_embedding)):
        if not required.exists():
            raise FileNotFoundError(required)
    processor = VideoProcessor(video_folder="data", annotation_dir="data/embodiedscan/",
                               frame_sampling_strategy="uniform")
    image_processor = SigLipImageProcessor()
    cfg = SimpleNamespace(
        generative_vision_tower_type="wan22_ti2v_online",
        generative_vision_tower_task="ti2v-5B",
        generative_vision_tower_checkpoint=args.checkpoint,
        generative_vision_tower_size=args.size,
        generative_vision_tower_timestep=args.timestep,
        generative_vision_tower_shift=5.0,
        generative_vision_tower_feat_block_idx=args.block,
        generative_vision_tower_prompt_emb_path=args.prompt_embedding,
        generative_vision_tower_dtype="bf16",
    )
    # Standalone process: no LLM, no DeepSpeed ZeRO initialization context.
    tower = build_generative_vision_tower(cfg).eval().requires_grad_(False)
    scene_pool = plan["scenes"][:args.scene_limit] if getattr(args, "scene_limit", None) else plan["scenes"]
    worker_scenes = scene_pool[getattr(args, "shard_index", 0)::getattr(args, "num_shards", 1)]
    for scene in worker_scenes:
        target = root / scene.split("/")[-1] / "wan22_ti2v.pt"
        frame_files = [os.path.normpath(p) for p in processor.sample_frame_files(
            scene, force_sample=True, frames_upbound=args.frames)]
        if target.exists():
            payload = torch.load(target, map_location="cpu", weights_only=True)
            assert payload["metadata"]["plan"] == plan, f"Stale feature: {target}"
            assert payload["metadata"]["frame_files"] == frame_files, f"Changed frames: {target}"
            existing = payload["features"]
            assert tuple(existing.shape) == (args.frames, 3072, 14, 14)
            assert torch.isfinite(existing).all(), f"Non-finite feature: {target}"
            print("Validated existing feature:", target, flush=True)
            continue
        print("Extracting:", scene, flush=True)
        video = processor.process_3d_video(
            scene, image_processor=image_processor, force_sample=True,
            frames_upbound=args.frames, strategy="resize", generative_feature_source="none")
        frames = video["images"]
        assert tuple(frames.shape) == (args.frames, 3, 384, 384), frames.shape
        seed = scene_seed(scene, args.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        parts = []
        with torch.inference_mode():
            for chunk in frames.split(args.chunk_frames):
                chunk = chunk.to(device=device, dtype=torch.bfloat16)
                feature = tower(chunk, split_sizes=[len(chunk)])
                assert tuple(feature.shape) == (len(chunk), 3072, 14, 14), feature.shape
                assert torch.isfinite(feature).all(), "Wan produced non-finite features"
                parts.append(feature.to(device="cpu", dtype=torch.bfloat16))
                print(f"  Extracted {sum(len(p) for p in parts)}/{args.frames} frames", flush=True)
        features = torch.cat(parts).contiguous()
        metadata = {"video_id": scene, "frame_files": frame_files, "plan": plan,
                    "scene_seed": seed}
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save({"features": features, "metadata": metadata}, temporary)
        temporary.replace(target)
        print("Saved:", target, tuple(features.shape), flush=True)
    print("Offline extraction finished. Run configure_wan_offline.py next.", flush=True)


def validate_full_features(plan, root):
    import torch
    from llava.video_utils import VideoProcessor
    processor = VideoProcessor(video_folder="data", annotation_dir="data/embodiedscan/",
                               frame_sampling_strategy="uniform")
    for i, scene in enumerate(plan["scenes"], 1):
        path = root / scene.split("/")[-1] / "wan22_ti2v.pt"
        if not path.is_file():
            raise FileNotFoundError("Incomplete full feature coverage: " + str(path))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        frames = [os.path.normpath(p) for p in processor.sample_frame_files(
            scene, force_sample=True, frames_upbound=plan["frames"])]
        if metadata["plan"] != plan or metadata["video_id"] != scene or metadata["frame_files"] != frames:
            raise ValueError("Stale feature or frame mismatch: " + str(path))
        features = payload["features"]
        if tuple(features.shape) != (plan["frames"], 3072, 14, 14) or not torch.isfinite(features).all():
            raise ValueError("Invalid shape or non-finite features: " + str(path))
        for frame in frames:
            if not all(Path(p).is_file() for p in (frame, frame.replace(".jpg", ".png"), frame.replace("jpg", "txt"))):
                raise FileNotFoundError("Missing RGB/depth input: " + frame)
        if i % 25 == 0 or i == len(plan["scenes"]):
            print(f"[Qwen3.5] Full feature coverage verified: {i}/{len(plan['scenes'])}", flush=True)
    print("[Qwen3.5] All full-training scene features are valid.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--timestep", type=int, default=300)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-frames", type=int, default=2)
    parser.add_argument("--checkpoint", default="data/models/Wan2.2-TI2V-5B")
    parser.add_argument("--prompt-embedding", default="llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--all-train", action="store_true", help="Scene UNION of all five full training datasets")
    parser.add_argument("--data-path", default="scripts/3d/train/multi.yaml")
    parser.add_argument("--check-only", action="store_true", help="Validate all saved features without CUDA extraction")
    parser.add_argument("--scene-limit", type=int, help="Extract only first N planned scenes for a preflight")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if min(args.frames, args.scenes, args.chunk_frames) < 1:
        parser.error("frames, scenes and chunk-frames must be positive")
    if args.scene_limit is not None and args.scene_limit < 1:
        parser.error("scene-limit must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard index/count")
    if args.check_only and not args.all_train:
        parser.error("--check-only requires --all-train")
    # The installed tools directory lives directly under the repository root.
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    plan, root = prepare_plan(args)
    if args.check_only:
        validate_full_features(plan, root)
    elif not args.plan_only:
        extract(args, plan, root)


if __name__ == "__main__":
    main()
