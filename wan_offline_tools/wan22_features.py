"""Wan2.2 offline feature extraction shared by ScanQA preprocessing tools."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace


def scene_seed(scene, base_seed):
    return (base_seed + int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16)) % (2**31)


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
    print("Offline extraction finished.", flush=True)

