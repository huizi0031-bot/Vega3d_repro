#!/usr/bin/env python3
"""Prepare matched ScanQA QA subsets and heldout-view supervision manifests."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import random
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--val-scenes", type=int, default=5)
    parser.add_argument("--questions-per-scene", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.output}")

    # This script uses CPU tensors only; no model checkpoint is instantiated.
    import numpy as np
    import torch
    from PIL import Image

    root = args.project.resolve()
    feature_id = "wan22_ti2v_offline_uniform32_1280x704_t300_b20_s42_v1"
    cache_root = root / "data/scannet" / feature_id
    train_path = root / "data/processed/scanqa_train_llava_style.json"
    val_path = root / "data/processed/scanqa_val_llava_style.json"
    train, val = read_json(train_path), read_json(val_path)
    train_groups, val_groups = collections.defaultdict(list), collections.defaultdict(list)
    for row in train:
        train_groups[row["video"]].append(row)
    for row in val:
        val_groups[row["video"]].append(row)
    assert not (set(train_groups) & set(val_groups)), "Scene split overlap"
    assert not ({x.split("_")[0] for x in train_groups} &
                {x.split("_")[0] for x in val_groups}), "Physical-room split overlap"

    rng = random.Random(args.seed)
    candidates = sorted(k for k, rows in train_groups.items()
                        if len(rows) >= args.questions_per_scene)
    rng.shuffle(candidates)
    pairs, selected_train, skipped = [], [], []
    for video in candidates:
        if len(pairs) == args.train_scenes:
            break
        scene = video.split("/")[-1]
        cache_path = cache_root / scene / "wan22_ti2v.pt"
        if not cache_path.is_file():
            skipped.append({"scene": scene, "reason": "missing_input_feature_cache"})
            continue
        payload = torch.load(cache_path, map_location="meta", weights_only=True)
        assert tuple(payload["features"].shape) == (32, 3072, 14, 14)
        metadata = payload["metadata"]
        assert metadata["video_id"] == video
        observed = [str(Path(x)) for x in metadata["frame_files"]]
        assert len(observed) == len(set(observed)) == 32
        assert all((root / f).is_file() for f in observed)
        scene_dir = root / "data/scannet/posed_images" / scene
        frames = sorted(scene_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        index = {p.name: i for i, p in enumerate(frames)}
        observed_names = {Path(f).name for f in observed}
        anchors = [index[Path(f).name] for f in observed]
        # Prefer the middle of the scan, but preserve exactly the cached input.
        anchors.sort(key=lambda i: (abs(i - (len(frames) - 1) / 2), i))
        picked = None
        for anchor_index in anchors:
            clip = frames[anchor_index:anchor_index + 5]
            if len(clip) != 5 or any(p.name in observed_names for p in clip[1:]):
                continue
            ids = [int(p.stem) for p in clip]
            if any(b - a != 10 for a, b in zip(ids, ids[1:])):
                continue
            try:
                poses = [np.loadtxt(p.with_suffix(".txt")) for p in clip]
                if any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
                    continue
                origin = poses[0]
                displacement = max(float(np.linalg.norm(p[:3, 3] - origin[:3, 3]))
                                   for p in poses[1:])
                rotations = [origin[:3, :3].T @ p[:3, :3] for p in poses[1:]]
                rotation_deg = max(math.degrees(math.acos(float(np.clip(
                    (np.trace(rotation) - 1) / 2, -1, 1)))) for rotation in rotations)
                if displacement > 0.5 or rotation_deg > 30:
                    continue
                for p in clip:
                    with Image.open(p) as image:
                        image.verify()
                picked = (clip, ids, displacement, rotation_deg)
                break
            except (OSError, ValueError):
                continue
        if picked is None:
            skipped.append({"scene": scene, "reason": "no_suitable_disjoint_clip"})
            continue
        clip, ids, displacement, rotation_deg = picked
        clip_paths = [str(p.relative_to(root)) for p in clip]
        pair = {
            "scene": scene, "video": video,
            "observed_frames": observed,
            "observed_feature_cache": str(cache_path.relative_to(root)),
            "anchor_frame": clip_paths[0],
            "heldout_target_frames": clip_paths[1:],
            "teacher_clip_frames": clip_paths,
            "teacher_clip_frame_ids": ids,
            "teacher_clip_pose_files": [str(Path(p).with_suffix(".txt")) for p in clip_paths],
            "max_translation_from_anchor_m": displacement,
            "max_rotation_from_anchor_deg": rotation_deg,
            "target_mode": "heldout_view_clip_prediction_not_strict_future",
            "first_frame_is_observed_anchor": True,
            "heldout_rgb_input_overlap": [],
            "teacher_clip_encoding": "joint temporal VAE encoding; not independent still-image encodes",
        }
        assert pair["anchor_frame"] in observed
        assert not (set(observed) & set(pair["heldout_target_frames"]))
        pairs.append(pair)
        chosen = list(train_groups[video])
        rng.shuffle(chosen)
        selected_train.extend(chosen[:args.questions_per_scene])
        print("PAIRED", len(pairs), scene, "target_ids", ids[1:], flush=True)

    assert len(pairs) == args.train_scenes, "Insufficient eligible training scenes"
    val_candidates = sorted(k for k, rows in val_groups.items()
                            if len(rows) >= args.questions_per_scene)
    rng.shuffle(val_candidates)
    assert len(val_candidates) >= args.val_scenes
    selected_val = []
    for video in val_candidates[:args.val_scenes]:
        rows = list(val_groups[video])
        rng.shuffle(rows)
        selected_val.extend(rows[:args.questions_per_scene])
    pair_by_video = {pair["video"]: pair for pair in pairs}
    paired_train = [dict(row, foresight=pair_by_video[row["video"]]) for row in selected_train]
    assert len({row["id"] for row in selected_train}) == len(selected_train)
    assert not ({row["video"] for row in selected_train} &
                {row["video"] for row in selected_val})

    summary = {
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "project": str(root), "output": str(args.output), "seed": args.seed,
        "status": "prepared",
        "train_records": len(selected_train), "train_scenes": len(pairs),
        "validation_records": len(selected_val),
        "validation_scenes": len({row["video"] for row in selected_val}),
        "train_val_scene_overlap": 0,
        "observed_frames_per_train_sample": 32,
        "teacher_clip_frames": 5, "new_target_frames": 4,
        "target_mode": "heldout views inside a globally observed scan; NOT causal future prediction",
        "image_checks": "all observed train paths exist; all teacher-clip JPEGs pass PIL verify",
        "camera_filter": {"max_translation_m": 0.5, "max_rotation_deg": 30},
        "frame_rate_note": "filenames spaced by 10; repository assumes 30 Hz source / 3 Hz extracted; verify before temporal claims",
        "train_input_cache_alignment": "exact frame list from each existing payload metadata",
        "validation_cache_note": "original QA protocol; uniform32 validation feature caches must be checked/extracted separately",
        "sources": {str(p): digest(p) for p in [train_path, val_path]},
        "feature_id": feature_id, "skipped_candidates": skipped,
        "limitations": [
            "Pilot checks data/gradient plumbing only; not a publication-scale or effect-estimation dataset.",
            "One target clip per scene is a debugging convenience; questions from a scene are not independent video targets.",
            "Heldout RGB exclusion does not exclude scene-level geometry/boxes from existing 3D metadata; keep these equal across baselines and report them.",
            "VSI-Bench scene-overlap audit is still required before reporting external benchmark generalization.",
            "This is a proposed auxiliary view-prediction objective, not evidence of improved reasoning or dynamics.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "train_original.json", selected_train)
    write_json(args.output / "val_original.json", selected_val)
    write_json(args.output / "scene_pairs.json", pairs)
    with (args.output / "train_paired.jsonl").open("w") as handle:
        for row in paired_train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output / "summary.json", summary)
    (args.output / "scanqa_only.yaml").write_text(
        "datasets:\n  - json_path: " + str(args.output / "train_original.json") +
        '\n    sampling_strategy: "all"\n')
    (args.output / "prepare_scanqa_foresight_pilot.py").write_text(Path(__file__).read_text())
    readme = """# ScanQA data manifests

- train_original.json: matched baseline QA records.
- train_paired.jsonl: the same records plus foresight frame-pair metadata.
- val_original.json: QA records from separate official validation scenes.
- scene_pairs.json: one observed-anchor/heldout-target plan per training scene.
- summary.json: source hashes, counts, frame checks and camera thresholds.

Targets are heldout views within a globally observed static scan, not strictly
causal future frames. Keep the original observation-frame order and geometry
metadata unchanged. Training entry: scripts/3d/train/train_scanqa_foresight.py.
"""
    (args.output / "README.md").write_text(readme)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
