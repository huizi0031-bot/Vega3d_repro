"""Read teacher-only RGB clips without changing VEGA's QA observation stream."""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


TARGET_MODE = "heldout_view_clip_prediction_not_strict_future"


def validate_pair(row, observed_frames=None, expected_frames=5):
    pair = row.get("foresight")
    if not isinstance(pair, dict) or pair.get("video") != row.get("video"):
        raise ValueError("Missing or mismatched foresight pair")
    if pair.get("target_mode") != TARGET_MODE:
        raise ValueError("Unsupported target protocol; this pilot is heldout-view prediction")
    normalize = lambda paths: [os.path.normpath(str(p)) for p in paths]
    observed = normalize(pair["observed_frames"])
    clip = normalize(pair["teacher_clip_frames"])
    targets = normalize(pair["heldout_target_frames"])
    if len(clip) != expected_frames or len(set(clip)) != len(clip) or targets != clip[1:]:
        raise ValueError("Teacher clip must contain one anchor and distinct ordered target frames")
    if clip[0] != os.path.normpath(pair["anchor_frame"]) or clip[0] not in observed:
        raise ValueError("Teacher anchor must be one of the observed QA frames")
    if set(targets) & set(observed):
        raise ValueError("Teacher targets overlap QA input RGB frames")
    if observed_frames is not None and observed != normalize(observed_frames):
        raise ValueError("Actual QA frame selection differs from the prepared foresight manifest")
    ids = [int(Path(p).stem) for p in clip]
    if any(b-a != 10 for a, b in zip(ids, ids[1:])):
        raise ValueError("Teacher frames are not in the recorded consecutive ScanNet order")
    return pair


def load_teacher_pixels(row, height, width, expected_frames=5, observed_frames=None):
    pair = validate_pair(row, observed_frames, expected_frames)
    if height % 32 or width % 32 or expected_frames < 5 or (expected_frames-1) % 4:
        raise ValueError("Wan2.2 requires H,W divisible by 32 and 4k+1 RGB frames")
    frames = []
    for name in pair["teacher_clip_frames"]:
        with Image.open(name) as image:
            image = ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.BICUBIC)
            frames.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
    # [C,T,H,W], RGB float32 [-1,1]; not SigLIP-normalized images.
    return torch.stack(frames, dim=1).float().div_(127.5).sub_(1.)
