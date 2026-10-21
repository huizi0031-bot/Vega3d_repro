"""Validate and load teacher-only clips independently of the task's dataset name."""
import os
from pathlib import Path

TARGET_MODE = 'heldout_view_clip_prediction_not_strict_future'
TARGET_MODES = {TARGET_MODE, 'heldout_view_clip_prediction'}


def validate_pair(row, observed_frames=None, expected_frames=5):
    pair = row.get('foresight')
    if not isinstance(pair, dict) or pair.get('video') != row.get('video'):
        raise ValueError('Missing or mismatched foresight pair')
    if pair.get('target_mode') not in TARGET_MODES:
        raise ValueError('Unsupported target protocol: expected heldout-view clip prediction')
    def normalize(paths):
        if not isinstance(paths, (list, tuple)) or not paths or any(not isinstance(p, str) or not p for p in paths):
            raise ValueError('Frame paths must be a nonempty ordered list of strings')
        return [os.path.normcase(str(Path(p).expanduser().resolve())) for p in paths]
    observed = normalize(pair.get('observed_frames'))
    clip = normalize(pair.get('teacher_clip_frames'))
    targets = normalize(pair.get('heldout_target_frames'))
    if len(observed) != len(set(observed)):
        raise ValueError('Observed frame paths must be distinct')
    if len(clip) != expected_frames or len(set(clip)) != len(clip) or targets != clip[1:]:
        raise ValueError('Teacher clip must contain one anchor and distinct ordered target frames')
    anchor = pair.get('anchor_frame')
    if not isinstance(anchor, str) or normalize([anchor])[0] != clip[0] or clip[0] not in observed:
        raise ValueError('Teacher anchor must be one of the observed input frames')
    if set(targets) & set(observed):
        raise ValueError('Teacher targets overlap input RGB frames')
    if observed_frames is not None and observed != normalize(observed_frames):
        raise ValueError('Actual observation frame selection differs from the foresight manifest')
    # Explicit list order defines the clip. Datasets may use arbitrary filenames
    # and frame intervals; the data preparation code is responsible for that order.
    return pair


def load_teacher_pixels(row, height, width, expected_frames=5, observed_frames=None):
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    pair = validate_pair(row, observed_frames, expected_frames)
    if height <= 0 or width <= 0 or height % 32 or width % 32 or expected_frames < 5 or (expected_frames-1) % 4:
        raise ValueError('Wan2.2 requires positive H,W divisible by 32 and 4k+1 RGB frames')
    frames = []
    for name in pair['teacher_clip_frames']:
        with Image.open(name) as image:
            image = ImageOps.fit(image.convert('RGB'), (width, height), method=Image.Resampling.BICUBIC)
            frames.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
    return torch.stack(frames, dim=1).float().div_(127.5).sub_(1.)
