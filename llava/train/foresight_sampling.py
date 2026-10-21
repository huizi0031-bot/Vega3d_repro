"""Deterministic heldout-view sampling on the original ScanNet scene layout."""
import json
import math
import os
from pathlib import Path

import numpy as np

from llava.train.foresight_data import TARGET_MODE, validate_pair


def pairing_settings(frames=5, max_translation=0.5, max_rotation=30.0):
    if frames < 5 or (frames - 1) % 4:
        raise ValueError('Teacher clips require 4k+1 frames, k >= 1')
    if not math.isfinite(max_translation) or max_translation < 0:
        raise ValueError('max_translation must be finite and nonnegative')
    if not math.isfinite(max_rotation) or not 0 <= max_rotation <= 180:
        raise ValueError('max_rotation must be between 0 and 180 degrees')
    return dict(teacher_frames=frames, max_translation=max_translation,
                max_rotation=max_rotation, selection='middle_anchor_following_saved_frames')


def normalized(paths):
    return [os.path.normcase(str(Path(p).resolve())) for p in paths]


class ScenePairSampler:
    """Reuse observations verbatim; select teacher RGB without changing annotations.

    A supplied snapshot is strict: missing scenes are errors, never a reason to
    silently choose a different clip. Without a snapshot, a scene is selected
    once per loader process and memoized. Existing explicit pairs take priority.
    """

    def __init__(self, processor, frames=5, max_translation=0.5, max_rotation=30.0,
                 cache_path=None):
        self.processor = processor
        self.settings = pairing_settings(frames, max_translation, max_rotation)
        self.frames = frames
        self.pairs = {}
        self.snapshot = None
        if cache_path:
            document = json.loads(Path(cache_path).read_text())
            if document.get('schema_version') != 1 or document.get('settings') != self.settings:
                raise ValueError('Foresight selection snapshot settings differ from this run')
            if not isinstance(document.get('scenes'), dict):
                raise ValueError('Invalid foresight selection snapshot')
            self.snapshot = document['scenes']

    def for_row(self, row, observed):
        if 'foresight' in row:
            return validate_pair(row, observed, self.frames)
        video = row['video']
        if video not in self.pairs:
            if self.snapshot is not None:
                if video not in self.snapshot:
                    raise ValueError(f'No saved foresight selection for {video}')
                pair = self.snapshot[video]
            else:
                pair = self.select(video, observed)
            self.pairs[video] = pair
        pair = self.pairs[video]
        validate_pair(dict(video=video, foresight=pair), observed, self.frames)
        return pair

    def select(self, video, observed):
        if len(set(normalized(observed))) != len(observed):
            raise ValueError(f'{video}: observations must be distinct')
        if video not in self.processor.scene:
            raise ValueError(f'{video}: scene is absent from EmbodiedScan metadata')
        scene_dir = Path(self.processor.video_folder) / 'scannet/posed_images' / video.split('/')[-1]
        frames = list(scene_dir.glob('*.jpg'))
        try:
            frames.sort(key=lambda p: int(p.stem))
        except ValueError as exc:
            raise ValueError(f'{video}: automatic ScanNet sampling requires numeric JPEG names') from exc
        indices = {name: i for i, name in enumerate(normalized(frames))}
        try:
            anchors = [indices[name] for name in normalized(observed)]
        except KeyError as exc:
            raise ValueError(f'{video}: observed frame is absent from posed_images: {exc}') from exc
        anchors.sort(key=lambda i: (abs(i - (len(frames) - 1) / 2), i))
        observed_set = set(normalized(observed))
        for index in anchors:
            clip = frames[index:index + self.frames]
            if len(clip) != self.frames or observed_set.intersection(normalized(clip[1:])):
                continue
            try:
                poses = [np.loadtxt(p.with_suffix('.txt')) for p in clip]
                if any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
                    continue
                origin = poses[0]
                translation = max(float(np.linalg.norm(p[:3, 3] - origin[:3, 3])) for p in poses[1:])
                rotation = max(math.degrees(math.acos(float(np.clip(
                    (np.trace(origin[:3, :3].T @ p[:3, :3]) - 1) / 2, -1, 1)))) for p in poses[1:])
                if translation > self.settings['max_translation'] or rotation > self.settings['max_rotation']:
                    continue
                from PIL import Image
                for path in clip:
                    with Image.open(path) as image:
                        image.verify()
            except (OSError, ValueError):
                continue
            paths = [str(p) for p in clip]
            return dict(video=video, target_mode=TARGET_MODE, observed_frames=list(observed),
                        anchor_frame=paths[0], teacher_clip_frames=paths,
                        heldout_target_frames=paths[1:],
                        max_translation_from_anchor_m=translation,
                        max_rotation_from_anchor_deg=rotation)
        raise ValueError(f'{video}: no valid disjoint {self.frames}-frame clip '
                         f'within {self.settings["max_translation"]} m / {self.settings["max_rotation"]} deg; '
                         'check saved RGB/poses or explicitly revise the selection thresholds')


def check_offline_header(path, video, observed, feature_id):
    """Inspect shape/order without loading large feature storage into CPU RAM."""
    import torch
    payload = torch.load(path, map_location='meta', mmap=True, weights_only=True)
    meta = payload['metadata']
    if meta['video_id'] != video or meta['plan']['feature_id'] != feature_id:
        raise ValueError(f'{video}: offline feature protocol or scene mismatch')
    # Match the runtime VideoProcessor comparison, including path spelling.
    if meta['frame_files'] != [os.path.normpath(p) for p in observed]:
        raise ValueError(f'{video}: offline feature order differs from the original uniform sampling')
    if tuple(payload['features'].shape) != (len(observed), 3072, 14, 14):
        raise ValueError(f'{video}: invalid offline feature dimensions')
    return meta
