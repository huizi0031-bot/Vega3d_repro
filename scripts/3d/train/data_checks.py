"""Dataset manifests and CPU preflight shared by scene-understanding training."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

import yaml

REPO = Path(__file__).resolve().parents[3]


def resolve_path(value, root=REPO):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else root / path


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    path = Path(path)
    text = path.read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Expected a nonempty array/JSONL: {path}')
    return rows


def dataset_specification(path):
    """Resolve JSON/JSONL or a VEGA datasets YAML, relative to the repository."""
    path = resolve_path(path)
    if path.suffix in {'.yaml', '.yml'}:
        datasets = yaml.safe_load(path.read_text()).get('datasets')
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f'Empty datasets specification: {path}')
    else:
        datasets = [dict(json_path=str(path), sampling_strategy='all')]
    normalized = []
    for item in datasets:
        if not isinstance(item, dict) or 'json_path' not in item:
            raise ValueError('Each dataset needs json_path')
        normalized.append(dict(json_path=str(resolve_path(item['json_path'])),
                               sampling_strategy=item.get('sampling_strategy', 'all')))
    return {'datasets': normalized}


def selected_rows(specification, seed):
    rng = random.Random(seed)
    rows, hashes = [], {}
    for dataset in specification['datasets']:
        path = Path(dataset['json_path'])
        current = read_rows(path)
        hashes[str(path)] = file_digest(path)
        strategy = dataset['sampling_strategy']
        if strategy != 'all':
            try:
                method, count = strategy.split(':')
                count = math.ceil(int(count[:-1]) * len(current) / 100) if count.endswith('%') else int(count)
            except (AttributeError, ValueError):
                raise ValueError(f'Invalid sampling strategy: {strategy}') from None
            if count < 1 or method not in {'first', 'end', 'random'}:
                raise ValueError(f'Invalid sampling strategy: {strategy}')
            if method == 'random':
                rng.shuffle(current)
            current = current[-count:] if method == 'end' else current[:count]
        rows.extend(current)
    if not rows:
        raise ValueError('No selected training examples')
    return rows, hashes


def check_weight_files(directory, prefix='model'):
    directory = Path(directory)
    index = directory / f'{prefix}.safetensors.index.json'
    if index.is_file():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if not names:
            raise ValueError(f'Empty model index: {index}')
    else:
        names = {f'{prefix}.safetensors'}
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing or empty model weights: {path}')


def validate_rows(rows, config, experiment):
    scenes = set()
    for row in rows:
        video = row.get('video')
        turns = row.get('conversations')
        if not isinstance(video, str) or not video.startswith('scannet/'):
            raise ValueError('Current 3D loader requires a scannet/<scene> video ID and matching scene assets')
        if not isinstance(turns, list) or not turns:
            raise ValueError('Each example needs conversations')
        if experiment['queries_enabled']:
            if (len(turns) != 2 or turns[0].get('from') != 'human'
                    or turns[1].get('from') != 'gpt'
                    or any(not isinstance(t.get('value'), str) or not t['value'].strip() for t in turns)):
                raise ValueError('Foresight currently requires one human prompt and one nonempty gpt answer')
            if row.get('metadata', {}).get('dataset', '').lower() in {'scanrefer', 'multi3drefer'}:
                raise ValueError('Object-proposal grounding is not connected to the foresight forward')
        scenes.add(video)
    return scenes


def audit_scene_assets(processor, video, data, files):
    import numpy as np
    from PIL import Image
    from llava.train.foresight_sampling import check_offline_header
    observed = processor.sample_frame_files(video, force_sample=True, frames_upbound=data['frame_count'])
    if len(set(observed)) != data['frame_count']:
        raise ValueError('Not enough distinct observation frames')
    meta = processor.scene[video]
    for key in ['axis_align_matrix', 'depth_cam2img']:
        value = np.asarray(meta[key])
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f'Invalid scene geometry: {key}')
        if key == 'depth_cam2img' and (value[0, 0] == 0 or value[1, 1] == 0):
            raise ValueError('Invalid focal length')
    errors = []
    for name in observed:
        path = resolve_path(name)
        for asset in [path, path.with_suffix('.png'), path.with_suffix('.txt')]:
            files.add(asset)
            if not asset.is_file():
                errors.append(f'missing {asset}')
                continue
            try:
                if asset.suffix == '.txt':
                    pose = np.loadtxt(asset)
                    if pose.shape != (4, 4) or not np.isfinite(pose).all():
                        raise ValueError('invalid pose')
                else:
                    with Image.open(asset) as image:
                        image.verify()
            except (OSError, ValueError) as exc:
                errors.append(f'{asset}: {exc}')
    cache = REPO / 'data/scannet' / data['feature_id'] / video.split('/')[-1] / 'wan22_ti2v.pt'
    if not cache.is_file():
        errors.append(f'missing offline features: {cache}')
    else:
        try:
            check_offline_header(cache, video, observed, data['feature_id'])
        except Exception as exc:
            errors.append(f'offline feature header: {exc}')
    return observed, errors


def preflight(config, experiment):
    """Audit original annotations and select teacher clips without rewriting QA."""
    data, model = config['data'], config['model']
    specification = dataset_specification(data['train'])
    rows, hashes = selected_rows(specification, config['seed'])
    hashes[str(resolve_path(data['train']))] = file_digest(resolve_path(data['train']))
    scenes = validate_rows(rows, config, experiment)
    for key in ['initial_checkpoint', 'vision_checkpoint']:
        directory = resolve_path(model[key])
        if not (directory / 'config.json').is_file():
            raise FileNotFoundError(directory / 'config.json')
        check_weight_files(directory)
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        if not (resolve_path(model['initial_checkpoint']) / name).is_file():
            raise FileNotFoundError(name)
    for split in ['train', 'val', 'test']:
        path = REPO / 'data/embodiedscan' / f'embodiedscan_infos_{split}.pkl'
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ['scannet_train_gt_box.json', 'scannet_val_pred_box.json']:
        path = REPO / 'data/metadata' / name
        if not path.is_file():
            raise FileNotFoundError(path)
    language = json.loads((resolve_path(model['initial_checkpoint']) / 'config.json').read_text())
    if language.get('model_type') not in {'qwen3_5_text', 'llava_qwen35'} or language.get('hidden_size') != model['language_hidden_size']:
        raise ValueError('Language checkpoint does not match model.language_hidden_size / Qwen3.5 architecture')
    if experiment['queries_enabled'] and not 1 <= model.get('queries', 32) <= 512:
        raise ValueError('Query count must be between 1 and 512')
    if experiment['video_loss_weight'] > 0:
        h, w, frames = model['teacher_height'], model['teacher_width'], model['teacher_frames']
        if h <= 0 or w <= 0 or h % 32 or w % 32 or frames < 5 or (frames - 1) % 4:
            raise ValueError('Wan requires H,W divisible by 32 and 4k+1 frames (k>=1)')
        directory = resolve_path(model['wan_checkpoint'])
        check_weight_files(directory, 'diffusion_pytorch_model')
        wan = json.loads((directory / 'config.json').read_text())
        expected = dict(model_type='ti2v', dim=3072, in_dim=48, out_dim=48, num_layers=30)
        if any(wan.get(key) != value for key, value in expected.items()):
            raise ValueError('Expected Wan2.2-TI2V-5B video teacher')
        if model['wan_context_dim'] != wan.get('text_dim', 4096) or not (directory / 'Wan2.2_VAE.pth').is_file():
            raise ValueError('Wan conditioning dimension or VAE asset mismatch')
    validation, validation_scenes = [], set()
    if data.get('validation'):
        validation, val_hashes = selected_rows(dataset_specification(data['validation']), config['seed'])
        hashes.update(val_hashes)
        hashes[str(resolve_path(data['validation']))] = file_digest(resolve_path(data['validation']))
        validation_scenes = {row['video'] for row in validation}
        if data.get('require_disjoint_validation_scenes', False) and scenes & validation_scenes:
            raise ValueError('Training and validation scenes overlap')
    from llava.video_utils import VideoProcessor
    from llava.train.foresight_sampling import ScenePairSampler, pairing_settings
    processor = VideoProcessor(video_folder='data', annotation_dir='data/embodiedscan/',
                               frame_sampling_strategy='uniform')
    settings = pairing_settings(model.get('teacher_frames', 5), data.get('max_translation', 0.5),
                                data.get('max_rotation', 30.0))
    sampler = ScenePairSampler(processor, settings['teacher_frames'], settings['max_translation'], settings['max_rotation'])
    representatives = {}
    for row in rows:
        representatives.setdefault(row['video'], row)
        if 'foresight' not in row:
            representatives[row['video']] = row
    files, clips, plans = set(), set(), {}
    errors, validation_errors, observed_by_scene = [], [], {}
    for index, video in enumerate(sorted(scenes | validation_scenes)):
        destination = errors if video in scenes else validation_errors
        try:
            observed, issues = audit_scene_assets(processor, video, data, files)
            observed_by_scene[video] = observed
            destination.extend(dict(scene=video, reason=issue) for issue in issues)
            if experiment['video_loss_weight'] > 0 and video in scenes:
                row = representatives[video]
                pair = sampler.for_row(row, observed)
                if 'foresight' not in row:
                    plans[video] = pair
                clips.add(tuple(pair['teacher_clip_frames']))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            destination.append(dict(scene=video, reason=str(exc)))
        if (index + 1) % 50 == 0:
            import sys
            print(f'Audited {index + 1}/{len(scenes | validation_scenes)} scenes', file=sys.stderr, flush=True)
    # Explicit manifests remain supported, including multiple clips per scene.
    if experiment['video_loss_weight'] > 0:
        for row in rows:
            if 'foresight' not in row or row['video'] not in observed_by_scene:
                continue
            try:
                pair = sampler.for_row(row, observed_by_scene[row['video']])
                clip = tuple(pair['teacher_clip_frames'])
                clips.add(clip)
            except (ValueError, KeyError) as exc:
                errors.append(dict(scene=row['video'], row_id=row.get('id'), reason=str(exc)))
        from PIL import Image
        for clip in sorted(clips):
            for name in clip:
                path = resolve_path(name)
                files.add(path)
                files.add(path.with_suffix('.txt'))
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    errors.append(dict(scene='teacher_clip', reason=f'{path}: {exc}'))
    pair_plan = dict(schema_version=1, settings=settings, scenes=plans)
    plan_text = json.dumps(pair_plan, sort_keys=True)
    hashes['foresight_selection'] = hashlib.sha256(plan_text.encode()).hexdigest()
    # Fingerprint the geometry and selected assets used by this audit. Large
    # feature storage is not byte-hashed or scanned for finite values here.
    asset_state = []
    for video, observed in sorted(observed_by_scene.items()):
        meta = processor.scene[video]
        asset_state.append([video, observed, meta['axis_align_matrix'], meta['depth_cam2img']])
        cache = REPO / 'data/scannet' / data['feature_id'] / video.split('/')[-1] / 'wan22_ti2v.pt'
        if cache.is_file():
            stat = cache.stat()
            asset_state.append([str(cache), stat.st_size, stat.st_mtime_ns])
    for path in sorted(files):
        if path.is_file():
            stat = path.stat()
            asset_state.append([str(path), stat.st_size, stat.st_mtime_ns,
                                file_digest(path) if path.suffix == '.txt' else None])
    hashes['scene_assets'] = hashlib.sha256(json.dumps(asset_state, sort_keys=True, default=lambda x: x.tolist()).encode()).hexdigest()
    return dict(train_rows=len(rows), train_scenes=len(scenes), validation_rows=len(validation),
                validation_scenes=len(validation_scenes), datasets=specification,
                train_path=str(resolve_path(data['train'])), data_hashes=hashes,
                distinct_teacher_clips=len(clips), auto_paired_scenes=len(plans),
                explicit_pair_rows=sum('foresight' in row for row in rows), checked_files=len(files),
                ready_for_training=not errors, errors=errors, validation_errors=validation_errors,
                validation_missing_feature_scenes=[s for s in sorted(validation_scenes) if not cache_path(s, data).is_file()],
                train_validation_scene_overlap=sorted(scenes & validation_scenes), _pair_plan=pair_plan)


def cache_path(scene, data):
    return REPO / 'data/scannet' / data['feature_id'] / scene.split('/')[-1] / 'wan22_ti2v.pt'
