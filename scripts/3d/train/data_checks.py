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
    # The current VideoProcessor consumes ScanNet/EmbodiedScan scene assets.
    # Dataset names and question wording are independent of this asset layout.
    from llava.train.foresight_data import validate_pair
    scenes, clips, files = set(), set(), set()
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
        if experiment['video_loss_weight'] > 0:
            pair = validate_pair(row, expected_frames=config['model']['teacher_frames'])
            if len(pair['observed_frames']) != config['data']['frame_count']:
                raise ValueError('Recorded observations differ from data.frame_count')
            clip = tuple(str(resolve_path(p).resolve()) for p in pair['teacher_clip_frames'])
            clips.add(clip)
            for name in pair['observed_frames']:
                path = resolve_path(name)
                files.update((path, path.with_suffix('.png'), path.with_suffix('.txt')))
            files.update(Path(p) for p in clip)
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(path)
    if clips:
        from PIL import Image
        for clip in clips:
            for name in clip:
                with Image.open(name) as image:
                    image.verify()
    return scenes, clips, files


def preflight(config, experiment):
    """Validate selected annotations, teacher pairs and local assets without CUDA."""
    data, model = config['data'], config['model']
    specification = dataset_specification(data['train'])
    rows, hashes = selected_rows(specification, config['seed'])
    hashes[str(resolve_path(data['train']))] = file_digest(resolve_path(data['train']))
    scenes, clips, files = validate_rows(rows, config, experiment)
    cache = lambda scene: REPO / 'data/scannet' / data['feature_id'] / scene.split('/')[-1] / 'wan22_ti2v.pt'
    missing = [str(cache(s)) for s in sorted(scenes) if not cache(s).is_file()]
    if missing:
        raise FileNotFoundError('Missing training offline features: ' + str(missing[:5]))
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
    return dict(train_rows=len(rows), train_scenes=len(scenes), validation_rows=len(validation),
                validation_scenes=len(validation_scenes), datasets=specification,
                train_path=str(resolve_path(data['train'])), data_hashes=hashes,
                distinct_teacher_clips=len(clips), checked_files=len(files),
                validation_missing_feature_scenes=[s for s in sorted(validation_scenes) if not cache(s).is_file()],
                train_validation_scene_overlap=sorted(scenes & validation_scenes))
