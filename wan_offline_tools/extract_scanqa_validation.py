#!/usr/bin/env python3
"""ScanQA pilot validation caches, reusing the original GPU extractor unchanged.

Preview/check mode is CPU-only. Execution requires an explicitly chosen idle GPU.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_KEYS = (
    'format_version', 'feature_id', 'frames', 'size', 'timestep', 'block',
    'seed', 'shift', 'frame_chunk_size', 'strategy', 'sampling', 'feature_dim',
    'spatial', 'input_dtype', 'image_processor', 'checkpoint', 'prompt_embedding',
)


def idle_gpu_snapshot(gpu):
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = [list(map(str.strip, row)) for row in csv.reader(raw.splitlines())]
    row = next((row for row in rows if row[0] == str(gpu)), None)
    if row is None:
        raise ValueError('Unknown physical GPU: ' + str(gpu))
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    busy = [line for line in processes.splitlines() if line.split(',')[0].strip() == row[1]]
    result = dict(index=int(row[0]), uuid=row[1], total_mib=int(row[2]),
                  used_mib=int(row[3]), utilization=int(row[4]), processes=busy)
    result['idle'] = not busy and result['used_mib'] <= 512 and result['utilization'] <= 5
    return result


def inspect_cache(torch, path, scene, frames, protocol):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata, features = payload['metadata'], payload['features']
    if metadata['video_id'] != scene or metadata['frame_files'] != frames:
        raise ValueError('Scene/frame mismatch: ' + str(path))
    if any(metadata['plan'].get(key) != value for key, value in protocol.items()):
        raise ValueError('Extraction protocol mismatch: ' + str(path))
    if tuple(features.shape) != (32, 3072, 14, 14) or features.dtype != torch.bfloat16:
        raise ValueError('Invalid feature shape/dtype: ' + str(path))
    if not torch.isfinite(features).all():
        raise ValueError('Non-finite features: ' + str(path))
    return dict(scene=scene, path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/foresight/scanqa_pilot.yaml')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.execute and (args.gpu is None or args.gpu < 0 or args.check_only):
        parser.error('--execute needs one nonnegative --gpu and excludes --check-only')
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu) if args.execute else '',
                      TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false')
    if args.execute:
        first = idle_gpu_snapshot(args.gpu)
        print(json.dumps({'gpu_before_loading': first}), flush=True)
        if not first['idle']:
            print('WAITING_FOR_IDLE_GPU: no GPU work started.', flush=True)
            return 75
    import torch
    import yaml
    from llava.video_utils import VideoProcessor
    from wan_offline_tools.wan22_features import extract
    torch.set_num_threads(args.cpu_threads)
    config = yaml.safe_load(Path(args.config).read_text())
    data = config['data']
    root = Path('data/scannet') / data['feature_id']
    annotations = Path(data['root']) / data['validation']
    rows = json.loads(annotations.read_text())
    scenes = sorted({row['video'] for row in rows})
    train = json.loads((Path(data['root']) / data['train_original']).read_text())
    train_scenes = sorted({row['video'] for row in train})
    if not scenes or set(scenes) & set(train_scenes):
        raise ValueError('Empty validation set or train/validation scene overlap')
    reference = root / train_scenes[0].split('/')[-1] / 'wan22_ti2v.pt'
    reference_payload = torch.load(reference, map_location='cpu', weights_only=True)
    original = reference_payload['metadata']['plan']
    protocol = {key: original[key] for key in PROTOCOL_KEYS}
    if (protocol['feature_id'] != data['feature_id'] or protocol['frames'] != 32
            or protocol['frame_chunk_size'] != 2 or protocol['input_dtype'] != 'bfloat16'):
        raise ValueError('Unexpected pilot extraction protocol')
    processor = VideoProcessor(video_folder='data', annotation_dir='data/embodiedscan/',
                               frame_sampling_strategy='uniform')
    frames_by_scene, complete, missing = {}, [], []
    for scene in scenes:
        frames = [os.path.normpath(x) for x in processor.sample_frame_files(
            scene, force_sample=True, frames_upbound=32)]
        frames_by_scene[scene] = frames
        for name in frames:
            for file in [name, name.replace('.jpg', '.png'), name.replace('.jpg', '.txt')]:
                if not Path(file).is_file():
                    raise FileNotFoundError(file)
        target = root / scene.split('/')[-1] / 'wan22_ti2v.pt'
        if target.is_file():
            complete.append(inspect_cache(torch, target, scene, frames, protocol))
        else:
            missing.append(scene)
    report = dict(validation_questions=len(rows), validation_scenes=scenes,
                  protocol=protocol, missing=missing, complete=complete,
                  cuda_initialized=torch.cuda.is_initialized(), training_started=False)
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return 1 if args.check_only and missing else 0
    lock_path = root / '.scanqa_validation_extract.lock'
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another validation extraction already holds the lock')
        if missing:
            # Check again after CPU/data preparation; reject every existing process.
            for _ in range(2):
                snapshot = idle_gpu_snapshot(args.gpu)
                if not snapshot['idle']:
                    print('WAITING_FOR_IDLE_GPU: no GPU work started.', flush=True)
                    return 75
                time.sleep(2)
            plan = dict(protocol, scenes=missing, scope='ScanQA pilot validation only',
                        validation_annotation=str(annotations),
                        validation_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
                        sample_counts={str(annotations): len(rows)})
            settings = SimpleNamespace(gpu=str(args.gpu), frames=32, size=protocol['size'],
                timestep=protocol['timestep'], block=protocol['block'], seed=protocol['seed'],
                chunk_frames=protocol['frame_chunk_size'], checkpoint=protocol['checkpoint'],
                prompt_embedding=protocol['prompt_embedding'], scene_limit=None,
                shard_index=0, num_shards=1)
            run = root / ('scanqa_validation_extract_' + time.strftime('%Y%m%d-%H%M%S'))
            run.mkdir()
            (run/'plan.json').write_text(json.dumps(plan, indent=2))
            try:
                extract(settings, plan, root)
            except Exception:
                (run/'status.json').write_text(json.dumps({'complete':False,
                    'training_started':False,'gpu':snapshot},indent=2))
                raise
            complete = [inspect_cache(torch, root/scene.split('/')[-1]/'wan22_ti2v.pt',
                        scene, frames_by_scene[scene], protocol) for scene in scenes]
            report.update(missing=[], complete=complete, gpu=snapshot)
        (root/'scanqa_validation_completion.json').write_text(json.dumps(report, indent=2))
        print('VALIDATION_FEATURES_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
