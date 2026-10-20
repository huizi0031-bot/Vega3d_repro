#!/usr/bin/env python3
"""Small, held-out five-task training/evaluation run using the existing repository."""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS = {
    'scanrefer': 'scanrefer_vg_val_llava_style.json',
    'multi3drefer': 'multi3drefer_val_llava_style.json',
    'sqa3d': 'sqa3d_test_llava_style.json',
    'scan2cap': 'scan2cap_val_llava_style.json',
    'scanqa': 'scanqa_val_llava_style.json',
}
TASK_MAPPING = {'scanqa': 0, 'sqa3d': 0, 'scan2cap': 1, 'scanrefer': 2, 'multi3drefer': 2}


def fix_answer_file_lock(source):
    """Pass a filename to fasteners; a stream object creates bogus repr paths."""
    old = 'fasteners.InterProcessLock(ans_file)'
    new = 'fasteners.InterProcessLock(answer_file)'
    if source.count(old) == 1:
        return source.replace(old, new)
    if source.count(old) == 0 and source.count(new) == 1:
        return source
    raise ValueError('Unexpected evaluation answer-file lock implementation')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command, log, env):
    print('RUN:', shlex.join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as output:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line, end='', flush=True)
            output.write(line)
            output.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log}')


def run_parallel(commands):
    """Start only this run's two workers; preserve all output in per-worker logs."""
    workers = []
    try:
        for command, log, env in commands:
            print('INFERENCE GPU', env['CUDA_VISIBLE_DEVICES'], 'LOG:', log, flush=True)
            output = log.open('w')
            try:
                process = subprocess.Popen(list(map(str, command)), stdout=output,
                                           stderr=subprocess.STDOUT, env=env)
            except BaseException:
                output.close()
                raise
            workers.append((process, output, log))
        while any(process.poll() is None for process, _, _ in workers):
            for process, _, log in workers:
                code = process.poll()
                if code not in (None, 0):
                    print('\n'.join(log.read_text(errors='replace').splitlines()[-100:]), flush=True)
                    raise RuntimeError(f'Inference failed ({code}); see {log}')
            import time
            time.sleep(1)
        for process, _, log in workers:
            if process.returncode:
                print('\n'.join(log.read_text(errors='replace').splitlines()[-100:]), flush=True)
                raise RuntimeError(f'Inference failed ({process.returncode}); see {log}')
    finally:
        for process, output, _ in workers:
            if process.poll() is None:
                process.terminate()
        for process, output, _ in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            output.close()


def require_files(paths):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing files:\n' + '\n'.join(missing))


def verify_core(repo, state):
    changed = [name for name, digest in state['source_hashes'].items()
               if file_hash(repo / name) != digest]
    if changed:
        raise RuntimeError('Repository source changed since preparation: ' + ', '.join(changed))


def prepare(repo, folder, run_name, args):
    import torch
    from llava.train.llava_trainer import LengthGroupedSampler
    required = [repo / 'wan_offline_tools/generated/qwen35_wan_offline_smoke.json',
                repo / 'wan_offline_tools/generated/qwen35_wan_offline_manifest.json',
                repo / 'scripts/3d/train/train_qwen35_wan22.sh']
    for task, filename in TASKS.items():
        required += [repo / 'data/processed' / filename,
                     repo / f'llava/eval/model_{task}.py', repo / f'llava/eval/eval_{task}.py']
    require_files(required)
    pointer = json.loads(required[1].read_text())
    plan = json.loads((repo / pointer['manifest']).read_text())
    if plan['frames'] != 8 or plan['frame_chunk_size'] != 2:
        raise ValueError('This run expects the existing eight-frame/two-frame-chunk extraction plan')
    train = json.loads(required[0].read_text())
    if Counter(row['metadata']['dataset'].lower() for row in train).keys() != TASK_MAPPING.keys():
        raise ValueError('Training subset must contain all five tasks')
    lengths = [(TASK_MAPPING[row['metadata']['dataset'].lower()],
                sum(len(c['value'].split()) for c in row['conversations'])) for row in train]
    torch.manual_seed(42)
    gpu_ids = args.gpus.split(',')
    num_gpus = len(gpu_ids)
    sampled = list(LengthGroupedSampler(batch_size=1, world_size=num_gpus, lengths=lengths, group_by_task=True))
    if len(sampled) % num_gpus or not sampled:
        raise ValueError('Original sampler did not produce full GPU batches')
    retained = Counter(train[i]['metadata']['dataset'].lower() for i in sampled)
    if any(retained.get(task, 0) == 0 for task in TASK_MAPPING):
        raise ValueError(f'Original sampler drops an entire task in this subset: {retained}')
    expected_steps = len(sampled) // num_gpus
    rows = {task: json.loads((repo / 'data/processed' / name).read_text())
            for task, name in TASKS.items()}
    common = set.intersection(*({row['video'] for row in items} for items in rows.values()))
    train_scenes = {row['video'] for row in train}
    scenes = sorted(common - train_scenes)[:args.eval_scenes]
    if len(scenes) != args.eval_scenes:
        raise ValueError('Not enough held-out scenes shared by all five evaluation splits')
    counts = {}
    question_files = {}
    for task, items in rows.items():
        selected = []
        for scene in scenes:
            candidates = [r for r in items if r['video'] == scene]
            if task == 'scan2cap':
                candidates = [r for r in candidates if r.get('box_input') is not None]
            # Distinct official sample IDs are required for exact prediction coverage.
            seen = set()
            picked = []
            for row in candidates:
                if str(row['id']) not in seen:
                    picked.append(row); seen.add(str(row['id']))
                if len(picked) == args.samples_per_scene:
                    break
            if len(picked) != args.samples_per_scene:
                raise ValueError(f'Not enough usable {task} samples in {scene}')
            selected.extend(picked)
        require_unique = [str(r['id']) for r in selected]
        if len(require_unique) != len(set(require_unique)):
            raise ValueError(f'Duplicate IDs in {task}')
        path = folder / 'questions' / (task + '.json')
        write_json(path, selected)
        counts[task] = len(selected)
        question_files[task] = str(path)
    train_path = folder / 'train.json'
    write_json(train_path, train)
    evaluation_plan = dict(plan, scenes=scenes,
                           sample_counts={TASKS[t]: counts[t] for t in TASKS})
    write_json(folder / 'eval_feature_plan.json', evaluation_plan)
    source_names = ['llava/train/llava_trainer.py', 'llava/train/train_3d.py',
                    'llava/model/builder.py', 'llava/model/language_model/llava_qwen.py',
                    'llava/video_utils.py', 'llava/model/llava_arch.py',
                    'llava/eval/caption_eval/meteor/meteor.py']
    source_names += [f'llava/eval/{prefix}_{t}.py' for t in TASKS for prefix in ('model', 'eval')]
    state = {'run_name': run_name, 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             'scope': 'small held-out subset; original scoring algorithms; not a full benchmark score',
             'train_samples': len(train), 'expected_train_steps': expected_steps,
             'sampler_retained_by_dataset': dict(retained), 'train_scenes': sorted(train_scenes),
             'eval_scenes': scenes, 'eval_counts': counts, 'question_files': question_files,
             'train_file': str(train_path), 'feature_id': plan['feature_id'],
             'source_hashes': {name: file_hash(repo / name) for name in source_names},
             'gpu_ids': gpu_ids, 'checkpoint_root': str((repo / args.output_root).resolve()), 'completed': {}}
    # Delegate training to the one maintained Qwen3.5 launcher; no per-run shell copy.
    state['train_command'] = ['bash', str(required[2]), '--mode', 'train',
                              '--feature-source', 'offline', '--gpus', args.gpus,
                              '--data-path', str(train_path), '--run-name', run_name,
                              '--max-steps', '-1', '--epochs', '1', '--frames', '8',
                              '--feature-id', plan['feature_id'],
                              '--output-root', state['checkpoint_root']]
    write_json(folder / 'training_command.json', state['train_command'])
    (folder / 'eval').mkdir()
    compat = Path(__file__).with_name('qwen35_eval_compat.py')
    shutil.copy2(compat, folder / 'eval' / compat.name)
    for task in TASKS:
        path = repo / f'llava/eval/model_{task}.py'
        source = path.read_text()
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_qwen']
        if len(nodes) != 1:
            raise ValueError(f'Unexpected evaluation prompt implementation: {path}')
        node = nodes[0]
        lines = source.splitlines(keepends=True)
        adapter = 'preprocess_grounding' if task in ('scanrefer', 'multi3drefer') else 'preprocess_generation'
        lines[node.lineno-1:node.end_lineno] = [f'from qwen35_eval_compat import {adapter} as preprocess_qwen\n']
        source = ''.join(lines)
        source = source.replace('args.conv_mode = "qwen_1_5"', 'args.conv_mode = "qwen35"')
        from no_ray import without_ray
        source = without_ray(source)
        source = fix_answer_file_lock(source)
        compile(source, str(path), 'exec')
        (folder / 'eval' / f'model_{task}.py').write_text(source)
    write_json(folder / 'state.json', state)
    print('Prepared:', folder, flush=True)
    print('Original sampler expected steps:', expected_steps, 'held-out scenes:', scenes, flush=True)
    print('Evaluation sample counts:', counts, flush=True)
    return state


def extract(repo, folder, state):
    plan = json.loads((folder / 'eval_feature_plan.json').read_text())
    path = repo / 'wan_offline_tools/extract_wan22_offline.py'
    spec = importlib.util.spec_from_file_location('wan_extractor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(gpu=state.get('gpu_ids', ['2', '3'])[0], frames=plan['frames'], size=plan['size'], timestep=plan['timestep'],
                           block=plan['block'], seed=plan['seed'], chunk_frames=plan['frame_chunk_size'],
                           checkpoint=plan['checkpoint'], prompt_embedding=plan['prompt_embedding'])
    module.extract(args, plan, repo / 'data/scannet' / plan['feature_id'])
    state['completed']['extract'] = True
    write_json(folder / 'state.json', state)


def verify_training(repo, folder, state):
    run = Path(state.get('checkpoint_root', str(repo.parents[1] / 'checkpoints'))) / state['run_name']
    require_files([run / 'trainer_state.json', run / 'config.json', run / 'tokenizer_config.json'])
    saved = json.loads((run / 'trainer_state.json').read_text())
    if saved['global_step'] != state['expected_train_steps']:
        raise ValueError(f"Expected {state['expected_train_steps']} actual original-sampler steps, got {saved['global_step']}")
    losses = [r['loss'] for r in saved.get('log_history', []) if 'loss' in r]
    norms = [r['grad_norm'] for r in saved.get('log_history', []) if 'grad_norm' in r]
    if len(losses) != state['expected_train_steps'] or not all(math.isfinite(x) for x in losses + norms):
        raise ValueError('Missing or non-finite training loss/gradient records')
    config = json.loads((run / 'config.json').read_text())
    if config['model_type'] != 'llava_qwen35' or config.get('generative_feature_source') != 'offline':
        raise ValueError('Saved checkpoint has unexpected model/feature configuration')
    from safetensors import safe_open
    index = run / 'model.safetensors.index.json'
    if index.is_file():
        weight_map = json.loads(index.read_text())['weight_map']
        require_files([run / name for name in set(weight_map.values())])
        keys = list(weight_map)
    elif (run / 'model.safetensors').is_file():
        with safe_open(str(run / 'model.safetensors'), framework='pt', device='cpu') as f:
            keys = list(f.keys())
    else:
        raise ValueError('No final safetensors checkpoint found')
    for part in ('generative_projector', 'feature_fusion', 'ground_head'):
        if not any(part in key for key in keys):
            raise ValueError('Saved model is missing ' + part)
    state['completed']['train'] = {'global_step': saved['global_step'], 'losses': losses,
                                  'checkpoint': str(run), 'weights_verified': True}
    write_json(folder / 'state.json', state)
    print('[Qwen3.5] TRAIN VERIFIED:', saved['global_step'], 'steps, finite loss/gradients, saved multimodal weights', flush=True)


def predict_count(path, question_path):
    questions = json.loads(Path(question_path).read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    actual = Counter(str(r['sample_id']) for r in rows)
    expected = Counter(str(r['id']) for r in questions)
    if actual != expected:
        raise ValueError(f'Prediction coverage mismatch: {path}; expected {sum(expected.values())}, got {sum(actual.values())}')
    for row in rows:
        if row.get('dataset', '').lower() == 'multi3drefer':
            scores, objects = row.get('scores'), row.get('objects')
            if not isinstance(scores, list) or not isinstance(objects, list):
                raise ValueError('Missing Multi3DRefer scores/objects in ' + str(path))
            if len(scores) != len(objects) + 1 or not all(math.isfinite(v) for v in scores):
                raise ValueError('Invalid Multi3DRefer score vector in ' + str(path))
            if not all(isinstance(box, list) and len(box) == 6 and all(math.isfinite(v) for v in box) for box in objects):
                raise ValueError('Invalid Multi3DRefer candidate boxes in ' + str(path))
        elif row.get('pred_response') is None:
            raise ValueError('Null prediction in ' + str(path))
    return len(rows)


def score_with_safe_meteor(repo, task, input_file):
    """Keep native scorers; serialize CR/LF safely in the METEOR stdio protocol."""
    if task in ('scan2cap', 'scanqa'):
        from llava.eval.caption_eval.meteor.meteor import Meteor
        original_stat = Meteor._stat
        def one_protocol_line(text):
            return text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        def transport_stat(self, hypothesis, references):
            return original_stat(self, one_protocol_line(hypothesis),
                                 [one_protocol_line(reference) for reference in references])
        Meteor._stat = transport_stat
        print('[Qwen3.5] METEOR transport: CR/LF -> spaces; raw predictions and native scoring formula preserved.', flush=True)
    import runpy
    entry = repo / f'llava/eval/eval_{task}.py'
    sys.argv = [str(entry), '--input-file', str(input_file)]
    runpy.run_path(str(entry), run_name='__main__')


def evaluate(repo, folder, state, env):
    verify_training(repo, folder, state)
    for task in TASKS:
        if state['completed'].get(task):
            record = state['completed'][task]
            predict_count(Path(record['answers']), state['question_files'][task])
            print('[Qwen3.5] Already verified:', task, flush=True)
            continue
        pending = state.setdefault('pending_predictions', {}).get(task)
        if pending:
            answer = Path(pending['answers'])
            count = predict_count(answer, state['question_files'][task])
            metric_log = answer.parent / ('metrics_retry-' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f') + '.log')
            print('[Qwen3.5] Reusing verified raw predictions:', task, answer, flush=True)
        else:
            attempt = folder / 'results' / task / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
            attempt.mkdir(parents=True)
            answer = attempt / 'answers.jsonl'
            infer_log = attempt / 'inference.log'
            metric_log = attempt / 'metrics.log'
            questions = json.loads(Path(state['question_files'][task]).read_text())
            commands = []
            shards = []
            gpu_ids = state.get('gpu_ids', ['2', '3'])
            for rank, gpu in enumerate(gpu_ids):
                shard_questions = attempt / f'questions_gpu{gpu}.json'
                shard_answers = attempt / f'answers_gpu{gpu}.jsonl'
                write_json(shard_questions, questions[rank::len(gpu_ids)])
                if not questions[rank::len(gpu_ids)]:
                    continue
                command = [sys.executable, folder / 'eval' / f'model_{task}.py',
                           '--model-path', Path(state.get('checkpoint_root', str(repo.parents[1] / 'checkpoints'))) / state['run_name'], '--video-folder', 'data',
                           '--embodiedscan-folder', 'data/embodiedscan', '--n_gpu', '1',
                           '--question-file', shard_questions, '--conv-mode', 'qwen35',
                           '--answer-file', shard_answers, '--frame_sampling_strategy', 'uniform',
                           '--max_frame_num', '8', '--overwrite_cfg', 'false',
                           '--generative_model_id', state['feature_id'], '--generative_feature_source', 'offline']
                infer_env = dict(env)
                infer_env['CUDA_VISIBLE_DEVICES'] = gpu
                infer_env['PYTHONPATH'] = str(folder / 'eval') + os.pathsep + env.get('PYTHONPATH', '')
                commands.append((command, attempt / f'inference_gpu{gpu}.log', infer_env))
                shards.append((shard_questions, shard_answers))
            run_parallel(commands)
            with answer.open('w') as merged:
                for question_path, answer_path in shards:
                    predict_count(answer_path, question_path)
                    merged.write(answer_path.read_text().rstrip('\n') + '\n')
            infer_log.write_text('Standalone Qwen3.5 workers completed; see inference_gpu*.log\n')
            count = predict_count(answer, state['question_files'][task])
            state['pending_predictions'][task] = {'answers': str(answer), 'samples': count}
            write_json(folder / 'state.json', state)
        run_command([sys.executable, Path(__file__).resolve(), '--stage', 'score',
                     '--task', task, '--input-file', answer], metric_log, env)
        text = metric_log.read_text()
        tokens = {
            'scanrefer': ['all iou@0.25:', 'all iou@0.5:'],
            'multi3drefer': ['all IoU25:', 'all IoU50:'], 'sqa3d': ['all:'],
            'scan2cap': ['CIDER:', 'BLEU:', 'METEOR:', 'Rouge:'],
            'scanqa': ['CIDER:', 'BLEU:', 'METEOR:', 'Rouge:', 'EM:'],
        }[task]
        for token in tokens:
            matches = re.findall(re.escape(token) + r'\s*([^\n]+)', text)
            if len(matches) != 1:
                raise ValueError('Missing/duplicate metric: ' + token + '; see ' + str(metric_log))
            values = [float(x.strip()) for x in matches[0].split(',')]
            expected = 4 if task == 'scanqa' and token == 'BLEU:' else 1
            if len(values) != expected or not all(math.isfinite(x) for x in values):
                raise ValueError('Non-finite/incomplete metric: ' + token + '; see ' + str(metric_log))
        state['completed'][task] = {'samples': count, 'answers': str(answer), 'metrics_log': str(metric_log),
                                    'metrics_text': text,
                                    'meteor_transport': 'CR/LF to spaces only' if task in ('scan2cap', 'scanqa') else None}
        state.get('pending_predictions', {}).pop(task, None)
        write_json(folder / 'state.json', state)
        print('[Qwen3.5] EVAL VERIFIED:', task, count, 'samples', flush=True)
    verify_core(repo, state)
    summary = {'scope': state['scope'], 'run_name': state['run_name'],
               'train': state['completed']['train'], 'train_scenes': state['train_scenes'],
               'eval_scenes': state['eval_scenes'], 'original_source_unchanged': True,
               'evaluations': {task: state['completed'][task] for task in TASKS}, 'all_five_passed': True}
    write_json(folder / 'summary.json', summary)
    print('[Qwen3.5] FULL SMOKE VERIFIED: training, saved-model reload, and all five official scorers completed.', flush=True)
    print('Summary:', folder / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'prepare', 'extract', 'train', 'eval', 'score'], default='all')
    parser.add_argument('--run-name')
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--output-root', default=str(Path(__file__).resolve().parents[3] / 'checkpoints'))
    parser.add_argument('--task', choices=list(TASKS))
    parser.add_argument('--input-file')
    parser.add_argument('--eval-scenes', type=int, default=2)
    parser.add_argument('--samples-per-scene', type=int, default=2)
    args = parser.parse_args()
    if min(args.eval_scenes, args.samples_per_scene) < 1:
        parser.error('Scene and sample counts must be positive')
    tools = Path(__file__).resolve().parent
    repo = tools.parent
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.stage == 'score':
        if not args.task or not args.input_file:
            parser.error('--task and --input-file are required for score')
        score_with_safe_meteor(repo, args.task, Path(args.input_file))
        return
    if not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', args.gpus) or len(set(args.gpus.split(','))) != len(args.gpus.split(',')):
        parser.error('Invalid/duplicate GPU indices')
    # Limit visibility before importing torch.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus.split(',')[0] if args.stage == 'extract' else args.gpus
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['PYTHONPATH'] = str(repo) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.pop('RAY_ADDRESS', None)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.run_name:
        run_name = args.run_name
    elif args.stage in ('all', 'prepare'):
        run_name = 'qwen35-wan22-offline-smoke-gpu' + args.gpus.replace(',', '') + '-' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    else:
        parser.error('--run-name is required to resume a prepared run')
    if not run_name.startswith('qwen35-'):
        run_name = 'qwen35-' + run_name
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_name):
        parser.error('Run name must contain only letters, numbers, underscores and hyphens')
    folder = tools / 'runs' / run_name
    state_file = folder / 'state.json'
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        verify_core(repo, state)
        env['CUDA_VISIBLE_DEVICES'] = ','.join(state.get('gpu_ids', ['2', '3']))
    else:
        if args.stage not in ('all', 'prepare'):
            parser.error('Prepare this run first')
        if (repo / args.output_root / run_name).exists() or folder.exists():
            raise FileExistsError('Run name already used; choose a new name')
        folder.mkdir(parents=True)
        state = prepare(repo, folder, run_name, args)
    if args.stage == 'prepare':
        return
    if args.stage in ('all', 'extract') and not state['completed'].get('extract'):
        # Separate subprocess prevents earlier torch imports from changing GPU visibility.
        if args.stage == 'all':
            run_command([sys.executable, Path(__file__), '--stage', 'extract', '--run-name', run_name,
                         '--gpus', ','.join(state['gpu_ids']), '--output-root', state['checkpoint_root']],
                        folder / 'extraction.log', env)
            state = json.loads(state_file.read_text())
        else:
            extract(repo, folder, state)
    if args.stage in ('all', 'train') and not state['completed'].get('train'):
        if not state['completed'].get('extract'):
            raise RuntimeError('Extract held-out features first')
        command = state.get('train_command', ['bash', str(folder / 'train.sh')])
        run_command(command, folder / 'training.log', env)
        verify_training(repo, folder, state)
    if args.stage in ('all', 'eval'):
        if not state['completed'].get('extract'):
            raise RuntimeError('Extract held-out features first')
        evaluate(repo, folder, state, env)


if __name__ == '__main__':
    main()
