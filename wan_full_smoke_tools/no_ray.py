"""Run a copied repository inference entry point without a Ray cluster."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo


def without_ray(source):
    tree = ast.parse(source)
    initializers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'init'
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ray']
    if not initializers:
        if not any(isinstance(n, ast.Name) and n.id == 'ray' for n in ast.walk(tree)):
            old = '    print(f"time: {np.mean(inference_time)}")'
            new = '    if inference_time is not None and len(inference_time) > 0:\n        print(f"time: {np.mean(inference_time)}")'
            if old in source and new not in source:
                source = source.replace(old, new, 1)
            compile(source, '<standalone-inference>', 'exec')
            return source
        raise ValueError('Ray references remain but the initialization is not identifiable')
    if len(initializers) != 1:
        raise ValueError('Expected one Ray initializer')
    call = initializers[0]
    line = source.splitlines()[call.lineno - 1]
    if not line.startswith('    ray.init('):
        raise ValueError('Unexpected inference dispatcher indentation')
    # The parser/data loading above this call stays intact. Replace only scheduling.
    source = ''.join(source.splitlines(keepends=True)[:call.lineno - 1])
    source += '    inference_time = eval_model(questions, args)\n    if inference_time is not None and len(inference_time) > 0:\n        print(f"time: {np.mean(inference_time)}")\n'
    if source.count('import ray\n') != 1 or source.count('@ray.remote(num_gpus=1)\n') != 1:
        raise ValueError('Unexpected Ray imports/decorator; no file has been changed')
    source = source.replace('import ray\n', '', 1).replace('@ray.remote(num_gpus=1)\n', '', 1)
    tree = ast.parse(source)
    if any(isinstance(n, ast.Name) and n.id == 'ray' for n in ast.walk(tree)):
        raise ValueError('Unexpected remaining Ray usage')
    compile(source, '<standalone-inference>', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run_name):
        parser.error('Invalid run name')
    tools = Path(__file__).resolve().parent
    folder = tools / 'runs' / args.run_name
    from run_full_smoke import TASKS, verify_core
    state = json.loads((folder / 'state.json').read_text())
    verify_core(tools.parent, state)
    prepared = []
    for task in TASKS:
        path = folder / 'eval' / f'model_{task}.py'
        original = path.read_text()
        prepared.append((path, original, without_ray(original)))
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    for path, original, updated in prepared:
        if original != updated:
            backup = path.with_name(path.name + '.bak-ray-' + stamp)
            shutil.copy2(path, backup)
            path.write_text(updated)
            print('Updated copied entry point:', path, 'Backup:', backup, flush=True)
        else:
            print('Already repaired:', path, flush=True)
    print('Original repository sources are unchanged. Resume --stage eval.', flush=True)


if __name__ == '__main__':
    main()
