#!/usr/bin/env python3
"""Verify an independently restored Qwen3.5 Python environment on CPU."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import site
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    repo = args.repo.resolve()
    prefix = Path(sys.prefix).resolve()
    if platform.python_version() != manifest['python']:
        raise RuntimeError('Python version mismatch')
    if Path(sys.base_prefix).resolve() != prefix or site.ENABLE_USER_SITE:
        raise RuntimeError('Environment still inherits a base or user site-packages')
    errors = []
    for package in manifest['wheels']:
        try:
            current = metadata.version(package['name'])
        except metadata.PackageNotFoundError:
            current = None
        if current != package['version']:
            errors.append([package['name'], package['version'], current])
    if errors:
        raise RuntimeError('Package versions differ: '+str(errors))
    bad_paths = [p for p in sys.path if 'site-packages' in p and prefix not in Path(p).resolve().parents]
    if bad_paths:
        raise RuntimeError('External site-packages detected: '+str(bad_paths))
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0,str(repo))
    modules = {}
    for name in ['torch','torchvision','torchaudio','PIL','numpy','transformers','accelerate',
                 'deepspeed','diffusers','torch_scatter','cv2','av','decord','yaml','timm',
                 'peft','datasets','tensorboard','typer',
                 'llava.model.language_model.llava_qwen','llava.train.train_3d',
                 'llava.model.multimodal_generative_encoder.wan22_foresight']:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not (prefix in path.parents or repo in path.parents):
            raise RuntimeError('Module loaded outside restored environment/repository: '+str(path))
        modules[name] = str(path.relative_to(repo if repo in path.parents else prefix))
    import torch
    if torch.version.cuda != manifest['torch_cuda']:
        raise RuntimeError('Torch CUDA build mismatch: ' + str(torch.version.cuda))
    from torchvision.ops import nms
    from torch_scatter import scatter_mean
    torch.testing.assert_close(scatter_mean(torch.tensor([1.,3.,7.]),torch.tensor([0,0,1])),torch.tensor([2.,7.]))
    assert nms(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]),torch.tensor([.9,.8]),.5).tolist() == [0]
    subprocess.run([str(prefix/'bin/ninja'),'--version'],check=True)
    shm = Path(torch.__file__).parent/'bin/torch_shm_manager'
    if not shm.is_file() or not os.access(shm,os.X_OK):
        raise RuntimeError('Torch shared-memory manager is missing or not executable')
    import triton
    ptxas = Path(triton.__file__).parent/'backends/nvidia/bin/ptxas'
    if not ptxas.is_file() or not os.access(ptxas,os.X_OK):
        raise RuntimeError('Triton ptxas is missing or not executable')
    subprocess.run([str(ptxas),'--version'],check=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU verification unexpectedly initialized CUDA')
    mappings = Path('/proc/self/maps').read_text().splitlines()
    old_libraries = [line for line in mappings if any(
        old.rstrip('/')+'/' in line for old in manifest['source_prefixes'])
        and str(prefix)+'/' not in line]
    if old_libraries:
        raise RuntimeError('Native libraries still loaded from source environment: '+str(old_libraries))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1',
           'PYTHONPATH':str(repo)+os.pathsep+str(repo/'tests'),
           'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    tokenizer = Path(env.get('QWEN35_TEST_TOKENIZER', str(repo/'data/models/Qwen3.5-2B-Language')))
    if tokenizer.is_dir():
        env['QWEN35_TEST_TOKENIZER'] = str(tokenizer)
    else:
        env.pop('QWEN35_TEST_TOKENIZER', None)
    commands = []
    if args.test:
        commands.append(('cpu_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v']))
    if args.check_data:
        commands.append(('data_loader',[sys.executable,'scripts/3d/train/train_scene_understanding.py',
                         '--mode','video','--check-loader']))
    checks = {}
    for name,command in commands:
        log = args.report.parent/(name+'.log')
        with log.open('w') as stream:
            result = subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
        checks[name] = {'exit_code':result.returncode,'log':str(log)}
        if result.returncode:
            raise RuntimeError(name+' failed; inspect '+str(log))
    report = dict(python=platform.python_version(),packages_verified=len(manifest['wheels']),
        torch_version=str(torch.__version__),torch_cuda=torch.version.cuda,
        independent_prefix=True,user_site_enabled=site.ENABLE_USER_SITE,
        source_shared_libraries_loaded=False,
        cuda_initialized=torch.cuda.is_initialized(),modules=modules,checks=checks)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
