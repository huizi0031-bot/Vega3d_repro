#!/usr/bin/env python3
"""Prepare model assets for the independent Qwen3.5-language + Wan2.2 variant.

Without --download this only uses already downloaded local model files.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from export_qwen35_language import export_language


def link_directory(path, target):
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError('Model link points to a different checkpoint: '+str(path))
    elif path.exists():
        if path.resolve() != target.resolve():
            raise FileExistsError('Refusing to overwrite '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target.resolve(), target_is_directory=True)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=repo.parents[1]/'models')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--endpoint', default=None, help='Optional user-selected Hugging Face endpoint')
    args = parser.parse_args()
    root = args.model_root.resolve()
    sources = {'Qwen3.5-2B':'Qwen/Qwen3.5-2B',
               'Wan2.2-TI2V-5B':'Wan-AI/Wan2.2-TI2V-5B'}
    revisions = {'Qwen3.5-2B':'15852e8c16360a2fea060d615a32b45270f8a8fc',
                 'Wan2.2-TI2V-5B':'921dbaf3f1674a56f47e83fb80a34bac8a8f203e'}
    if args.download:
        from huggingface_hub import snapshot_download
        for name, model_id in sources.items():
            snapshot_download(model_id, local_dir=root/name, endpoint=args.endpoint, revision=revisions[name])
    for name in sources:
        if not (root/name/'config.json').is_file():
            raise FileNotFoundError(f'Missing {root/name}; download the official model or use --download')
    language = root/'Qwen3.5-2B-Language'
    if (language/'language_export.json').is_file():
        report = json.loads((language/'language_export.json').read_text())
        if Path(report['source']).resolve() != (root/'Qwen3.5-2B').resolve():
            raise ValueError('Existing language export came from a different source')
    else:
        report = export_language(root/'Qwen3.5-2B', language)
    link_directory(repo/'data/models/Qwen3.5-2B-Language', language)
    link_directory(repo/'data/models/Wan2.2-TI2V-5B', root/'Wan2.2-TI2V-5B')
    prompt = repo/'llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt'
    if not prompt.is_file():
        subprocess.run([sys.executable, str(repo/'scripts/3d/preprocessing/export_wan22_prompt_embedding.py'),
                        '--checkpoint-dir',str(root/'Wan2.2-TI2V-5B'), '--output',str(prompt), '--device','cpu'],
                       cwd=repo,check=True)
    print(json.dumps({'language_model':str(language),'wan_model':str(root/'Wan2.2-TI2V-5B'),
                      'prompt_embedding':str(prompt),'language_export':report},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
