#!/usr/bin/env python3
"""Explicit new-variant full-run wrapper; never launches a hardcoded old run."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--run-name')
    parser.add_argument('--stage', choices=['all','extract','train'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9]+(,[0-9]+)*', args.gpus):
        parser.error('Use comma-separated GPU indices')
    if not args.run_name:
        if args.resume:
            parser.error('--resume requires --run-name')
        args.run_name = 'qwen35-wan22-full-gpu'+args.gpus.replace(',','')+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')
    command = ['bash',str(repo/'scripts/3d/train/train_qwen35_wan22_full.sh'),
               '--gpus',args.gpus,'--run-name',args.run_name,'--stage',args.stage]
    if args.resume: command.append('--resume')
    if args.dry_run: command.append('--dry-run')
    env = dict(os.environ)
    env['PATH'] = str(repo.parents[1]/'envs/vega3d-qwen35/bin')+':'+env['PATH']
    result = subprocess.run(command,cwd=repo,env=env)
    if not args.dry_run:
        record = repo/'wan_full_smoke_tools/full_training_runs'/args.run_name
        record.mkdir(parents=True,exist_ok=True)
        (record/'supervisor_exit.json').write_text(json.dumps(
            {'command':command,'exit_code':result.returncode,'finished_at':datetime.now().isoformat()},indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
