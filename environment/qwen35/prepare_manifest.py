#!/usr/bin/env python3
"""Finalize the offline environment manifest and dependency locks."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    root, spec = args.bundle, args.specification
    observed = json.loads((root / 'observed-environment.json').read_text())
    wheels = json.loads((root / 'wheels.json').read_text())
    recorded = {item['filename'] for item in wheels}
    for path in sorted((root / 'wheelhouse').glob('*.whl')):
        if path.name in recorded:
            continue
        with zipfile.ZipFile(path) as archive:
            entry = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
            metadata = BytesParser().parsebytes(archive.read(entry))
        name = metadata['Name'].lower().replace('_', '-')
        if name not in {'absl-py', 'werkzeug'}:
            raise RuntimeError('Unexpected additional wheel: ' + path.name)
        wheels.append(dict(name=name, version=metadata['Version'], layer='addition',
                           filename=path.name, sha256=digest(path), bytes=path.stat().st_size,
                           source='PyPI wheel; fills dependency reported by source pip check'))
    assert len({p['name'] for p in wheels}) == len(wheels)
    assert {'absl-py', 'werkzeug'} <= {p['name'] for p in wheels}
    manifest = dict(schema_version=1, captured_at=observed['captured_at'],
                    python=observed['python'], system='Linux', machine=observed['machine'],
                    torch_cuda='12.4',
                    source_platform=observed['platform'],
                    source_prefixes=observed['source_prefixes'],
                    runtime_sha256=digest(root / 'python-runtime.tar.gz'),
                    wheels=wheels,
                    excluded=[p for p in observed['packages'] if p['excluded']],
                    source_pip_check=observed['original_pip_check'])
    text = json.dumps(manifest, indent=2) + '\n'
    (root / 'manifest.json').write_text(text)
    spec.mkdir(parents=True, exist_ok=True)
    (spec / 'manifest.json').write_text(text)
    (spec / 'observed-environment.json').write_text(json.dumps(observed, indent=2) + '\n')
    ordered = sorted(wheels, key=lambda p: p['name'])
    header = '# Python ' + observed['python'] + '; Linux x86_64. See README.md for offline installation.\n'
    (spec / 'requirements.lock.txt').write_text(header + ''.join(
        p['name'] + '==' + p['version'] + '\n' for p in ordered))
    (spec / 'requirements-offline.lock.txt').write_text(header + ''.join(
        p['name'] + '==' + p['version'] + ' --hash=sha256:' + p['sha256'] + '\n' for p in ordered))
    print(json.dumps(dict(packages=len(wheels), python=observed['python'], bundle=str(root))))


if __name__ == '__main__':
    main()
