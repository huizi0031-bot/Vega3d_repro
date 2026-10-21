#!/usr/bin/env python3
"""Snapshot the effective environment and preserve installed distributions as wheels."""
import argparse
import base64
import csv
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import zipfile
from packaging.tags import parse_tag, sys_tags
from packaging.utils import parse_wheel_filename


EXCLUDED = {
    'flash-attn': 'Unused legacy extension built for Torch 2.4; current model uses SDPA.',
    'vega-3d': 'Obsolete editable installation; install the supplied current repository instead.',
    'llava': 'Legacy placeholder distribution; current llava package comes from the repository.',
}


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def layer(path):
    path = Path(path).resolve()
    if Path(sys.prefix).resolve() in path.parents:
        return 'overlay'
    if Path(sys.base_prefix).resolve() in path.parents:
        return 'base'
    return 'user'


def preserve_wheel(distribution, destination, repair_scripts=False):
    name, version = distribution.metadata['Name'], distribution.version
    wheel_text = distribution.read_text('WHEEL')
    if not wheel_text:
        raise ValueError('Missing wheel metadata: '+name)
    wheel_meta = Parser().parsestr(wheel_text)
    wheel_tags = wheel_meta.get_all('Tag')
    supported = set(sys_tags())
    tag = next((value for value in wheel_tags if parse_tag(value) & supported), None)
    if tag is None:
        # Some installed distributions have stale internal tags. Use a compatible
        # same-version upstream wheel rather than inventing a binary ABI tag.
        for candidate in sorted(destination.glob('*.whl')):
            candidate_name, candidate_version, _, candidate_tags = parse_wheel_filename(candidate.name)
            if canonical(str(candidate_name)) != canonical(name) or str(candidate_version) != version:
                continue
            if not candidate_tags & supported:
                continue
            old = destination/(re.sub(r'[-_.]+', '_', name)+'-'+version+'-'+wheel_tags[0]+'.whl')
            if old != candidate and old.exists():
                old.unlink()
            return dict(filename=candidate.name, sha256=sha256(candidate), bytes=candidate.stat().st_size,
                        source='Compatible same-version upstream wheel; installed internal tag unsupported: '+str(wheel_tags))
        raise ValueError('No compatible installed wheel tag: '+name+': '+str(wheel_tags))
    wheel_name = re.sub(r'[-_.]+', '_', name)+'-'+version+'-'+tag+'.whl'
    target = destination/wheel_name
    previous = destination/(re.sub(r'[-_.]+', '_', name)+'-'+version+'-'+wheel_tags[0]+'.whl')
    if previous != target and previous.exists():
        if target.exists():
            raise FileExistsError('Conflicting wheel tags: '+name)
        previous.rename(target)
    entry_points = {entry.name for entry in distribution.entry_points
                    if entry.group in {'console_scripts', 'gui_scripts'}}
    bin_files = [item for item in distribution.files if str(item).startswith('../')
                 and '/bin/' in str(distribution.locate_file(item).absolute())
                 and Path(item).name not in entry_points]
    if target.exists() and repair_scripts and bin_files:
        target.unlink()
    if target.exists():
        if repair_scripts:
            # Update the central directory without recompressing large CUDA files.
            # RECORD hashes concern file contents, which remain unchanged here.
            with zipfile.ZipFile(target, 'a') as archive:
                for item in distribution.files:
                    relative = str(item).replace(os.sep, '/')
                    if relative.startswith('../'):
                        continue
                    try:
                        entry = archive.getinfo(relative)
                    except KeyError:
                        continue
                    source = Path(distribution.locate_file(item))
                    mode = (source.stat().st_mode & 0xffff) << 16
                    if entry.external_attr != mode:
                        entry.external_attr = mode
                        archive._didModify = True
        return dict(filename=wheel_name, sha256=sha256(target), bytes=target.stat().st_size)
    prefix = re.sub(r'[-_.]+', '_', name)+'-'+version
    info_dir = Path(distribution._path).name
    site = Path(distribution._path).parent
    records, copied, missing = [], set(), []
    temporary = target.with_suffix('.whl.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in distribution.files:
            relative = str(item).replace(os.sep, '/')
            if '__pycache__' in relative or relative.endswith(('.pyc', '.pyo')):
                continue
            if relative.startswith(info_dir+'/') and relative.rsplit('/',1)[-1] in {
                    'RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json'}:
                continue
            source = Path(distribution.locate_file(item))
            script = False
            if relative.startswith('../'):
                if '/bin/' in str(source.resolve()) or '/bin/' in str(source.absolute()):
                    # pip recreates declared entry points. Other scripts and binaries
                    # (notably ninja) must be preserved in the wheel's scripts scheme.
                    if source.name in entry_points:
                        continue
                    relative = prefix+'.data/scripts/'+source.name
                    script = True
                else:
                    environment = site.parents[2]
                    try:
                        data_name = source.resolve().relative_to(environment.resolve()).as_posix()
                    except ValueError:
                        raise ValueError(f'External distribution data: {name}: {item}')
                    relative = prefix+'.data/data/'+data_name
            if not source.is_file():
                missing.append(str(item)); continue
            if relative in copied:
                continue
            copied.add(relative)
            digest, size = hashlib.sha256(), 0
            info = zipfile.ZipInfo(relative)
            info.external_attr = (source.stat().st_mode & 0xffff) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 1
            with source.open('rb') as input_stream, archive.open(info, 'w', force_zip64=True) as output:
                if script:
                    first = input_stream.readline(4096)
                    if first.startswith(b'#!') and b'python' in first:
                        first = b'#!python\n'
                    digest.update(first); size += len(first); output.write(first)
                for chunk in iter(lambda: input_stream.read(8*1024*1024), b''):
                    digest.update(chunk); size += len(chunk); output.write(chunk)
            records.append([relative, 'sha256='+base64.urlsafe_b64encode(digest.digest()).decode().rstrip('='), str(size)])
        if missing:
            raise FileNotFoundError(name+': missing installed files: '+str(missing[:12]))
        record_name = info_dir+'/RECORD'
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator='\n').writerows(records+[[record_name,'','']])
        archive.writestr(record_name, buffer.getvalue())
    temporary.replace(target)
    return dict(filename=wheel_name, sha256=sha256(target), bytes=target.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--wheels', action='store_true')
    parser.add_argument('--repair-scripts', action='store_true',
                        help='Restore executable modes and preserve non-entry-point scripts')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wheelhouse = output/'wheelhouse'
    wheelhouse.mkdir(exist_ok=True)
    selected = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name')
        if name:
            selected.setdefault(canonical(name), distribution)
    packages = []
    for key, distribution in sorted(selected.items()):
        packages.append(dict(name=key, version=distribution.version,
            layer=layer(distribution._path), excluded=EXCLUDED.get(key)))
    manifest = dict(schema_version=1, captured_at=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        source_prefixes=[sys.prefix, sys.base_prefix],
        packages=packages, original_pip_check=subprocess.run([sys.executable,'-m','pip','check'],
            capture_output=True,text=True).stdout.strip())
    (output/'observed-environment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.runtime and not (output/'python-runtime.tar.gz').exists():
        import conda_pack
        print('Packing Python runtime without site-packages', flush=True)
        conda_pack.pack(prefix=sys.base_prefix, output=str(output/'python-runtime.tar.gz'),
            filters=[('exclude','*'),('include','bin/python*'),('include','lib/*'),
                     ('exclude','lib/python3.10/site-packages/*'),
                     ('include','include/python3.10/*'),('include','ssl/*')],
            ignore_editable_packages=True, ignore_missing_files=True,
            compress_level=1, n_threads=2, verbose=False)
        print('Python runtime packed', flush=True)
    wheels = []
    if args.wheels:
        active = [(key,d) for key,d in sorted(selected.items()) if key not in EXCLUDED]
        for index,(key,distribution) in enumerate(active,1):
            info = preserve_wheel(distribution,wheelhouse,args.repair_scripts)
            info.update(name=key,version=distribution.version,layer=layer(distribution._path))
            info.setdefault('source','installed distribution; files copied without re-compiling')
            wheels.append(info)
            (output/'wheels.json').write_text(json.dumps(wheels,indent=2)+'\n')
            print(f'{index}/{len(active)} {key}=={distribution.version} {info["bytes"]}',flush=True)
    print(json.dumps({'observed':len(packages),'preserved_wheels':len(wheels),'output':str(output)}),flush=True)


if __name__ == '__main__':
    main()
