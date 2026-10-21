#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PREFIX=""
BUNDLE=""
TEST=0
CHECK_DATA=0
while (($#)); do
    case "$1" in
        --prefix|--bundle|--repo)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --prefix) PREFIX="$2";;
                --bundle) BUNDLE="$2";;
                --repo) REPO="$2";;
            esac
            shift 2;;
        --test) TEST=1; shift;;
        --check-data) CHECK_DATA=1; shift;;
        --help|-h)
            echo "Usage: bash environment/qwen35/install.sh --prefix /new/environment --bundle /offline/bundle [--repo /checkout] [--test] [--check-data]"
            exit 0;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
[[ -n "$PREFIX" && -n "$BUNDLE" ]] || { echo "--prefix and --bundle are required" >&2; exit 2; }
[[ "$PREFIX" = /* && "$BUNDLE" = /* && "$REPO" = /* ]] || { echo "Use absolute paths" >&2; exit 2; }
[[ ! -e "$PREFIX" && ! -L "$PREFIX" ]] || { echo "Destination already exists: $PREFIX" >&2; exit 2; }
[[ -f "$BUNDLE/manifest.json" && -f "$BUNDLE/python-runtime.tar.gz" ]] || { echo "Incomplete offline bundle" >&2; exit 2; }
cmp -s "$SCRIPT_DIR/manifest.json" "$BUNDLE/manifest.json" || { echo "Bundle does not match this environment specification" >&2; exit 2; }

# The host Python only checks checksums; package installation uses the bundled Python.
python3 - "$BUNDLE" <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
root=Path(sys.argv[1]);manifest=json.loads((root/'manifest.json').read_text())
if platform.system()!='Linux' or platform.machine()!='x86_64':
    raise SystemExit('This bundle targets Linux x86_64')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
if digest(root/'python-runtime.tar.gz')!=manifest['runtime_sha256']:
    raise SystemExit('Python runtime checksum mismatch')
for item in manifest['wheels']:
    if digest(root/'wheelhouse'/item['filename'])!=item['sha256']:
        raise SystemExit('Wheel checksum mismatch: '+item['filename'])
print('Offline bundle checksums verified')
PY

mkdir -p "$PREFIX"
touch "$PREFIX/.installation-incomplete"
tar -xzf "$BUNDLE/python-runtime.tar.gz" -C "$PREFIX"
export PATH="$PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
"$PREFIX/bin/python" "$PREFIX/bin/conda-unpack"
"$PREFIX/bin/python" -m ensurepip --upgrade
mkdir -p "$PREFIX/reproduction"

"$PREFIX/bin/python" - "$BUNDLE" "$PREFIX" <<'PY'
import json,subprocess,sys
from pathlib import Path
root,prefix=map(Path,sys.argv[1:])
manifest=json.loads((root/'manifest.json').read_text())
# Overlay packages must win over user/base packages that share a namespace.
for layer in ['base','user','overlay','addition']:
    packages=[p for p in manifest['wheels'] if p['layer']==layer]
    if not packages:continue
    lock=prefix/'reproduction'/('requirements-'+layer+'.txt')
    lock.write_text(''.join(p['name']+'=='+p['version']+' --hash=sha256:'+p['sha256']+'\n' for p in packages))
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--require-hashes',
        '--find-links',str(root/'wheelhouse'),'-r',str(lock)],check=True)
PY

"$PREFIX/bin/python" -m pip install --no-index --no-deps --no-build-isolation -e "$REPO"
ARGS=(--manifest "$BUNDLE/manifest.json" --repo "$REPO" --report "$PREFIX/reproduction/verification.json")
((TEST == 0)) || ARGS+=(--test)
((CHECK_DATA == 0)) || ARGS+=(--check-data)
"$PREFIX/bin/python" "$SCRIPT_DIR/verify.py" "${ARGS[@]}"
cat > "$PREFIX/reproduction/activate.sh" <<'SH'
QWEN35_PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$QWEN35_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
unset QWEN35_PREFIX
SH
rm "$PREFIX/.installation-incomplete"
echo "Environment ready: $PREFIX"
echo "Python: $PREFIX/bin/python"
