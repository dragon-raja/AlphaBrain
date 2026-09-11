#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENV_ROOT=${1:-/share/longjunyu/alphabrain/envs/libero-plus-data-v1}
REQUIREMENTS="$REPO_ROOT/docs/cabi_vla/configs/libero_plus_data_requirements.txt"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 1
fi
if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "requirements file is missing: $REQUIREMENTS" >&2
  exit 1
fi
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  if [[ -e "$ENV_ROOT" ]]; then
    echo "refusing to reuse an incomplete environment: $ENV_ROOT" >&2
    exit 1
  fi
  uv venv --python 3.12 "$ENV_ROOT"
fi

UV_LINK_MODE=copy uv pip install \
  --python "$ENV_ROOT/bin/python" \
  --requirement "$REQUIREMENTS"

"$ENV_ROOT/bin/python" - <<'PY'
import importlib.metadata

for package in ("crc32c", "numpy", "pillow", "protobuf", "tfrecord"):
    print(f"{package}=={importlib.metadata.version(package)}")
PY
