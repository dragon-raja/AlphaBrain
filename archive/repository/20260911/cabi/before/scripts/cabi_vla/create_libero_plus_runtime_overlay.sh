#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BASE_PYTHON=${LIBERO_PLUS_BASE_PYTHON:-/share/longjunyu/capt-vla/envs/libero/bin/python}
OVERLAY_ROOT=${1:-/share/longjunyu/alphabrain/envs/libero-plus-runtime-overlay-v1}
REQUIREMENTS="$REPO_ROOT/docs/cabi_vla/configs/libero_plus_runtime_overlay_requirements.txt"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "LIBERO base Python is missing: $BASE_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "requirements file is missing: $REQUIREMENTS" >&2
  exit 1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libMagickWand'; then
  echo "MagickWand runtime is missing; install libmagickwand-6.q16-7t64 first" >&2
  exit 1
fi

verify_overlay() {
  PYTHONPATH="$OVERLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$BASE_PYTHON" - <<'PY'
import h5py
import pywt
import skimage
from wand.api import library
from wand.version import VERSION as wand_version

assert h5py.__version__ == "3.11.0"
assert skimage.__version__ == "0.21.0"
assert wand_version == "0.6.13"
assert pywt.__version__ == "1.4.1"
assert library.MagickMotionBlurImage
print("libero_plus_runtime_overlay", "ok")
print("h5py", h5py.__version__)
print("skimage", skimage.__version__)
print("wand", wand_version)
print("pywt", pywt.__version__)
PY
}

if verify_overlay >/dev/null 2>&1; then
  echo "LIBERO-Plus runtime overlay already prepared: $OVERLAY_ROOT"
  verify_overlay
  exit 0
fi

mkdir -p "$OVERLAY_ROOT"
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  "$BASE_PYTHON" -m pip install \
  --target "$OVERLAY_ROOT" \
  --no-deps \
  --disable-pip-version-check \
  --requirement "$REQUIREMENTS"

verify_overlay
