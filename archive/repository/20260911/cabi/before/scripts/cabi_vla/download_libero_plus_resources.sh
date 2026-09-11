#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DOWNLOAD_DIR=${LIBERO_PLUS_DOWNLOAD_DIR:-/workspace/.downloads/libero-plus}
VERIFIED_DIR=${LIBERO_PLUS_VERIFIED_DIR:-/share/longjunyu/alphabrain/datasets/libero-plus/verified}
CONNECTIONS=${LIBERO_PLUS_DOWNLOAD_CONNECTIONS:-16}
MAX_ATTEMPTS=${LIBERO_PLUS_DOWNLOAD_ATTEMPTS:-30}
RANGE_WORKERS=${LIBERO_PLUS_RANGE_WORKERS:-5}

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required" >&2
  exit 1
fi
if [[ ! "$CONNECTIONS" =~ ^[1-9][0-9]*$ || "$CONNECTIONS" -gt 32 ]]; then
  echo "LIBERO_PLUS_DOWNLOAD_CONNECTIONS must be in [1, 32]" >&2
  exit 2
fi
if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIBERO_PLUS_DOWNLOAD_ATTEMPTS must be positive" >&2
  exit 2
fi
if [[ ! "$RANGE_WORKERS" =~ ^[1-8]$ ]]; then
  echo "LIBERO_PLUS_RANGE_WORKERS must be in [1, 8]" >&2
  exit 2
fi

mkdir -p "$DOWNLOAD_DIR" "$VERIFIED_DIR"
exec 9>"$DOWNLOAD_DIR/download.lock"
if ! flock -n 9; then
  echo "another LIBERO-Plus resource download is already running" >&2
  exit 1
fi

verify_file() {
  local path=$1
  local expected_size=$2
  local expected_sha=$3
  [[ -f "$path" ]] || return 1
  [[ $(stat -c %s "$path") == "$expected_size" ]] || return 1
  [[ $(sha256sum "$path" | awk '{print $1}') == "$expected_sha" ]]
}

download_one() {
  local filename=$1
  local url=$2
  local expected_size=$3
  local expected_sha=$4
  local source="$DOWNLOAD_DIR/$filename"
  local destination="$VERIFIED_DIR/$filename"
  local log="$DOWNLOAD_DIR/${filename}.download.log"

  if verify_file "$destination" "$expected_size" "$expected_sha"; then
    echo "already verified: $destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "refusing to overwrite unverified destination: $destination" >&2
    exit 1
  fi

  if [[ -f "$source" && -f "$source.aria2" ]]; then
    echo "repairing existing aria2 partial file: $filename"
    python3 "$REPO_ROOT/scripts/cabi_vla/resume_aria2_ranges.py" \
      --file "$source" \
      --control "$source.aria2" \
      --url "$url" \
      --expected-size "$expected_size" \
      --expected-sha256 "$expected_sha" \
      --part-dir "$DOWNLOAD_DIR/range-parts-$filename" \
      --workers "$RANGE_WORKERS"
  fi

  if ! verify_file "$source" "$expected_size" "$expected_sha"; then
    local attempt
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
      echo "download attempt $attempt/$MAX_ATTEMPTS: $filename"
      # Resolve the public Hugging Face LFS URL with curl first. aria2 handles
      # fresh signed object URLs more reliably than the public redirect.
      local resolved_url
      if ! resolved_url=$(curl -fsSL --max-time 60 --range 0-1048575 \
        -o /dev/null -w '%{url_effective}' "$url"); then
        echo "object URL resolution failed for $filename; retrying" >&2
        sleep 10
        continue
      fi
      if [[ "$resolved_url" != https://* ]]; then
        echo "failed to resolve HTTPS object URL for $filename" >&2
        sleep 10
        continue
      fi
      # aria2 rejects the inherited SOCKS-style ALL_PROXY value before applying
      # the working HTTP(S) proxy variables. Keep HTTP(S)_PROXY and drop only
      # ALL_PROXY for this subprocess.
      if env -u all_proxy -u ALL_PROXY aria2c \
        --continue=true \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --file-allocation=none \
        --max-connection-per-server="$CONNECTIONS" \
        --split="$CONNECTIONS" \
        --min-split-size=4M \
        --max-tries=0 \
        --retry-wait=5 \
        --connect-timeout=60 \
        --timeout=60 \
        --console-log-level=warn \
        --summary-interval=0 \
        --download-result=hide \
        --dir="$DOWNLOAD_DIR" \
        --out="$filename" \
        "$resolved_url" >>"$log" 2>&1; then
        if [[ $(stat -c %s "$source" 2>/dev/null || true) == "$expected_size" ]]; then
          break
        fi
      fi
      sleep 10
    done
  fi

  if [[ $(stat -c %s "$source" 2>/dev/null || true) != "$expected_size" ]]; then
    echo "download did not reach expected size: $filename" >&2
    exit 1
  fi
  echo "verifying local SHA256: $filename"
  if ! verify_file "$source" "$expected_size" "$expected_sha"; then
    echo "local verification failed: $source" >&2
    exit 1
  fi

  local staging="$VERIFIED_DIR/.${filename}.copy-$$"
  trap 'rm -f "$staging"' RETURN
  cp "$source" "$staging"
  if ! verify_file "$staging" "$expected_size" "$expected_sha"; then
    echo "destination verification failed: $staging" >&2
    exit 1
  fi
  mv "$staging" "$destination"
  trap - RETURN
  echo "verified resource ready: $destination"
}

download_one \
  assets.zip \
  https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip \
  6395849578 \
  96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf

download_one \
  libero_plus_camparam_rlds.zip \
  https://huggingface.co/datasets/Sylvest/libero_plus_camparam_rlds/resolve/main/libero_plus_camparam_rlds.zip \
  16607835331 \
  a99466a1bb7eab4d0c55094d64d53ef6794ee835ba0db003fcee3e3fa6568e73

echo "all LIBERO-Plus resources are verified"
