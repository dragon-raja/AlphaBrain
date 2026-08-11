#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

mkdir -p "${TEMP_DIR}/run_base_vla"
cp "${SCRIPT_DIR}/train.sh" "${TEMP_DIR}/run_base_vla/train.sh"
cat > "${TEMP_DIR}/run_finetune.sh" <<'EOF'
#!/usr/bin/env bash
printf '<%s>\n' "$@"
EOF

actual="$(bash "${TEMP_DIR}/run_base_vla/train.sh" pi05_custom configs/custom.yaml)"
expected=$'<configs/custom.yaml>\n<--mode>\n<pi05_custom>'
[[ "${actual}" == "${expected}" ]] || {
    printf 'custom config forwarding mismatch:\n%s\n' "${actual}" >&2
    exit 1
}

actual="$(bash "${TEMP_DIR}/run_base_vla/train.sh" pi05_default)"
expected='<pi05_default>'
[[ "${actual}" == "${expected}" ]] || {
    printf 'default config forwarding mismatch:\n%s\n' "${actual}" >&2
    exit 1
}

echo "train.sh forwarding tests passed"
