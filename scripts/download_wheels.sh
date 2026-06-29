#!/usr/bin/env bash
# Download GPU wheels required for Docker build (not stored in git).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/wheels"
cd "$ROOT/wheels"

download() {
  local url="$1" out="$2"
  echo "Downloading $out ..."
  curl -L --fail --retry 3 -o "$out" "$url"
}

download \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  "causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# flash-attn and mamba_ssm wheels: copy from an existing machine or build locally.
# Pre-built URLs vary by release; if missing, copy manually:
#   cp /path/to/wheels/flash_attn-*.whl wheels/
#   cp /path/to/wheels/mamba_ssm-*.whl wheels/

for whl in \
  flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
  mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
do
  if [[ ! -f "$whl" ]]; then
    echo "MISSING: wheels/$whl"
    echo "  Copy manually from your working environment (see README)."
  fi
done

ls -lh "$ROOT/wheels/"*.whl 2>/dev/null || true
