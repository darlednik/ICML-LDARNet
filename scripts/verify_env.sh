#!/usr/bin/env bash
# Smoke-test imports and a tiny forward pass inside the container or venv.
set -euo pipefail

cd /workspace/ldar 2>/dev/null || cd "$(dirname "$0")/.."

echo "=== versions ==="
python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count", torch.cuda.device_count())
PY

echo "=== imports ==="
python - <<'PY'
import flash_attn
import mamba_ssm
from mamba_ssm.modules.mamba2 import Mamba2
from ldar.models.mixer_seq import LDarForMaskedLM
from pretrain import ldar_110m_config
print("imports ok")
PY

echo "=== forward pass (GPU 0) ==="
python - <<'PY'
import torch
from ldar.models.mixer_seq import LDarForMaskedLM
from pretrain import ldar_110m_config

if not torch.cuda.is_available():
    raise SystemExit("CUDA required for this check")

cfg = ldar_110m_config()
model = LDarForMaskedLM(cfg).cuda()
x = torch.randint(0, 7, (1, 512), device="cuda")
mask = torch.ones(1, 512, dtype=torch.bool, device="cuda")
out = model(x, attention_mask=mask)
print("logits", tuple(out.logits.shape))
print("env ok")
PY

echo "=== data paths ==="
for p in ldar_data/ldar_data.fa ldar_data/human-sequences.bed ldar_data/multi_species_genomes_dataset; do
  if [[ -e "$p" ]]; then echo "ok  $p"; else echo "MISSING  $p"; fi
done
