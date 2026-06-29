# Wheels (not in git)

Docker build needs three local wheels (~776 MB). They are listed in `.gitignore`.

**Option A — copy from a working machine:**
```bash
cp /path/to/full/project/wheels/*.whl wheels/
```

**Option B — partial download:**
```bash
bash scripts/download_wheels.sh
# then manually add flash_attn and mamba_ssm wheels
```

Required files:
- `flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`
- `mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`
- `causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`
