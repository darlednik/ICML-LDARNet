# LDARNet

**LDARNet** (DNA Adaptive Representation Network) is a hierarchical foundation
model for genomic sequences with **learnable tokenization**. Instead of fixed
k-mer or byte tokenization, it learns content-aware sequence boundaries through
*dynamic chunking*, processes the compressed sequence with a bidirectional
state-space backbone (**BiMamba-2**), and reconstructs base-resolution
representations through *dechunking*. The encoder additionally uses a single
local attention layer for fine-grained motif recognition. LDARNet is pretrained
with masked language modeling (MLM) on the human reference genome together with
a multispecies collection.

📄 **Paper:** [LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling](https://arxiv.org/abs/2606.04552) (ICML 2026)

> **Note on the paper:** the published manuscript contains a typo regarding the
> model size. The released checkpoint is the **110M-parameter** LDARNet; we will
> update the arXiv version accordingly.

This repository contains the model implementation and the MLM pretraining
pipeline used to train LDARNet from scratch.

## Architecture at a glance

| Component | Layout | `d_model` |
|---|---|---|
| Encoder | `m3t1` — 3× BiMamba-2 + 1 local-attention layer | 512 |
| Backbone | `M10` — 10× BiMamba-2 (+ SwiGLU) | 768 |
| Decoder | `m4` — 4× BiMamba-2 | 512 |

- Single-stage hierarchy, compression ratio **N = 4**, ≈**110M** parameters
  (run `count_params` for the exact figure).
- Bidirectional throughout: BiMamba-2 (forward + reverse Mamba-2 with shared
  projections, mean fusion), non-causal local attention, a bidirectional router,
  and a bidirectional EMA dechunker.
- Byte-level vocabulary: `{A, C, G, T, N, [MASK], [PAD]}`.
- Reverse-complement augmentation during pretraining supplies the biological
  RC symmetry that weight tying alone does not encode.

## Repo layout

```
ldar/           model + datasets + collator
pretrain.py     multi-GPU entry point
notebooks/      boundary interpretability (Figures 1–6)
Dockerfile      CUDA 12.9, PyTorch 2.7.1, pinned transformers
wheels/         GPU wheels for Docker (not in git — see wheels/README.md)
scripts/        verify_env.sh, download_wheels.sh
```

## Pretrained weights

The 110M checkpoint is hosted on Hugging Face:
**[darlednik/LDARNet-110M](https://huggingface.co/darlednik/LDARNet-110M)**

Download into `models_ckpts/` (used by evaluation and optional local caching):

```bash
pip install "huggingface_hub>=0.24.0,<1.0"
huggingface-cli download darlednik/LDARNet-110M model_ckpt_110m.pt --local-dir models_ckpts
```

Load in Python (config is embedded in the checkpoint):

```python
import torch
from ldar.utils.ckpt import load_ldar_from_ckpt

model, cfg = load_ldar_from_ckpt(
    "models_ckpts/model_ckpt_110m.pt",
    device="cuda",
    dtype=torch.bfloat16,
)
```

## Biological interpretability notebook

[`notebooks/boundary_interpretability.ipynb`](notebooks/boundary_interpretability.ipynb)
reproduces the boundary-analysis figures from the paper:

- **Fig 1** — average router boundary profiles centered on regulatory motifs (TATA, CAAT, Kozak, Inr)
- **Fig 2** — splice donor/acceptor boundary enrichment (± strand)
- **Fig 3** — curated loci (HBB promoter from GRCh38 when FASTA is mounted; SV40 / splice controls)
- **Fig 4** — native vs dinucleotide-shuffled motif backgrounds
- **Fig 5** — boundary density along sequence length
- **Fig 6** — HBB TSS window from the reference genome

The notebook **downloads weights automatically** from
[darlednik/LDARNet-110M](https://huggingface.co/darlednik/LDARNet-110M) on first run
(or reuses `models_ckpts/model_ckpt_110m.pt` if already present).

```bash
jupyter notebook notebooks/boundary_interpretability.ipynb
```

**Optional:** mount GRCh38 for Fig 3 (HBB) and Fig 6 — see [Data layout](#data-layout-mount-separately)
(`ldar_data/ldar_data.fa`). NT downstream tasks (Figs 2, search pools) require `datasets`
and internet on first fetch.

Figures are saved under `figures/boundary_interpretability/`.

## Setup from a clean clone

```bash
git clone git@github.com:darlednik/ICML-LDARNet.git
cd ICML-LDARNet

# wheels (not in git)
cp /path/to/wheels/*.whl wheels/
# or: bash scripts/download_wheels.sh  (+ copy flash_attn + mamba_ssm manually)

# data (not in git) — symlink or copy
ln -s /path/to/ldar_data ldar_data

# build image (transformers pinned in Dockerfile — no manual downgrade)
docker build -t ldar:latest .
```

## Run container

```bash
docker run --name ldar_train --gpus all -it \
  --ipc=host --shm-size=32g \
  -v "$PWD":/workspace/ldar \
  -v /tmp:/tmp \
  -w /workspace/ldar \
  -e HF_DATASETS_CACHE=/tmp/hf-cache \
  -e TRANSFORMERS_CACHE=/tmp/hf-cache \
  -e TRITON_CACHE_DIR=/tmp/triton-cache \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/ti-cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ldar bash
```

Inside the container:

```bash
bash scripts/verify_env.sh
tmux new -s train
```

## Train

**Smoke test** (1 GPU, tiny — verifies the forward/backward path, including the
bidirectional dechunker on a padded compressed batch):

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py \
  --epochs 1 --batch_size 2 --accum_steps 1 \
  --seq_len 512 --num_bytes_per_token 4 --max_steps 2
```

**Full training** (multi-GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 pretrain.py \
  --epochs 15 --batch_size 32 --accum_steps 16 \
  --seq_len 4096 --num_bytes_per_token 4 \
  --lr 5e-4 --alpha_ratio 0.03 \
  2>&1 | tee logs/pretrain.log
```

Notes:
- `--num_bytes_per_token` **is** the compression ratio `N` (the paper uses
  `N = 4`). It also sets the outer-layer LR multiplier
  `λ = sqrt(N) · d_back/d_outer = sqrt(4) · 768/512 = 3.0`.
- **Effective batch size** = `batch_size × accum_steps × num_gpus`. The command
  above yields `32 × 16 × 3 = 1536`. Set this deliberately to the regime you
  want to report; for a single GPU, `32 × 16 = 512`.

## `transformers` version

`mamba_ssm` breaks with `transformers>=5`. This repo pins
**`transformers>=4.45,<4.49`** in `Dockerfile` and `pyproject.toml`.

Verify inside the container:

```bash
python -c "import transformers; print(transformers.__version__)"
python -c "from mamba_ssm.modules.mamba2 import Mamba2; print('ok')"
```

## Outputs

Checkpoints are written under `models_ckpts/` (each checkpoint stores the model
config, so it is self-describing for reloading).
Logs: `logs/*_rank*.csv`.

## Data layout (mount separately)

```
ldar_data/
├── ldar_data.fa
├── ldar_data.fa.fai
├── human-sequences.bed          # 4th column: train / valid / test split
└── multi_species_genomes_dataset/
```

## Citation

If you use LDARNet, please cite:

```bibtex
@misc{ledneva2026ldarnetdnaadaptiverepresentation,
      title={LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling},
      author={Daria Ledneva and Denis Kuznetsov},
      year={2026},
      eprint={2606.04552},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.04552},
}
```

## License

Code released under the [Apache License 2.0](LICENSE). Note that the
pretraining data (the human reference genome and the Nucleotide Transformer
multispecies collection) are distributed under their own respective licenses.

## Acknowledgements

The dynamic-chunking design builds on
[H-Net](https://github.com/goombalab/hnet); the bidirectional Mamba construction
follows [Caduceus](https://github.com/kuleshov-group/caduceus); state-space
layers use [mamba-ssm](https://github.com/state-spaces/mamba) and attention uses
[flash-attention](https://github.com/Dao-AILab/flash-attention).