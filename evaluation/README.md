# Nucleotide Transformer downstream evaluation

> **TL;DR.** After the camera-ready version, we identified and fixed a numerical
> precision bug in the model's EMA dechunker (bf16 → fp32 scan). On the fixed
> checkpoint, LDARNet's results on the Nucleotide Transformer benchmark improve:
> **15/18 tasks are now best among models <300M** (up from 11/18 in the paper).
> All numbers below are on the released checkpoint and reproducible with the code
> in this repo. **The paper is being updated to reflect these numbers.**

## What changed

- **Bug fix (dechunker precision).** The bidirectional EMA scan in the dechunker
  was running in bf16, which caused the state to collapse on long sequences. We
  forced the scan to fp32. This gives a more stable model without any change to
  architecture or parameter count.

> Note: LDARNet has ~110M parameters (110.69M). The camera-ready lists "120M";
> this is a labeling correction, not a change of model.

## Results (MCC ×100, mean ± 95% CI over 10 folds)

**Bold** marks tasks where LDARNet is the best model under 300M parameters.
`Old` is the camera-ready LDARNet number. `Best <300M` and `Best overall` are the
strongest *other* models (LDARNet excluded), under 300M and of any size respectively.

| Task | LDARNet (updated) | LR | BS | Old | Best <300M (other) | Best overall (other) |
|------|:-----------------:|:--:|:--:|:---:|:-------------------:|:--------------------:|
| H3 | **81.7 ± 0.8** | 1e-4 | 64 | 78.2 | 79.4 (Caduceus-Ph) | 80.6 (Generator) |
| H4 | **81.2 ± 0.5** | 5e-5 | 64 | 81.3 | 79.9 (Caduceus-PS) | 81.5 (Generator) |
| H3K9ac | **64.6 ± 0.7** | 1e-4 | 64 | 60.3 | 58.6 (HyenaDNA) | 61.2 (Generator) |
| H3K14ac | **64.0 ± 0.7** | 1e-4 | 64 | 58.9 | 60.8 (HyenaDNA) | 60.8 (HyenaDNA) |
| H4ac | **65.7 ± 0.7** | 2e-4 | 64 | 62.3 | 58.5 (Caduceus-PS) | 59.2 (Generator) |
| H3K4me1 | **59.5 ± 0.7** | 2e-4 | 64 | 58.3 | 51.2 (DNABERT-2) | 55.3 (Generator) |
| H3K4me2 | **54.2 ± 0.9** | 2e-4 | 64 | 49.6 | 45.5 (HyenaDNA) | 45.5 (HyenaDNA) |
| H3K4me3 | **60.2 ± 1.0** | 2e-4 | 64 | 57.6 | 55.0 (HyenaDNA) | 55.0 (HyenaDNA) |
| H3K36me3 | **68.7 ± 0.4** | 1e-4 | 64 | 62.4 | 61.4 (HyenaDNA) | 65.7 (Generator) |
| H3K79me3 | **69.5 ± 0.6** | 1e-4 | 64 | 68.7 | 68.2 (Caduceus-PS) | 68.2 (Caduceus-PS) |
| Promoter all | 94.3 ± 0.1 | 1e-4 | 64 | 93.9 | 94.5 (DNABERT-2) | 96.2 (Generator) |
| Promoter non-TATA | **94.4 ± 0.3** | 2e-4 | 64 | 94.4 | 94.4 (DNABERT-2) | 96.2 (Generator) |
| Promoter TATA | 91.5 ± 0.6 | 1e-4 | 64 | 92.3 | 92.0 (Enformer) | 94.8 (Generator) |
| Enhancer | **54.3 ± 1.1** | 2e-5 | 64 | 57.7 | 52.5 (DNABERT-2) | 58.0 (Generator) |
| Enhancer type | **44.2 ± 2.2** | 5e-5 | 64 | 42.0 | 43.3 (GROVER) | 47.7 (Generator) |
| Splice all | **95.8 ± 0.3** | 2e-4 | 64 | 94.2 | 95.3 (Caduceus-PS) | 97.8 (Generator) |
| Splice acceptor | 91.5 ± 0.9 | 2e-4 | 64 | 92.7 | 93.5 (HyenaDNA) | 98.1 (Generator) |
| Splice donor | **93.3 ± 1.9** | 2e-4 | 64 | 92.8 | 93.0 (Caduceus-PS) | 97.8 (Generator) |

- **Best among models <300M: 15/18 tasks** (camera-ready: 11/18).
- **Best of all models, any size (incl. NT-2.5B, NT-v2-500M, Generator-1.2B): 9/18 tasks**
  — H3, H3K9ac, H3K14ac, H4ac, H3K4me1, H3K4me2, H3K4me3, H3K36me3, H3K79me3.

## Fold-level proof (committed)

Per-fold `test_mcc` values for all 18 canonical configs are committed under
[`fold_results/`](fold_results/) (180 JSON files, ~52 KB total). Structure:

```
evaluation/fold_results/<task>/lr<lr>_bs<bs>/fold{0..9}/test_results.json
```

## Reproducing from scratch

Download the checkpoint (if not already present):

```bash
huggingface-cli download darlednik/LDARNet-110M model_ckpt_110m.pt --local-dir models_ckpts
```

Run all 18 tasks (edit `GPUS` in `run_eval.sh` to match your machine):

```bash
bash evaluation/run_eval.sh
```

Fresh runs write fold-level metrics under
`results/nt_110m_ldarnet/<task>/lr<lr>_bs<bs>/fold*/test_results.json`.
The committed numbers in the table above come from [`fold_results/`](fold_results/).
