"""Load LDARNet checkpoints using the config embedded at save time."""

from dataclasses import fields

import torch

from ldar.models.mixer_seq import LDarForMaskedLM
from ldar.models.config_ldar import LDarConfig, SSMConfig, AttnConfig


def _build_config(payload, ckpt_path: str) -> LDarConfig:
    if not isinstance(payload, dict) or "config" not in payload or payload["config"] is None:
        raise ValueError(
            f"No embedded 'config' found in {ckpt_path}. Re-save with "
            "save_ckpt(..., config=cfg), or construct LDarConfig manually."
        )
    cfg_dict = dict(payload["config"])

    if isinstance(cfg_dict.get("ssm_cfg"), dict):
        cfg_dict["ssm_cfg"] = SSMConfig(**cfg_dict["ssm_cfg"])
    if isinstance(cfg_dict.get("attn_cfg"), dict):
        cfg_dict["attn_cfg"] = AttnConfig(**cfg_dict["attn_cfg"])

    valid = {f.name for f in fields(LDarConfig) if f.init}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
    return LDarConfig(**cfg_dict)


def load_ldar_from_ckpt(
    ckpt_path: str,
    *,
    for_training: bool = False,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    verbose: bool = True,
):
    """Load LDarForMaskedLM from a self-describing checkpoint.

    Args:
        ckpt_path: Path to checkpoint (.pt).
        for_training: If True, return the model on CPU in float32 train mode
            for HuggingFace Trainer fine-tuning. Otherwise move to ``device`` in
            ``dtype`` (default bfloat16) and call ``.eval()`` for inference.
        device: Target device when ``for_training=False``.
        dtype: Target dtype when ``for_training=False`` (default bfloat16).
        verbose: Print checkpoint summary.

    Returns:
        (model, cfg)
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _build_config(payload, ckpt_path)

    model = LDarForMaskedLM(cfg)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(
            f"Checkpoint mismatch for {ckpt_path}: {len(missing)} missing keys "
            f"(first: {missing[:5]})"
        )
    if verbose:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        step = payload.get("step") if isinstance(payload, dict) else None
        print(f"[ckpt] {ckpt_path}")
        if step is not None:
            print(f"[ckpt] step/epoch = {step}")
        print(
            f"[ckpt] layout={cfg.arch_layout}  d_model={cfg.d_model}  "
            f"d_intermediate={cfg.d_intermediate}  params={n_params:.2f}M"
        )
        print(f"[ckpt] load_state_dict: missing=0  unexpected={len(unexpected)}")
        if unexpected:
            print(f"[ckpt]   unexpected[:8] = {unexpected[:8]}")

    if not for_training:
        model.to(device=device, dtype=dtype or torch.bfloat16)
        model.eval()

    return model, cfg
