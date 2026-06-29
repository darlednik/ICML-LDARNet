import re
import copy
from dataclasses import dataclass, field

import optree

from typing import Optional

import torch
import torch.nn as nn

from flash_attn.ops.triton.layer_norm import RMSNorm

from ldar.modules.block import create_block
from ldar.modules.utils import get_seq_idx, get_stage_cfg

from ldar.models.config_ldar import LDarConfig

class Isotropic(nn.Module):
    def __init__(
        self,
        config: LDarConfig,
        pos_idx: int,
        stage_idx: int,
        mode: str = "mlm",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.mode = mode
        self.stage_idx = stage_idx
        self.d_model = config.d_model[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T")
            assert n_layer.isdigit()
            layers += [
                create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
                    layer_idx=(layer_idx + i),
                    mode=self.mode,
                    **factory_kwargs,
                )
                for i in range(int(n_layer))
            ]
            if arch.islower():
                self.height += int(n_layer)
            else:
                self.height += 2 * int(n_layer)
            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)

        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)

    def forward(
        self,
        hidden_states,
        *,
        padding_mask=None, 
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert (padding_mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        
        packed = False
        assert (
            hidden_states.dim() == 3
        ), "Hidden states must be (B, L, D) in unpacked mode"
        # kwargs passed to Block / Mixer
        attn_mixer_kwargs.update({
            "attention_mask": padding_mask,
            "padding_mask":   padding_mask,
        })
        ssm_mixer_kwargs.update({
            "padding_mask": padding_mask,
        })

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("t", "T"):
                layer_mixer_kwargs = attn_mixer_kwargs
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)
            else:
                # Currently supporting only Mamba2 and MHA
                raise NotImplementedError

            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=None,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Setting prenorm=False ignores the residual
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        return hidden_states