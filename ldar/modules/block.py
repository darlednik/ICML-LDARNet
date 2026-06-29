from functools import partial
from typing import Optional

from torch import nn, Tensor
import torch

from flash_attn.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.mamba2 import Mamba2

from .mha import MHA
from .mlp import SwiGLU

from typing import Literal

class BiMamba2Wrapper(nn.Module):

    def __init__(self, d_model: int, layer_idx: int = 0, combine: str = "mean", **mamba_kwargs):
        super().__init__()
        self.fwd = Mamba2(d_model=d_model, layer_idx=layer_idx, **mamba_kwargs)
        self.bwd = Mamba2(d_model=d_model, layer_idx=layer_idx, **mamba_kwargs)

        self.bwd.in_proj.weight = self.fwd.in_proj.weight
        self.bwd.in_proj.bias = self.fwd.in_proj.bias
        self.bwd.out_proj.weight = self.fwd.out_proj.weight
        self.bwd.out_proj.bias = self.fwd.out_proj.bias
        
        self.combine = combine
        self.d_model = d_model
        if combine == "gated":
            self.gate = nn.Parameter(torch.zeros(1, 1, d_model))

    @staticmethod
    def _apply_mask(x: Tensor, mask: Optional[Tensor]) -> Tensor:

        if mask is None:
            return x
        return x * mask.unsqueeze(-1).to(x.dtype)

    def forward(
        self,
        hidden_states: Tensor,                         # [B, T, D]
        *,
        padding_mask: Optional[Tensor] = None,         # bool [B, T]
        attention_mask: Optional[Tensor] = None,       # alias
        **unused,
    ) -> Tensor:
        if padding_mask is None:
            padding_mask = attention_mask

        x = self._apply_mask(hidden_states, padding_mask)

        y_f = self.fwd(x, inference_params=None)       # [B,T,D]
        y_f = self._apply_mask(y_f, padding_mask)

        x_rev = torch.flip(x, [1])
        pm_rev = None if padding_mask is None else torch.flip(padding_mask, [1])
        y_b = self.bwd(x_rev, inference_params=None)
        y_b = self._apply_mask(y_b, pm_rev)
        y_b = torch.flip(y_b, [1])

        if self.combine == "mean":
            y = 0.5 * (y_f + y_b)
        elif self.combine == "sum":
            y = y_f + y_b
        else:  # gated
            g = torch.sigmoid(self.gate)
            y = g * y_f + (1 - g) * y_b
        return y

def create_block(
    arch,
    d_model,
    d_intermediate=None,
    ssm_cfg=dict(),
    attn_cfg=dict(),
    norm_epsilon: float = 1e-5,
    layer_idx: int | None = None,
    residual_in_fp32: bool = True,
    device=None,
    dtype=None,
    *,
    mode: Literal["causal", "mlm"] = "mlm",  
):

    factory_kwargs = {"device": device, "dtype": dtype}
    is_mlm = mode == "mlm"

    if arch in ("t", "T"):  
        attn_cfg_local = dict(attn_cfg)

        mixer_cls = partial(
            MHA,
            **attn_cfg_local,
            **factory_kwargs,
            layer_idx=layer_idx,
        )

    elif arch in ("m", "M"):
        assert is_mlm, (
            "LDARNet ships an MLM-only (bidirectional) configuration; "
            "the causal Mamba2 path is not implemented."
        )
        mixer_cls = partial(
            BiMamba2Wrapper,
            **ssm_cfg,
            **factory_kwargs,
            layer_idx=layer_idx,
        )
    else:
        raise NotImplementedError(f"unknown arch '{arch}'")

    if arch in ("T", "M"):
        mlp_cls = partial(
            SwiGLU,
            d_intermediate=d_intermediate,
            **factory_kwargs,
        )
    elif arch in ("t", "m"):
        mlp_cls = nn.Identity
    else:
        raise NotImplementedError

    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
    )
    return block


class Block(nn.Module):
    def __init__(
        self,
        d_model,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=None,
        residual_in_fp32=True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(d_model)
            self.mlp = mlp_cls(d_model)
        else:
            self.mlp = None

        assert RMSNorm is not None, "Triton is not installed"
        assert isinstance(self.norm1, RMSNorm), "Only RMSNorm is supported"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        mixer_kwargs=None,
        *,                              
        attention_mask: Optional[Tensor] = None, 
        padding_mask:  Optional[Tensor] = None,  
    ):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )

        if mixer_kwargs is None:
            mixer_kwargs = {}

        if padding_mask is None:
            padding_mask = attention_mask
        if padding_mask is not None:
            mixer_kwargs = {
                **mixer_kwargs,
                "attention_mask": padding_mask,
                "padding_mask":   padding_mask, 
            }

        hidden_states = self.mixer(
            hidden_states,
            **mixer_kwargs,
        )

        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual