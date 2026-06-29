
import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange

from flash_attn import (
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

from .rotary import RotaryEmbedding


def _flash_compute_dtype(device: torch.device) -> torch.dtype:
    try:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _mask_to_cu_seqlens(attn_mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    lengths = attn_mask.long().sum(dim=1).to(torch.int32)  # [B]
    cu = torch.zeros(attn_mask.size(0) + 1, dtype=torch.int32, device=attn_mask.device)
    cu[1:] = torch.cumsum(lengths, dim=0)
    return cu, int(lengths.max().item())


class FlashSelfAttentionMLM(nn.Module):

    def __init__(
        self,
        softmax_scale=None,
        window_size=(-1, -1),
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.window_size = window_size

    def forward(self, qkv, cu_seqlens=None, max_seqlen=None):
        orig_dtype = qkv.dtype
        compute_dtype = _flash_compute_dtype(qkv.device)
        if orig_dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(compute_dtype)

        if cu_seqlens is not None:
            out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                softmax_scale=self.softmax_scale,
                causal=False,                
                window_size=self.window_size,
            )
        else:
            out = flash_attn_qkvpacked_func(
                qkv,
                softmax_scale=self.softmax_scale,
                causal=False,             
                window_size=self.window_size,
            )

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out


class LinearResidual(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


class MHA(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        qkv_proj_bias=False,
        out_proj_bias=False,
        window_size=-1,  # -1: global
        softmax_scale=None,
        layer_idx=None,              
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim

        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        qkv_dim = self.head_dim * (3 * self.num_heads)

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.Wqkv = nn.Linear(d_model, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        win = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.inner_attn = FlashSelfAttentionMLM(
            softmax_scale=softmax_scale,
            window_size=win,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, **factory_kwargs)

    def forward(
        self,
        x,
        *,
        attention_mask=None,  
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None, 
        **kwargs,
    ):

        qkv = self.Wqkv(x)  # (..., 3*h*d)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim).contiguous()

        assert qkv.dim() == 5, "expected (B, L, 3, H, D) with attention_mask"
        B, L = attention_mask.shape

        attention_mask = attention_mask.bool()
        cu, mx = _mask_to_cu_seqlens(attention_mask)

        # print(qkv.shape)
        qkv_packed = qkv[attention_mask].contiguous()
        # print(qkv_packed.shape)

        if self.rotary_emb_dim > 0:
            qkv_packed = self.rotary_emb(
                qkv_packed, seqlen_offset=0, cu_seqlens=cu, max_seqlen=mx
            )

        ctx_packed = self.inner_attn(qkv_packed, cu_seqlens=cu, max_seqlen=mx)  # (total, H, D)

        context = torch.zeros(
            B, L, self.num_heads, self.head_dim,
            dtype=ctx_packed.dtype, device=ctx_packed.device,
        )
        context[attention_mask] = ctx_packed

        out = self.out_proj(
            rearrange(context.to(self.out_proj.weight.dtype), "... h d -> ... (h d)")
        )
        return out
