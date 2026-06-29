from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from ldar.modules.utils import get_seq_idx


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor   # (B,L,2), soft distribution [non-boundary, boundary]
    boundary_mask: torch.Tensor   # (B,L), hard mask for inference
    selected_probs: torch.Tensor  # (B,L,1), probability of the chosen class


@dataclass
class RoutingModuleState:
    has_seen_tokens: torch.Tensor
    last_hidden_state: torch.Tensor


@dataclass
class DeChunkState:
    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModule(nn.Module):
    """
    Bidirectional routing for MLM.
    - Computes similarity to both left and right neighbors.
    - Turns local discontinuities into boundary probabilities.
    - Provides both soft (for training) and hard (for inference) outputs.
    """

    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}

        # Linear projections for q/k
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Identity initialization (stabilizes early training)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(self, hidden_states, mask=None, cu_seqlens=None, inference=False):
        """
        Args:
            hidden_states: (B,L,D) padded mode
            mask: (B,L) bool, valid tokens (True=valid)
            cu_seqlens: (optional) ragged support (not used here)
            inference: if True, returns hard boundary_mask
        """
        assert (mask is not None) or (cu_seqlens is not None), \
            "Either mask or cu_seqlens must be provided"
        B, L, D = hidden_states.shape
        valid = mask.bool()

        # Projections
        q = F.normalize(self.q_proj_layer(hidden_states), dim=-1)
        k = F.normalize(self.k_proj_layer(hidden_states), dim=-1)

        # Cosine sim with right neighbor
        cos_fwd = torch.einsum("b l d, b l d -> b l", q[:, :-1], k[:, 1:])

        # Cosine sim with left neighbor
        cos_bwd = torch.einsum("b l d, b l d -> b l", q[:, 1:], k[:, :-1])

        # Symmetric similarity: average of forward/backward
        cos_sim = 0.5 * (cos_fwd + cos_bwd)

        # Boundary probability: discontinuity = (1 - cos)/2
        p_boundary = torch.clamp((1.0 - cos_sim) / 2.0, 0.0, 1.0)

        PAD_PROB = 1.0
        boundary_prob = F.pad(p_boundary, (1, 0), "constant", PAD_PROB)

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = (selected_idx == 1) & valid                         # [B,L], False on PAD

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,   # hard/soft depending on mode
            selected_probs=selected_probs, # chosen probability
        )


class ChunkLayer(nn.Module):
    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        if (cu_seqlens is None) and (mask is not None):
            boundary_mask = boundary_mask & mask.bool()  # idempotent if already applied upstream

        if cu_seqlens is not None:
            # Packed mode: boundary_mask is (T,), hidden_states is (T, D)
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None  # staying in packed mode downstream
        else:
            # Padded (B, L, D) + (B, L)
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)  # (B,)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            # Push non-selected (and padded) positions to the right
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None  # unused in padded path

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask


class DeChunkLayer(nn.Module):
    """
    Bidirectional EMA-style dechunker.

    Forward EMA:   z_j = p_j z_j + (1 - p_j) z_{j-1}
    Backward EMA:  z_j = p_j z_j + (1 - p_j) z_{j+1}
    Output is the mean of the two passes (when bidirectional=True).
    """

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
        bidirectional=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        self.bidirectional = bidirectional
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        mask=None,                 # (B, L) full-resolution padding mask (True = valid)
        compressed_mask=None,      # (B, M) compressed padding mask (True = valid)
    ):
        B, L = boundary_mask.shape
        M = hidden_states.shape[1]
        device = hidden_states.device
        original_dtype = hidden_states.dtype

        p_full = torch.clamp(boundary_prob[..., 1].float(), 1e-4, 1.0 - 1e-4)  # (B,L)
        token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        seq_sorted_indices = torch.argsort(token_idx, dim=1)  # (B,L)
        p = torch.gather(p_full, 1, seq_sorted_indices[:, :M])  # (B,M)

        p = p.float()
        dt = torch.log1p(-p).neg()                             # (B,M) fp32, dt = -log(1-p)
        x = (hidden_states.to(torch.float32) / dt[..., None])  # (B,M,D) fp32
        A = -torch.ones((self.nheads,), device=device, dtype=torch.float32)
        b = p                                                  # (B,M) fp32
        c = torch.ones_like(b)

        def run_scan(x, dt, b, c):
            out = mamba_chunk_scan_combined(
                rearrange(x, "b m (h d) -> b m h d", h=self.nheads),
                repeat(dt, "b m -> b m h", h=self.nheads),
                A,
                rearrange(b, "b m -> b m 1 1"),
                rearrange(c, "b m -> b m 1 1"),
                chunk_size=self.block_size,
                seq_idx=None,
            )
            return rearrange(out, "b m h d -> b m (h d)")

        if self.bidirectional:
            if compressed_mask is None:
                num_boundaries = boundary_mask.sum(dim=1)  # (B,)
                compressed_mask = (
                    torch.arange(M, device=device)[None, :] < num_boundaries[:, None]
                )
            pad = ~compressed_mask.bool()  # (B,M)

            x_masked = x.masked_fill(pad.unsqueeze(-1), 0.0)

            out_fwd = run_scan(x_masked, dt, b, c)

            x_rev  = torch.flip(x_masked, dims=[1])
            dt_rev = torch.flip(dt, dims=[1])
            b_rev  = torch.flip(b,  dims=[1])
            c_rev  = torch.flip(c,  dims=[1])
            out_bwd = torch.flip(run_scan(x_rev, dt_rev, b_rev, c_rev), dims=[1])

            out = 0.5 * (out_fwd + out_bwd)
        else:
            out = run_scan(x, dt, b, c)

        plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1
        plug_back_idx = plug_back_idx.clamp(min=0, max=M - 1)  # OOB guard
        out = torch.gather(
            out,
            1,
            plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        )  # (B,L,D)

        if (cu_seqlens is None) and (mask is not None):
            out = out.masked_fill(~mask.bool().unsqueeze(-1), 0)

        return out.to(original_dtype)