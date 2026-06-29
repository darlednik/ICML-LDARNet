from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn

from ldar.modules.isotropic import Isotropic
from ldar.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState,
)
from ldar.modules.utils import apply_optimization_params

from .config_ldar import LDarConfig


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)


@dataclass
class LDarState:
    routing_module_state: Optional[RoutingModuleState] = None
    dechunk_state: Optional[DeChunkState] = None


@dataclass
class LDarOutput:
    """
    Extended LDar output with optional intermediate states.
    Backward compatible via __iter__.
    """
    hidden_states: torch.Tensor  # main output (after trimming to [:D])
    boundary_predictions: list  # boundary routing predictions
    encoder_output: Optional[torch.Tensor] = None  # encoder output (full width)
    decoder_output_full: Optional[torch.Tensor] = None  # decoder output before trimming
    
    def __iter__(self):
        """Backward compatible tuple unpacking."""
        return iter((self.hidden_states, self.boundary_predictions))
    
    def __getitem__(self, idx):
        """Backward compatible index access."""
        if idx == 0:
            return self.hidden_states
        elif idx == 1:
            return self.boundary_predictions
        else:
            raise IndexError("LDarOutput index out of range")


class LDar(nn.Module):
    def __init__(
        self,
        config: LDarConfig,
        stage_idx: int,
        mode: str = "mlm",  
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.mode = mode     

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = None
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = LDar
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                **_pos_idx_dict,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)

            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - config.d_model[stage_idx - 1], **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None
    

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0) -> None:
        n_residuals = parent_residuals
        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)

        else:
            n_residuals += self.encoder.height + self.decoder.height
            for name, m in self.encoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            for name, m in self.decoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    
            self.main_network._init_weights(initializer_range, n_residuals)
    

    def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        """
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
        
        if not self.is_innermost:
            self.main_network._apply_lr_multiplier(lr_multiplier)

    def forward(
        self,
        hidden_states,
        *,
        padding_mask=None,      
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None,
        return_all_outputs=False,
        **mixer_kwargs,
    ):
        assert (padding_mask is not None) ^ (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either padding_mask OR (cu_seqlens & max_seqlen) must be provided"

        assert (
            padding_mask is not None
        ), "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states,
                padding_mask=padding_mask,            
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                inference_params=None, 
                mode=self.mode,   
                **mixer_kwargs,
            )
            hidden_states_trimmed = hidden_states[..., :D]
            
            if return_all_outputs:
                return LDarOutput(
                    hidden_states=hidden_states_trimmed,
                    boundary_predictions=[],
                    encoder_output=None,  # no encoder at innermost level
                    decoder_output_full=hidden_states  # full output before trimming
                )
            else:
                return hidden_states_trimmed, []

        # ENCODER
        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            padding_mask=padding_mask,               
            inference_params=None,
            **mixer_kwargs,
        )
        
        # save encoder output (respecting padding)
        encoder_output = hidden_states[..., :D].clone() if return_all_outputs else None

        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        # ROUTING AND COMPRESSION
        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=padding_mask,               
        )
        hidden_states, next_cu_seqlens, next_max_seqlen, next_padding_mask = self.chunk_layer(
            hidden_states, 
            bpred_output.boundary_mask,
            cu_seqlens=cu_seqlens, 
            mask=padding_mask
        )

        bpred_output.next_padding_mask = next_padding_mask

        # MAIN NETWORK (compressed representation)
        hidden_states, prev_boundary_predictions = self.main_network(
            hidden_states,
            cu_seqlens=next_cu_seqlens,
            max_seqlen=next_max_seqlen,
            padding_mask=next_padding_mask,                
            inference_params=None,
            return_all_outputs=return_all_outputs,
            **mixer_kwargs,
        )

        # DECOMPRESSION
        hidden_states = self.dechunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            cu_seqlens=next_cu_seqlens,
            mask=padding_mask                
        )

        # RESIDUAL CONNECTION
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        # DECODER
        hidden_states = self.decoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            padding_mask=padding_mask,
            inference_params=None,
            mode=self.mode,
            **mixer_kwargs,
        )

        # save full decoder output before trimming
        decoder_output_full = hidden_states.clone() if return_all_outputs else None
        
        # trim back to original hidden size
        hidden_states_trimmed = hidden_states[..., :D]
        
        if return_all_outputs:
            return LDarOutput(
                hidden_states=hidden_states_trimmed,
                boundary_predictions=[bpred_output, *prev_boundary_predictions],
                encoder_output=encoder_output,
                decoder_output_full=decoder_output_full
            )
        else:
            # backward compatibility: return tuple as before
            return hidden_states_trimmed, [bpred_output, *prev_boundary_predictions]