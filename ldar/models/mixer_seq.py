from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .ldar import LDar, LDarState, LDarOutput
from .config_ldar import LDarConfig

from ldar.modules.dc import RoutingModuleOutput
from ldar.modules.utils import apply_optimization_params

@dataclass
class MaskedLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    last_hidden_state: torch.Tensor
    encoder_states: Optional[torch.Tensor] = None  # encoder output (optional)
    decoder_states_full: Optional[torch.Tensor] = None  # full decoder output (optional)


class LDarForMaskedLM(nn.Module):
    def __init__(
        self,
        config: LDarConfig,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)

        self.backbone = LDar(
            config=config,
            stage_idx=0,
            mode=config.mode,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
    
    def init_weights(self, initializer_range: float = 0.02) -> None:
        """
        Initializes the weights of the model.
        """
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=initializer_range)
        # embeddings are initialized differently from linears
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=initializer_range)
        self.backbone._init_weights(initializer_range)

    def apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        for param in self.embeddings.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        for param in self.lm_head.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[0])
        self.backbone._apply_lr_multiplier(lr_multiplier)


    def forward(
        self,
        input_ids,
        attention_mask=None,    
        position_ids=None,
        labels=None,                # accepted for HF-style callers; loss is computed externally
        return_all_outputs=False,
        **mixer_kwargs,
    ):
        # `labels` is intentionally not forwarded into the backbone. It is accepted
        # here only so external callers can pass it without it leaking into
        # **mixer_kwargs (which would be deep-copied at every layer).
        hidden_states = self.embeddings(input_ids)

        B, L, D = hidden_states.shape

        assert (
            position_ids is None
        ), "Position ids are not supported for LDar due to the subsampling hierarchical structure"

        backbone_output = self.backbone(
            hidden_states,
            cu_seqlens=None,
            max_seqlen=None,
            padding_mask=attention_mask,   
            inference_params=None,
            return_all_outputs=return_all_outputs,
            **mixer_kwargs,
        )

        if return_all_outputs:
            # backbone_output is LDarOutput
            hidden_states = backbone_output.hidden_states
            bpred_output = backbone_output.boundary_predictions
            encoder_states = backbone_output.encoder_output
            decoder_states_full = backbone_output.decoder_output_full
        else:
            # backward compatibility: backbone_output is a tuple
            hidden_states, bpred_output = backbone_output
            encoder_states = None
            decoder_states_full = None

        hidden_states = hidden_states.view(B, L, D)

        lm_logits = self.lm_head(hidden_states)

        return MaskedLMOutput(
            logits=lm_logits,
            bpred_output=bpred_output,
            last_hidden_state=hidden_states,
            encoder_states=encoder_states,
            decoder_states_full=decoder_states_full
        )