from dataclasses import dataclass, field
from typing import List, Union, Literal


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class LDarConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 7
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
    mode: Literal["mlm", "causal"] = "mlm"
    pad_token_id: int = 6
    mask_token_id: int = 5