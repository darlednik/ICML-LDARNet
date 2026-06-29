
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class MlmCollatorConfig:
    pad_id: int
    mask_id: int
    vocab_tokens: int  # e.g. 5 for A,C,G,T,N
    n_idx: int         # index of 'N' in tokenizer
    p_mlm: float = 0.15

class MlmCollator:
    def __init__(self, cfg: MlmCollatorConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device

    def __call__(self, batch):
        ids = [b["input_ids"] for b in batch]
        max_len = max(t.size(0) for t in ids)
        B = len(ids)

        input_ids = torch.full((B, max_len), self.cfg.pad_id, dtype=torch.long)
        for i, t in enumerate(ids):
            input_ids[i, :t.size(0)] = t

        # float32 attention mask is safer downstream
        attention_mask = (input_ids != self.cfg.pad_id).to(torch.float32)
        labels = torch.full_like(input_ids, -100)

        # only A/C/G/T are mask candidates (exclude PAD and N)
        is_predictable = (input_ids < self.cfg.vocab_tokens) & (input_ids != self.cfg.n_idx) & (attention_mask > 0)

        probs = torch.rand_like(input_ids, dtype=torch.float32)
        masked = (probs < self.cfg.p_mlm) & is_predictable
        labels[masked] = input_ids[masked]

        rnd = torch.rand_like(input_ids, dtype=torch.float32)
        replace_mask = masked & (rnd < 0.8)
        input_ids[replace_mask] = self.cfg.mask_id

        rand_mask = masked & ~replace_mask & (rnd < 0.9)
        if rand_mask.any():
            # sample only from A/C/G/T (assume indices 0..3)
            rand_tokens = torch.randint(0, self.cfg.vocab_tokens - 1, (rand_mask.sum().item(),),
                                        device=input_ids.device)
            input_ids[rand_mask] = rand_tokens

        raw_texts = [b.get("chunk", None) for b in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_texts": raw_texts,   # <— NEW
        }