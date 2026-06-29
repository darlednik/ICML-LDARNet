import numpy as np
import torch
from typing import List, Dict, Any

class ByteTokenizer:
    def __init__(self):
        self.vocab = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4,    
            "[MASK]": 5,
            "[PAD]": 6,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.mask_idx = self.vocab["[MASK]"]
        self.pad_idx  = self.vocab["[PAD]"]
        self.dtype = np.uint8  

    def encode_str(self, seq: str) -> torch.Tensor:
        ids = [self.vocab.get(ch, self.vocab["N"]) for ch in seq]
        return torch.tensor(ids, dtype=torch.long)
    def encode(
        self,
        seqs: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, np.ndarray]]:
        outputs: List[Dict[str, np.ndarray]] = []
        for seq in seqs:
            ids = [self.vocab.get(ch, self.vocab["N"]) for ch in seq]
            arr = np.array(ids, dtype=self.dtype)
            outputs.append({"input_ids": arr})
        return outputs

    def decode(self, tokens: np.ndarray | List[int], **kwargs: Any) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        filtered = [int(t) for t in tokens if t != self.mask_idx and t != self.pad_idx]
        return "".join(self.inv_vocab.get(t, "N") for t in filtered)
