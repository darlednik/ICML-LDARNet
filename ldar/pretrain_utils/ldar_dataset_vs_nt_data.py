import os
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
from torch.utils.data import Dataset
import pyfaidx  # random access FASTA (uppercasing supported)
from datasets import load_from_disk


# ---- reverse complement (for optional RC augmentation) ----
_RC = str.maketrans("ACGTN", "TGCAN")
def revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1]


@dataclass
class LDarBedDatasetConfig:
    fasta_path: str
    bed_path: str
    seq_len: int
    tokenizer: "ByteTokenizer"
    rc_aug: bool = True                  # reverse-complement augmentation
    skip_N_windows: bool = True          # True: drop any window containing 'N'
    expect_exact_len: bool = True        # True: require end-start == seq_len
    # If BED contains longer intervals and you want to tile them inside the dataset,
    # set expect_exact_len=False and provide tile_stride below:
    tile_stride: Optional[int] = None    # used only when expect_exact_len=False
    split: Optional[str] = None          # "train" | "valid" | "test" (or any tag); None -> no filtering


class LDarBedDataset(Dataset):
    """
    Map-style dataset that enumerates BED windows and random-accesses the genome
    via pyfaidx. Each item yields dict(chunk=str, input_ids=LongTensor[seq_len]).
    """

    def __init__(self, cfg: LDarBedDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self._fa: Optional[pyfaidx.Fasta] = None  # created lazily per worker
        self._intervals: List[Tuple[str, int, int]] = self._load_intervals(cfg)

        if len(self._intervals) == 0:
            msg = "No usable windows found in BED (after filtering"
            if cfg.split is not None:
                msg += f" by split='{cfg.split}'"
            msg += ")."
            raise RuntimeError(msg)

    # ---- worker-safe lazy FASTA handle ----
    def _ensure_fa(self):
        if self._fa is None:
            self._fa = pyfaidx.Fasta(
                self.cfg.fasta_path,
                as_raw=True,
                sequence_always_upper=True,
                rebuild=False,           # assume .fai exists or let pyfaidx build it once
                read_ahead=1_000_000,    # speed hint
            )

    # ---- BED loader + (optional) tiling for long intervals ----
    def _load_intervals(self, cfg: LDarBedDatasetConfig) -> List[Tuple[str, int, int]]:
        ivals: List[Tuple[str, int, int]] = []
        want_split = (cfg.split is not None)

        with open(cfg.bed_path) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue

                chrom, s, e = parts[0], int(parts[1]), int(parts[2])

                # if split filtering requested, require a 4th column and match it
                if want_split:
                    if len(parts) < 4:
                        continue
                    row_split = parts[3]
                    if row_split != cfg.split:
                        continue

                length = e - s

                if cfg.expect_exact_len:
                    if length != cfg.seq_len:
                        continue
                    ivals.append((chrom, s, e))
                else:
                    # tile a longer interval into fixed windows
                    win = cfg.seq_len
                    stride = cfg.tile_stride or win
                    if length < win:
                        continue
                    for start in range(s, e - win + 1, stride):
                        ivals.append((chrom, start, start + win))
        return ivals

    def __len__(self) -> int:
        return len(self._intervals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        self._ensure_fa()
        chrom, s, e = self._intervals[idx]
        # pyfaidx slice is end-exclusive
        seq = str(self._fa[chrom][s:e])

        # Optionally drop windows with any N (recommended for clean MLM)
        if self.cfg.skip_N_windows and ("N" in seq):
            # simple re-sample behavior: pick a random replacement (keeps epoch length stable)
            ridx = random.randrange(len(self._intervals))
            return self.__getitem__(ridx)

        if self.cfg.rc_aug and (random.random() < 0.5):
            seq = revcomp(seq)

        ids = self.cfg.tokenizer.encode_str(seq)  # LongTensor[seq_len]
        return {"chunk": seq, "input_ids": ids}


@dataclass
class FastCombinedDatasetConfig:
    bed_dataset_config: LDarBedDatasetConfig
    multi_species_path: str


class FastCombinedGenomeDataset(Dataset):
    """
    Fast combined dataset with minimal __getitem__ work and simple indexing:
    all BED windows first, then all multi-species sequences.
    """

    def __init__(self, cfg: FastCombinedDatasetConfig):
        super().__init__()
        self.cfg = cfg

        # BED dataset
        self.bed_dataset = LDarBedDataset(cfg.bed_dataset_config)

        # multi-species dataset
        self.multi_species_dataset = load_from_disk(cfg.multi_species_path)["train"]

        # cache sizes
        self.bed_size = len(self.bed_dataset)
        self.multi_species_size = len(self.multi_species_dataset)
        self.total_size = self.bed_size + self.multi_species_size

        # cache tokenizer
        self.tokenizer = cfg.bed_dataset_config.tokenizer

        print(f"Fast combined dataset: {self.bed_size} BED + {self.multi_species_size} multi-species = {self.total_size} total")

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        if idx < self.bed_size:
            # BED sample (already in target format)
            return self.bed_dataset[idx]
        else:
            # multi-species sample (minimal processing)
            ms_idx = idx - self.bed_size
            item = self.multi_species_dataset[ms_idx]
            seq = item['sequence']  # string of length 4096
            if len(seq) > 4096:
                seq = seq[:4096]

            if self.cfg.bed_dataset_config.skip_N_windows and ("N" * 50 in seq):
                ridx = random.randrange(self.total_size)
                return self.__getitem__(ridx)

            if self.cfg.bed_dataset_config.rc_aug and (random.random() < 0.5):
                seq = revcomp(seq)

            ids = self.cfg.bed_dataset_config.tokenizer.encode_str(seq)
            return {"chunk": seq, "input_ids": ids}