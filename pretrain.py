#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, math, time, contextlib, random, argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---- LDAR-specific imports (must be available in your env) -----------------
from ldar.models.mixer_seq import LDarForMaskedLM
from ldar.models.config_ldar import LDarConfig, SSMConfig, AttnConfig
from ldar.utils.train import load_balancing_loss, group_params
from ldar.utils.tokenizers import ByteTokenizer
from ldar.pretrain_utils.ldar_dataset_vs_nt_data import (
    LDarBedDatasetConfig,
    LDarBedDataset,
    FastCombinedDatasetConfig,
    FastCombinedGenomeDataset,
)
from ldar.pretrain_utils.mlm_collator import MlmCollatorConfig, MlmCollator

# ============================ Utilities =====================================

@torch.no_grad()
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m

def seed_worker(worker_id: int):
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)

def init_distributed(backend: str = "nccl"):
    """Auto-init DDP if launched with torchrun. Returns (is_dist, rank, world_size, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, world_size, local_rank, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device

def is_main_process() -> bool:
    return (not dist.is_available() or not dist.is_initialized()) or dist.get_rank() == 0

def ddp_all_reduce_tensor(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def save_ckpt(model: nn.Module, optimizer: torch.optim.Optimizer, path: str,
              config=None, step=None, args=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = unwrap_model(model)
    state_dict_cpu = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    opt_state = optimizer.state_dict()
    payload = {"model": state_dict_cpu, "opt": opt_state}
    # Storing the config (and step/args) makes checkpoints self-describing
    if config is not None:
        from dataclasses import asdict
        try:
            payload["config"] = asdict(config)
        except TypeError:
            payload["config"] = config
    if step is not None:
        payload["step"] = step
    if args is not None:
        payload["args"] = vars(args) if hasattr(args, "__dict__") else args
    torch.save(payload, path)
    if is_main_process():
        print(f"✔ saved checkpoint to {path}")

def visualize_boundaries(ro, name, step, seq_idx=0, max_len=200):
    os.makedirs("images", exist_ok=True)

    probs = ro.boundary_prob[seq_idx, :max_len]
    if probs.ndim == 2:
        probs = probs[:, -1]  # keep your last-channel choice

    # --- Cast to float32 before numpy (matplotlib can't plot bfloat16) ---
    probs_np = probs.detach().to(torch.float32).cpu().numpy()
    mask_np  = ro.boundary_mask[seq_idx, :max_len].float().cpu().numpy()

    x = np.arange(len(probs_np))

    plt.figure(figsize=(12, 3))
    plt.plot(x, probs_np, label="boundary_prob")
    plt.scatter(x, mask_np, label="boundary_mask", marker="x")
    plt.ylim(-0.1, 1.1)
    plt.title(f"Routing boundaries for seq {seq_idx}")
    plt.xlabel("Position"); plt.ylabel("Keep prob / mask"); plt.legend()
    out_path = f"images/boundaries_{name}_{step}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_wsd_scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    decay_ratio: float  = 0.20,
    min_lr_ratio: float = 0.05,
):
    warmup = max(1, int(total_steps * warmup_ratio))
    decay  = max(1, int(total_steps * decay_ratio))
    stable = max(0, total_steps - warmup - decay)
    decay_start = warmup + stable

    # normalize inverse-sqrt so: t=1 -> 1.0 ; t=decay -> min_lr_ratio
    denom = (1.0 - 1.0 / math.sqrt(decay)) if decay > 1 else 1.0
    scale = (1.0 - min_lr_ratio) / denom if denom > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / float(warmup)  # linear 0→1
        if step < decay_start:
            return 1.0                          # plateau
        t = (step - decay_start) + 1            # 1..decay
        raw = 1.0 / math.sqrt(max(1.0, float(t)))
        return 1.0 - scale * (1.0 - raw)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def ldar_110m_config() -> LDarConfig:
    """
    Single-stage hierarchical LDARNet: encoder [m3t1] / backbone [M10] / decoder [m4].
    d_model widens 512 -> 768 in the compressed backbone (H-Net convention).
    """
    return LDarConfig(
        arch_layout=["m3t1", ["M10"], "m4"],
        d_model=[512, 768],
        d_intermediate=[0, 2560],         # backbone SwiGLU hidden
        vocab_size=7,                     # A,C,G,T,N + [PAD] + [MASK]
        ssm_cfg=SSMConfig(
            chunk_size=256,
            d_conv=4,
            d_state=128,
            expand=2,
        ),
        attn_cfg=AttnConfig(
            num_heads=[16, 16],
            rotary_emb_dim=[32, 48],
            window_size=[1023, -1],       # local window in the encoder; global elsewhere
        ),
        tie_embeddings=False,
    )

def ldar_2m_config() -> LDarConfig:
    """Micro-LDARNet (~2.01M params) for the NT-tasks weight class.
    Encoder 'm3t1' is identical to the 110m model; decoder m4->m3, backbone M10->M6.
    Width 512/768 -> 64/128 (ratio 2.0). λ_outer auto-derives to sqrt(4)*128/64 = 4.0."""
    return LDarConfig(
        arch_layout=["m3t1", ["M6"], "m3"],
        d_model=[64, 128],
        d_intermediate=[0, 384],          # backbone SwiGLU hidden (di/d_back = 3.0, as in big model)
        vocab_size=7,
        ssm_cfg=SSMConfig(chunk_size=256, d_conv=4, d_state=128, expand=2),  # unchanged
        attn_cfg=AttnConfig(
            num_heads=[2, 4],             # head_dim = 64/2=32, 128/4=32 (mirrors big-model head_dim=32)
            rotary_emb_dim=[32, 32],      # = head_dim
            window_size=[1023, -1],       # positional, NOT width-dependent -> unchanged
        ),
        tie_embeddings=False,
    )

def compute_outer_backbone_lrs(d_outer: int, d_back: int, N_down: float) -> List[float]:
    lam_outer = math.sqrt(float(N_down)) * (float(d_back) / float(d_outer))
    return [lam_outer, 1.0]  # [outer, backbone]

# ============================ Training / Validation =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    name: str,
    alpha_ratio: float,
    downsample_ratios: List[float],
    accum_steps: int = 1,
    max_grad_norm: Optional[float] = 1.0,
    log_every: int = 100,
    max_steps: Optional[int] = None,
    log_file: Optional[str] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_step_per_batch: bool = False,  # we step per optimizer update by default
):
    main = is_main_process()
    # Whether we are running under DDP (used to gate no_sync()).
    is_dist = dist.is_available() and dist.is_initialized()

    if log_file is not None and main:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss", "mlm", "ratio"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else (torch.amp.GradScaler("cuda") if use_cuda else None)

    def autocast_ctx():
        if use_cuda:
            return torch.amp.autocast("cuda", dtype=amp_dtype)
        return contextlib.nullcontext()

    model.train()
    step = 0
    optim_steps = 0

    running_loss  = torch.zeros((), device=device, dtype=torch.float32)
    running_mlm   = torch.zeros((), device=device, dtype=torch.float32)
    running_ratio = torch.zeros((), device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)

    for j, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True).bool()
        labels         = batch["labels"].to(device, non_blocking=True)

        do_step = ((j + 1) % accum_steps == 0)

        # Under DDP, only synchronize gradients on the micro-batch that actually
        # triggers an optimizer step. This avoids `accum_steps` redundant
        # all-reduces per optimizer update.
        sync_ctx = (
            model.no_sync()
            if (is_dist and not do_step and hasattr(model, "no_sync"))
            else contextlib.nullcontext()
        )

        with autocast_ctx():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits

            active = (labels != -100)
            if active.any():
                mlm_loss = F.cross_entropy(logits[active].contiguous(), labels[active], reduction="mean")
            else:
                mlm_loss = logits.new_zeros(())

            ratio_losses = []
            prev_mask = attention_mask
            for i, ro in enumerate(out.bpred_output):
                N = downsample_ratios[min(i, len(downsample_ratios) - 1)]
                rloss = load_balancing_loss(ro, N=N, padding_mask=prev_mask)
                ratio_losses.append(rloss)

                # Visualize occasionally (rank 0 only)
                if i == 0 and main and (step % 500 == 0):
                    visualize_boundaries(ro, name=name, step=step, seq_idx=0, max_len=200)

                stage_mask = getattr(ro, "next_padding_mask", None)
                if stage_mask is not None:
                    prev_mask = stage_mask

            ratio_loss = torch.stack(ratio_losses).sum() if ratio_losses else logits.new_zeros(())
            loss = mlm_loss + alpha_ratio * ratio_loss

        # Gradient accumulation
        loss_to_backprop = loss / max(1, accum_steps)

        if scaler is not None and amp_dtype == torch.float16:
            with sync_ctx:
                scaler.scale(loss_to_backprop).backward()
            if do_step:
                scaler.unscale_(optimizer)
                if (max_grad_norm is not None) and (max_grad_norm > 0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optim_steps += 1
                if scheduler is not None and not scheduler_step_per_batch:
                    scheduler.step()
        else:
            with sync_ctx:
                loss_to_backprop.backward()
            if do_step:
                if (max_grad_norm is not None) and (max_grad_norm > 0):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optim_steps += 1
                if scheduler is not None and not scheduler_step_per_batch:
                    scheduler.step()

        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        # Running stats
        running_loss  += loss.detach().to(torch.float32)
        running_mlm   += mlm_loss.detach().to(torch.float32)
        running_ratio += ratio_loss.detach().to(torch.float32)
        step += 1

        if (step % log_every == 0) and main:
            denom = float(log_every)
            avg_loss  = (running_loss  / denom).item()
            avg_mlm   = (running_mlm   / denom).item()
            avg_ratio = (running_ratio / denom).item()
            print(f"[step {step}] loss={avg_loss:.4f}  mlm={avg_mlm:.4f}  ratio={avg_ratio:.4f}")
            if log_file is not None:
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, avg_loss, avg_mlm, avg_ratio])
            running_loss.zero_(); running_mlm.zero_(); running_ratio.zero_()

        if (max_steps is not None) and (step >= max_steps):
            break

        # Periodic mid-epoch checkpoint (rank 0 only)
        if main and j > 0 and (j % 10_000 == 0):
            save_ckpt(model, optimizer, "models_ckpts/last_checkpoint_one_stage.pt")

    return {"train_steps": float(step), "optim_steps": float(optim_steps)}

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    alpha_ratio: float,
    downsample_ratios: List[float],
    log_file: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    main = is_main_process()

    loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    mlm_sum   = torch.zeros((), device=device, dtype=torch.float64)
    ratio_sum = torch.zeros((), device=device, dtype=torch.float64)
    steps     = 0

    for j, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True).bool()
        labels         = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        active = (labels != -100)
        if active.any():
            mlm_loss = F.cross_entropy(logits[active].contiguous(), labels[active], reduction="mean")
        else:
            mlm_loss = logits.new_zeros((), dtype=torch.float32, device=logits.device)

        ratio_losses = []
        prev_mask = attention_mask
        for i, ro in enumerate(out.bpred_output):
            N = downsample_ratios[min(i, len(downsample_ratios) - 1)]
            rloss = load_balancing_loss(ro, N=N, padding_mask=prev_mask)
            ratio_losses.append(rloss)
            stage_mask = getattr(ro, "next_padding_mask", None)
            if stage_mask is not None:
                prev_mask = stage_mask

        ratio_loss = torch.stack(ratio_losses).sum() if ratio_losses else logits.new_zeros(())
        total_loss = mlm_loss + alpha_ratio * ratio_loss

        loss_sum  += total_loss.to(torch.float64)
        mlm_sum   += mlm_loss.to(torch.float64)
        ratio_sum += ratio_loss.to(torch.float64)
        steps     += 1

        if (max_steps is not None) and (steps >= max_steps):
            break

    # Distributed aggregation
    pack = torch.tensor([loss_sum.item(), mlm_sum.item(), ratio_sum.item(), float(steps)],
                        device=device, dtype=torch.float64)
    pack = ddp_all_reduce_tensor(pack)
    total_loss_sum, total_mlm_sum, total_ratio_sum, total_steps = pack.tolist()
    denom = max(1.0, total_steps)

    metrics = {
        "val_loss":  total_loss_sum  / denom,
        "val_mlm":   total_mlm_sum   / denom,
        "val_ratio": total_ratio_sum / denom,
        "val_steps": total_steps,
    }

    if log_file is not None and main:
        new_file = not os.path.exists(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["val_loss", "val_mlm", "val_ratio", "val_steps"])
            writer.writerow([metrics["val_loss"], metrics["val_mlm"],
                             metrics["val_ratio"], int(metrics["val_steps"])])
    return metrics

# ============================ Main ==========================================

def main():
    print("Script ran correctly.")
    parser = argparse.ArgumentParser(description="LDar pretraining")
    parser.add_argument("--num_bytes_per_token", type=str, default="4",
                        help="Downsample ratios (comma-separated) or single value, e.g., '4' or '4,4'")
    parser.add_argument("--interval_len", type=int, default=17)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--waiting_hours", type=int, default=0)
    parser.add_argument("--accum_steps", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha_ratio", type=float, default=0.03)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--model_size", type=str, default="110m", choices=["110m", "2m"])
    args_cli = parser.parse_args()

    # Optional wait (cluster scheduling)
    if args_cli.waiting_hours > 0:
        time.sleep(3600 * args_cli.waiting_hours)

    # Seeds / perf knobs
    seed = 1337
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    with contextlib.suppress(Exception):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    # DDP init
    is_dist, rank, world_size, local_rank, device = init_distributed()
    main_rank = is_main_process()
    print(f"[DDP] dist={is_dist} rank={rank}/{world_size} local_rank={local_rank} device={device}")

    # Tokenizer / datasets
    tok = ByteTokenizer()
    n_idx = tok.vocab["N"]

    train_bed_config = LDarBedDatasetConfig(
        fasta_path="ldar_data/ldar_data.fa",
        bed_path="ldar_data/human-sequences.bed",
        seq_len=args_cli.seq_len,
        tokenizer=tok,
        split="train",
        expect_exact_len=False,
    )
    val_bed_config = LDarBedDatasetConfig(
        fasta_path="ldar_data/ldar_data.fa",
        bed_path="ldar_data/human-sequences.bed",
        seq_len=args_cli.seq_len,
        tokenizer=tok,
        split="valid",
        expect_exact_len=False,
    )

    train_ds = FastCombinedGenomeDataset(FastCombinedDatasetConfig(
        bed_dataset_config=train_bed_config,
        multi_species_path="ldar_data/multi_species_genomes_dataset"
    ))
    val_ds = LDarBedDataset(val_bed_config)

    collate = MlmCollator(MlmCollatorConfig(
        pad_id=tok.pad_idx, mask_id=tok.mask_idx, vocab_tokens=5, n_idx=n_idx, p_mlm=0.15
    ))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None

    # Reproducible shuffling for the single-process path.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + rank)

    num_workers = max(2, (os.cpu_count() or 8) // 8)
    train_loader = DataLoader(
        train_ds, batch_size=args_cli.batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True,
        collate_fn=collate, sampler=train_sampler,
        worker_init_fn=seed_worker, generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True,
        collate_fn=collate, sampler=val_sampler,
        worker_init_fn=seed_worker,
    )

    if main_rank:
        try:
            print(f"train_len={len(train_ds)}, val_len={len(val_ds)}")
        except Exception:
            pass

    # Model
    cfg = ldar_2m_config() if args_cli.model_size == "2m" else ldar_110m_config()
    model = LDarForMaskedLM(cfg)
    model.init_weights(initializer_range=0.02)
    model = model.to(device)

    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    base_model = unwrap_model(model)
    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if main_rank:
        print(f"Model parameters: {n_params/1e6:.3f}M")

    # Stage-wise LR multipliers (one-stage + backbone)
    Ns = [float(x) for x in str(args_cli.num_bytes_per_token).split(",")]
    if len(Ns) == 0:
        Ns = [1.0]
    N_down = Ns[0]
    d_outer, d_back = int(cfg.d_model[0]), int(cfg.d_model[-1])
    lambdas = compute_outer_backbone_lrs(d_outer, d_back, N_down)  # [lam_outer, 1.0]
    base_model.apply_lr_multiplier(lambdas)
    if main_rank:
        print(f"[LR modulation] N={N_down} d_outer={d_outer} d_back={d_back} -> λ={lambdas}")

    # Optimizer with param groups from group_params()
    pgroups_raw = group_params(base_model)
    opt_param_groups = []
    for g in pgroups_raw:
        lr = args_cli.lr * g.get("lr_multiplier", 1.0)
        wd = g.get("weight_decay", 0.01)
        opt_param_groups.append({"params": g["params"], "lr": lr, "weight_decay": wd})

    try:
        opt = torch.optim.AdamW(opt_param_groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(opt_param_groups, betas=(0.9, 0.95), eps=1e-8, foreach=True)

    # Scheduler — step per optimizer update
    batches_per_epoch = len(train_loader)
    optim_steps_per_epoch = math.ceil(batches_per_epoch / max(1, args_cli.accum_steps))
    total_steps = args_cli.epochs * optim_steps_per_epoch
    scheduler = build_wsd_scheduler(
        opt, total_steps=total_steps, warmup_ratio=0.10, decay_ratio=0.20, min_lr_ratio=0.05
    )
    scheduler_step_per_batch = False  # IMPORTANT: per optimizer step

    # Save dir and log files
    save_dir = f"ckpts_{args_cli.model_size}_ratio_{args_cli.num_bytes_per_token}_steps_{args_cli.max_steps}_interval_{args_cli.interval_len}"
    train_log = f"logs/{save_dir}_rank{rank}.csv"
    val_log   = f"logs/{save_dir}_val_rank{rank}.csv"

    # Train
    for ep in range(1, args_cli.epochs + 1):
        if is_dist and (train_sampler is not None):
            train_sampler.set_epoch(ep)

        if main_rank:
            print(f"\n===== Epoch {ep}/{args_cli.epochs} =====")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=opt,
            device=device,
            name=save_dir,
            alpha_ratio=args_cli.alpha_ratio,
            downsample_ratios=Ns,
            accum_steps=args_cli.accum_steps,
            max_grad_norm=(None if args_cli.max_grad_norm <= 0 else args_cli.max_grad_norm),
            log_every=50,
            max_steps=args_cli.max_steps,
            log_file=train_log,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )

        # Keep steps aligned before validation
        if is_dist:
            dist.barrier()

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            alpha_ratio=args_cli.alpha_ratio,
            downsample_ratios=Ns,
            log_file=val_log,
        )

        if main_rank:
            print(f"[val] loss={val_metrics['val_loss']:.4f}  "
                  f"mlm={val_metrics['val_mlm']:.4f}  "
                  f"ratio={val_metrics['val_ratio']:.4f}  "
                  f"steps={int(val_metrics['val_steps'])}")

            ckpt_path = os.path.join("models_ckpts", save_dir, f"ep{ep:02d}.pt")
            save_ckpt(model, opt, ckpt_path, config=cfg, step=ep, args=args_cli)

    if is_dist:
        dist.destroy_process_group()
    if main_rank:
        print("Done.")

if __name__ == "__main__":
    main()