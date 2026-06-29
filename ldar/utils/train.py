import torch

from ldar.modules.dc import RoutingModuleOutput
from ldar.models.mixer_seq import LDarForMaskedLM
from ldar.modules.utils import apply_optimization_params


def load_balancing_loss(
    router_output,               # RoutingModuleOutput
    N: float,
    padding_mask: torch.Tensor | None = None, 
) -> torch.Tensor:
    assert float(N) > 1.0, "N must be > 1"

    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask.bool()

    if padding_mask is None:
        valid = torch.ones_like(boundary_mask, dtype=torch.bool)
    else:
        assert padding_mask.shape == boundary_mask.shape, \
            f"padding_mask {padding_mask.shape} vs boundary_mask {boundary_mask.shape}"
        valid = padding_mask.bool()

    F = (boundary_mask & valid).float().sum() / valid.float().sum()
    G = (tokenized_prob * valid.float()).sum() / valid.float().sum()

    return ((1.0 - F) * (1.0 - G) + (N - 1.0) * F * G) * (N / (N - 1.0))


def group_params(
    model: LDarForMaskedLM,
) -> list[dict[str, list[torch.Tensor] | float]]:
    param_groups = []
    all_keys = set()

    for name, param in model.named_parameters():
        # No weight decay on biases, norms, or params the module explicitly marks
        # as decay-exempt (mamba_ssm sets _no_weight_decay=True on A_log, D, dt_bias).
        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or getattr(param, "_no_weight_decay", False)
        ):
            apply_optimization_params(param, weight_decay=0.0)

        all_keys.update(param._optim.keys())
    
    all_keys = list(all_keys)
    all_tuples = []
    param_groups = []

    for name, param in model.named_parameters():
        current_tuple = tuple(param._optim.get(key, None) for key in all_keys)
        if current_tuple not in all_tuples:
            all_tuples.append(current_tuple)
            param_groups.append({
                "params": [param],
                **param._optim,
            })
        else:
            idx = all_tuples.index(current_tuple)
            param_groups[idx]["params"].append(param)
    
    return param_groups

def group_params_downstream(
    model,
    base_weight_decay: float = 0.03,
    head_lr_mul: float = 10.0,
):
    def is_no_decay(name: str) -> bool:
        lname = name.lower()
        return lname.endswith(".bias") or ("norm" in lname or "layernorm" in lname or "ln" in lname)

    def is_head(name: str) -> bool:
        return name.startswith("base_model.classifier.") or name.startswith("classifier.")

    groups = {
        ("backbone_decay",):   {"params": [], "weight_decay": base_weight_decay, "lr_multiplier": 1.0},
        ("backbone_nodecay",): {"params": [], "weight_decay": 0.0,               "lr_multiplier": 1.0},
        ("head_decay",):       {"params": [], "weight_decay": base_weight_decay, "lr_multiplier": head_lr_mul},
        ("head_nodecay",):     {"params": [], "weight_decay": 0.0,               "lr_multiplier": head_lr_mul},
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_head(name):
            key = ("head_nodecay",) if is_no_decay(name) else ("head_decay",)
        else:
            key = ("backbone_nodecay",) if is_no_decay(name) else ("backbone_decay",)
        groups[key]["params"].append(p)

    return [g for g in groups.values() if g["params"]]