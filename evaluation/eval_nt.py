import argparse
import os

import time
import json
from typing import Dict, Tuple, Callable
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.distributed as dist
from scipy import stats

from ldar.utils.train import group_params_downstream

import shutil


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


PAD_ID = 6
ch2id = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def dna_to_ids(s: str):
    """Convert DNA sequence to token IDs."""
    s = s.upper()
    return torch.tensor([ch2id.get(c, 4) for c in s], dtype=torch.long)


class LDarTokenizer:
    """Simple tokenizer wrapper for LDar model."""
    
    def __init__(self):
        self.pad_token_id = PAD_ID
        self.eos_token_id = PAD_ID
        self.pad_token = "[PAD]"
        self.eos_token = "[PAD]"
        self.padding_side = "right"
        self.truncation_side = "right"
    
    def __call__(self, sequences, max_length=None, truncation=True, padding=False, return_tensors=None, **kwargs):
        """Tokenize sequences."""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        token_ids = []
        for seq in sequences:
            ids = dna_to_ids(seq)
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            token_ids.append(ids.tolist())
        
        attention_masks = [[1] * len(ids) for ids in token_ids]
        
        result = {
            "input_ids": token_ids,
            "attention_mask": attention_masks
        }
        
        if return_tensors == "pt":
            max_len = max(len(ids) for ids in token_ids)
            
            padded_ids = []
            padded_masks = []
            for ids, mask in zip(token_ids, attention_masks):
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [PAD_ID] * pad_len)
                padded_masks.append(mask + [0] * pad_len)
            
            result = {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_masks, dtype=torch.long)
            }
        
        return result

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config = {
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)


class LDarForSequenceClassification(nn.Module):
    """Wrapper to adapt LDar model for sequence classification with configurable pooling."""
    
    def __init__(self, ldar_model, num_labels, problem_type="single_label_classification", 
                 pooling_strategy="mean"):
        super().__init__()
        self.ldar_model = ldar_model
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.pooling_strategy = pooling_strategy
        
        self.hidden_size = ldar_model.config.d_model[0]  # Use LAST stage dimension
        
        # Adjust hidden size for concat pooling strategies
        if pooling_strategy in ["first_last", "first_mean_last"]:
            classifier_input_size = self.hidden_size * 2
        else:
            classifier_input_size = self.hidden_size
        
        # Attention pooling layer
        if pooling_strategy == "attention":
            self.attention_pool = nn.Linear(self.hidden_size, 1)
        
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        
        self.config = type('Config', (), {
            'num_labels': num_labels,
            'problem_type': problem_type,
        })()
    
    def pool_hidden_states(self, hidden_states, attention_mask):
        """Apply pooling strategy to hidden states."""
        
        if self.pooling_strategy == "mean":
            # Standard mean pooling over non-padded tokens
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            denom = mask.sum(1).clamp_min(1.0)
            return (hidden_states * mask).sum(1) / denom
        
        elif self.pooling_strategy == "max":
            # Max pooling over non-padded tokens
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            hidden_states_masked = hidden_states.masked_fill(~mask.bool(), -1e9)
            return hidden_states_masked.max(1)[0]
        
        elif self.pooling_strategy == "first":
            # First token (CLS-like)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == "last":
            # Last non-padded token
            lengths = attention_mask.sum(1) - 1
            lengths = lengths.clamp(min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, lengths]
        
        elif self.pooling_strategy == "first_last":
            # Concatenate first and last tokens
            first = hidden_states[:, 0, :]
            lengths = attention_mask.sum(1) - 1
            lengths = lengths.clamp(min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last = hidden_states[batch_indices, lengths]
            return torch.cat([first, last], dim=-1)
        
        elif self.pooling_strategy == "attention":
            # Learnable attention-based pooling
            attn_weights = self.attention_pool(hidden_states).squeeze(-1)  # [B, L]
            attn_weights = attn_weights.masked_fill(~attention_mask.bool(), -1e9)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=1).unsqueeze(-1)  # [B, L, 1]
            return (hidden_states * attn_weights).sum(1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.ldar_model(input_ids=input_ids, attention_mask=attention_mask, return_all_outputs=True)
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling
        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            if self.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def load_ldar_model(checkpoint_path: str):
    """Load LDar backbone from the config embedded in the checkpoint."""
    from ldar.utils.ckpt import load_ldar_from_ckpt

    model, _ = load_ldar_from_ckpt(
        checkpoint_path,
        for_training=True,
        verbose=is_main_process(),
    )
    return model


def setup_dataset(
    dataset_name: str,
    subset_name: str,
    tokenizer: LDarTokenizer,
    max_length: int = 16384,
    problem_type: str = "single_label_classification",
    seed: int = 42,
    num_folds: int = 10,
    fold_id: int = 0
) -> Tuple[Dataset, int]:
    """Setup dataset with proper train/val split for fold."""
    dist_print(f"Loading dataset {dataset_name}...")
    start_time = time.time()

    if subset_name is None:
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name)
        dataset = dataset.filter(lambda e: e["task"] == subset_name)
        
    dist_print(f"Dataset loaded in {time.time() - start_time:.2f}s")

    # Determine number of labels
    if problem_type == "single_label_classification":
        assert isinstance(dataset["train"]["label"][0], int), "Label must be int for single-label"
        max_label = max(dataset["train"]["label"])
        num_labels = max_label + 1
    elif problem_type == "multi_label_classification":
        assert isinstance(dataset["train"]["label"][0], list), "Label must be list for multi-label"
        num_labels = len(dataset["train"]["label"][0])
    elif problem_type == "regression":
        if isinstance(dataset["train"]["label"][0], list):
            num_labels = len(dataset["train"]["label"][0])
        elif isinstance(dataset["train"]["label"][0], float):
            num_labels = 1
        else:
            raise NotImplementedError("Regression with non-float labels not supported")
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Create validation split if not present
    if not any(x in dataset for x in ["validation", "valid", "val"]):
        assert (num_folds > 0 and 0 <= fold_id < num_folds), "Invalid fold settings"
        
        dist_print(f"Performing {num_folds}-fold cross-validation (using fold {fold_id})")
        
        kfold = KFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed,
        )
        
        train_data_list = list(dataset["train"])
        splits = list(kfold.split(train_data_list))
        train_idx, valid_idx = splits[fold_id]
        
        dataset["validation"] = dataset["train"].select(valid_idx)
        dataset["train"] = dataset["train"].select(train_idx)

    def _process_function(examples):
        # Find the correct field containing the sequence
        if "sequence" in examples:
            sequences = examples["sequence"]
        elif "seq" in examples:
            sequences = examples["seq"]
        elif "dna_sequence" in examples:
            sequences = examples["dna_sequence"]
        elif "dna_seq" in examples:
            sequences = examples["dna_seq"]
        elif "text" in examples:
            sequences = examples["text"]
        else:
            raise ValueError(
                "No sequence column found in dataset. Expected 'sequence', 'seq', 'dna_sequence', 'dna_seq', or 'text'."
            )
    
        tokenized = tokenizer(sequences, return_tensors='pt')
        tokenized["label"] = examples["label"]
        return tokenized

    # Apply tokenization to dataset
    dataset = dataset.map(
        _process_function,
        batched=True,
        remove_columns=[
            col
            for col in dataset["train"].column_names
            if col not in ["input_ids", "attention_mask", "label"]
        ],
        num_proc=16,
    )

    return dataset, num_labels


def get_compute_metrics_func(problem_type: str, num_labels: int) -> Callable:

    def _compute_metrics_single_label_classification(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Apply softmax to logits to get probabilities
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()

        accuracy = (predictions == labels).mean()
        f1 = f1_score(labels, predictions, average="weighted")
        mcc = matthews_corrcoef(labels, predictions)

        # Calculate AUROC
        if num_labels == 2:
            # Binary classification: use probabilities of the positive class
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            # Multi-class classification: use One-vs-Rest strategy
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")

        return {"accuracy": accuracy, "f1_score": f1, "mcc": mcc, "auroc": auroc}

    def _compute_metrics_multi_label_classification(eval_pred):
        predictions, labels = eval_pred

        return {
            "f1_max": f1_max(torch.tensor(predictions), torch.tensor(labels)),
            "auprc_micro": area_under_prc(
                torch.tensor(predictions).flatten(),
                torch.tensor(labels).long().flatten(),
            ),
        }

    def _compute_metrics_regression(eval_pred):
        logits, labels = eval_pred
        predictions = logits.squeeze()
        labels = labels.squeeze()

        # Reshape if needed
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, num_labels)
        if labels.ndim == 1:
            labels = labels.reshape(-1, num_labels)

        results = {}

        # Calculate metrics per dimension if multi-dimensional
        if num_labels > 1:
            label_names = [f"label_{i}" for i in range(num_labels)]

            for idx, label in enumerate(label_names):
                pred = predictions[:, idx]
                true = labels[:, idx]

                # MSE
                mse = np.mean((pred - true) ** 2)
                results[f"mse_{label}"] = mse

                # MAE
                mae = np.mean(np.abs(pred - true))
                results[f"mae_{label}"] = mae

                # R²
                y_mean = np.mean(true)
                ss_tot = np.sum((true - y_mean) ** 2)
                ss_res = np.sum((true - pred) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
                results[f"r2_{label}"] = r2

                # Pearson
                x_mean = np.mean(pred)
                numerator = np.sum((pred - x_mean) * (true - y_mean))
                denominator = np.sqrt(
                    np.sum((pred - x_mean) ** 2) * np.sum((true - y_mean) ** 2)
                )
                pearson = numerator / denominator if denominator != 0 else float("nan")
                results[f"pearson_{label}"] = pearson

        # Calculate overall metrics across all dimensions
        total_mse = np.mean((predictions - labels) ** 2)
        total_mae = np.mean(np.abs(predictions - labels))
        total_y_mean = np.mean(labels)
        total_ss_tot = np.sum((labels - total_y_mean) ** 2)
        total_ss_res = np.sum((labels - predictions) ** 2)
        total_r2 = 1 - (total_ss_res / total_ss_tot) if total_ss_tot != 0 else float("nan")
        
        # Overall Pearson
        total_x_mean = np.mean(predictions)
        total_numerator = np.sum((predictions - total_x_mean) * (labels - total_y_mean))
        total_denominator = np.sqrt(
            np.sum((predictions - total_x_mean) ** 2) * np.sum((labels - total_y_mean) ** 2)
        )
        total_pearson = total_numerator / total_denominator if total_denominator != 0 else float("nan")

        results["mse"] = total_mse
        results["mae"] = total_mae
        results["r2"] = total_r2
        results["pearson"] = total_pearson

        return results

    def area_under_prc(pred, target):
        order = pred.argsort(descending=True)
        target = target[order]
        precision = target.cumsum(0) / torch.arange(
            1, len(target) + 1, device=target.device
        )
        auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
        return auprc

    def f1_max(pred, target):
        order = pred.argsort(descending=True, dim=1)
        target = target.gather(1, order)
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)

        all_order = pred.flatten().argsort(descending=True)
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()
        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
        return all_f1.max()

    # Return the appropriate metrics function based on problem type
    if problem_type == "single_label_classification":
        return _compute_metrics_single_label_classification
    elif problem_type == "multi_label_classification":
        return _compute_metrics_multi_label_classification
    elif problem_type == "regression":
        return _compute_metrics_regression
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


class DataCollator:
    """Data collator for LDar model."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Extract sequences and labels
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["label"] for f in features]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        padded_ids = []
        padded_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [PAD_ID] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long if isinstance(labels[0], int) else torch.float),
        }


class MetricsLoggerCallback(TrainerCallback):
    """Callback to log metrics after each evaluation."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics_history = []
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is not None:
            metrics_with_step = {
                "epoch": state.epoch,
                "step": state.global_step,
                **metrics
            }
            self.metrics_history.append(metrics_with_step)
            
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        with open(self.log_file, 'w') as f:
            json.dump({
                "training_history": self.metrics_history,
                "total_epochs": state.epoch,
                "total_steps": state.global_step,
            }, f, indent=2)


def train_single_fold(
    checkpoint_path: str,
    dataset_name: str,
    subset_name: str,
    output_dir: str,
    learning_rate: float,
    effective_batch_size: int,
    physical_batch_size: int,
    max_length: int,
    problem_type: str,
    main_metrics: str,
    seed: int,
    num_folds: int,
    fold_id: int,
    weight_decay: float,
    epochs: int,
    pooling_strategy: str = "mean",
):
    """Train a single fold with given hyperparameters."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gradient_accumulation_steps = max(1, effective_batch_size // physical_batch_size)
    
    dist_print(f"Effective BS={effective_batch_size}, physical BS={physical_batch_size}, grad acc={gradient_accumulation_steps}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass
    
    tokenizer = LDarTokenizer()
    
    dataset, num_labels = setup_dataset(
        dataset_name,
        subset_name,
        tokenizer,
        max_length,
        problem_type,
        seed,
        num_folds,
        fold_id
    )
    
    dist_print(f"Fold {fold_id}: {len(dataset['train'])} train, {len(dataset['validation'])} val")
    
    ldar_model = load_ldar_model(checkpoint_path)
    model = LDarForSequenceClassification(
        ldar_model=ldar_model,
        num_labels=num_labels,
        problem_type=problem_type,
        pooling_strategy=pooling_strategy,
    )
    model.to(device)

    if subset_name != "" and subset_name is not None:
        fold_output_dir = os.path.join(output_dir, subset_name, f"lr{learning_rate}_bs{effective_batch_size}", f"fold{fold_id}")
    else:
        fold_output_dir = os.path.join(output_dir, dataset_name.split("/")[-1], f"lr{learning_rate}_bs{effective_batch_size}", f"fold{fold_id}")
        
    os.makedirs(fold_output_dir, exist_ok=True)
    
    metrics_log_file = os.path.join(fold_output_dir, "training_metrics.json")

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        per_device_train_batch_size=physical_batch_size,
        per_device_eval_batch_size=physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        logging_steps=10,
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        metric_for_best_model="mcc",
        greater_is_better=main_metrics not in ["mse", "mae"],
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={"factor": 0.5, "patience": 2},
        seed=seed,
        data_seed=seed,
        bf16=torch.cuda.get_device_capability()[0] >= 8 if torch.cuda.is_available() else False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=False,
        save_safetensors=False
    )

    base_lr = learning_rate
    pgroups_raw = group_params_downstream(
        model,     
        base_weight_decay=weight_decay,
        head_lr_mul=1.0
    )
    
    opt_param_groups = []
    for g in pgroups_raw:
        lr = base_lr * g.get("lr_multiplier", 1.0)
        wd = g.get("weight_decay", weight_decay)
        opt_param_groups.append({"params": g["params"], "lr": lr, "weight_decay": wd})

    try:
        opt = torch.optim.AdamW(opt_param_groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(opt_param_groups, betas=(0.9, 0.95), eps=1e-8, foreach=True)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5),
        MetricsLoggerCallback(metrics_log_file)
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollator(tokenizer=tokenizer),
        compute_metrics=get_compute_metrics_func(problem_type, num_labels),
        callbacks=callbacks,
        optimizers=(opt, None)
    )
    
    dist_print(f"Training fold {fold_id}...")
    train_result = trainer.train()
    
    train_info = {
        "fold_id": fold_id,
        "learning_rate": learning_rate,
        "effective_batch_size": effective_batch_size,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "total_epochs": train_result.metrics.get("epoch", epochs),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }
    
    with open(os.path.join(fold_output_dir, "train_info.json"), 'w') as f:
        json.dump(train_info, f, indent=2)
    
    test_results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    
    with open(os.path.join(fold_output_dir, "test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=2)

    if fold_id == 0:
        dist_print(f"Fold 0 checkpoint kept in {fold_output_dir}")
    else:
        dist_print(f"Removing checkpoints from {fold_output_dir}...")
        for item in os.listdir(fold_output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path_to_remove = os.path.join(fold_output_dir, item)
                if os.path.isdir(checkpoint_path_to_remove):
                    shutil.rmtree(checkpoint_path_to_remove)
                    dist_print(f"  Removed {item}")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LDARNet on Nucleotide Transformer downstream tasks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset_name", type=str, default="InstaDeepAI/nucleotide_transformer_downstream_tasks")
    parser.add_argument("--subset_name", type=str, default="", help="NT task subset (optional for non-NT datasets)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max_length", type=int, default=1_280)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--problem_type", type=str, default="single_label_classification", 
                       choices=["single_label_classification", "multi_label_classification", "regression"])
    parser.add_argument("--main_metrics", type=str, default="mcc", help="Main metric for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical_batch_size", type=int, default=128, help="Physical batch size")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, required=True, help="Fine-tuning learning rate")
    parser.add_argument("--effective_batch_size", type=int, required=True, help="Effective batch size (physical × grad acc)")
    
    args = parser.parse_args()
    
    subset_name = args.subset_name.strip() if args.subset_name else ""
    subset_for_filter = subset_name if subset_name else None
    lr = args.learning_rate
    eff_bs = args.effective_batch_size
    
    print("LDARNet NT evaluation")
    print(f"Task: {subset_name if subset_name else '<all>'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Physical batch size: {args.physical_batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Effective batch size: {eff_bs}")
    print(f"Gradient accumulation steps: {eff_bs // args.physical_batch_size}")
    print("=" * 60)
    
    dataset_tag = subset_name if subset_name else "all"
    dataset_output_dir = os.path.join(args.output_dir, dataset_tag)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    config_key = f"lr{lr}_bs{eff_bs}"
    config_dir = os.path.join(dataset_output_dir, config_key)
    os.makedirs(config_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"Running {args.num_folds}-fold cross-validation")
    print(f"   LR={lr}, effective BS={eff_bs}")
    print(f"   physical BS={args.physical_batch_size}")
    print(f"   grad acc={eff_bs // args.physical_batch_size}")
    print("=" * 60 + "\n")
    
    fold_results = []
    
    for fold_id in range(args.num_folds):
        print(f"\n{'='*60}")
        print(f"Processing fold {fold_id + 1}/{args.num_folds}")
        print(f"{'='*60}")
        
        result = train_single_fold(
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset_name,
            subset_name=subset_for_filter,
            output_dir=args.output_dir,
            learning_rate=lr,
            effective_batch_size=eff_bs,
            physical_batch_size=args.physical_batch_size,
            max_length=args.max_length,
            problem_type=args.problem_type,
            main_metrics=args.main_metrics,
            seed=args.seed,
            num_folds=args.num_folds,
            fold_id=fold_id,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            pooling_strategy="mean"
        )
        fold_results.append(result)
        
        metric_val = result[f"test_{args.main_metrics}"]
        print(f"Fold {fold_id}: {args.main_metrics}={metric_val:.4f}")
    
    if not fold_results:
        print("All folds failed.")
        return
    
    print("\n" + "=" * 60)
    print("Computing statistics across folds")
    print("=" * 60)
    
    all_metrics = {}
    
    for metric_name in fold_results[0].keys():
        if metric_name.startswith("test_"):
            values = [r[metric_name] for r in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            
            n = len(values)
            ci_95 = stats.t.ppf(0.975, n-1) * std_val / np.sqrt(n)
            
            all_metrics[metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95": ci_95,
                "values": values
            }
    
    results_summary = {
        "task": subset_name if subset_name else "all",
        "checkpoint": args.checkpoint,
        "config": {
            "learning_rate": lr,
            "effective_batch_size": eff_bs,
            "physical_batch_size": args.physical_batch_size,
            "gradient_accumulation_steps": eff_bs // args.physical_batch_size,
        },
        "num_folds": args.num_folds,
        "num_folds_completed": len(fold_results),
        "main_metric": args.main_metrics,
        "metrics": {}
    }
    
    for metric_name, metric_data in all_metrics.items():
        short_name = metric_name.replace("test_", "")
        results_summary["metrics"][short_name] = {
            "mean": float(metric_data["mean"]),
            "std": float(metric_data["std"]),
            "ci_95": float(metric_data["ci_95"]),
            "values": [float(v) for v in metric_data["values"]]
        }
    
    results_file = os.path.join(dataset_output_dir, "results_summary.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    summary_md = os.path.join(dataset_output_dir, "RESULTS.md")
    with open(summary_md, 'w') as f:
        f.write(f"# Results: {subset_name if subset_name else 'all'}\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Checkpoint:** `{args.checkpoint}`\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Learning Rate:** {lr}\n")
        f.write(f"- **Effective Batch Size:** {eff_bs}\n")
        f.write(f"- **Physical Batch Size:** {args.physical_batch_size}\n")
        f.write(f"- **Gradient Accumulation Steps:** {eff_bs // args.physical_batch_size}\n")
        f.write(f"- **Number of Folds:** {len(fold_results)}\n\n")
        
        f.write("## Metrics Summary\n\n")
        f.write("| Metric | Mean | Std | CI 95% |\n")
        f.write("|--------|------|-----|--------|\n")
        
        main_metric_name = f"test_{args.main_metrics}"
        if main_metric_name in all_metrics:
            metric_data = all_metrics[main_metric_name]
            short_name = args.main_metrics
            f.write(f"| **{short_name}** | **{metric_data['mean']:.4f}** | **{metric_data['std']:.4f}** | **±{metric_data['ci_95']:.4f}** |\n")
        
        for metric_name, metric_data in sorted(all_metrics.items()):
            if metric_name != main_metric_name:
                short_name = metric_name.replace("test_", "")
                f.write(f"| {short_name} | {metric_data['mean']:.4f} | {metric_data['std']:.4f} | ±{metric_data['ci_95']:.4f} |\n")
        
        f.write("\n## Individual Fold Results\n\n")
        f.write("| Fold | " + " | ".join([m.replace("test_", "") for m in all_metrics.keys()]) + " |\n")
        f.write("|------|" + "|".join(["------" for _ in all_metrics]) + "|\n")
        
        for fold_id in range(len(fold_results)):
            values = [f"{fold_results[fold_id][m]:.4f}" for m in all_metrics.keys()]
            f.write(f"| {fold_id} | " + " | ".join(values) + " |\n")
    
    print(f"Summary saved to: {summary_md}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nTask: {args.subset_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Configuration: LR={lr}, BS={eff_bs}")
    print(f"Folds completed: {len(fold_results)}/{args.num_folds}\n")
    
    main_metric_name = f"test_{args.main_metrics}"
    if main_metric_name in all_metrics:
        metric_data = all_metrics[main_metric_name]
        print(f"{args.main_metrics.upper()}: {metric_data['mean']:.4f} ± {metric_data['ci_95']:.4f}")
    
    print("\nAll metrics:")
    for metric_name, metric_data in sorted(all_metrics.items()):
        short_name = metric_name.replace("test_", "")
        print(f"   {short_name}: {metric_data['mean']:.4f} ± {metric_data['ci_95']:.4f}")
    
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()