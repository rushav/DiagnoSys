"""
train.py — DiagnoSys ML Classification
QLoRA fine-tuning of Llama 3.1 8B for multi-label domain classification.
Targets RTX 3080 (10 GB VRAM) via 4-bit BitsAndBytes quantisation + PEFT LoRA.
"""

import os
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from dataset import DatasetBuilder, DOMAIN_LABELS, NUM_LABELS

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "meta-llama/Meta-Llama-3.1-8B"
CHECKPOINT_DIR   = Path("ml/classification/checkpoints")
DEFAULT_DB_URL   = os.environ.get("DATABASE_URL", "postgresql://user:pass@localhost/diagnosys")

# ── QLoRA / BnB config ────────────────────────────────────────────────────────
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # nested quantisation saves ~0.4 GB
)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                                # rank — balance quality vs VRAM
    lora_alpha=32,                       # scaling = alpha / r = 2
    lora_dropout=0.05,
    bias="none",
    # Target the attention projection matrices in every transformer block
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(model_name: str = MODEL_NAME) -> tuple:
    """
    Returns (model, tokenizer) with 4-bit quantisation + LoRA adapters.
    VRAM budget on RTX 3080 (10 GB):
      - 4-bit weights  : ~4.5 GB
      - LoRA adapters  : ~0.1 GB
      - Activations    : ~2.0 GB  (batch=4, seq=512)
      - Optimizer state: ~0.8 GB  (paged AdamW)
      Total            : ~7.4 GB  ✓ fits in 10 GB
    """
    logger.info("Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model in 4-bit …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        quantization_config=BNB_CONFIG,
        device_map="auto",               # auto-places layers across GPU/CPU
        torch_dtype=torch.bfloat16,
        problem_type="multi_label_classification",
    )
    # Point pad token id so the model doesn't warn
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    logger.info("Injecting LoRA adapters …")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Training loop ─────────────────────────────────────────────────────────────
def train(
    db_url: str = DEFAULT_DB_URL,
    jsonl_path: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 4,          # RTX 3080 sweet-spot with grad accumulation
    grad_accum_steps: int = 8,    # effective batch = 32
    lr: float = 2e-4,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    pos_weight_scale: float = 5.0,  # upweight positives for imbalanced labels
    max_grad_norm: float = 1.0,
    save_every_n_epochs: int = 1,
    wandb_project: str = "diagnosys-classification",
    run_name: str = "llama31-8b-qlora",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.init(
        project=wandb_project,
        name=run_name,
        config=dict(
            model=MODEL_NAME, epochs=epochs, batch_size=batch_size,
            grad_accum=grad_accum_steps, lr=lr, lora_r=LORA_CONFIG.r,
            lora_alpha=LORA_CONFIG.lora_alpha,
        ),
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    builder = DatasetBuilder(db_url=db_url, model_name=MODEL_NAME)
    if jsonl_path:
        loaders = builder.build_from_jsonl(jsonl_path, batch_size=batch_size, num_workers=2)
    else:
        loaders = builder.build(batch_size=batch_size, num_workers=2)

    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model, _ = build_model()

    # ── Loss: BCEWithLogitsLoss with positive class weighting ─────────────────
    pos_weight = torch.full((NUM_LABELS,), pos_weight_scale, device="cuda")
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiser: paged AdamW (BnB) keeps optimizer states in CPU RAM ────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        logger.info("Using PagedAdamW8bit optimiser")
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        logger.warning("bitsandbytes PagedAdamW unavailable — falling back to AdamW")

    total_steps   = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = GradScaler()

    global_step = 0
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels         = batch["labels"].cuda()

            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits
                loss    = criterion(logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = running_loss / (step + 1)
                    lr_now   = scheduler.get_last_lr()[0]
                    logger.info(
                        "Epoch %d | step %d | loss %.4f | lr %.2e",
                        epoch, global_step, avg_loss, lr_now,
                    )
                    wandb.log({"train/loss": avg_loss, "train/lr": lr_now}, step=global_step)

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = evaluate_loader(model, val_loader, criterion)
        logger.info(
            "Epoch %d | val_loss %.4f | val_f1 %.4f | val_precision %.4f | val_recall %.4f",
            epoch, val_metrics["loss"], val_metrics["f1"],
            val_metrics["precision"], val_metrics["recall"],
        )
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % save_every_n_epochs == 0:
            ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:02d}"
            model.save_pretrained(str(ckpt_path))
            logger.info("Checkpoint saved → %s", ckpt_path)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_path   = CHECKPOINT_DIR / "best"
            model.save_pretrained(str(best_path))
            logger.info("New best model (F1=%.4f) saved → %s", best_val_f1, best_path)
            wandb.run.summary["best_val_f1"] = best_val_f1

    wandb.finish()
    logger.info("Training complete. Best val F1: %.4f", best_val_f1)
    return best_val_f1


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_loader(model, loader, criterion, threshold: float = 0.5) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels         = batch["labels"].cuda()

            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits
                loss    = criterion(logits, labels)

            total_loss  += loss.item()
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.float().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds      = (all_logits > threshold).astype(int)

    return {
        "loss":      total_loss / len(loader),
        "f1":        f1_score(all_labels, preds, average="micro", zero_division=0),
        "precision": precision_score(all_labels, preds, average="micro", zero_division=0),
        "recall":    recall_score(all_labels, preds, average="micro", zero_division=0),
    }


# ── Optional type hint fix for older Python ───────────────────────────────────
from typing import Optional  # noqa: E402 (placed after imports for clarity)


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train DiagnoSys classifier")
    parser.add_argument("--db-url",      default=DEFAULT_DB_URL)
    parser.add_argument("--jsonl",       default=None, help="Path to JSONL dataset (offline mode)")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--run-name",    default="llama31-8b-qlora")
    parser.add_argument("--wandb-project", default="diagnosys-classification")
    args = parser.parse_args()

    train(
        db_url=args.db_url,
        jsonl_path=args.jsonl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
    )
