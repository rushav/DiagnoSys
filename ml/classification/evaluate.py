"""
evaluate.py — DiagnoSys ML Classification
Evaluation metrics: precision, recall, F1 per domain, confusion matrix.
Target: 80%+ precision on held-out validation set.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    multilabel_confusion_matrix, classification_report,
    average_precision_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from dataset import DiagnoSysDataset, DatasetBuilder

DOMAIN_LABELS = [
    "databases", "backend", "frontend", "ml", "devops", "security",
    "mobile", "systems", "networking", "web", "cloud", "testing",
    "performance", "architecture", "data_engineering", "nlp",
    "computer_vision", "rl", "robotics", "other",
]

CHECKPOINT_DIR = Path("ml/classification/checkpoints")
EVAL_OUTPUT_DIR = Path("ml/classification/eval_results")


def load_model_for_eval(checkpoint_path: str, base_model: str = "meta-llama/Meta-Llama-3.1-8B"):
    """Load fine-tuned model for evaluation."""
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(DOMAIN_LABELS),
        quantization_config=bnb_config,
        device_map="auto",
        problem_type="multi_label_classification",
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model, tokenizer


def predict_batch(
    model, tokenizer, texts: List[str], batch_size: int = 16, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a list of texts, return (preds, probs)."""
    device = next(model.parameters()).device
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    probs = np.vstack(all_probs)
    preds = (probs >= threshold).astype(int)
    return preds, probs


def compute_per_domain_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1, AP per domain label."""
    metrics = {}
    for i, label in enumerate(DOMAIN_LABELS):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        ap = average_precision_score(y_true[:, i], y_prob[:, i]) if y_true[:, i].sum() > 0 else 0.0
        metrics[label] = {"precision": p, "recall": r, "f1": f, "ap": ap}
    return metrics


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """Plot per-label confusion matrices (binary)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cms = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    for i, (cm, label) in enumerate(zip(cms, DOMAIN_LABELS)):
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[i], cmap="Blues",
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        axes[i].set_title(label, fontsize=9)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=100)
    plt.close(fig)
    print(f"Confusion matrices saved to {output_dir}/confusion_matrices.png")


def plot_per_domain_metrics(metrics: Dict[str, Dict[str, float]], output_dir: Path):
    """Bar chart: precision / recall / F1 per domain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(metrics.keys())
    precisions = [metrics[l]["precision"] for l in labels]
    recalls = [metrics[l]["recall"] for l in labels]
    f1s = [metrics[l]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.80, color="red", linestyle="--", label="80% target")
    ax.legend()
    ax.set_title("Per-Domain Precision / Recall / F1")
    plt.tight_layout()
    fig.savefig(output_dir / "per_domain_metrics.png", dpi=100)
    plt.close(fig)
    print(f"Per-domain metrics chart saved to {output_dir}/per_domain_metrics.png")


def evaluate(
    checkpoint_path: str,
    val_data_path: Optional[str] = None,
    threshold: float = 0.5,
    batch_size: int = 16,
):
    """Full evaluation pipeline."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path} ...")
    model, tokenizer = load_model_for_eval(checkpoint_path)

    # Load validation data
    if val_data_path:
        import pandas as pd
        df = pd.read_json(val_data_path, lines=True)
        texts = df["text"].tolist()
        labels_raw = df["labels"].tolist()
        y_true = np.zeros((len(labels_raw), len(DOMAIN_LABELS)), dtype=int)
        for i, labs in enumerate(labels_raw):
            for lab in labs:
                if lab in DOMAIN_LABELS:
                    y_true[i, DOMAIN_LABELS.index(lab)] = 1
    else:
        # Synthetic validation data for testing
        print("No val_data_path provided; using synthetic data for demo.")
        texts = [
            "How do I optimize a PostgreSQL query with multiple joins?",
            "React useState hook causing infinite re-renders",
            "Docker container OOM killed in production",
        ]
        y_true = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=int)

    print(f"Evaluating on {len(texts)} examples ...")
    y_pred, y_prob = predict_batch(model, tokenizer, texts, batch_size=batch_size, threshold=threshold)

    # Compute metrics
    per_domain = compute_per_domain_metrics(y_true, y_pred, y_prob)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Macro  — Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")
    print(f"Micro  — Precision: {micro_p:.4f} | Recall: {micro_r:.4f} | F1: {micro_f1:.4f}")
    print("\nPer-Domain Metrics:")
    for label, m in per_domain.items():
        flag = " ✓" if m["precision"] >= 0.80 else " ✗"
        print(f"  {label:<20} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}{flag}")

    # Save plots
    plot_confusion_matrices(y_true, y_pred, EVAL_OUTPUT_DIR)
    plot_per_domain_metrics(per_domain, EVAL_OUTPUT_DIR)

    # Save JSON results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "checkpoint": checkpoint_path,
        "threshold": threshold,
        "n_samples": len(texts),
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "per_domain": per_domain,
        "meets_target": macro_p >= 0.80,
    }
    results_path = EVAL_OUTPUT_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--val-data", default=None, help="Path to validation JSONL file")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.val_data, args.threshold, args.batch_size)
