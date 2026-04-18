"""
evaluation.py
=============
Evaluation metrics and visualisation helpers.

Produces:
  - per-class classification_report (sklearn)
  - accuracy / F1-macro / F1-weighted
  - confusion-matrix heatmaps  (seaborn)
  - training-curve plots       (matplotlib)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

from models.config_simplified import DEVICE


# ── inference on test set ─────────────────────────────────────────────────────

def test_model(
    model: nn.Module,
    dataloader,
    device: torch.device = DEVICE,
) -> tuple:
    """
    Run inference on the given dataloader.

    Returns:
        (sent_preds, sent_labels, emo_preds, emo_labels)  — all np.ndarray
    """
    model.eval()
    all_sent_preds, all_sent_labels = [], []
    all_emo_preds, all_emo_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  test", leave=False):
            input_ids = batch["input_ids"].to(device)
            s_labels = batch["sentiment_label"]
            e_labels = batch["emotion_label"]

            sent_logits, emo_logits = model(input_ids)

            all_sent_preds.extend(sent_logits.argmax(dim=1).cpu().numpy())
            all_sent_labels.extend(s_labels.numpy())
            all_emo_preds.extend(emo_logits.argmax(dim=1).cpu().numpy())
            all_emo_labels.extend(e_labels.numpy())

    return (
        np.array(all_sent_preds),
        np.array(all_sent_labels),
        np.array(all_emo_preds),
        np.array(all_emo_labels),
    )


# ── metric computation ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, task_name: str = "Task") -> dict:
    """
    Print classification report and return summary metrics dict.

    Returns:
        {"accuracy": float, "f1_macro": float, "f1_weighted": float}
    """
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wgt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {task_name}")
    print(sep)
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 Macro     : {f1_mac:.4f}")
    print(f"  F1 Weighted  : {f1_wgt:.4f}")

    return {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wgt}


# ── visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list,
    task_name: str,
    save_path: str = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {task_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)


def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: str = None,
) -> None:
    """Plot train/val loss and accuracy curves side-by-side."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "s--", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy (Sentiment)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
