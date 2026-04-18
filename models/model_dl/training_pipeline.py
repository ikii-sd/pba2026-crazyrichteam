"""
training_pipeline.py
====================
Training loop with early stopping and LR scheduler.

Note: Adapted to actual model_design.py API:
  - SimpleBiLSTM.forward() takes only input_ids (no 'lengths' arg)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from models.config_simplified import (
    DEVICE, GRAD_CLIP_MAX_NORM,
    USE_LR_SCHEDULER, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
)


# ── per-epoch helpers ───────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion_sentiment: nn.Module,
    criterion_emotion: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        (avg_loss, avg_sentiment_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch in tqdm(dataloader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)
        emotion_labels = batch["emotion_label"].to(device)

        optimizer.zero_grad()

        # forward — model takes input_ids only
        sentiment_logits, emotion_logits = model(input_ids)

        loss_sent = criterion_sentiment(sentiment_logits, sentiment_labels)
        loss_emo = criterion_emotion(emotion_logits, emotion_labels)
        loss = loss_sent + loss_emo

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        bs = sentiment_labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        correct += (sentiment_logits.argmax(dim=1) == sentiment_labels).sum().item()

    return total_loss / total_samples, correct / total_samples


def validate(
    model: nn.Module,
    dataloader,
    criterion_sentiment: nn.Module,
    criterion_emotion: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """Validate / evaluate model (no gradient update)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="    val", leave=False):
            input_ids = batch["input_ids"].to(device)
            sentiment_labels = batch["sentiment_label"].to(device)
            emotion_labels = batch["emotion_label"].to(device)

            sentiment_logits, emotion_logits = model(input_ids)

            loss_sent = criterion_sentiment(sentiment_logits, sentiment_labels)
            loss_emo = criterion_emotion(emotion_logits, emotion_labels)
            loss = loss_sent + loss_emo

            bs = sentiment_labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            correct += (sentiment_logits.argmax(dim=1) == sentiment_labels).sum().item()

    return total_loss / total_samples, correct / total_samples


# ── full training loop ───────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    learning_rate: float,
    patience: int,
    model_name: str = "model",
    device: torch.device = DEVICE,
) -> Dict:
    """
    Full training loop with early stopping and optional LR scheduler.

    Returns:
        history dict with train/val loss & accuracy per epoch,
        plus 'total_time_minutes'.
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion_sentiment = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()

    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )

    history: Dict = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Training : {model_name}")
    print(f"  Device   : {device} | LR: {learning_rate} | Patience: {patience}")
    print(f"{bar}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer,
            criterion_sentiment, criterion_emotion, device,
        )
        val_loss, val_acc = validate(
            model, val_loader,
            criterion_sentiment, criterion_emotion, device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
            f"LR {lr_now:.2e} | {elapsed:.1f}s"
        )

        # early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)\n")
                break

    total_time = time.time() - start_time
    print(f"  ✅ Done in {total_time / 60:.1f} min | best val loss {best_val_loss:.4f}")

    # restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history["total_time_minutes"] = total_time / 60
    return history


# ── checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    vocab,
    sentiment_label2id: dict,
    emotion_label2id: dict,
    metadata: dict,
    path: str,
) -> None:
    """
    Save model + vocab meta + class mappings as a single .pt checkpoint.
    Also writes a companion _metadata.json next to the .pt file.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_size": len(vocab),
        "sentiment_label2id": sentiment_label2id,
        "emotion_label2id": emotion_label2id,
        "metadata": metadata,
    }
    torch.save(checkpoint, path)
    print(f"  ✅ Checkpoint saved : {path}")

    meta_path = str(path).replace(".pt", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Metadata saved   : {meta_path}")
