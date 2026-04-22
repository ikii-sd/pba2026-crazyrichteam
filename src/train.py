"""
src/train.py
============
Modul training loop terpusat untuk model NLP analisis sentimen & emosi.

Fitur utama:
    - `train_one_epoch()` : iterasi data training per epoch.
    - `validate()`        : evaluasi model pada validation set tanpa gradient.
    - `fit()`             : loop utuh mencakup epoch, early stopping,
                            LR scheduling, checkpointing (best & last),
                            dan tracking history.

Mendukung otomatisasi:
    - Mixed precision (torch.cuda.amp) opsional jika tersedia hardware.
    - Gradient clipping opsional.

Author : Crazy Rich Team — PBA 2026
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.logger import get_logger
from src.utils import ensure_dir, save_json

logger = get_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_sent: nn.Module,
    criterion_emo: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float, float]:
    """Melakukan 1 epoch training pada dataloader yang diberikan."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    correct_sent = 0
    correct_emo = 0
    
    pbar = tqdm(dataloader, desc="  🚀 Train", leave=False)
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        sent_labels = batch["sentiment_label"].to(device)
        emo_labels = batch["emotion_label"].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                sent_logits, emo_logits = model(input_ids)
                loss_s = criterion_sent(sent_logits, sent_labels)
                loss_e = criterion_emo(emo_logits, emo_labels)
                loss = loss_s + loss_e
                
            scaler.scale(loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            sent_logits, emo_logits = model(input_ids)
            loss_s = criterion_sent(sent_logits, sent_labels)
            loss_e = criterion_emo(emo_logits, emo_labels)
            loss = loss_s + loss_e
            
            loss.backward()
            
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
        
        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        
        preds_s = sent_logits.argmax(dim=1)
        preds_e = emo_logits.argmax(dim=1)
        
        c_sent = (preds_s == sent_labels).sum().item()
        c_emo = (preds_e == emo_labels).sum().item()
        
        correct_sent += c_sent
        correct_emo += c_emo
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc_S": f"{c_sent/bs:.2f}"})
        
    avg_loss = total_loss / total_samples
    acc_s = correct_sent / total_samples
    acc_e = correct_emo / total_samples
    
    return avg_loss, acc_s, acc_e


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion_sent: nn.Module,
    criterion_emo: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Melakukan 1 epoch evaluasi pada validation/test set."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    correct_sent = 0
    correct_emo = 0
    
    pbar = tqdm(dataloader, desc="  🔍 Val", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            sent_labels = batch["sentiment_label"].to(device)
            emo_labels = batch["emotion_label"].to(device)
            
            sent_logits, emo_logits = model(input_ids)
            
            loss_s = criterion_sent(sent_logits, sent_labels)
            loss_e = criterion_emo(emo_logits, emo_labels)
            loss = loss_s + loss_e
            
            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            
            preds_s = sent_logits.argmax(dim=1)
            preds_e = emo_logits.argmax(dim=1)
            
            correct_sent += (preds_s == sent_labels).sum().item()
            correct_emo += (preds_e == emo_labels).sum().item()
            
    avg_loss = total_loss / total_samples
    acc_s = correct_sent / total_samples
    acc_e = correct_emo / total_samples
    
    return avg_loss, acc_s, acc_e


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    config: Dict[str, Any],
    run_env: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Menjalankan full training pipeline (epoch loop)."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    
    # Param Konfigurasi
    epochs = config.get("num_epochs", 15)
    patience = config.get("early_stopping_patience", 3)
    min_delta = config.get("early_stopping_min_delta", 1e-4)
    grad_clip = config.get("grad_clip_max_norm", 1.0)
    use_amp = config.get("use_amp", False)
    mod_name = config.get("model_name", "generic")
    
    # Path dari run_env
    chk_dir = Path(run_env["chk_dir"])
    best_path = chk_dir / f"best_{mod_name}.pt"
    last_path = chk_dir / f"last_{mod_name}.pt"
    hist_path = Path(run_env["log_dir"]) / f"history_{mod_name}.json"
    
    crit_s = nn.CrossEntropyLoss()
    crit_e = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler(device.type) if (use_amp and device.type == "cuda") else None
    
    history = {
        "train_loss": [], "train_sent_acc": [], "train_emo_acc": [],
        "val_loss": [], "val_sent_acc": [], "val_emo_acc": [], "lr": []
    }
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    logger.info(f"🚀 MEMULAI TRAINING : {mod_name.upper()}")
    logger.info(f"   Device   : {device}")
    logger.info(f"   Epochs   : {epochs}")
    logger.info(f"   Patience : {patience}")
    logger.info(f"   AMP      : {'Ya' if scaler else 'Tidak'}")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # TRAIN
        tr_loss, tr_s_acc, tr_e_acc = train_one_epoch(
            model, train_loader, optimizer, crit_s, crit_e, device, grad_clip, scaler
        )
        
        # VAL
        vl_loss, vl_s_acc, vl_e_acc = validate(
            model, val_loader, crit_s, crit_e, device
        )
        
        # SCHEDULER
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl_loss)
            else:
                scheduler.step()
                
        lr_now = optimizer.param_groups[0]["lr"]
        
        # History track
        history["train_loss"].append(tr_loss)
        history["train_sent_acc"].append(tr_s_acc)
        history["train_emo_acc"].append(tr_e_acc)
        history["val_loss"].append(vl_loss)
        history["val_sent_acc"].append(vl_s_acc)
        history["val_emo_acc"].append(vl_e_acc)
        history["lr"].append(lr_now)
        
        elapsed = time.time() - t0
        
        logger.info(
            f"[Ep {epoch:3d}/{epochs}] "
            f"T-Loss={tr_loss:.4f} V-Loss={vl_loss:.4f} | "
            f"S-AccV={vl_s_acc:.3f} E-AccV={vl_e_acc:.3f} | {elapsed:.1f}s"
        )
        
        # EVAL BEST
        if vl_loss < best_val_loss - min_delta:
            best_val_loss = vl_loss
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, best_path)
            
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                logger.info(f"🛑 Early Stopping dipicu di epoch {epoch}! Val Loss stagnan ({patience} epoch).")
                break
                
    # End of run tasks
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": vl_loss,
        "config": config,
    }, last_path)
    
    save_json(hist_path, history)
    
    total_m = (time.time() - start_time) / 60
    logger.info(f"✅ Selesai dalam {total_m:.1f} menit. Best Val Loss: {best_val_loss:.4f} (ep {best_epoch})")
    
    return history
