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
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.utils import ensure_dir, save_json, compute_metrics


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
    """
    Melakukan 1 epoch training pada dataloader yang diberikan.

    Parameters
    ----------
    model          : nn.Module
        Model PyTorch (contoh: dari src.models).
    dataloader     : DataLoader
        DataLoader untuk training set.
    optimizer      : Optimizer
    criterion_sent : Loss function untuk sentimen.
    criterion_emo  : Loss function untuk emosi.
    device         : "cpu" atau "cuda"
    grad_clip      : float
        Batas gradient clipping. Jika <= 0, tidak ada kliping.
    scaler         : GradScaler (opsional)
        Untuk mixed precision training.

    Returns
    -------
    Tuple[float, float, float]
        (rata-rata_loss_total, akurasi_sentimen, akurasi_emosi)
    """
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
        
        # Opsi: Automatic Mixed Precision
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
        
        # Hitung akurasi batch
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
    """
    Melakukan 1 epoch evaluasi pada validation/test set.

    Parameters
    ----------
    Mirip dengan train_one_epoch, tanpa optimizer/scaler.

    Returns
    -------
    Tuple[float, float, float]
        (rata-rata_loss_total, akurasi_sentimen, akurasi_emosi)
    """
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
    out_dir: str | Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Menjalankan full training pipeline (epoch loop).

    Fitur:
      - Otomatis transfer model ke device (jika belum).
      - Menjalankan Train & Val tiap epoch.
      - Early Stopping monitoring val_loss.
      - Menyimpan checkpoint best & last model.
      - Log metrics dan menyimpan format JSON (`history.json`).

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    optimizer : torch.optim.Optimizer
    scheduler : LR Scheduler (opsional)
    config : Dict
        Kamus pengaturan. Parameter utama:
        - "num_epochs": batas maksimum epoch.
        - "early_stopping_patience": kesabaran early stopping (0 = nonaktif).
        - "early_stopping_min_delta": perbedaan minimal perbaikan loss.
        - "grad_clip_max_norm": batas gradient clipping.
        - "use_amp": apakah menggunakan mixed precision.
        - "model_name": sekadar pembeda format file output.
    out_dir : str | Path
        Direktori target menyimpan checkpoint, history, dll.
    device : torch.device, opsional
        Device target. Jika None, diambil otomatis cuda jika ada.

    Returns
    -------
    Dict[str, Any]
        History dictionary metrik per epoch.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    
    # ── Param Konfigurasi
    epochs = config.get("num_epochs", 15)
    patience = config.get("early_stopping_patience", 3)
    min_delta = config.get("early_stopping_min_delta", 1e-4)
    grad_clip = config.get("grad_clip_max_norm", 1.0)
    use_amp = config.get("use_amp", False)
    mod_name = config.get("model_name", "generic")
    
    # Checkpoint dir
    chk_dir = ensure_dir(Path(out_dir) / "checkpoints")
    best_path = chk_dir / f"best_{mod_name}.pt"
    last_path = chk_dir / f"last_{mod_name}.pt"
    hist_path = ensure_dir(Path(out_dir) / "logs") / f"history_{mod_name}.json"
    
    # Loss functions
    crit_s = nn.CrossEntropyLoss()
    crit_e = nn.CrossEntropyLoss()
    
    # Scaler
    scaler = torch.amp.GradScaler(device.type) if (use_amp and device.type == "cuda") else None
    
    # Trackers
    history = {
        "train_loss": [], "train_sent_acc": [], "train_emo_acc": [],
        "val_loss": [], "val_sent_acc": [], "val_emo_acc": [], "lr": []
    }
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print(f"  🚀  MEMULAI TRAINING : {mod_name.upper()}")
    print("="*70)
    print(f"  Device   : {device}")
    print(f"  Epochs   : {epochs}")
    print(f"  Patience : {patience}")
    print(f"  AMP      : {'Ya' if scaler else 'Tidak'}")
    print("-"*70)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # ── TRAIN
        tr_loss, tr_s_acc, tr_e_acc = train_one_epoch(
            model, train_loader, optimizer, crit_s, crit_e, device, grad_clip, scaler
        )
        
        # ── VAL
        vl_loss, vl_s_acc, vl_e_acc = validate(
            model, val_loader, crit_s, crit_e, device
        )
        
        # ── SCHEDULER
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(vl_loss)
            else:
                scheduler.step()
                
        lr_now = optimizer.param_groups[0]["lr"]
        
        # Log History
        history["train_loss"].append(tr_loss)
        history["train_sent_acc"].append(tr_s_acc)
        history["train_emo_acc"].append(tr_e_acc)
        history["val_loss"].append(vl_loss)
        history["val_sent_acc"].append(vl_s_acc)
        history["val_emo_acc"].append(vl_e_acc)
        history["lr"].append(lr_now)
        
        elapsed = time.time() - t0
        
        # Log info
        print(
            f" [Ep {epoch:3d}/{epochs}] "
            f"T-Loss: {tr_loss:.4f}  V-Loss: {vl_loss:.4f} | "
            f"S-Acc V: {vl_s_acc:.3f}  E-Acc V: {vl_e_acc:.3f} | "
            f"{elapsed:.1f}s"
        )
        
        # ── EVAL BEST & EARLY STOPPING
        if vl_loss < best_val_loss - min_delta:
            best_val_loss = vl_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best
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
                print(f"\n  🛑 Early Stopping dipicu di epoch {epoch}!")
                print(f"     Val Loss tidak membaik selama {patience} epoch.")
                break
                
    # Save last
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": vl_loss,
        "config": config,
    }, last_path)
    
    # Save history
    save_json(hist_path, history)
    
    total_m = (time.time() - start_time) / 60
    print("-"*70)
    print(f"  ✅ Selesai dalam {total_m:.1f} menit.")
    print(f"     Best Val Loss : {best_val_loss:.4f} (di epoch {best_epoch})")
    print(f"     Model terbaik : {best_path}")
    print("="*70 + "\n")
    
    return history
