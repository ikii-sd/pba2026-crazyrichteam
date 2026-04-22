"""
src/utils.py
============
Modul utility (helper) untuk proyek NLP — Analisis Sentimen & Emosi.

Berisi fungsi-fungsi umum yang sering digunakan agar tidak berserakan di notebook:
    - setup_run_env()            : membuat run directory & config otomatis
    - set_seed()                 : mengatur random seed
    - ensure_dir()               : memastikan direktori ada
    - save_json() / load_json()  : I/O JSON
    - get_timestamp()            : format waktu untuk penamaan file/folder
    - compute_metrics()          : menghitung accuracy, F1-macro, F1-weighted
    - plot_training_curves()     : plotting loss & accuracy
    - plot_confusion_matrix()    : plotting confusion matrix heatmap

Author : Crazy Rich Team — PBA 2026
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.logger import get_logger, setup_run_logger

logger = get_logger(__name__)

# ── Opsional: PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Opsional: Sklearn & Matplotlib/Seaborn ────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    _PLOT_METRICS_AVAILABLE = True
except ImportError:
    _PLOT_METRICS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# A. GENERAL HELPERS (Run Env & File System)
# ══════════════════════════════════════════════════════════════════════════════


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Mengambil timestamp saat ini dalam format string."""
    return datetime.now().strftime(fmt)


def setup_run_env(
    experiment_name: str,
    base_out_dir: str | Path = "outputs"
) -> Dict[str, Path]:
    """
    Menyiapkan seluruh folder environment (figures, checkpoints, logs) untuk satu RUN.
    Termasuk juga inisiasi master Logger.
    
    Parameters
    ----------
    experiment_name : str
        Nama eksperimen (e.g. "baseline", "model_large_aug").
    base_out_dir : str | Path
        Root directory (default "outputs").
        
    Returns
    -------
    dict:
        "run_name"   : Nama dinamis run (timestamp_experiment).
        "chk_dir"    : path ke checkpoint folder.
        "log_dir"    : path ke log folder.
        "fig_dir"    : path ke figure folder.
    """
    run_name = f"{get_timestamp()}_{experiment_name}"
    
    # Path preparation
    chk_dir = ensure_dir(Path(base_out_dir) / "checkpoints" / run_name)
    fig_dir = ensure_dir(Path(base_out_dir) / "figures" / run_name)
    log_dir_root = ensure_dir(Path(base_out_dir) / "logs")
    
    # Setup Logger specific for this run
    setup_run_logger(run_name, log_dir=log_dir_root)
    
    logger.info(f"📁 Run Environment Setup Selesai")
    logger.info(f"   Checkpoints : {chk_dir}")
    logger.info(f"   Figures     : {fig_dir}")
    logger.info(f"   Logs        : {log_dir_root / run_name}")
    
    return {
        "run_name": run_name,
        "chk_dir": chk_dir,
        "log_dir": log_dir_root / run_name,
        "fig_dir": fig_dir,
    }


def set_seed(seed: int = 42) -> None:
    """Mengatur random seed untuk Python, NumPy, dan PyTorch demi reproduktifitas."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    logger.info(f"🌱 Random seed diatur ke: {seed}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Memastikan sebuah direktori ada. Jika tidak, direktori akan dibuat."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Union[str, Path], obj: Any) -> None:
    """Menyimpan objek Python ke file JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    logger.info(f"💾 JSON tersimpan di: {p}")


def load_json(path: Union[str, Path]) -> Any:
    """Memuat objek Python dari file JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# B. METRICS & PLOTTING
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    task_name: str = "Task",
    verbose: bool = True,
) -> Dict[str, float]:
    """Menghitung beberapa metrik klasifikasi standar."""
    if not _PLOT_METRICS_AVAILABLE:
        raise ImportError("[compute_metrics] Membutuhkan scikit-learn.")

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wgh = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    res = {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wgh}

    if verbose:
        logger.info(f"📊 Metrics ({task_name}): Acc={acc:.4f} | F1-Mac={f1_mac:.4f} | F1-Wgh={f1_wgh:.4f}")

    return res


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: Optional[Union[str, Path]] = None,
    filename: str = "loss_curves.png",
    title: str = "Training Curves",
) -> None:
    """Plotting kurva Loss dan Metrics (Accuracy/F1) lalu simpan ke folder figures."""
    if not _PLOT_METRICS_AVAILABLE:
        logger.warning("[plot_training_curves] Matplotlib tidak tersedia.")
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    train_metric = history.get("train_sent_acc", history.get("train_acc", []))
    val_metric = history.get("val_sent_acc", history.get("val_acc", []))

    n_cols = 2 if train_metric else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
    if n_cols == 1: axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="tab:blue")
    if val_loss:
        axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if train_metric:
        axes[1].plot(epochs, train_metric, "o-", label="Train Metric", color="tab:green")
        if val_metric:
            axes[1].plot(epochs, val_metric, "s--", label="Val Metric", color="tab:red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Metric (Acc/F1)")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    plt.tight_layout()

    if save_dir:
        p = Path(save_dir) / filename
        ensure_dir(p.parent)
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor='w')
        logger.info(f"📈 Kurva disimpan ke: {p}")
        
    plt.close()


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_dir: Optional[Union[str, Path]] = None,
    filename: str = "confusion_matrix.png",
) -> None:
    """Plotting confusion matrix dalam bentuk heatmap lalu simpan."""
    if not _PLOT_METRICS_AVAILABLE:
        logger.warning("[plot_confusion_matrix] Sklearn/Matplotlib tidak tersedia.")
        return

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)*0.8), max(4, len(class_names)*0.6)))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar=False
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        ensure_dir(p.parent)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  [utils] 📈 Confusion Matrix disimpan ke: {p}")
        
    plt.show()

# Run simple check if executed directly
if __name__ == "__main__":
    t = get_timestamp()
    print(f"Testing timestamp: {t}")
    set_seed(123)
