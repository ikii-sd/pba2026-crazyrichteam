"""
src/utils.py
============
Modul utility (helper) untuk proyek NLP — Analisis Sentimen & Emosi.

Berisi fungsi-fungsi umum yang sering digunakan agar tidak berserakan di notebook:
    - set_seed()                 : mengatur random seed
    - ensure_dir()               : memastikan direktori ada
    - save_json() / load_json()  : I/O JSON
    - get_timestamp()            : format waktu untuk penamaan file/folder
    - compute_metrics()          : menghitung accuracy, F1-macro, F1-weighted
    - plot_training_curves()     : plotting loss & accuracy
    - plot_confusion_matrix()    : plotting confusion matrix heatmap

Semua fungsi dilengkapi type hints dan docstring.

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
# A. GENERAL HELPERS (File System & Utils)
# ══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int = 42) -> None:
    """
    Mengatur random seed untuk Python, NumPy, dan PyTorch demi reproduktifitas.

    Parameters
    ----------
    seed : int
        Nilai seed. Default: 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    print(f"  [utils] 🌱 Random seed diatur ke: {seed}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Memastikan sebuah direktori ada. Jika tidak, direktori akan dibuat.

    Parameters
    ----------
    path : str | Path
        Path direktori tujuan.

    Returns
    -------
    Path
        Path object dari direktori yang dipastikan ada.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Union[str, Path], obj: Any) -> None:
    """
    Menyimpan objek Python ke file JSON.

    Parameters
    ----------
    path : str | Path
        Path file JSON tujuan.
    obj : Any
        Objek yang akan disimpan (dict, list, dll.)
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    print(f"  [utils] 💾 JSON tersimpan di: {p}")


def load_json(path: Union[str, Path]) -> Any:
    """
    Memuat objek Python dari file JSON.

    Parameters
    ----------
    path : str | Path
        Path file JSON.

    Returns
    -------
    Any
        Hasil parsing JSON (dict, list, dll.)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Mengambil timestamp saat ini dalam format string.
    Berguna untuk penamaan folder run / checkpoint.

    Parameters
    ----------
    fmt : str
        Format datetime klasik. Default: "%Y%m%d_%H%M%S"

    Returns
    -------
    str
    """
    return datetime.now().strftime(fmt)


# ══════════════════════════════════════════════════════════════════════════════
# B. METRICS & PLOTTING
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    task_name: str = "Task",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Menghitung beberapa metrik klasifikasi standar.

    Parameters
    ----------
    y_true : list/array
        Label sebenarnya.
    y_pred : list/array
        Label prediksi model.
    task_name : str
        Nama task (untuk log). Default: "Task".
    verbose : bool
        Apakah mencetak hasil ke layar. Default: True.

    Returns
    -------
    Dict[str, float]
        { "accuracy": ..., "f1_macro": ..., "f1_weighted": ... }
    """
    if not _PLOT_METRICS_AVAILABLE:
        raise ImportError(
            "[compute_metrics] Membutuhkan scikit-learn. Install: pip install scikit-learn"
        )

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wgh = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    res = {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wgh}

    if verbose:
        print(f"\n  📊 Metrics ({task_name}):")
        print(f"     Accuracy    : {acc:.4f}")
        print(f"     F1 Macro    : {f1_mac:.4f}")
        print(f"     F1 Weighted : {f1_wgh:.4f}")

    return res


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
) -> None:
    """
    Plotting kurva Loss dan Metrics (Accuracy/F1) dari history dictionary.

    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary berisi riwayat training per epoch.
        Contoh minimal: {"train_loss": [...], "val_loss": [...]}
    save_path : str | Path, opsional
        Jika diberikan, gambar akan disimpan ke path ini.
    title : str
        Judul utama plot.
    """
    if not _PLOT_METRICS_AVAILABLE:
        print("[plot_training_curves] Matplotlib tidak tersedia. Lewati plot.")
        return

    # Ekstrak loss
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    # Ekstrak metrik ke-2 (jika ada, misalnya akurasi gabungan atau setidaknya sentimen)
    train_metric = history.get("train_sent_acc", history.get("train_acc", []))
    val_metric = history.get("val_sent_acc", history.get("val_acc", []))

    n_cols = 2 if train_metric else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
    if n_cols == 1: axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Plot 1: Loss
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="tab:blue")
    if val_loss:
        axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Plot 2: Metric (opsional)
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

    if save_path:
        p = Path(save_path)
        ensure_dir(p.parent)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  [utils] 📈 Plot kurva disimpan ke: {p}")
        
    plt.show()


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plotting confusion matrix dalam bentuk heatmap.

    Parameters
    ----------
    y_true : list/array
        Label ground truth.
    y_pred : list/array
        Label prediksi.
    class_names : List[str]
        Nama kelas secara berurutan.
    title : str
        Judul plot.
    save_path : str | Path, opsional
        Path tujuan simpan.
    """
    if not _PLOT_METRICS_AVAILABLE:
        print("[plot_confusion_matrix] Sklearn/Matplotlib/Seaborn tidak tersedia. Lewati plot.")
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
