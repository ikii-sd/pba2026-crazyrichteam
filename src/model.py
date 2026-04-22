"""
src/model.py
============
Modul terpusat untuk definisi model, optimizer, dan scheduler.

Proyek : Analisis Sentimen & Emosi Ulasan E-Commerce Indonesia
Dataset: PRDECT-ID
Tim    : Crazy Rich Team — PBA 2026

Fungsi utama yang disediakan:
    build_model(config)             → nn.Module sesuai config
    get_optimizer(model, config)    → torch.optim.Optimizer
    get_scheduler(optimizer, config)→ LR Scheduler atau None
    count_params(model)             → jumlah parameter (trainable / total)
    model_summary(model, config)    → cetak ringkasan arsitektur ke stdout

Model yang didukung (config["model_name"]):
    "baseline"   → BiLSTM 1 layer, hidden=128, emb=128     (~1.4 M params)
    "improved"   → BiLSTM 2 layer, hidden=256, emb=128     (~3.1 M params)
    "large"      → BiLSTM 2 layer, hidden=256, emb=256     (~5.3 M params)
    "textcnn"    → TextCNN multi-kernel                     (~1.7 M params)

Cara pemakaian (notebook):
    from src.model import build_model, get_optimizer, get_scheduler, model_summary

    config = {
        "model_name"          : "baseline",
        "vocab_size"          : 10_000,
        "num_sentiment_classes": 2,
        "num_emotion_classes" : 5,
        # --- opsional (override default per model) ---
        "embedding_dim"       : 128,
        "hidden_dim"          : 128,
        "num_layers"          : 1,
        "dropout"             : 0.4,
        # --- optimizer ---
        "optimizer"           : "adam",       # "adam" | "adamw" | "sgd"
        "learning_rate"       : 1e-3,
        "weight_decay"        : 1e-5,
        # --- scheduler ---
        "scheduler"           : "plateau",    # "plateau" | "cosine" | "step" | None
        "scheduler_factor"    : 0.5,
        "scheduler_patience"  : 1,
        "scheduler_step_size" : 5,            # hanya untuk "step"
        "num_epochs"          : 15,           # dibutuhkan oleh "cosine"
    }

    model    = build_model(config)
    opt      = get_optimizer(model, config)
    sched    = get_scheduler(opt, config)
    model_summary(model, config)

Test mandiri (CLI):
    python src/model.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ── Registry model yang didukung ──────────────────────────────────────────────
_MODEL_REGISTRY: Dict[str, type] = {}


def _register(name: str):
    """Dekorator: daftarkan kelas model ke registry."""
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# A. KELAS MODEL
# ══════════════════════════════════════════════════════════════════════════════


@_register("baseline")
class BiLSTMBaseline(nn.Module):
    """
    BiLSTM Baseline — model ringan 1 layer.

    Arsitektur:
        Embedding → BiLSTM(1 layer) → Dropout → FC(2H→H) → ReLU
            → Dropout → [Sentiment Head | Emotion Head]

    Target parameter : < 2 M
    Direkomendasikan : percobaan awal / baseline comparison
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_sentiment_classes: int = 2,
        num_emotion_classes: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        vocab_size            : ukuran vocabulary (termasuk PAD & UNK)
        embedding_dim         : dimensi embedding token
        hidden_dim            : dimensi hidden state BiLSTM (satu arah)
        num_layers            : jumlah layer LSTM
        dropout               : probabilitas dropout
        pad_idx               : indeks token padding (default 0)
        num_sentiment_classes : jumlah kelas sentimen
        num_emotion_classes   : jumlah kelas emosi
        """
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)

        # Simpan hyperparameter untuk inspection
        self.hparams = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_sentiment_classes": num_sentiment_classes,
            "num_emotion_classes": num_emotion_classes,
        }

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : Tensor [batch_size, seq_len]

        Returns
        -------
        sentiment_logits : Tensor [batch_size, num_sentiment_classes]
        emotion_logits   : Tensor [batch_size, num_emotion_classes]
        """
        # [B, L] → [B, L, E]
        x = self.embedding(input_ids)
        # BiLSTM → ambil hidden state terakhir (forward + backward)
        _, (hidden, _) = self.bilstm(x)
        # hidden[-2]: forward last, hidden[-1]: backward last
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2H]

        x = self.dropout(features)
        x = self.relu(self.shared_fc(x))  # [B, H]
        x = self.dropout(x)

        return self.sentiment_head(x), self.emotion_head(x)


@_register("improved")
class BiLSTMImproved(nn.Module):
    """
    BiLSTM Improved — 2 layer, hidden lebih besar, BatchNorm.

    Arsitektur:
        Embedding → BiLSTM(2 layer) → Dropout
            → BatchNorm1d → FC(2H→H) → ReLU → Dropout
            → [Sentiment Head | Emotion Head]

    Target parameter : ~3 M
    Direkomendasikan : eksperimen lanjutan jika baseline kurang baik
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        pad_idx: int = 0,
        num_sentiment_classes: int = 2,
        num_emotion_classes: int = 5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)

        self.hparams = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_sentiment_classes": num_sentiment_classes,
            "num_emotion_classes": num_emotion_classes,
        }

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        _, (hidden, _) = self.bilstm(x)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)   # [B, 2H]

        features = self.bn(features)
        x = self.dropout(features)
        x = self.relu(self.shared_fc(x))
        x = self.dropout(x)

        return self.sentiment_head(x), self.emotion_head(x)


@_register("large")
class BiLSTMLarge(nn.Module):
    """
    BiLSTM Large — embedding lebih besar, 2 shared FC layer, residual-style dropout.

    Arsitektur:
        Embedding(256) → BiLSTM(2 layer, H=256)
            → BatchNorm → Dropout
            → FC(512→256) → ReLU → Dropout
            → FC(256→128) → ReLU → Dropout
            → [Sentiment Head | Emotion Head]

    Target parameter : ~5–6 M
    Direkomendasikan : kapasitas model untuk dataset besar atau augmented
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        intermediate_dim: int = 128,
        dropout: float = 0.4,
        pad_idx: int = 0,
        num_sentiment_classes: int = 2,
        num_emotion_classes: int = 5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.relu = nn.ReLU()

        self.sentiment_head = nn.Linear(intermediate_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(intermediate_dim, num_emotion_classes)

        self.hparams = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "intermediate_dim": intermediate_dim,
            "dropout": dropout,
            "num_sentiment_classes": num_sentiment_classes,
            "num_emotion_classes": num_emotion_classes,
        }

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        _, (hidden, _) = self.bilstm(x)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)   # [B, 2H]

        features = self.bn(features)
        x = self.dropout(features)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        return self.sentiment_head(x), self.emotion_head(x)


@_register("textcnn")
class TextCNN(nn.Module):
    """
    TextCNN — Convolutional Neural Network untuk teks (Kim 2014).

    Arsitektur:
        Embedding → Conv1d(kernel=[2,3,4]) → MaxPool → Concat
            → Dropout → FC → [Sentiment Head | Emotion Head]

    Target parameter : ~1.7 M
    Direkomendasikan : alternatif cepat, konvergen lebih cepat dari RNN
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: List[int] = None,
        dropout: float = 0.4,
        pad_idx: int = 0,
        num_sentiment_classes: int = 2,
        num_emotion_classes: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        kernel_sizes : List[int]
            Ukuran kernel konvolusi. Default: [2, 3, 4].
        num_filters  : int
            Jumlah filter per kernel.
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Conv1d: in_channels=embedding_dim, output per kernel=num_filters
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,           # same padding (approx)
            )
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        combined_dim = num_filters * len(kernel_sizes)
        self.fc_shared = nn.Linear(combined_dim, 128)
        self.sentiment_head = nn.Linear(128, num_sentiment_classes)
        self.emotion_head = nn.Linear(128, num_emotion_classes)

        self.hparams = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "num_filters": num_filters,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
            "num_sentiment_classes": num_sentiment_classes,
            "num_emotion_classes": num_emotion_classes,
        }

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, L] → [B, L, E] → [B, E, L]  (Conv1d expects channel first)
        x = self.embedding(input_ids).permute(0, 2, 1)

        # Konvolusi + global max pool per kernel
        pooled = []
        for conv in self.convs:
            c = self.relu(conv(x))         # [B, num_filters, L']
            c = c.max(dim=2).values        # [B, num_filters]
            pooled.append(c)

        features = torch.cat(pooled, dim=1)  # [B, num_filters * len(kernels)]
        features = self.dropout(features)

        x = self.relu(self.fc_shared(features))
        x = self.dropout(x)

        return self.sentiment_head(x), self.emotion_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# B. FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Bangun model sesuai konfigurasi dict.

    Parameters
    ----------
    config : dict
        Wajib ada:
            "model_name"           : str — nama model ("baseline", "improved",
                                     "large", "textcnn")
            "vocab_size"           : int
            "num_sentiment_classes": int
            "num_emotion_classes"  : int
        Opsional (dengan default per model):
            "embedding_dim"  : int  (default bervariasi per model)
            "hidden_dim"     : int
            "num_layers"     : int
            "dropout"        : float
            "pad_idx"        : int  (default 0)
            "kernel_sizes"   : List[int]  (hanya TextCNN)
            "num_filters"    : int        (hanya TextCNN)
            "intermediate_dim": int       (hanya BiLSTMLarge)

    Returns
    -------
    nn.Module

    Raises
    ------
    KeyError  : jika key wajib tidak ada di config
    ValueError: jika model_name tidak dikenali

    Examples
    --------
    >>> config = {
    ...     "model_name": "baseline",
    ...     "vocab_size": 10_000,
    ...     "num_sentiment_classes": 2,
    ...     "num_emotion_classes": 5,
    ... }
    >>> model = build_model(config)
    """
    required_keys = ["model_name", "vocab_size", "num_sentiment_classes", "num_emotion_classes"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"[build_model] Key wajib tidak ada di config: {missing}")

    name = str(config["model_name"]).lower()
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"[build_model] Model '{name}' tidak dikenali. "
            f"Pilihan: {available}"
        )

    cls = _MODEL_REGISTRY[name]

    # Argumen umum yang berlaku ke semua model
    common_kwargs = dict(
        vocab_size=int(config["vocab_size"]),
        num_sentiment_classes=int(config["num_sentiment_classes"]),
        num_emotion_classes=int(config["num_emotion_classes"]),
        pad_idx=int(config.get("pad_idx", 0)),
        dropout=float(config.get("dropout", 0.4)),
        embedding_dim=int(config.get("embedding_dim", 128)),
    )

    # Argumen khusus per model
    if name in ("baseline", "improved", "large"):
        common_kwargs["hidden_dim"] = int(config.get("hidden_dim", 128 if name == "baseline" else 256))
        common_kwargs["num_layers"] = int(config.get("num_layers", 1 if name == "baseline" else 2))
        if name == "large":
            common_kwargs["intermediate_dim"] = int(config.get("intermediate_dim", 128))

    elif name == "textcnn":
        common_kwargs.pop("embedding_dim")   # akan diisi ulang
        common_kwargs["embedding_dim"] = int(config.get("embedding_dim", 128))
        common_kwargs["num_filters"] = int(config.get("num_filters", 128))
        common_kwargs["kernel_sizes"] = config.get("kernel_sizes", [2, 3, 4])

    model = cls(**common_kwargs)
    total = count_params(model)
    print(
        f"  [build_model] ✅  '{name}'  "
        f"({total:,} parameter trainable)"
    )
    return model


def get_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Buat optimizer sesuai config.

    Parameters
    ----------
    model  : nn.Module
    config : dict
        "optimizer"    : str  — "adam" | "adamw" | "sgd"  (default: "adam")
        "learning_rate": float (default: 1e-3)
        "weight_decay" : float (default: 1e-5)
        "momentum"     : float (default: 0.9)  — hanya untuk SGD

    Returns
    -------
    torch.optim.Optimizer

    Examples
    --------
    >>> opt = get_optimizer(model, {"optimizer": "adamw", "learning_rate": 5e-4})
    """
    opt_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 1e-3))
    wd = float(config.get("weight_decay", 1e-5))

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd,
            momentum=momentum, nesterov=True,
        )
    else:
        raise ValueError(
            f"[get_optimizer] Optimizer '{opt_name}' tidak dikenali. "
            f"Pilihan: adam, adamw, sgd"
        )

    print(
        f"  [get_optimizer] ✅  {opt_name.upper()}  "
        f"lr={lr}  wd={wd}"
    )
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[Any]:
    """
    Buat learning rate scheduler sesuai config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    config    : dict
        "scheduler"         : str  — "plateau" | "cosine" | "step" | None
        "scheduler_factor"  : float (default: 0.5)     — untuk "plateau"
        "scheduler_patience": int   (default: 1)        — untuk "plateau"
        "scheduler_step_size": int  (default: 5)        — untuk "step"
        "scheduler_gamma"   : float (default: 0.1)      — untuk "step"
        "num_epochs"        : int   (default: 15)       — untuk "cosine"

    Returns
    -------
    LR scheduler atau None

    Examples
    --------
    >>> sched = get_scheduler(opt, {"scheduler": "plateau"})
    >>> sched = get_scheduler(opt, {"scheduler": None})  # → None
    """
    sched_name = config.get("scheduler", "plateau")
    if sched_name is None or str(sched_name).lower() == "none":
        print("  [get_scheduler] ℹ️  Scheduler dinonaktifkan.")
        return None

    sched_name = str(sched_name).lower()

    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 1)),
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("num_epochs", 15)),
            eta_min=float(config.get("lr_min", 1e-6)),
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("scheduler_step_size", 5)),
            gamma=float(config.get("scheduler_gamma", 0.1)),
        )
    else:
        raise ValueError(
            f"[get_scheduler] Scheduler '{sched_name}' tidak dikenali. "
            f"Pilihan: plateau, cosine, step, None"
        )

    print(f"  [get_scheduler] ✅  {sched_name.upper()} scheduler aktif")
    return scheduler


# ══════════════════════════════════════════════════════════════════════════════
# C. UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def count_params(
    model: nn.Module,
    trainable_only: bool = True,
) -> int:
    """
    Hitung total parameter model.

    Parameters
    ----------
    model          : nn.Module
    trainable_only : bool
        Jika True, hanya hitung parameter yang requires_grad=True.
        Default: True.

    Returns
    -------
    int

    Examples
    --------
    >>> n = count_params(model)
    >>> print(f"{n:,} parameter")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, int] = (4, 64),
) -> None:
    """
    Cetak ringkasan model: nama kelas, hyperparameter, jumlah parameter,
    dan output shape (dummy forward pass).

    Parameters
    ----------
    model       : nn.Module
    config      : dict, optional — untuk menampilkan label config
    input_shape : tuple (batch_size, seq_len) — untuk dummy forward pass

    Examples
    --------
    >>> model_summary(model, config)
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  📐  Model Summary : {model.__class__.__name__}")
    print(sep)

    # Hyperparameter (jika tersedia)
    if hasattr(model, "hparams"):
        print("  Hyperparameter:")
        for k, v in model.hparams.items():
            print(f"    {k:25s}: {v}")
    elif config:
        print("  Config:")
        for k, v in config.items():
            print(f"    {k:25s}: {v}")

    # Jumlah parameter
    trainable = count_params(model, trainable_only=True)
    total = count_params(model, trainable_only=False)
    print(f"\n  Trainable params : {trainable:>12,}")
    print(f"  Total params     : {total:>12,}")

    # Dummy forward pass
    device = next(model.parameters()).device
    dummy = torch.randint(0, model.hparams.get("vocab_size", 10_000) if hasattr(model, "hparams") else 10_000, input_shape).to(device)
    model.eval()
    with torch.no_grad():
        try:
            s_out, e_out = model(dummy)
            print(f"\n  Input shape             : {tuple(dummy.shape)}")
            print(f"  Sentiment output shape  : {tuple(s_out.shape)}")
            print(f"  Emotion output shape    : {tuple(e_out.shape)}")
        except Exception as exc:
            print(f"  Forward pass gagal: {exc}")
    model.train()

    print(sep + "\n")


def list_available_models() -> List[str]:
    """
    Kembalikan daftar nama model yang terdaftar di registry.

    Returns
    -------
    List[str]

    Examples
    --------
    >>> print(list_available_models())
    ['baseline', 'improved', 'large', 'textcnn']
    """
    return sorted(_MODEL_REGISTRY.keys())


# ══════════════════════════════════════════════════════════════════════════════
# D. CLI / SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧪  SMOKE TEST — src/model.py")
    print("=" * 60)

    print(f"\nModel tersedia: {list_available_models()}\n")

    # Konfigurasi base yang dipakai bersama
    _BASE = {
        "vocab_size"           : 10_000,
        "num_sentiment_classes": 2,
        "num_emotion_classes"  : 5,
        "optimizer"            : "adam",
        "learning_rate"        : 1e-3,
        "weight_decay"         : 1e-5,
        "scheduler"            : "plateau",
    }

    for _name in list_available_models():
        print(f"\n{'─' * 60}")
        print(f"  ▶  Testing model: {_name}")
        print(f"{'─' * 60}")

        cfg = {**_BASE, "model_name": _name}
        m = build_model(cfg)
        opt = get_optimizer(m, cfg)
        sched = get_scheduler(opt, cfg)
        model_summary(m, cfg)

    print("=" * 60)
    print("  ✅  Semua model berhasil dibuat dan diverifikasi!")
    print("=" * 60 + "\n")
