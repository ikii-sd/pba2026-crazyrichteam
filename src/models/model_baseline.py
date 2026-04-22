"""
src/models/model_baseline.py
=============================
Varian Model Baseline — BiLSTM 1 layer ringan.

Profil:
    - Arsitektur : Embedding → BiLSTM(1 layer, H=128) → FC → 2 Heads
    - ~1.4 M parameter
    - Cocok untuk : percobaan awal, benchmark cepat, perangkat terbatas

Antarmuka standar yang WAJIB ada di setiap model_*.py:
    MODEL_NAME   : str  — identifier unik
    DEFAULT_CFG  : dict — hyperparameter default model ini
    build(config): nn.Module — factory function

Cara pemakaian:
    # Langsung dari file ini
    from src.models.model_baseline import build, DEFAULT_CFG
    model = build({**DEFAULT_CFG, "vocab_size": 10_000})

    # Atau lewat src/models/__init__.py (direkomendasikan)
    from src.models import build_model
    model = build_model({"model_name": "baseline", "vocab_size": 10_000, ...})

Author : Crazy Rich Team — PBA 2026
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# IDENTITAS & KONFIGURASI DEFAULT
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME: str = "baseline"

DEFAULT_CFG: Dict[str, Any] = {
    # ── Model ─────────────────────────────────────────────
    "model_name"             : "baseline",
    "embedding_dim"          : 128,
    "hidden_dim"             : 128,
    "num_layers"             : 1,
    "dropout"                : 0.3,
    "pad_idx"                : 0,
    # ── Output (diisi dari data pipeline) ────────────────
    "vocab_size"             : 10_000,
    "num_sentiment_classes"  : 2,
    "num_emotion_classes"    : 5,
    # ── Optimizer ─────────────────────────────────────────
    "optimizer"              : "adam",
    "learning_rate"          : 1e-3,
    "weight_decay"           : 1e-5,
    # ── Scheduler ─────────────────────────────────────────
    "scheduler"              : "plateau",
    "scheduler_factor"       : 0.5,
    "scheduler_patience"     : 1,
    # ── Training ──────────────────────────────────────────
    "num_epochs"             : 15,
    "batch_size"             : 64,
    "early_stopping_patience": 3,
    "grad_clip_max_norm"     : 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# DEFINISI MODEL
# ══════════════════════════════════════════════════════════════════════════════


class BiLSTMBaseline(nn.Module):
    """
    BiLSTM Baseline — model ringan untuk klasifikasi sentimen & emosi.

    Arsitektur:
        Embedding(vocab_size, emb_dim)
        → BiLSTM(emb_dim, H, num_layers=1, bidirectional=True)
        → Concat [forward_last, backward_last]  → [B, 2H]
        → Dropout → Linear(2H→H) → ReLU → Dropout
        → [Linear(H, n_sent) | Linear(H, n_emo)]
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
        super().__init__()
        # Dropout di dalam LSTM hanya berlaku jika num_layers > 1
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

        # Simpan hyperparameter untuk inspeksi / serialisasi
        self.hparams: Dict[str, Any] = dict(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_sentiment_classes=num_sentiment_classes,
            num_emotion_classes=num_emotion_classes,
        )

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : Tensor [batch_size, seq_len]

        Returns
        -------
        sentiment_logits : Tensor [batch_size, num_sentiment_classes]
        emotion_logits   : Tensor [batch_size, num_emotion_classes]
        """
        x = self.embedding(input_ids)             # [B, L, E]
        _, (hidden, _) = self.bilstm(x)           # hidden: [2*L, B, H]
        # Ambil hidden terakhir forward + backward
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2H]
        x = self.dropout(features)
        x = self.relu(self.shared_fc(x))          # [B, H]
        x = self.dropout(x)
        return self.sentiment_head(x), self.emotion_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION (antarmuka standar)
# ══════════════════════════════════════════════════════════════════════════════


def build(config: Dict[str, Any]) -> nn.Module:
    """
    Buat instance BiLSTMBaseline dari config dict.

    Parameters
    ----------
    config : dict
        Minimal: "vocab_size", "num_sentiment_classes", "num_emotion_classes".
        Sisanya menggunakan DEFAULT_CFG jika tidak disediakan.

    Returns
    -------
    nn.Module (BiLSTMBaseline)

    Examples
    --------
    >>> from src.models.model_baseline import build, DEFAULT_CFG
    >>> model = build({**DEFAULT_CFG, "vocab_size": 8_000})
    """
    merged = {**DEFAULT_CFG, **config}
    return BiLSTMBaseline(
        vocab_size=int(merged["vocab_size"]),
        embedding_dim=int(merged.get("embedding_dim", 128)),
        hidden_dim=int(merged.get("hidden_dim", 128)),
        num_layers=int(merged.get("num_layers", 1)),
        dropout=float(merged.get("dropout", 0.3)),
        pad_idx=int(merged.get("pad_idx", 0)),
        num_sentiment_classes=int(merged["num_sentiment_classes"]),
        num_emotion_classes=int(merged["num_emotion_classes"]),
    )


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _cfg = {**DEFAULT_CFG, "vocab_size": 10_000}
    _m = build(_cfg)
    _params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    _dummy = torch.randint(0, 10_000, (4, 64))
    _s, _e = _m(_dummy)
    print(f"[{MODEL_NAME}] params={_params:,}  sent={tuple(_s.shape)}  emo={tuple(_e.shape)}")
