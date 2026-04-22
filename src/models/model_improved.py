"""
src/models/model_improved.py
=============================
Varian Model Improved — BiLSTM 2 layer dengan BatchNorm.

Profil:
    - Arsitektur : Embedding → BiLSTM(2 layer, H=256) → BN → FC → 2 Heads
    - ~3.1 M parameter
    - Cocok untuk : percobaan lanjutan setelah baseline, lebih dalam tapi masih cepat

Antarmuka standar:
    MODEL_NAME   : str
    DEFAULT_CFG  : dict
    build(config): nn.Module

Author : Crazy Rich Team — PBA 2026
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# IDENTITAS & KONFIGURASI DEFAULT
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME: str = "improved"

DEFAULT_CFG: Dict[str, Any] = {
    # ── Model ─────────────────────────────────────────────
    "model_name"             : "improved",
    "embedding_dim"          : 128,
    "hidden_dim"             : 256,
    "num_layers"             : 2,
    "dropout"                : 0.4,
    "pad_idx"                : 0,
    "use_batch_norm"         : True,
    # ── Output ────────────────────────────────────────────
    "vocab_size"             : 10_000,
    "num_sentiment_classes"  : 2,
    "num_emotion_classes"    : 5,
    # ── Optimizer ─────────────────────────────────────────
    "optimizer"              : "adamw",
    "learning_rate"          : 5e-4,
    "weight_decay"           : 1e-5,
    # ── Scheduler ─────────────────────────────────────────
    "scheduler"              : "plateau",
    "scheduler_factor"       : 0.5,
    "scheduler_patience"     : 2,
    # ── Training ──────────────────────────────────────────
    "num_epochs"             : 20,
    "batch_size"             : 64,
    "early_stopping_patience": 4,
    "grad_clip_max_norm"     : 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# DEFINISI MODEL
# ══════════════════════════════════════════════════════════════════════════════


class BiLSTMImproved(nn.Module):
    """
    BiLSTM Improved — 2 layer dengan BatchNorm untuk training lebih stabil.

    Arsitektur:
        Embedding(vocab_size, emb_dim)
        → BiLSTM(emb_dim, H, num_layers=2, bidirectional=True)
        → Concat [forward_last, backward_last]  → [B, 2H]
        → (opsional) BatchNorm1d(2H)
        → Dropout → Linear(2H→H) → ReLU → Dropout
        → [Linear(H, n_sent) | Linear(H, n_emo)]
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        pad_idx: int = 0,
        use_batch_norm: bool = True,
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
            dropout=dropout,          # dropout antarlayer (num_layers > 1)
        )
        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)

        self.hparams: Dict[str, Any] = dict(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
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
        x = self.embedding(input_ids)
        _, (hidden, _) = self.bilstm(x)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2H]

        if self.use_bn:
            features = self.bn(features)

        x = self.dropout(features)
        x = self.relu(self.shared_fc(x))
        x = self.dropout(x)
        return self.sentiment_head(x), self.emotion_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


def build(config: Dict[str, Any]) -> nn.Module:
    """
    Buat instance BiLSTMImproved dari config dict.

    Parameters
    ----------
    config : dict
        Minimal: "vocab_size", "num_sentiment_classes", "num_emotion_classes".

    Returns
    -------
    nn.Module (BiLSTMImproved)

    Examples
    --------
    >>> from src.models.model_improved import build, DEFAULT_CFG
    >>> model = build({**DEFAULT_CFG, "vocab_size": 10_000})
    """
    merged = {**DEFAULT_CFG, **config}
    return BiLSTMImproved(
        vocab_size=int(merged["vocab_size"]),
        embedding_dim=int(merged.get("embedding_dim", 128)),
        hidden_dim=int(merged.get("hidden_dim", 256)),
        num_layers=int(merged.get("num_layers", 2)),
        dropout=float(merged.get("dropout", 0.4)),
        pad_idx=int(merged.get("pad_idx", 0)),
        use_batch_norm=bool(merged.get("use_batch_norm", True)),
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
