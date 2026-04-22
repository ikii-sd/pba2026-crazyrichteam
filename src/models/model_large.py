"""
src/models/model_large.py
==========================
Varian Model Large — BiLSTM 2 layer, embedding besar, 2 shared FC.

Profil:
    - Arsitektur : Embedding(256) → BiLSTM(2 layer, H=256) → BN → 2×FC → 2 Heads
    - ~5–6 M parameter
    - Cocok untuk : kapasitas penuh, data yang lebih besar atau di-augmented

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

MODEL_NAME: str = "large"

DEFAULT_CFG: Dict[str, Any] = {
    # ── Model ─────────────────────────────────────────────
    "model_name"             : "large",
    "embedding_dim"          : 256,
    "hidden_dim"             : 256,
    "num_layers"             : 2,
    "intermediate_dim"       : 128,
    "dropout"                : 0.4,
    "pad_idx"                : 0,
    "use_batch_norm"         : True,
    # ── Output ────────────────────────────────────────────
    "vocab_size"             : 10_000,
    "num_sentiment_classes"  : 2,
    "num_emotion_classes"    : 5,
    # ── Optimizer ─────────────────────────────────────────
    "optimizer"              : "adamw",
    "learning_rate"          : 3e-4,
    "weight_decay"           : 1e-4,
    # ── Scheduler ─────────────────────────────────────────
    "scheduler"              : "cosine",
    "num_epochs"             : 25,
    # ── Training ──────────────────────────────────────────
    "batch_size"             : 32,
    "early_stopping_patience": 5,
    "grad_clip_max_norm"     : 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# DEFINISI MODEL
# ══════════════════════════════════════════════════════════════════════════════


class BiLSTMLarge(nn.Module):
    """
    BiLSTM Large — kapasitas penuh dengan dua shared FC layer.

    Arsitektur:
        Embedding(vocab_size, emb_dim=256)
        → BiLSTM(256, 256, num_layers=2, bidirectional=True)
        → Concat [forward_last, backward_last]   → [B, 512]
        → (opsional) BatchNorm1d(512)
        → Dropout → Linear(512→256) → ReLU
        → Dropout → Linear(256→128) → ReLU → Dropout
        → [Linear(128, n_sent) | Linear(128, n_emo)]
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
            dropout=dropout,
        )
        self.use_bn = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)

        # Dua FC layer bersama
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.relu = nn.ReLU()

        self.sentiment_head = nn.Linear(intermediate_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(intermediate_dim, num_emotion_classes)

        self.hparams: Dict[str, Any] = dict(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
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
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        return self.sentiment_head(x), self.emotion_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


def build(config: Dict[str, Any]) -> nn.Module:
    """
    Buat instance BiLSTMLarge dari config dict.

    Parameters
    ----------
    config : dict
        Minimal: "vocab_size", "num_sentiment_classes", "num_emotion_classes".

    Returns
    -------
    nn.Module (BiLSTMLarge)

    Examples
    --------
    >>> from src.models.model_large import build, DEFAULT_CFG
    >>> model = build({**DEFAULT_CFG, "vocab_size": 10_000})
    """
    merged = {**DEFAULT_CFG, **config}
    return BiLSTMLarge(
        vocab_size=int(merged["vocab_size"]),
        embedding_dim=int(merged.get("embedding_dim", 256)),
        hidden_dim=int(merged.get("hidden_dim", 256)),
        num_layers=int(merged.get("num_layers", 2)),
        intermediate_dim=int(merged.get("intermediate_dim", 128)),
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
