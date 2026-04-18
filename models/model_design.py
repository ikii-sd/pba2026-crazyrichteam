"""
model_design.py
Implementasi model ringan multi-output BiLSTM untuk:
1) Sentiment classification
2) Emotion classification
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleBiLSTM(nn.Module):
    """
    Lightweight multi-output BiLSTM:
    Embedding -> BiLSTM -> Shared Dense -> 2 Output Heads
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.4,
        pad_idx: int = 0,
        num_sentiment_classes: int = 2,
        num_emotion_classes: int = 5,
    ) -> None:
        super().__init__()

        # Jika num_layers=1, dropout internal LSTM diabaikan oleh PyTorch
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

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

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Tensor shape [batch_size, seq_len]

        Returns:
            sentiment_logits: [batch_size, num_sentiment_classes]
            emotion_logits:   [batch_size, num_emotion_classes]
        """
        # [B, L] -> [B, L, E]
        embedded = self.embedding(input_ids)

        # outputs: [B, L, 2H]
        # hidden:  [2*num_layers, B, H]
        _, (hidden, _) = self.bilstm(embedded)

        # Ambil hidden state layer terakhir untuk arah forward & backward
        # hidden[-2] = forward terakhir, hidden[-1] = backward terakhir
        forward_last = hidden[-2]  # [B, H]
        backward_last = hidden[-1]  # [B, H]

        # [B, 2H]
        features = torch.cat((forward_last, backward_last), dim=1)

        # Shared projection
        x = self.dropout(features)
        x = self.shared_fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Multi-output heads
        sentiment_logits = self.sentiment_head(x)
        emotion_logits = self.emotion_head(x)

        return sentiment_logits, emotion_logits


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Hitung total parameter model.

    Args:
        model: PyTorch model
        trainable_only: jika True, hanya hitung parameter yang requires_grad=True

    Returns:
        int: jumlah parameter
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Smoke test singkat
    model = SimpleBiLSTM()
    total_params = count_parameters(model)

    print("Model created!")
    print(f"Total parameters: {total_params:,}")

    # Dummy forward pass
    batch_size, seq_len = 4, 64
    dummy_input = torch.randint(0, 10_000, (batch_size, seq_len))
    sent_logits, emo_logits = model(dummy_input)

    print(f"Sentiment logits shape: {tuple(sent_logits.shape)}")
    print(f"Emotion logits shape: {tuple(emo_logits.shape)}")
