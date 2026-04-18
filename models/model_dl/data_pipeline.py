"""
data_pipeline.py
================
Setup PyTorch DataLoaders untuk training.

Note: Adapted to match actual Peran 1 API:
  - Uses ProcessedData dataclass from data_processor.py
  - Uses actual Vocabulary API (text_to_indices with max_seq_len kwarg)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from models.data_processor import ProcessedData
from models.vocabulary_builder import Vocabulary
from models.config_simplified import BATCH_SIZE, MAX_SEQ_LEN


class SentimentEmotionDataset(Dataset):
    """Custom PyTorch Dataset for sentiment & emotion classification."""

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary):
        """
        Args:
            df: DataFrame with columns [text, sentiment_id, emotion_id]
            vocab: Built Vocabulary object
        """
        self.data = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = MAX_SEQ_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        sentiment_label = int(row["sentiment_id"])
        emotion_label = int(row["emotion_id"])

        # Convert text to padded/truncated indices
        indices = self.vocab.text_to_indices(text, max_seq_len=self.max_len)

        # Actual (non-padding) length
        tokens = text.split()
        length = min(len(tokens), self.max_len)
        length = max(length, 1)  # at least 1

        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
        }


def create_dataloaders(
    processed: ProcessedData,
    vocab: Vocabulary,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Create train/val/test DataLoaders from ProcessedData.

    Args:
        processed: ProcessedData dataclass from data_processor.process_data()
        vocab: Built Vocabulary object
        batch_size: samples per batch

    Returns:
        dict with keys "train", "val", "test"
    """
    splits = {
        "train": processed.train_df,
        "val": processed.val_df,
        "test": processed.test_df,
    }

    dataloaders = {}
    for split_name, split_df in splits.items():
        dataset = SentimentEmotionDataset(split_df, vocab)
        shuffle = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # 0 is safe on Windows
        )
        print(f"  ✅ {split_name.upper()} loader: {len(dataset)} samples, "
              f"{len(dataloaders[split_name])} batches")

    return dataloaders


# ── smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from models.data_processor import process_data
    from models.config_simplified import (
        DATASET_PATH, TEXT_COLUMN, SENTIMENT_LABEL_COLUMN,
        EMOTION_LABEL_COLUMN, RANDOM_SEED, VOCAB_PATH,
    )

    processed = process_data(
        csv_path=DATASET_PATH,
        text_column=TEXT_COLUMN,
        sentiment_column=SENTIMENT_LABEL_COLUMN,
        emotion_column=EMOTION_LABEL_COLUMN,
        random_state=RANDOM_SEED,
    )

    vocab = Vocabulary()
    all_texts = pd.concat(
        [processed.train_df, processed.val_df, processed.test_df]
    )["text"].tolist()
    vocab.build_from_texts(all_texts)

    loaders = create_dataloaders(processed, vocab)

    batch = next(iter(loaders["train"]))
    print(f"\nBatch keys:       {list(batch.keys())}")
    print(f"input_ids shape:  {batch['input_ids'].shape}")
    print(f"sentiment_label:  {batch['sentiment_label'][:8]}")
