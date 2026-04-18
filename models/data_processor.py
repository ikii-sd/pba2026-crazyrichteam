"""
data_processor.py
Utility untuk:
1) load data dari CSV
2) clean text sederhana
3) encode label sentiment & emotion
4) split train/val/test secara stratified (80/10/10)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ProcessedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    sentiment_label2id: Dict[str, int]
    sentiment_id2label: Dict[int, str]
    emotion_label2id: Dict[str, int]
    emotion_id2label: Dict[int, str]
    metadata: Dict


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load dataset CSV ke DataFrame.

    Args:
        csv_path: path ke file CSV

    Returns:
        pd.DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset kosong.")

    return df


def clean_text(text: str) -> str:
    """
    Cleaning teks sederhana:
    - ubah ke string
    - lowercase
    - hapus URL
    - hapus karakter non-alfanumerik (kecuali spasi)
    - rapikan spasi

    Args:
        text: input text

    Returns:
        str: text bersih
    """
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_label_mapping(series: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build mapping label <-> id dari sebuah kolom label.
    Urutan label dibuat stabil dengan sorting alfabetis.

    Args:
        series: kolom label (string)

    Returns:
        (label2id, id2label)
    """
    labels = sorted(series.astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def process_data(
    csv_path: str | Path,
    text_column: str,
    sentiment_column: str,
    emotion_column: str,
    random_state: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> ProcessedData:
    """
    Proses end-to-end:
    - load data
    - rename kolom -> text, sentiment, emotion
    - drop missing & text kosong
    - clean text
    - encode label
    - split train/val/test secara stratified

    Stratifikasi dilakukan pada gabungan label:
        stratify_key = sentiment + "__" + emotion

    Args:
        csv_path: path dataset CSV
        text_column: nama kolom text asli
        sentiment_column: nama kolom sentiment asli
        emotion_column: nama kolom emotion asli
        random_state: seed untuk split
        train_size: default 0.8
        val_size: default 0.1
        test_size: default 0.1

    Returns:
        ProcessedData dataclass
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size harus = 1.0")

    # 1) load
    df = load_data(csv_path)

    # 2) validasi kolom
    required_cols = [text_column, sentiment_column, emotion_column]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Kolom tidak ditemukan di dataset: {missing_cols}")

    # 3) rename -> standar
    df = df.rename(
        columns={
            text_column: "text",
            sentiment_column: "sentiment",
            emotion_column: "emotion",
        }
    )

    # 4) pilih kolom inti
    df = df[["text", "sentiment", "emotion"]].copy()

    # 5) drop missing
    df = df.dropna(subset=["text", "sentiment", "emotion"]).copy()

    # 6) clean text + drop kosong
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()

    if len(df) < 20:
        raise ValueError("Data tersisa terlalu sedikit setelah cleaning.")

    # 7) pastikan label string
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()

    # 8) mapping label
    sentiment_label2id, sentiment_id2label = _build_label_mapping(df["sentiment"])
    emotion_label2id, emotion_id2label = _build_label_mapping(df["emotion"])

    df["sentiment_id"] = df["sentiment"].map(sentiment_label2id)
    df["emotion_id"] = df["emotion"].map(emotion_label2id)

    # 9) stratify key gabungan
    df["stratify_key"] = df["sentiment"].astype(str) + "__" + df["emotion"].astype(str)

    # Cek apakah semua kombinasi punya >=2 sampel agar aman untuk stratified split
    key_counts = df["stratify_key"].value_counts()
    rare_keys = key_counts[key_counts < 2]
    if not rare_keys.empty:
        # fallback: stratify hanya sentiment jika kombinasi terlalu jarang
        stratify_col = df["sentiment"]
    else:
        stratify_col = df["stratify_key"]

    # 10) split train vs temp (val+test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # 11) split temp -> val/test
    # proporsi di temp
    val_ratio_in_temp = val_size / (val_size + test_size)

    # tentukan stratify untuk temp split
    temp_key_counts = temp_df["stratify_key"].value_counts()
    temp_rare_keys = temp_key_counts[temp_key_counts < 2]
    if not temp_rare_keys.empty:
        temp_stratify_col = temp_df["sentiment"]
    else:
        temp_stratify_col = temp_df["stratify_key"]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=random_state,
        stratify=temp_stratify_col,
    )

    # 12) rapikan kolom output
    keep_cols = ["text", "sentiment", "emotion", "sentiment_id", "emotion_id"]
    train_df = train_df[keep_cols].reset_index(drop=True)
    val_df = val_df[keep_cols].reset_index(drop=True)
    test_df = test_df[keep_cols].reset_index(drop=True)

    metadata = {
        "total_samples_after_cleaning": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "sentiment_num_classes": int(len(sentiment_label2id)),
        "emotion_num_classes": int(len(emotion_label2id)),
        "sentiment_distribution": train_df["sentiment"].value_counts().to_dict(),
        "emotion_distribution": train_df["emotion"].value_counts().to_dict(),
    }

    return ProcessedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sentiment_label2id=sentiment_label2id,
        sentiment_id2label=sentiment_id2label,
        emotion_label2id=emotion_label2id,
        emotion_id2label=emotion_id2label,
        metadata=metadata,
    )


def save_label_encoders(
    output_path: str | Path,
    sentiment_label2id: Dict[str, int],
    emotion_label2id: Dict[str, int],
) -> None:
    """
    Simpan label encoder ke JSON.

    Args:
        output_path: path file JSON
        sentiment_label2id: mapping sentiment
        emotion_label2id: mapping emotion
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sentiment_label2id": sentiment_label2id,
        "emotion_label2id": emotion_label2id,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Contoh penggunaan sederhana (silakan sesuaikan path/kolom jika berbeda)
    default_csv = (
        Path(__file__).resolve().parents[2] / "data" / "clean" / "cleaned_dataset.csv"
    )

    result = process_data(
        csv_path=default_csv,
        text_column="clean_review",
        sentiment_column="Sentiment",
        emotion_column="Emotion",
        random_state=42,
    )

    print("Data processed successfully!")
    print(
        "Train/Val/Test:", len(result.train_df), len(result.val_df), len(result.test_df)
    )
    print("Sentiment classes:", result.sentiment_label2id)
    print("Emotion classes:", result.emotion_label2id)
