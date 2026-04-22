"""
src/dataloader.py
=================
Modul data pipeline terpusat untuk proyek NLP — Analisis Sentimen & Emosi
Ulasan E-Commerce Bahasa Indonesia (PRDECT-ID).

Memindahkan SEMUA logika data (baca file, split, preprocessing, tokenisasi,
PyTorch Dataset/DataLoader) ke satu tempat sehingga notebook hanya memanggil
fungsi saja tanpa kode duplikat.

Pipeline utama:
    1. load_raw_data()          → baca CSV mentah / bersih ke DataFrame
    2. preprocess()             → clean text + validasi kolom + encode label
    3. train_val_test_split()   → split stratified (train / val / test)
    4. build_vocab()            → bangun / muat Vocabulary dari split train
    5. build_dataset()          → buat PyTorch Dataset dari satu split
    6. build_dataloader()       → buat PyTorch DataLoader dari Dataset
    7. build_full_pipeline()    → orchestrasi satu-panggil semua langkah di atas

Cara pemakaian cepat (notebook):
    from src.dataloader import build_full_pipeline

    pipeline = build_full_pipeline(
        csv_path="data/clean/cleaned_dataset.csv",
        text_column="clean_review",
        sentiment_column="Sentiment",
        emotion_column="Emotion",
    )
    train_loader = pipeline["loaders"]["train"]
    vocab        = pipeline["vocab"]
    meta         = pipeline["metadata"]

Test mandiri (CLI):
    python src/dataloader.py

Author  : Crazy Rich Team — PBA 2026
Dataset : PRDECT-ID
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── PyTorch (opsional — hanya dibutuhkan untuk Dataset/DataLoader) ────────────
try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Sklearn untuk stratified split ────────────────────────────────────────────
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════════════════
# A. KONFIGURASI DEFAULT
# ══════════════════════════════════════════════════════════════════════════════

# Nilai-nilai ini bisa dioverride lewat parameter fungsi. Tidak ada magic number.

_DEFAULT_TEXT_COL: str = "clean_review"
_DEFAULT_SENTIMENT_COL: str = "Sentiment"
_DEFAULT_EMOTION_COL: str = "Emotion"

_DEFAULT_TRAIN_SIZE: float = 0.8
_DEFAULT_VAL_SIZE: float = 0.1
_DEFAULT_TEST_SIZE: float = 0.1

_DEFAULT_RANDOM_SEED: int = 42
_DEFAULT_BATCH_SIZE: int = 64
_DEFAULT_MAX_SEQ_LEN: int = 64
_DEFAULT_MAX_VOCAB_SIZE: int = 10_000
_DEFAULT_NUM_WORKERS: int = 0  # 0 aman di Windows
_DEFAULT_MIN_TOKEN_LEN: int = 2


# ══════════════════════════════════════════════════════════════════════════════
# B. DATACLASS OUTPUT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SplitData:
    """Wadah untuk tiga split DataFrame hasil pemisahan dataset."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    sentiment_label2id: Dict[str, int] = field(default_factory=dict)
    sentiment_id2label: Dict[int, str] = field(default_factory=dict)
    emotion_label2id: Dict[str, int] = field(default_factory=dict)
    emotion_id2label: Dict[int, str] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# C. VOCABULARY (inline — tidak perlu import tambahan)
# ══════════════════════════════════════════════════════════════════════════════


class Vocabulary:
    """
    Kelas Vocabulary untuk konversi token ↔ indeks integer.

    Spesifikasi:
    - <PAD> = indeks 0
    - <UNK> = indeks 1
    - Ukuran vocab maksimal = max_vocab_size
    - Frekuensi token tertinggi diprioritaskan
    """

    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_IDX: int = 0
    UNK_IDX: int = 1

    def __init__(
        self,
        max_vocab_size: int = _DEFAULT_MAX_VOCAB_SIZE,
        lowercase: bool = True,
    ) -> None:
        """
        Inisialisasi Vocabulary kosong.

        Parameters
        ----------
        max_vocab_size : int
            Jumlah maksimum token dalam vocab (termasuk special tokens).
            Default: 10_000.
        lowercase : bool
            Jika True, semua token dikonversi ke lowercase. Default: True.
        """
        if max_vocab_size < 2:
            raise ValueError("max_vocab_size minimal 2 (untuk <PAD> dan <UNK>).")
        self.max_vocab_size = max_vocab_size
        self.lowercase = lowercase
        self.token2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2token: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        self._is_built: bool = False

    def __len__(self) -> int:
        return len(self.token2idx)

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={len(self)}, max={self.max_vocab_size}, "
            f"built={self._is_built})"
        )

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenisasi sederhana: split spasi (teks sudah pre-cleaned)."""
        if text is None:
            return []
        return str(text).strip().split()

    # ── public API ────────────────────────────────────────────────────────

    def build_from_texts(self, texts: Iterable[str]) -> None:
        """
        Bangun vocabulary dari iterable teks.

        Parameters
        ----------
        texts : Iterable[str]
            Koleksi teks (list, generator, pd.Series, dll.)
        """
        counter: Counter = Counter()
        for text in texts:
            if text is None:
                continue
            t = str(text).lower() if self.lowercase else str(text)
            counter.update(self._tokenize(t))

        available_slots = self.max_vocab_size - 2
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        top_tokens = [tok for tok, _ in sorted_items[:available_slots]]

        # reset lalu isi ulang
        self.token2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2token = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        for token in top_tokens:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        self._is_built = True

    def token_to_idx(self, token: str) -> int:
        """
        Konversi satu token ke indeks integer.

        Parameters
        ----------
        token : str

        Returns
        -------
        int
            Indeks token; UNK_IDX jika token tidak ada di vocab.
        """
        if token is None:
            return self.UNK_IDX
        tok = str(token).lower() if self.lowercase else str(token)
        return self.token2idx.get(tok, self.UNK_IDX)

    def text_to_indices(
        self,
        text: str,
        max_seq_len: Optional[int] = None,
    ) -> List[int]:
        """
        Konversi teks ke list indeks integer dengan optional padding/truncation.

        Parameters
        ----------
        text : str
        max_seq_len : int, optional
            Jika diberikan, hasil di-truncate atau di-pad ke panjang ini.

        Returns
        -------
        List[int]
        """
        t = str(text).lower() if self.lowercase else str(text)
        tokens = self._tokenize(t)
        indices = [self.token_to_idx(tok) for tok in tokens]

        if max_seq_len is not None:
            if max_seq_len <= 0:
                raise ValueError("max_seq_len harus > 0.")
            if len(indices) > max_seq_len:
                indices = indices[:max_seq_len]
            else:
                indices += [self.PAD_IDX] * (max_seq_len - len(indices))

        return indices

    def save(self, path: str | Path) -> None:
        """
        Simpan vocabulary ke file JSON.

        Parameters
        ----------
        path : str | Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_vocab_size": self.max_vocab_size,
            "lowercase": self.lowercase,
            "token2idx": self.token2idx,
            "special_tokens": {
                "PAD_TOKEN": self.PAD_TOKEN,
                "UNK_TOKEN": self.UNK_TOKEN,
                "PAD_IDX": self.PAD_IDX,
                "UNK_IDX": self.UNK_IDX,
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  [Vocabulary] Tersimpan: {path}  (size={len(self)})")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """
        Muat vocabulary dari file JSON.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        Vocabulary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File vocabulary tidak ditemukan: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        vocab = cls(
            max_vocab_size=int(payload.get("max_vocab_size", _DEFAULT_MAX_VOCAB_SIZE)),
            lowercase=bool(payload.get("lowercase", True)),
        )
        token2idx = {str(k): int(v) for k, v in payload.get("token2idx", {}).items()}
        if not token2idx:
            raise ValueError("File vocabulary tidak valid: token2idx kosong.")
        if token2idx.get(cls.PAD_TOKEN) != cls.PAD_IDX:
            raise ValueError("Vocab tidak valid: <PAD> harus indeks 0.")
        if token2idx.get(cls.UNK_TOKEN) != cls.UNK_IDX:
            raise ValueError("Vocab tidak valid: <UNK> harus indeks 1.")

        vocab.token2idx = token2idx
        vocab.idx2token = {idx: tok for tok, idx in token2idx.items()}
        vocab._is_built = True
        print(f"  [Vocabulary] Dimuat dari: {path}  (size={len(vocab)})")
        return vocab


# ══════════════════════════════════════════════════════════════════════════════
# D. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════


if _TORCH_AVAILABLE:

    class SentimentEmotionDataset(Dataset):
        """
        PyTorch Dataset untuk klasifikasi sentimen & emosi multi-output.

        Setiap item mengembalikan dict:
            {
              "input_ids"       : LongTensor [max_seq_len],
              "length"          : LongTensor (panjang asli sebelum padding),
              "sentiment_label" : LongTensor (indeks kelas sentimen),
              "emotion_label"   : LongTensor (indeks kelas emosi),
            }
        """

        def __init__(
            self,
            df: pd.DataFrame,
            vocab: Vocabulary,
            max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        ) -> None:
            """
            Parameters
            ----------
            df : pd.DataFrame
                DataFrame dengan kolom: text, sentiment_id, emotion_id.
            vocab : Vocabulary
                Vocabulary yang sudah di-build.
            max_seq_len : int
                Panjang sequence setelah padding/truncation. Default: 64.
            """
            self.data = df.reset_index(drop=True)
            self.vocab = vocab
            self.max_seq_len = max_seq_len

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict:
            row = self.data.iloc[idx]
            text = str(row["text"])
            sentiment_label = int(row["sentiment_id"])
            emotion_label = int(row["emotion_id"])

            indices = self.vocab.text_to_indices(text, max_seq_len=self.max_seq_len)
            tokens = text.split()
            length = max(min(len(tokens), self.max_seq_len), 1)

            return {
                "input_ids": torch.tensor(indices, dtype=torch.long),
                "length": torch.tensor(length, dtype=torch.long),
                "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
                "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            }


# ══════════════════════════════════════════════════════════════════════════════
# E. FUNGSI UTAMA DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int = _DEFAULT_RANDOM_SEED) -> None:
    """
    Set random seed untuk reproduktifitas di Python, NumPy, dan PyTorch.

    Parameters
    ----------
    seed : int
        Nilai seed. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    print(f"  [seed] Random seed diset ke {seed}")


def load_raw_data(
    csv_path: str | Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Baca file CSV ke pandas DataFrame.

    Parameters
    ----------
    csv_path : str | Path
        Path ke file CSV (data mentah atau yang sudah dibersihkan).
    encoding : str
        Encoding file. Default: "utf-8".

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan.
    ValueError
        Jika DataFrame kosong.

    Examples
    --------
    >>> df = load_raw_data("data/PRDECT-ID Dataset.csv")
    >>> print(df.shape)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[load_raw_data] Dataset tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    if df.empty:
        raise ValueError(f"[load_raw_data] Dataset kosong: {csv_path}")

    print(
        f"  [load_raw_data] ✅ Loaded: {csv_path.name}  "
        f"({len(df):,} baris × {len(df.columns)} kolom)"
    )
    return df


def preprocess(
    df: pd.DataFrame,
    text_column: str = _DEFAULT_TEXT_COL,
    sentiment_column: str = _DEFAULT_SENTIMENT_COL,
    emotion_column: str = _DEFAULT_EMOTION_COL,
    do_basic_clean: bool = False,
    min_text_length: int = 1,
) -> pd.DataFrame:
    """
    Validasi kolom, optional basic cleaning, encode label numerik.

    Jika `do_basic_clean=True` dan kolom 'text' belum bersih, maka akan
    dilakukan pembersihan ringan (lowercase + hapus karakter non-alfanumerik).
    Untuk pipeline full cleaning, gunakan `src.preprocessing.batch_clean()`
    sebelum memanggil fungsi ini.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil `load_raw_data()`.
    text_column : str
        Nama kolom teks. Default: "clean_review".
    sentiment_column : str
        Nama kolom label sentimen. Default: "Sentiment".
    emotion_column : str
        Nama kolom label emosi. Default: "Emotion".
    do_basic_clean : bool
        Jika True, terapkan basic cleaning ringan pada kolom teks. Default: False.
    min_text_length : int
        Jumlah karakter minimum teks agar tidak di-drop. Default: 1.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom standar:
        [text, sentiment, emotion, sentiment_id, emotion_id]

    Raises
    ------
    KeyError
        Jika kolom yang dibutuhkan tidak ada.
    ValueError
        Jika data tersisa terlalu sedikit setelah cleaning.

    Examples
    --------
    >>> df = load_raw_data("data/clean/cleaned_dataset.csv")
    >>> df_clean = preprocess(df, text_column="clean_review")
    """
    # ── Validasi kolom ────────────────────────────────────────────────────
    required = [text_column, sentiment_column, emotion_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"[preprocess] Kolom tidak ditemukan: {missing}.\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    # ── Rename ke kolom standar ───────────────────────────────────────────
    df = df.rename(
        columns={
            text_column: "text",
            sentiment_column: "sentiment",
            emotion_column: "emotion",
        }
    )
    df = df[["text", "sentiment", "emotion"]].copy()

    # ── Drop missing ──────────────────────────────────────────────────────
    original_len = len(df)
    df = df.dropna(subset=["text", "sentiment", "emotion"]).copy()

    # ── Optional basic cleaning ───────────────────────────────────────────
    if do_basic_clean:
        _re_url = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        _re_nonalpha = re.compile(r"[^a-z0-9\s]")
        _re_spaces = re.compile(r"\s+")

        def _basic_clean(text: str) -> str:
            if not isinstance(text, str):
                text = str(text)
            text = text.lower()
            text = _re_url.sub(" ", text)
            text = _re_nonalpha.sub(" ", text)
            text = _re_spaces.sub(" ", text).strip()
            return text

        df["text"] = df["text"].apply(_basic_clean)

    # ── Pastikan string ───────────────────────────────────────────────────
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= min_text_length].copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()

    if len(df) < 20:
        raise ValueError(
            f"[preprocess] Data sangat sedikit setelah cleaning: {len(df)} baris. "
            f"Periksa konfigurasi kolom dan path dataset."
        )

    # ── Encode label ─────────────────────────────────────────────────────
    def _make_mapping(series: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
        labels = sorted(series.unique().tolist())
        l2i = {lbl: i for i, lbl in enumerate(labels)}
        i2l = {i: lbl for lbl, i in l2i.items()}
        return l2i, i2l

    sentiment_l2i, sentiment_i2l = _make_mapping(df["sentiment"])
    emotion_l2i, emotion_i2l = _make_mapping(df["emotion"])

    df["sentiment_id"] = df["sentiment"].map(sentiment_l2i)
    df["emotion_id"] = df["emotion"].map(emotion_l2i)

    print(
        f"  [preprocess] ✅  {len(df):,} / {original_len:,} baris valid  |  "
        f"Sentimen: {list(sentiment_l2i.keys())}  |  "
        f"Emosi: {list(emotion_l2i.keys())}"
    )
    return df, sentiment_l2i, sentiment_i2l, emotion_l2i, emotion_i2l


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = _DEFAULT_TRAIN_SIZE,
    val_size: float = _DEFAULT_VAL_SIZE,
    test_size: float = _DEFAULT_TEST_SIZE,
    random_seed: int = _DEFAULT_RANDOM_SEED,
    stratify_column: Optional[str] = "sentiment",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pisahkan DataFrame menjadi train / val / test secara stratified.

    Stratifikasi dilakukan pada kolom sentimen (default). Jika konflik
    kelas sangat jarang ditemukan, fallback ke sentimen saja.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pra-diproses (wajib punya kolom 'sentiment' atau
        kolom yang ditentukan di `stratify_column`).
    train_size : float
        Proporsi data training. Default: 0.8.
    val_size : float
        Proporsi data validasi. Default: 0.1.
    test_size : float
        Proporsi data test. Default: 0.1.
    random_seed : int
        Seed untuk reproduktifitas. Default: 42.
    stratify_column : str, optional
        Nama kolom yang digunakan untuk stratifikasi. Default: "sentiment".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        Jika total proporsi tidak sama dengan 1.0.

    Examples
    --------
    >>> train_df, val_df, test_df = train_val_test_split(df)
    >>> print(len(train_df), len(val_df), len(test_df))
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"[train_val_test_split] train_size + val_size + test_size harus = 1.0, "
            f"dapat {total}."
        )

    stratify_col = df[stratify_column] if stratify_column in df.columns else None

    # split train vs temp (val + test)
    temp_ratio = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_col,
    )

    # split temp → val / test
    val_ratio_in_temp = val_size / (val_size + test_size)
    temp_stratify = (
        temp_df[stratify_column] if stratify_column in temp_df.columns else None
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=random_seed,
        stratify=temp_stratify,
    )

    keep_cols = [c for c in ["text", "sentiment", "emotion", "sentiment_id", "emotion_id"] if c in df.columns]
    train_df = train_df[keep_cols].reset_index(drop=True)
    val_df = val_df[keep_cols].reset_index(drop=True)
    test_df = test_df[keep_cols].reset_index(drop=True)

    print(
        f"  [split] ✅  Train: {len(train_df):,}  |  "
        f"Val: {len(val_df):,}  |  Test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


def build_vocab(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    text_column: str = "text",
    max_vocab_size: int = _DEFAULT_MAX_VOCAB_SIZE,
    lowercase: bool = True,
    save_path: Optional[str | Path] = None,
) -> Vocabulary:
    """
    Bangun Vocabulary dari teks pada split (direkomendasikan: hanya dari train).

    Parameters
    ----------
    train_df : pd.DataFrame
        Split training — vocab dibangun dari sini.
    val_df, test_df : pd.DataFrame, optional
        Jika diberikan, teks val dan test juga dimasukkan ke corpus vocab.
        **Catatan:** Untuk menghindari data leakage pada evaluation, biarkan
        None (default) agar vocab hanya dari train.
    text_column : str
        Nama kolom teks. Default: "text".
    max_vocab_size : int
        Ukuran vocab maksimum. Default: 10_000.
    lowercase : bool
        Jika True, token dikonversi ke lowercase. Default: True.
    save_path : str | Path, optional
        Jika diberikan, simpan vocab ke file JSON.

    Returns
    -------
    Vocabulary

    Examples
    --------
    >>> vocab = build_vocab(train_df, save_path="models/model_dl/artifacts/vocab.json")
    """
    corpus: List[str] = train_df[text_column].tolist()
    if val_df is not None:
        corpus += val_df[text_column].tolist()
    if test_df is not None:
        corpus += test_df[text_column].tolist()

    vocab = Vocabulary(max_vocab_size=max_vocab_size, lowercase=lowercase)
    vocab.build_from_texts(corpus)
    print(f"  [build_vocab] ✅  Vocab dibangun: {len(vocab):,} token")

    if save_path is not None:
        vocab.save(save_path)

    return vocab


def build_dataset(
    df: pd.DataFrame,
    vocab: Vocabulary,
    max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
) -> "SentimentEmotionDataset":
    """
    Buat PyTorch Dataset dari satu split DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Split DataFrame (train / val / test).
    vocab : Vocabulary
        Vocabulary yang telah di-build.
    max_seq_len : int
        Panjang sequence target (padding / truncation). Default: 64.

    Returns
    -------
    SentimentEmotionDataset

    Raises
    ------
    ImportError
        Jika PyTorch tidak terinstall.

    Examples
    --------
    >>> train_dataset = build_dataset(train_df, vocab)
    >>> print(len(train_dataset))
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "[build_dataset] PyTorch tidak terinstall. "
            "Install dengan: pip install torch"
        )
    return SentimentEmotionDataset(df=df, vocab=vocab, max_seq_len=max_seq_len)


def build_dataloader(
    dataset: "SentimentEmotionDataset",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    pin_memory: bool = False,
) -> "DataLoader":
    """
    Buat PyTorch DataLoader dari Dataset.

    Parameters
    ----------
    dataset : SentimentEmotionDataset
    batch_size : int
        Jumlah sampel per batch. Default: 64.
    shuffle : bool
        Set True untuk training loader. Default: False.
    num_workers : int
        Worker untuk loading paralel. Default: 0 (aman di Windows).
    pin_memory : bool
        Aktifkan pin_memory untuk transfer GPU lebih cepat. Default: False.

    Returns
    -------
    DataLoader

    Examples
    --------
    >>> train_loader = build_dataloader(train_dataset, batch_size=64, shuffle=True)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("[build_dataloader] PyTorch tidak terinstall.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_full_pipeline(
    csv_path: str | Path,
    text_column: str = _DEFAULT_TEXT_COL,
    sentiment_column: str = _DEFAULT_SENTIMENT_COL,
    emotion_column: str = _DEFAULT_EMOTION_COL,
    train_size: float = _DEFAULT_TRAIN_SIZE,
    val_size: float = _DEFAULT_VAL_SIZE,
    test_size: float = _DEFAULT_TEST_SIZE,
    random_seed: int = _DEFAULT_RANDOM_SEED,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
    max_vocab_size: int = _DEFAULT_MAX_VOCAB_SIZE,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    do_basic_clean: bool = False,
    vocab_save_path: Optional[str | Path] = None,
    label_enc_save_path: Optional[str | Path] = None,
) -> dict:
    """
    Orchestrasi satu-panggil seluruh data pipeline:
    load → preprocess → split → build vocab → build dataset → build dataloader.

    Parameters
    ----------
    csv_path : str | Path
        Path ke file CSV (mentah atau sudah di-clean).
    text_column : str
        Nama kolom teks. Default: "clean_review".
    sentiment_column : str
        Nama kolom sentimen. Default: "Sentiment".
    emotion_column : str
        Nama kolom emosi. Default: "Emotion".
    train_size : float
        Proporsi training. Default: 0.8.
    val_size : float
        Proporsi validasi. Default: 0.1.
    test_size : float
        Proporsi test. Default: 0.1.
    random_seed : int
        Seed global. Default: 42.
    batch_size : int
        Ukuran batch untuk DataLoader. Default: 64.
    max_seq_len : int
        Panjang sequence setelah padding. Default: 64.
    max_vocab_size : int
        Ukuran vocab maksimum. Default: 10_000.
    num_workers : int
        Worker DataLoader. Default: 0.
    do_basic_clean : bool
        Jika True, terapkan basic cleaning. Default: False.
    vocab_save_path : str | Path, optional
        Jika diberikan, simpan vocab ke path ini.
    label_enc_save_path : str | Path, optional
        Jika diberikan, simpan label encoders ke JSON.

    Returns
    -------
    dict
        {
          "splits"  : SplitData,    # DataFrames train/val/test + label mappings
          "vocab"   : Vocabulary,
          "loaders" : {             # hanya tersedia jika PyTorch ada
              "train": DataLoader,
              "val"  : DataLoader,
              "test" : DataLoader,
          },
          "metadata": dict,
        }

    Examples
    --------
    >>> pipeline = build_full_pipeline(
    ...     csv_path="data/clean/cleaned_dataset.csv",
    ...     text_column="clean_review",
    ... )
    >>> train_loader = pipeline["loaders"]["train"]
    >>> vocab = pipeline["vocab"]
    """
    print("\n" + "=" * 60)
    print("  🚀  DATA PIPELINE — build_full_pipeline()")
    print("=" * 60)

    # 0. Set seed
    set_seed(random_seed)

    # 1. Load
    df_raw = load_raw_data(csv_path)

    # 2. Preprocess
    df, sent_l2i, sent_i2l, emo_l2i, emo_i2l = preprocess(
        df_raw,
        text_column=text_column,
        sentiment_column=sentiment_column,
        emotion_column=emotion_column,
        do_basic_clean=do_basic_clean,
    )

    # 3. Split
    train_df, val_df, test_df = train_val_test_split(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_seed=random_seed,
    )

    splits = SplitData(
        train=train_df,
        val=val_df,
        test=test_df,
        sentiment_label2id=sent_l2i,
        sentiment_id2label=sent_i2l,
        emotion_label2id=emo_l2i,
        emotion_id2label=emo_i2l,
        metadata={
            "total_after_cleaning": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "num_sentiment_classes": len(sent_l2i),
            "num_emotion_classes": len(emo_l2i),
            "sentiment_distribution": train_df["sentiment"].value_counts().to_dict(),
            "emotion_distribution": train_df["emotion"].value_counts().to_dict(),
        },
    )

    # 4. Vocab
    vocab = build_vocab(
        train_df=train_df,
        max_vocab_size=max_vocab_size,
        save_path=vocab_save_path,
    )

    # 5. Optional: simpan label encoders
    if label_enc_save_path is not None:
        lep = Path(label_enc_save_path)
        lep.parent.mkdir(parents=True, exist_ok=True)
        with lep.open("w", encoding="utf-8") as f:
            json.dump(
                {"sentiment_label2id": sent_l2i, "emotion_label2id": emo_l2i},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  [label_enc] Tersimpan: {lep}")

    # 6. Dataset & DataLoader (opsional — butuh PyTorch)
    loaders: dict = {}
    if _TORCH_AVAILABLE:
        pin = torch.cuda.is_available()
        for split_name, split_df, shuffle in [
            ("train", train_df, True),
            ("val", val_df, False),
            ("test", test_df, False),
        ]:
            ds = build_dataset(split_df, vocab, max_seq_len=max_seq_len)
            loaders[split_name] = build_dataloader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin,
            )
            print(
                f"  [loader] {split_name.upper():5s}  "
                f"{len(ds):>6,} sampel  |  "
                f"{len(loaders[split_name]):>4} batch"
            )
    else:
        print(
            "  [pipeline] ⚠️  PyTorch tidak tersedia — "
            "loaders tidak dibuat (hanya splits & vocab)."
        )

    print("=" * 60)
    print("  ✅  Pipeline selesai!\n")

    return {
        "splits": splits,
        "vocab": vocab,
        "loaders": loaders,
        "metadata": splits.metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# F. CLI / SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Cari path dataset secara otomatis relatif dari repo root
    _HERE = Path(__file__).resolve().parent        # src/
    _REPO = _HERE.parent                           # pba2026-crazyrichteam/

    _CLEAN_CSV = _REPO / "data" / "clean" / "cleaned_dataset.csv"
    _RAW_CSV = _REPO / "data" / "PRDECT-ID Dataset.csv"

    # Pilih dataset yang tersedia
    if _CLEAN_CSV.exists():
        _CSV = _CLEAN_CSV
        _TEXT_COL = "clean_review"
        _DO_CLEAN = False
        print(f"🔍  Menggunakan dataset bersih: {_CSV}")
    elif _RAW_CSV.exists():
        _CSV = _RAW_CSV
        _TEXT_COL = "Customer Review"
        _DO_CLEAN = True
        print(f"🔍  Menggunakan dataset mentah: {_CSV}")
    else:
        print("❌  Dataset tidak ditemukan. Periksa folder data/", file=sys.stderr)
        sys.exit(1)

    # Jalankan pipeline lengkap
    result = build_full_pipeline(
        csv_path=_CSV,
        text_column=_TEXT_COL,
        sentiment_column="Sentiment",
        emotion_column="Emotion",
        random_seed=42,
        batch_size=64,
        max_seq_len=64,
        max_vocab_size=10_000,
        do_basic_clean=_DO_CLEAN,
    )

    # Tampilkan ringkasan
    meta = result["metadata"]
    print("\n📊  Ringkasan Pipeline:")
    print(f"  Total sampel valid : {meta['total_after_cleaning']:,}")
    print(f"  Train / Val / Test : {meta['train_samples']:,} / {meta['val_samples']:,} / {meta['test_samples']:,}")
    print(f"  Kelas sentimen     : {result['splits'].sentiment_label2id}")
    print(f"  Kelas emosi        : {result['splits'].emotion_label2id}")
    print(f"  Ukuran vocab       : {len(result['vocab']):,}")

    if result["loaders"]:
        batch = next(iter(result["loaders"]["train"]))
        print(f"\n  Batch keys         : {list(batch.keys())}")
        print(f"  input_ids shape    : {tuple(batch['input_ids'].shape)}")
        print(f"  sentiment_label    : {batch['sentiment_label'][:8]}")
        print(f"  emotion_label      : {batch['emotion_label'][:8]}")

    print("\n✅  src/dataloader.py smoke test selesai!\n")
