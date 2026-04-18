"""
vocabulary_builder.py
Utility class Vocabulary untuk:
- membangun vocab dari kumpulan teks
- konversi teks -> index token
- simpan / muat vocab ke JSON

Spesifikasi:
- <PAD> = 0
- <UNK> = 1
- max vocab default = 10_000
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(
        self,
        max_vocab_size: int = 10_000,
        lowercase: bool = True,
    ) -> None:
        if max_vocab_size < 2:
            raise ValueError("max_vocab_size minimal 2 (untuk <PAD> dan <UNK>).")

        self.max_vocab_size = max_vocab_size
        self.lowercase = lowercase

        # mapping utama
        self.token2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2token = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

        self._is_built = False

    def __len__(self) -> int:
        return len(self.token2idx)

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        if text is None:
            return []
        text = str(text).strip()
        if not text:
            return []
        return text.split()

    def build_from_texts(self, texts: Iterable[str]) -> None:
        """
        Bangun vocabulary dari iterable teks.

        Rules:
        - tokenisasi sederhana: split spasi
        - frekuensi tertinggi diprioritaskan
        - ukuran vocab maksimal = max_vocab_size
        - index 0 dan 1 disediakan untuk PAD/UNK
        """
        counter = Counter()

        for text in texts:
            if text is None:
                continue
            t = str(text)
            if self.lowercase:
                t = t.lower()
            tokens = self._simple_tokenize(t)
            if tokens:
                counter.update(tokens)

        # sisakan slot untuk PAD + UNK
        available_slots = self.max_vocab_size - 2
        if available_slots <= 0:
            # tetap hanya special tokens
            self._is_built = True
            return

        # urut berdasarkan frekuensi desc, lalu token asc (stabil)
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        top_tokens = [tok for tok, _ in sorted_items[:available_slots]]

        # reset mapping ke special tokens dulu
        self.token2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2token = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

        for token in top_tokens:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        self._is_built = True

    def token_to_idx(self, token: str) -> int:
        """
        Konversi 1 token ke index.
        Token OOV -> UNK_IDX
        """
        if token is None:
            return self.UNK_IDX
        tok = str(token)
        if self.lowercase:
            tok = tok.lower()
        return self.token2idx.get(tok, self.UNK_IDX)

    def text_to_indices(self, text: str, max_seq_len: int | None = None) -> List[int]:
        """
        Konversi teks menjadi list index token.

        Jika max_seq_len diberikan:
        - truncate jika lebih panjang
        - pad dengan PAD_IDX jika lebih pendek
        """
        if text is None:
            tokens = []
        else:
            t = str(text)
            if self.lowercase:
                t = t.lower()
            tokens = self._simple_tokenize(t)

        indices = [self.token_to_idx(tok) for tok in tokens]

        if max_seq_len is not None:
            if max_seq_len <= 0:
                raise ValueError("max_seq_len harus > 0.")
            if len(indices) > max_seq_len:
                indices = indices[:max_seq_len]
            elif len(indices) < max_seq_len:
                indices = indices + [self.PAD_IDX] * (max_seq_len - len(indices))

        return indices

    def save(self, path: str | Path) -> None:
        """
        Simpan vocabulary ke file JSON.
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

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """
        Muat vocabulary dari file JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File vocabulary tidak ditemukan: {path}")

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        vocab = cls(
            max_vocab_size=int(payload.get("max_vocab_size", 10_000)),
            lowercase=bool(payload.get("lowercase", True)),
        )

        token2idx = payload.get("token2idx", {})
        if not token2idx:
            raise ValueError("File vocabulary tidak valid: token2idx kosong.")

        # pastikan int index
        cleaned_token2idx = {}
        for token, idx in token2idx.items():
            cleaned_token2idx[str(token)] = int(idx)

        # validasi special token
        if cleaned_token2idx.get(cls.PAD_TOKEN) != cls.PAD_IDX:
            raise ValueError("Vocab tidak valid: <PAD> harus memiliki index 0.")
        if cleaned_token2idx.get(cls.UNK_TOKEN) != cls.UNK_IDX:
            raise ValueError("Vocab tidak valid: <UNK> harus memiliki index 1.")

        vocab.token2idx = cleaned_token2idx
        vocab.idx2token = {idx: tok for tok, idx in vocab.token2idx.items()}
        vocab._is_built = True
        return vocab


if __name__ == "__main__":
    # contoh penggunaan
    texts = [
        "barang bagus dan pengiriman cepat",
        "produk jelek tidak sesuai deskripsi",
        "mantap banget kualitas bagus",
    ]

    vocab = Vocabulary(max_vocab_size=10_000, lowercase=True)
    vocab.build_from_texts(texts)

    print("Vocab size:", len(vocab))
    print(
        "Indices:",
        vocab.text_to_indices("barang kualitas tidak dikenal", max_seq_len=8),
    )
