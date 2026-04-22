"""
src/
====
Package modul Python untuk proyek NLP — Analisis Sentimen & Emosi
Ulasan E-Commerce Bahasa Indonesia (PRDECT-ID).

Modul tersedia:
    - preprocessing : fungsi clean_text(), batch_clean() — 14-step NLP pipeline
    - dataloader    : data pipeline lengkap (load, split, vocab, Dataset, DataLoader)

Contoh penggunaan:
    from src.preprocessing import clean_text, batch_clean
    from src.dataloader import build_full_pipeline, load_raw_data
"""

from src.preprocessing import clean_text, batch_clean, get_stopwords, get_stemmer
from src.dataloader import (
    load_raw_data,
    preprocess,
    train_val_test_split,
    build_vocab,
    build_dataset,
    build_dataloader,
    build_full_pipeline,
    Vocabulary,
    set_seed,
)

__all__ = [
    # preprocessing
    "clean_text",
    "batch_clean",
    "get_stopwords",
    "get_stemmer",
    # dataloader
    "load_raw_data",
    "preprocess",
    "train_val_test_split",
    "build_vocab",
    "build_dataset",
    "build_dataloader",
    "build_full_pipeline",
    "Vocabulary",
    "set_seed",
]
__version__ = "1.1.0"
__author__ = "Crazy Rich Team — PBA 2026"
