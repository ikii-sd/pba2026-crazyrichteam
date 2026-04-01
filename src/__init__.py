"""
src/
====
Package preprocessing untuk proyek NLP — Analisis Sentimen & Emosi
Ulasan E-Commerce Bahasa Indonesia (PRDECT-ID).

Modul tersedia:
    - preprocessing : fungsi clean_text() dan batch_clean()

Contoh penggunaan:
    from src.preprocessing import clean_text, batch_clean
"""

from src.preprocessing import clean_text, batch_clean, get_stopwords, get_stemmer

__all__ = ["clean_text", "batch_clean", "get_stopwords", "get_stemmer"]
__version__ = "1.0.0"
__author__  = "Crazy Rich Team — PBA 2026"
