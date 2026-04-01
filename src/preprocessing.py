"""
src/preprocessing.py
====================
Modul preprocessing teks untuk proyek NLP — Analisis Sentimen & Emosi
Ulasan E-Commerce Bahasa Indonesia.

Dataset : PRDECT-ID
Author  : Crazy Rich Team — PBA 2026

Pipeline clean_text() — 14 langkah berurutan:
  1.  Lowercase
  2.  Hapus URL
  3.  Hapus HTML tags
  4.  Konversi emoji → teks deskriptif (opsional)
  5.  Normalisasi harga kontekstual (50k, 100rb, Rp50.000)
  6.  Hapus angka
  7.  Hapus tanda baca & karakter non-alfanumerik
  8.  Normalisasi karakter repetisi ("bagussss" → "baguss")
  9.  Normalisasi slang e-commerce Indonesia (kamus 80+ entri)
  10. Hapus token kosong
  11. Hapus stopword (Sastrawi + tambahan manual)
  12. Stemming morfologis (Sastrawi)
  13. Filter token terlalu pendek (default: min 2 karakter)
  14. Gabung token & normalisasi whitespace

Cara pemakaian:
    from src.preprocessing import clean_text, batch_clean

    teks_bersih   = clean_text("Barang bagussss bgt!! seller ramah 👍")
    series_bersih = batch_clean(df["Customer Review"])

Test mandiri (CLI):
    python src/preprocessing.py
"""

from __future__ import annotations

import re
import sys
from typing import Optional

import pandas as pd

# ── Lazy import: emoji (opsional) ─────────────────────────────────────────────
try:
    import emoji as _emoji_lib

    _EMOJI_AVAILABLE = True
except ImportError:
    _EMOJI_AVAILABLE = False

# ── Lazy import: Sastrawi ─────────────────────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
        StopWordRemoverFactory,
    )

    _SASTRAWI_AVAILABLE = True
except ImportError:
    _SASTRAWI_AVAILABLE = False
    print(
        "[preprocessing.py] WARNING: PySastrawi tidak ditemukan. "
        "Stemming dan stopword removal Sastrawi dinonaktifkan.\n"
        "Install dengan: pip install PySastrawi",
        file=sys.stderr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# A. KONSTANTA & KAMUS SLANG
# ══════════════════════════════════════════════════════════════════════════════

# ── Kamus normalisasi slang & singkatan ulasan e-commerce Indonesia ────────────
# Format: { "kata_tidak_baku": "kata_baku" }
# Nilai kosong "" → token akan dihapus pada langkah filter
SLANG_DICT: dict[str, str] = {
    # ── Negasi & modalitas ────────────────────────────────────────────────────
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "ndak": "tidak",
    "nda": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "g": "tidak",
    "blm": "belum",
    "belom": "belum",
    "udah": "sudah",
    "udh": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "emg": "memang",
    "emang": "memang",
    "hrs": "harus",
    "mesti": "harus",
    "bs": "bisa",
    "bsa": "bisa",
    # ── Intensifier & partikel ────────────────────────────────────────────────
    "bgt": "banget",
    "bngt": "banget",
    "bgtt": "banget",
    "bener": "benar",
    "bner": "benar",
    "aja": "saja",
    "doang": "saja",
    "doank": "saja",
    "sih": "",
    "deh": "",
    "kok": "",
    "dong": "",
    "loh": "",
    "lho": "",
    # ── Kata ganti & konjungsi ────────────────────────────────────────────────
    "yg": "yang",
    "yng": "yang",
    "dr": "dari",
    "dgn": "dengan",
    "dng": "dengan",
    "dngan": "dengan",
    "utk": "untuk",
    "tuk": "untuk",
    "buat": "untuk",
    "bwt": "untuk",
    "pd": "pada",
    "krn": "karena",
    "karna": "karena",
    "keren": "karena",  # typo umum
    "krena": "karena",
    "sm": "sama",
    "ama": "sama",
    "jg": "juga",
    "jga": "juga",
    "tp": "tapi",
    "tpi": "tapi",
    "tetapi": "tapi",
    "klo": "kalau",
    "klu": "kalau",
    "kalo": "kalau",
    "klw": "kalau",
    "kl": "kalau",
    "kmrn": "kemarin",
    "kemren": "kemarin",
    "skrg": "sekarang",
    "skrang": "sekarang",
    "bsk": "besok",
    "dll": "dan lain lain",
    "dsb": "dan sebagainya",
    "dst": "dan seterusnya",
    # ── Kata sifat & penilaian produk ─────────────────────────────────────────
    "bgs": "bagus",
    "bgus": "bagus",
    "mantep": "mantap",
    "mantul": "mantap",
    "mntap": "mantap",
    "kece": "keren",
    "kinclong": "bersih",
    "oke": "baik",
    "ok": "baik",
    "oce": "baik",
    "sip": "bagus",
    "jos": "bagus",
    "joss": "bagus",
    "josss": "bagus",
    "top": "bagus",
    "murah": "murah",
    "mura": "murah",
    "mahal": "mahal",
    "jelek": "buruk",
    "jlek": "buruk",
    "rusak": "rusak",
    "cacat": "cacat",
    "puas": "puas",
    "kecewa": "kecewa",
    "kzl": "kesal",
    "kesel": "kesal",
    "ksel": "kesal",
    "ori": "original",
    "orisinil": "original",
    "asli": "original",
    "sesuai": "sesuai",
    "cocok": "sesuai",
    "mulus": "mulus",
    # ── Transaksi & pengiriman ────────────────────────────────────────────────
    "seller": "penjual",
    "penjual": "penjual",
    "toko": "toko",
    "olshop": "toko online",
    "respon": "respons",
    "fast": "cepat",
    "slow": "lambat",
    "packing": "kemasan",
    "packaging": "kemasan",
    "pake": "pakai",
    "dipake": "dipakai",
    "cod": "bayar di tempat",
    "ongkir": "ongkos kirim",
    "rekomen": "rekomendasi",
    "rekomend": "rekomendasi",
    "rekom": "rekomendasi",
    "recommend": "rekomendasi",
    # ── Sapaan & ekspresi ─────────────────────────────────────────────────────
    "makasih": "terima kasih",
    "makasi": "terima kasih",
    "mksh": "terima kasih",
    "thx": "terima kasih",
    "thanks": "terima kasih",
    "thank": "terima kasih",
    "tq": "terima kasih",
    "ty": "terima kasih",
    "trims": "terima kasih",
    "tks": "terima kasih",
    "wkwk": "",
    "wkwkwk": "",
    "haha": "",
    "hehe": "",
    "hihi": "",
    "lol": "",
    "btw": "ngomong ngomong",
    # ── Satuan & harga ────────────────────────────────────────────────────────
    "rb": "ribu",
    "jt": "juta",
}

# ── Regex dikompilasi sekali di module-level (efisiensi) ──────────────────────
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_PRICE_RP = re.compile(r"[Rr][Pp]\.?\s*\d[\d.,]*")  # Rp50.000
_RE_PRICE_K = re.compile(r"(\d+)\s*[kK]\b")  # 50k
_RE_PRICE_RB = re.compile(r"(\d+)\s*rb\b", re.IGNORECASE)  # 100rb
_RE_PRICE_JT = re.compile(r"(\d+)\s*jt\b", re.IGNORECASE)  # 2jt
_RE_NUMBER = re.compile(r"\d+")
_RE_PUNCT = re.compile(r"[^\w\s]")
_RE_UNDERSCORE = re.compile(r"_+")
_RE_REPEAT = re.compile(r"(.)\1{2,}")  # ≥ 3 karakter berulang
_RE_WHITESPACE = re.compile(r"\s+")


# ══════════════════════════════════════════════════════════════════════════════
# B. SINGLETON — Stemmer & Stopwords
# ══════════════════════════════════════════════════════════════════════════════
# Inisialisasi Sastrawi memuat dictionary besar (~beberapa detik).
# Pola singleton memastikan objek dibuat SATU KALI saja selama runtime,
# tidak dibuat ulang pada setiap pemanggilan clean_text().

_stemmer_instance = None
_stopwords_instance = None


def get_stemmer():
    """
    Mengembalikan instance Sastrawi Stemmer (singleton).

    Returns
    -------
    Sastrawi.Stemmer.Stemmer | None
        None jika PySastrawi tidak terinstall.
    """
    global _stemmer_instance
    if _stemmer_instance is None and _SASTRAWI_AVAILABLE:
        factory = StemmerFactory()
        _stemmer_instance = factory.create_stemmer()
    return _stemmer_instance


def get_stopwords() -> set[str]:
    """
    Mengembalikan himpunan stopword gabungan (singleton):
    Sastrawi stopwords + tambahan partikel & kata tidak bermakna.

    Returns
    -------
    set[str]
    """
    global _stopwords_instance
    if _stopwords_instance is None:
        if _SASTRAWI_AVAILABLE:
            sw_factory = StopWordRemoverFactory()
            base_sw = set(sw_factory.get_stop_words())
        else:
            base_sw = set()

        # Tambahan partikel & kata tidak bermakna untuk analisis sentimen
        extra_sw = {
            "nya",
            "si",
            "pun",
            "lah",
            "kah",
            "tah",
            "ku",
            "mu",
            "kami",
            "kita",
            "kalian",
            "ini",
            "itu",
            "sana",
            "sini",
            "situ",
            "mau",
            "mau",
            "ada",
            "tidak",
            "bisa",
        }
        _stopwords_instance = base_sw | extra_sw
    return _stopwords_instance


# ══════════════════════════════════════════════════════════════════════════════
# C. PRIVATE HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def _remove_url(text: str) -> str:
    """Hapus URL (http, https, www)."""
    return _RE_URL.sub(" ", text)


def _remove_html(text: str) -> str:
    """Hapus HTML/XML tags."""
    return _RE_HTML.sub(" ", text)


def _handle_emoji(text: str, to_text: bool = True) -> str:
    """
    Tangani emoji dalam teks.

    Parameters
    ----------
    text    : str
    to_text : bool
        True  → konversi emoji ke deskripsi teks (😊 → "smiling face")
        False → hapus emoji

    Returns
    -------
    str
    """
    if not _EMOJI_AVAILABLE:
        # Hapus karakter non-ASCII sebagai fallback
        return text.encode("ascii", "ignore").decode("ascii")

    if to_text:
        # emoji.demojize: 😊 → ":smiling_face:"
        text = _emoji_lib.demojize(text, delimiters=(" ", " "))
        # Bersihkan tanda titik dua sisa dan ubah underscore → spasi
        text = re.sub(
            r":([a-zA-Z_]+):",
            lambda m: m.group(1).replace("_", " "),
            text,
        )
    else:
        text = "".join(ch for ch in text if ch not in _emoji_lib.EMOJI_DATA)
    return text


def _normalize_price(text: str) -> str:
    """
    Normalisasi notasi harga kontekstual sebelum angka dihapus.

    Contoh:
        Rp50.000 → harga
        50k      → 50 ribu
        100rb    → 100 ribu
        2jt      → 2 juta
    """
    text = _RE_PRICE_RP.sub(" harga ", text)
    text = _RE_PRICE_K.sub(r"\1 ribu ", text)
    text = _RE_PRICE_RB.sub(r"\1 ribu ", text)
    text = _RE_PRICE_JT.sub(r"\1 juta ", text)
    return text


def _normalize_repeat(text: str) -> str:
    """
    Normalisasi karakter berulang ≥ 3 kali → sisakan 2.

    Contoh:
        "bagussss"  → "baguss"
        "cepatttt"  → "cepatt"
        "mantaaaap" → "mantaap"
    """
    return _RE_REPEAT.sub(r"\1\1", text)


def _normalize_slang(
    tokens: list[str],
    slang_dict: dict[str, str],
) -> list[str]:
    """
    Ganti token slang/tidak-baku dengan bentuk bakunya menggunakan kamus.

    Nilai "" dalam kamus → token akan dihapus pada langkah berikutnya.
    """
    return [slang_dict.get(tok, tok) for tok in tokens]


def _remove_stopwords(
    tokens: list[str],
    stopwords: set[str],
) -> list[str]:
    """Hapus token yang ada dalam himpunan stopword."""
    return [t for t in tokens if t not in stopwords]


def _stem_tokens(tokens: list[str], stemmer) -> list[str]:
    """Terapkan stemming Sastrawi pada setiap token."""
    if stemmer is None:
        return tokens
    return [stemmer.stem(t) for t in tokens]


# ══════════════════════════════════════════════════════════════════════════════
# D. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


def clean_text(
    text,
    *,
    slang_dict: Optional[dict] = None,
    remove_stopwords: bool = True,
    do_stemming: bool = True,
    remove_numbers: bool = True,
    emoji_to_text: bool = True,
    min_token_len: int = 2,
) -> str:
    """
    Bersihkan satu string teks ulasan e-commerce bahasa Indonesia.

    Parameters
    ----------
    text             : str | float | None
        Teks ulasan mentah.
    slang_dict       : dict, optional
        Kamus slang kustom. Default: SLANG_DICT bawaan modul.
    remove_stopwords : bool
        Hapus stopword (Sastrawi + tambahan). Default: True.
    do_stemming      : bool
        Terapkan stemming Sastrawi. Default: True.
    remove_numbers   : bool
        Hapus angka dari teks. Default: True.
    emoji_to_text    : bool
        True  → konversi emoji ke teks.
        False → hapus emoji. Default: True.
    min_token_len    : int
        Panjang minimum token yang dipertahankan. Default: 2.

    Returns
    -------
    str
        Teks bersih, siap digunakan sebagai fitur model NLP.

    Examples
    --------
    >>> clean_text("Barang bagussss bgt!! seller ramah 👍")
    'barang baguss banget jual ramah thumbs up'

    >>> clean_text("KECEWA!! gak sesuai deskripsi. Harga 50k tapi jelek bgt")
    'kecewa sesuai deskripsi harga ribu buruk banget'
    """
    # ── Guard: tangani None / NaN / non-string ──────────────────────────────
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            import math

            if math.isnan(float(text)):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(text)

    if slang_dict is None:
        slang_dict = SLANG_DICT

    # ── Step 1: Lowercase ────────────────────────────────────────────────────
    text = text.lower()

    # ── Step 2: Hapus URL ────────────────────────────────────────────────────
    text = _remove_url(text)

    # ── Step 3: Hapus HTML tags ──────────────────────────────────────────────
    text = _remove_html(text)

    # ── Step 4: Handle emoji ─────────────────────────────────────────────────
    text = _handle_emoji(text, to_text=emoji_to_text)

    # ── Step 5: Normalisasi harga kontekstual ────────────────────────────────
    text = _normalize_price(text)

    # ── Step 6: Hapus angka ──────────────────────────────────────────────────
    if remove_numbers:
        text = _RE_NUMBER.sub(" ", text)

    # ── Step 7: Hapus tanda baca & karakter non-alfanumerik ──────────────────
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_UNDERSCORE.sub(" ", text)

    # ── Step 8: Normalisasi karakter repetisi ────────────────────────────────
    text = _normalize_repeat(text)

    # ── Tokenisasi (whitespace split) ────────────────────────────────────────
    tokens = text.split()

    # ── Step 9: Normalisasi slang ────────────────────────────────────────────
    tokens = _normalize_slang(tokens, slang_dict)

    # ── Step 10: Hapus token kosong (akibat slang → "") ──────────────────────
    tokens = [t for t in tokens if t.strip()]

    # ── Step 11: Hapus stopwords ─────────────────────────────────────────────
    if remove_stopwords:
        tokens = _remove_stopwords(tokens, get_stopwords())

    # ── Step 12: Stemming ────────────────────────────────────────────────────
    if do_stemming:
        tokens = _stem_tokens(tokens, get_stemmer())

    # ── Step 13: Hapus token terlalu pendek ──────────────────────────────────
    tokens = [t for t in tokens if len(t) >= min_token_len]

    # ── Step 14: Gabung & normalisasi whitespace ─────────────────────────────
    result = _RE_WHITESPACE.sub(" ", " ".join(tokens)).strip()
    return result


def batch_clean(
    series: "pd.Series",
    verbose: bool = True,
    **kwargs,
) -> "pd.Series":
    """
    Terapkan clean_text() ke seluruh pd.Series teks.

    Parameters
    ----------
    series  : pd.Series
        Kolom teks ulasan mentah.
    verbose : bool
        Tampilkan progress bar (butuh tqdm). Default: True.
    **kwargs
        Argumen tambahan diteruskan ke clean_text().

    Returns
    -------
    pd.Series
        Series teks bersih dengan indeks yang sama.

    Examples
    --------
    >>> df["clean_review"] = batch_clean(df["Customer Review"])
    """

    def _apply(text):
        return clean_text(text, **kwargs)

    if verbose:
        try:
            from tqdm.auto import tqdm

            tqdm.pandas(desc="🧹 Cleaning teks")
            return series.progress_apply(_apply)
        except ImportError:
            print(
                "[preprocessing.py] INFO: tqdm tidak terinstall — "
                "progress bar dinonaktifkan."
            )
    return series.apply(_apply)


# ══════════════════════════════════════════════════════════════════════════════
# E. CLI TEST — python src/preprocessing.py
# ══════════════════════════════════════════════════════════════════════════════


def _run_sanity_check() -> None:
    """Jalankan sanity check dengan 8 contoh kalimat ulasan nyata."""
    TEST_CASES = [
        # (input_teks, keterangan)
        (
            "Barang bagussss bgt!! Penjual ramah & respon cepat 👍😊",
            "Ulasan positif dengan emoji & karakter berulang",
        ),
        (
            "KECEWA!! Barang rusak, gak sesuai deskripsi. Harga 50k tapi jelek bgt",
            "Ulasan negatif dengan caps & harga",
        ),
        (
            "mantep paten joss, fast delivery, packing aman. Top deh seller nya!",
            "Slang + campur kode Inggris",
        ),
        (
            "udah 3x beli di sini, krn harganya mura tp kualitas oke. Makasih yg jual!",
            "Singkatan & slang ganda",
        ),
        (
            "lambat bgt pengirimannya, udah 2 minggu blm sampe. seller ga respon 😡",
            "Ulasan negatif dengan emoji marah",
        ),
        (
            "Alhamdulillah barang ori sesuai gambar, ongkir gratis lagi. recommended!",
            "Ulasan positif formal",
        ),
        (
            "bgs banget, harga Rp75.000 worth it. Kemasan aman, ga ada yg rusak.",
            "Singkatan 'bgs' + format harga Rupiah",
        ),
        (None, "Input None — harus mengembalikan string kosong"),
    ]

    SEP = "=" * 80
    print(f"\n{SEP}")
    print("  SANITY CHECK — src/preprocessing.py")
    print(SEP)

    for idx, (raw, keterangan) in enumerate(TEST_CASES, start=1):
        hasil = clean_text(raw)
        label = str(raw)[:65] + "…" if raw and len(str(raw)) > 65 else str(raw)
        print(f"\n[{idx}] {keterangan}")
        print(f"  IN : {label}")
        print(f"  OUT: {hasil}")

    print(f"\n{SEP}")
    print("  ✅  Sanity check selesai.")
    print(SEP)


if __name__ == "__main__":
    _run_sanity_check()
