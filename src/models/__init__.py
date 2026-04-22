"""
src/models/__init__.py
=======================
Registry terpusat semua varian model proyek.

Notebook / script hanya perlu mengubah satu nilai:
    config["model_name"] = "baseline" | "improved" | "large"

dan memanggil:
    from src.models import build_model, list_models, get_default_config

Cara pemakaian (notebook):
    from src.models import build_model, get_default_config

    # 1. Ambil config default untuk model pilihan
    config = get_default_config("improved")

    # 2. Sesuaikan parameter sesuai data
    config.update({
        "vocab_size"            : len(vocab),
        "num_sentiment_classes" : 2,
        "num_emotion_classes"   : 5,
    })

    # 3. Bangun model — satu baris
    model = build_model(config)

Model yang tersedia:
    "baseline"  → BiLSTMBaseline  (1 layer, H=128)       ~1.4 M params
    "improved"  → BiLSTMImproved  (2 layer, H=256, BN)   ~3.1 M params
    "large"     → BiLSTMLarge     (2 layer, H=256, 2×FC) ~5.6 M params

Menambahkan varian baru:
    1. Buat file src/models/model_<nama>.py
    2. Pastikan ada: MODEL_NAME, DEFAULT_CFG, dan fungsi build(config)
    3. Import dan daftarkan di bagian "REGISTRY" di file ini

Author : Crazy Rich Team — PBA 2026
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

# ── Import semua varian model ─────────────────────────────────────────────────
from src.models.model_baseline import (
    build as _build_baseline,
    DEFAULT_CFG as _CFG_BASELINE,
    MODEL_NAME as _NAME_BASELINE,
)
from src.models.model_improved import (
    build as _build_improved,
    DEFAULT_CFG as _CFG_IMPROVED,
    MODEL_NAME as _NAME_IMPROVED,
)
from src.models.model_large import (
    build as _build_large,
    DEFAULT_CFG as _CFG_LARGE,
    MODEL_NAME as _NAME_LARGE,
)

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY UTAMA
# format: { "nama_model": (factory_function, default_config_dict) }
# ══════════════════════════════════════════════════════════════════════════════
_REGISTRY: Dict[str, tuple] = {
    _NAME_BASELINE : (_build_baseline, _CFG_BASELINE),
    _NAME_IMPROVED : (_build_improved, _CFG_IMPROVED),
    _NAME_LARGE    : (_build_large,    _CFG_LARGE),
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


def list_models() -> List[str]:
    """
    Kembalikan daftar nama model yang terdaftar.

    Returns
    -------
    List[str]

    Examples
    --------
    >>> from src.models import list_models
    >>> print(list_models())
    ['baseline', 'improved', 'large']
    """
    return sorted(_REGISTRY.keys())


def get_default_config(model_name: str) -> Dict[str, Any]:
    """
    Kembalikan salinan config default untuk model tertentu.

    Config ini bisa langsung dimodifikasi tanpa merusak nilai aslinya.

    Parameters
    ----------
    model_name : str
        Nama model — "baseline" | "improved" | "large"

    Returns
    -------
    dict — salinan DEFAULT_CFG model tersebut

    Raises
    ------
    ValueError : jika model_name tidak dikenali

    Examples
    --------
    >>> from src.models import get_default_config
    >>> cfg = get_default_config("baseline")
    >>> cfg["vocab_size"] = 8_000
    """
    _check_name(model_name)
    _, default_cfg = _REGISTRY[model_name.lower()]
    return dict(default_cfg)  # salinan — aman dimodifikasi


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Bangun model sesuai config["model_name"].

    Ini adalah satu-satunya fungsi yang perlu dipanggil dari notebook.
    Notebook tidak perlu tahu file mana yang berisi model — cukup ubah
    config["model_name"].

    Parameters
    ----------
    config : dict
        Wajib ada:
            "model_name"             : str  — "baseline" | "improved" | "large"
            "vocab_size"             : int
            "num_sentiment_classes"  : int
            "num_emotion_classes"    : int
        Opsional lainnya diambil dari DEFAULT_CFG masing-masing model
        jika tidak disediakan.

    Returns
    -------
    nn.Module

    Raises
    ------
    KeyError  : jika "model_name" tidak ada di config
    ValueError: jika model_name tidak dikenali

    Examples
    --------
    >>> from src.models import build_model, get_default_config
    >>>
    >>> # Cara 1: mulai dari default config
    >>> config = get_default_config("improved")
    >>> config.update({"vocab_size": 10_000, "num_sentiment_classes": 2, "num_emotion_classes": 5})
    >>> model = build_model(config)
    >>>
    >>> # Cara 2: buat config sendiri (key tidak lengkap — fallback ke default)
    >>> model = build_model({
    ...     "model_name"            : "baseline",
    ...     "vocab_size"            : 10_000,
    ...     "num_sentiment_classes" : 2,
    ...     "num_emotion_classes"   : 5,
    ... })
    """
    if "model_name" not in config:
        raise KeyError(
            "[build_model] Key 'model_name' wajib ada di config. "
            f"Pilihan: {list_models()}"
        )

    name = str(config["model_name"]).lower()
    _check_name(name)

    factory, default_cfg = _REGISTRY[name]
    # Merge: default_cfg di-override oleh config yang diberikan
    merged = {**default_cfg, **config}

    model = factory(merged)

    # Hitung dan print ringkasan parameter
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [src.models] ✅  Model '{name}' dibuat  "
        f"({trainable:,} parameter trainable)"
    )
    return model


def compare_models(
    base_config: Dict[str, Any],
    model_names: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Bangun semua (atau sebagian) model dan bandingkan jumlah parameter.

    Berguna saat memilih arsitektur sebelum training.

    Parameters
    ----------
    base_config  : dict
        Config dasar (vocab_size, num_classes, dll.) yang dipakai bersama.
    model_names  : List[str], optional
        Subset model yang ingin dibandingkan. Default: semua model.

    Returns
    -------
    dict — { model_name: { "params": int, "model": nn.Module } }

    Examples
    --------
    >>> from src.models import compare_models
    >>> results = compare_models({
    ...     "vocab_size": 10_000,
    ...     "num_sentiment_classes": 2,
    ...     "num_emotion_classes": 5,
    ... })
    >>> for name, info in results.items():
    ...     print(f"{name}: {info['params']:,} params")
    """
    import torch
    targets = model_names or list_models()
    comparison: Dict[str, Dict[str, Any]] = {}

    print("\n" + "=" * 55)
    print("  📊  Perbandingan Arsitektur Model")
    print("=" * 55)
    print(f"  {'Model':<15} {'Parameter':>15}  {'Output'}")
    print("  " + "-" * 50)

    for name in targets:
        cfg = {**base_config, "model_name": name}
        try:
            factory, default_cfg = _REGISTRY[name]
            m = factory({**default_cfg, **cfg})
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)

            # Dummy forward untuk cek shape output
            dummy = torch.randint(0, int(cfg.get("vocab_size", 10_000)), (2, 64))
            m.eval()
            with torch.no_grad():
                s_out, e_out = m(dummy)
            out_str = f"sent={tuple(s_out.shape)}  emo={tuple(e_out.shape)}"

            comparison[name] = {"params": params, "model": m}
            print(f"  {name:<15} {params:>15,}  {out_str}")
        except Exception as exc:
            print(f"  {name:<15} ❌  Error: {exc}")

    print("=" * 55 + "\n")
    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPER
# ══════════════════════════════════════════════════════════════════════════════


def _check_name(name: str) -> None:
    """Validasi nama model ada di registry."""
    if name not in _REGISTRY:
        raise ValueError(
            f"[src.models] Model '{name}' tidak dikenali. "
            f"Pilihan: {list_models()}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "build_model",
    "list_models",
    "get_default_config",
    "compare_models",
]

__version__ = "1.0.0"
__author__ = "Crazy Rich Team — PBA 2026"


# ── CLI smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧪  SMOKE TEST — src/models/__init__.py")
    print("=" * 60)

    print(f"\nModel terdaftar: {list_models()}\n")

    _base = {
        "vocab_size"            : 10_000,
        "num_sentiment_classes" : 2,
        "num_emotion_classes"   : 5,
    }

    # Test build setiap model
    for _name in list_models():
        cfg = get_default_config(_name)
        cfg.update(_base)
        m = build_model(cfg)

    # Test compare
    compare_models(_base)

    print("✅  Semua model lolos smoke test!\n")
