"""
src/
====
Package modul Python untuk proyek NLP — Analisis Sentimen & Emosi
Ulasan E-Commerce Bahasa Indonesia (PRDECT-ID).

Modul tersedia:
    - preprocessing : fungsi clean_text(), batch_clean() — 14-step NLP pipeline
    - dataloader    : data pipeline lengkap (load, split, vocab, Dataset, DataLoader)
    - model         : build_model(config), get_optimizer, get_scheduler, count_params
    - models/       : registry multi-varian (baseline, improved, large)
    - train         : training pipeline lengkap (fit, validation)
    - utils         : helper functions (seed, json, metrics, plotting)

Contoh penggunaan:
    from src.preprocessing import clean_text, batch_clean
    from src.dataloader import build_full_pipeline
    from src.model import build_model, get_optimizer, get_scheduler, model_summary
    from src.models import build_model as build_model_v, list_models, compare_models
    from src.train import fit
    from src.utils import set_seed, plot_training_curves
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
)
from src.model import (
    build_model,
    get_optimizer,
    get_scheduler,
    count_params,
    model_summary,
    list_available_models,
)
from src.train import train_one_epoch, validate, fit
from src.utils import (
    set_seed,
    ensure_dir,
    save_json,
    load_json,
    get_timestamp,
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
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
    # model
    "build_model",
    "get_optimizer",
    "get_scheduler",
    "count_params",
    "model_summary",
    "list_available_models",
    # train
    "train_one_epoch",
    "validate",
    "fit",
    # utils
    "set_seed",
    "ensure_dir",
    "save_json",
    "load_json",
    "get_timestamp",
    "compute_metrics",
    "plot_training_curves",
    "plot_confusion_matrix",
]
__version__ = "1.3.0"
__author__ = "Crazy Rich Team — PBA 2026"
