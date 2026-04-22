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
    - utils         : helper functions & run environment manager
    - logger        : setup sentral logging pipeline (ke terminal + file run)

Contoh penggunaan:
    from src.logger import get_logger
    from src.utils import setup_run_env
    from src.dataloader import build_full_pipeline
    from src.train import fit
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
    setup_run_env,
    ensure_dir,
    save_json,
    load_json,
    get_timestamp,
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
)
from src.logger import get_logger, setup_run_logger

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
    # utils & logger
    "setup_run_env",
    "set_seed",
    "ensure_dir",
    "save_json",
    "load_json",
    "get_timestamp",
    "compute_metrics",
    "plot_training_curves",
    "plot_confusion_matrix",
    "get_logger",
    "setup_run_logger",
]
__version__ = "1.4.0"
__author__ = "Crazy Rich Team — PBA 2026"
