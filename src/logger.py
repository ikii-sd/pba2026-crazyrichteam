"""
src/logger.py
=============
Modul sistem logging tersentralisasi untuk proyek.

Fungsi:
    - Menyediakan `get_logger(name)` untuk digunakan di semua modul.
    - Mencatat log ke console (warna/standar) dan ke file (log.txt) di setiap run_name.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Konstanta format
LOG_FORMAT_CONSOLE = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_FORMAT_FILE    = "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
DATE_FORMAT        = "%H:%M:%S"

# Global run log path untuk setup dynamic per-run
_RUN_LOG_PATH: Optional[Path] = None
_ROOT_LOGGER_SETUP_DONE: bool = False

def setup_run_logger(run_name: str, log_dir: str | Path = "outputs/logs") -> Path:
    """
    Kaitkan logger root ke file spesifik untuk satu 'run'.
    Biasa dipanggil sekali di awal notebook / orchestrator.

    Parameters
    ----------
    run_name : str
        Nama run (misal: "20260422_153000_baseline")
    log_dir : str | Path
        Direktori utama log.

    Returns
    -------
    Path ke file log.txt yang dibuat.
    """
    global _RUN_LOG_PATH, _ROOT_LOGGER_SETUP_DONE
    
    log_path = Path(log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / "log.txt"
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Hapus semua handler lama (jika notebook running berkali-kali)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # ── Console Handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(LOG_FORMAT_CONSOLE, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # ── File Handler ──
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter(LOG_FORMAT_FILE)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    _RUN_LOG_PATH = file_path
    _ROOT_LOGGER_SETUP_DONE = True
    
    logger.info("=" * 60)
    logger.info(f"Dimulai Run: {run_name}")
    logger.info("=" * 60)
    
    return file_path

def get_logger(module_name: str) -> logging.Logger:
    """
    Fungsi standar yang dipanggil di setiap file .py (contoh: get_logger(__name__)).
    
    Jika `setup_run_logger` belum dipanggil, fallback menambahkan console handler saja
    agar pesan tetap muncul secara standar (tanpa file).
    """
    logger = logging.getLogger(module_name)
    
    if not _ROOT_LOGGER_SETUP_DONE and not logger.hasHandlers():
        # Fallback console
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_fmt = logging.Formatter(LOG_FORMAT_CONSOLE, datefmt=DATE_FORMAT)
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
        # Cegah propagasi duplikat kalau root nanti di-setup
        logger.propagate = False 

    # Jika setup_run_logger sudah dipanggil, logger ini akan inherit dari root automatically
    if _ROOT_LOGGER_SETUP_DONE:
        logger.propagate = True

    return logger

