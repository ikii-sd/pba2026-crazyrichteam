"""
config_simplified.py
Konfigurasi sederhana untuk pipeline Deep Learning
Sentiment & Emotion Analysis (E-Commerce Indonesia)
"""

from pathlib import Path

import torch

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # .../pba2026-crazyrichteam

DATASET_PATH = PROJECT_ROOT / "data" / "clean" / "cleaned_dataset.csv"
OUTPUT_DIR = BASE_DIR
MODEL_DIR = OUTPUT_DIR / "models"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
LOG_DIR = OUTPUT_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# DATASET COLUMNS (hasil analisis Step 1)
# =========================================================
TEXT_COLUMN = "clean_review"
SENTIMENT_LABEL_COLUMN = "Sentiment"
EMOTION_LABEL_COLUMN = "Emotion"

# =========================================================
# PREPROCESSING
# =========================================================
MAX_VOCAB_SIZE = 10_000
MAX_SEQ_LEN = 64  # rekomendasi dari distribusi panjang teks
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
LOWERCASE = True

# =========================================================
# MODEL HYPERPARAMETERS (constraint < 2M params)
# =========================================================
MODEL_NAME = "SimpleBiLSTM"
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.4

# output classes
NUM_SENTIMENT_CLASSES = 2
NUM_EMOTION_CLASSES = 5

# =========================================================
# TRAINING CONFIG
# =========================================================
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_MAX_NORM = 1.0

# split ratio (train/val/test = 80/10/10)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

# scheduler (opsional)
USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 1

# =========================================================
# OUTPUT FILES
# =========================================================
BEST_MODEL_PATH = MODEL_DIR / "best_simplified_bilstm.pt"
LAST_MODEL_PATH = MODEL_DIR / "last_simplified_bilstm.pt"
VOCAB_PATH = ARTIFACT_DIR / "vocab_simplified.json"
LABEL_ENCODERS_PATH = ARTIFACT_DIR / "label_encoders_simplified.json"
TRAIN_HISTORY_PATH = LOG_DIR / "train_history_simplified.json"

# =========================================================
# RUNTIME / DATALOADER
# =========================================================
NUM_WORKERS = 0  # aman untuk lintas OS
PIN_MEMORY = torch.cuda.is_available()

# =========================================================
# SANITY CHECKS
# =========================================================
assert abs((TRAIN_SIZE + VAL_SIZE + TEST_SIZE) - 1.0) < 1e-8, (
    "TRAIN_SIZE + VAL_SIZE + TEST_SIZE harus = 1.0"
)

assert MAX_VOCAB_SIZE <= 10_000, "MAX_VOCAB_SIZE tidak boleh > 10,000"
assert MAX_SEQ_LEN <= 128, "MAX_SEQ_LEN tidak boleh > 128"
assert EMBEDDING_DIM in (64, 128), "EMBEDDING_DIM harus 64 atau 128"
assert HIDDEN_DIM in (128, 256), "HIDDEN_DIM harus 128 atau 256"
assert 0.3 <= DROPOUT <= 0.5, "DROPOUT harus di rentang 0.3 - 0.5"
