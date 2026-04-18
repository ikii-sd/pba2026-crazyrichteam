"""
app.py — Gradio App untuk Analisis Sentimen & Emosi Ulasan E-Commerce Indonesia
================================================================================
Deploy di Hugging Face Spaces.
Model : PyTorch BiLSTM (Deep Learning)
Dataset: PRDECT-ID — Ulasan E-Commerce Bahasa Indonesia
Tim    : Crazy Rich Team — PBA 2026
"""

print("Starting app initialization...")

import pathlib
import json
import sys
import os

# Disable Gradio analytics untuk offline mode
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

print("Importing Gradio...")
import gradio as gr

print("Importing PyTorch...")
import torch

print("Importing Pandas...")
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# 📦 LOAD MODEL & VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = pathlib.Path("models")
MODEL_DL_DIR = MODEL_DIR / "model_dl"

print(f"Using device: {DEVICE}")

# ── Import model architecture ──────────────────────────────────────────────────
from models.model_design import SimpleBiLSTM
from models.vocabulary_builder import Vocabulary
from models.config_simplified import MAX_SEQ_LEN

# ── Load model metadata ────────────────────────────────────────────────────────
print("Loading model metadata...")
metadata_path = MODEL_DL_DIR / "saved_models" / "best_model_metadata.json"
with open(metadata_path, "r") as f:
    model_metadata = json.load(f)

vocab_size = model_metadata["vocab_size"]
hidden_dim = model_metadata["hidden_dim"]
num_layers = model_metadata["num_layers"]
dropout = model_metadata["dropout"]

# ── Build model ────────────────────────────────────────────────────────────────
print("Building BiLSTM model...")
model = SimpleBiLSTM(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    pad_idx=0,
    num_sentiment_classes=2,
    num_emotion_classes=5,
)

# ── Load pretrained weights ────────────────────────────────────────────────────
print("Loading pretrained weights...")
model_path = MODEL_DL_DIR / "saved_models" / "best_model.pt"
checkpoint = torch.load(model_path, map_location=DEVICE)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "embedding.weight" in checkpoint:
        # Direct state dict
        model.load_state_dict(checkpoint)
    else:
        # Try to load whatever is in the checkpoint
        print(f"Warning: Unknown checkpoint structure. Keys: {list(checkpoint.keys())}")
        if len(checkpoint) > 0:
            # Try the first value if it looks like a state dict
            first_val = list(checkpoint.values())[0]
            if isinstance(first_val, dict):
                model.load_state_dict(first_val)
            else:
                model.load_state_dict(checkpoint)
else:
    # Direct state dict
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

# ── Load vocabulary ────────────────────────────────────────────────────────────
print("Loading vocabulary...")
vocab_path = MODEL_DL_DIR / "artifacts" / "vocab_simplified.json"
vocab = Vocabulary.load(vocab_path)

print(f"✅ Model loaded successfully!")
print(f"   Vocab size: {vocab_size}")
print(f"   Hidden dim: {hidden_dim}")
print(f"   Device: {DEVICE}")

# ── Label mappings ────────────────────────────────────────────────────────────
SENTIMENT_LABELS = {0: "Negatif", 1: "Positif"}
EMOTION_LABELS = {0: "Bahagia", 1: "Sedih", 2: "Takut", 3: "Cinta", 4: "Marah"}


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 FUNGSI PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════


def predict_review(text: str) -> tuple:
    """
    Prediksi sentimen dan emosi dari teks ulasan e-commerce.

    Returns
    -------
    tuple[dict, dict]
        (sentimen_result, emosi_result)  — format dict {label: confidence}
        untuk komponen gr.Label.
    """
    if not text or not text.strip():
        return {"Error: teks kosong": 1.0}, {"Error: teks kosong": 1.0}

    try:
        # 1. Tokenize dan konversi ke indices
        indices = vocab.text_to_indices(text, max_seq_len=MAX_SEQ_LEN)
        input_ids = torch.tensor([indices], dtype=torch.long).to(DEVICE)

        # 2. Forward pass
        with torch.no_grad():
            sentiment_logits, emotion_logits = model(input_ids)

        # 3. Ambil probabilitas
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)[0].cpu().numpy()
        emotion_probs = torch.softmax(emotion_logits, dim=1)[0].cpu().numpy()

        # 4. Build hasil dictionary
        sentiment_result = {
            SENTIMENT_LABELS[i]: float(sentiment_probs[i])
            for i in range(len(SENTIMENT_LABELS))
        }

        emotion_result = {
            EMOTION_LABELS[i]: float(emotion_probs[i])
            for i in range(len(EMOTION_LABELS))
        }

        return sentiment_result, emotion_result

    except Exception as e:
        return {"Error": 1.0}, {"Error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 🎨 GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLES = [
    [
        "Barang bagus dan respon penjual cepat sekali, harga bersaing. Sangat puas dengan pembelian ini!"
    ],
    [
        "Kecewa banget! Barang rusak saat sampai dan tidak sesuai dengan deskripsi di toko."
    ],
    [
        "Alhamdulillah berfungsi dengan baik, packaging aman, seller dan kurir amanah. Terima kasih!"
    ],
    [
        "Pengiriman sangat lambat, sudah 2 minggu belum sampai. Seller tidak mau merespons chat sama sekali."
    ],
    [
        "Mantap! Kualitas produk bagus, harga murah, pelayanan memuaskan. Recommended banget buat semua!"
    ],
    [
        "Takut beli lagi, barang yang diterima beda dengan foto. Merasa ditipu oleh penjual ini."
    ],
]

demo = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(
        label="💬 Masukkan Ulasan Produk",
        placeholder="Ketik ulasan produk e-commerce di sini... (contoh: 'Barang bagus, seller ramah dan pengiriman cepat!')",
        lines=3,
    ),
    outputs=[
        gr.Label(
            label="🔎 Sentimen",
            num_top_classes=2,
        ),
        gr.Label(
            label="😊 Emosi",
            num_top_classes=5,
        ),
    ],
    title="🧺 Analisis Sentimen & Emosi Ulasan E-Commerce Indonesia",
    description=(
        "<div style='text-align: center; margin-bottom: 20px;'>"
        "Model NLP Deep Learning (BiLSTM) untuk menganalisis <b>sentimen</b> (Positif / Negatif) dan <b>emosi</b> "
        "(Bahagia / Sedih / Takut / Cinta / Marah) pada ulasan produk e-commerce berbahasa Indonesia.<br><br>"
        "Model dilatih menggunakan dataset <b>PRDECT-ID</b> dengan <i>PyTorch BiLSTM architecture</i>."
        "</div>"
    ),
    clear_btn="Bersihkan",
    submit_btn="Kirim",
    stop_btn="Berhenti",
    examples=EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
