# 📋 AUDIT & RENCANA REFACTOR — pba2026-crazyrichteam
**Proyek:** Analisis Sentimen & Emosi Ulasan E-Commerce Indonesia (PRDECT-ID)
**Tim:** Crazy Rich Team — PBA 2026
**Tanggal Audit:** 22 April 2026
**Status:** Siap Dieksekusi

---

## 1. DAFTAR NOTEBOOK & RINGKASAN ISI

### 📓 `notebooks/01_eda_preprocessing.ipynb` (~4.1 MB)
| Bagian | Isi |
|--------|-----|
| **Load Data** | Membaca `data/PRDECT-ID Dataset.csv`, cek shape, kolom, sample awal |
| **Preprocessing** | Pipeline `clean_text()` 14 langkah: lowercase → hapus URL/HTML → emoji → harga → angka → tanda baca → repeat → slang → stopword → stemming |
| **Visualisasi** | Distribusi label sentiment & emotion, word cloud, histogram panjang teks, bar chart frekuensi kata |
| **Output** | Menyimpan `data/clean/cleaned_dataset.csv` |

### 📓 `notebooks/02_pycaret_automl.ipynb` (~370 KB)
| Bagian | Isi |
|--------|-----|
| **Load Data** | Membaca `data/clean/cleaned_dataset.csv` hasil preprocessing |
| **Model** | PyCaret AutoML — TF-IDF + multiple ML classifiers (LR, SVC, NB, dll.) |
| **Training** | Stratified cross-validation, benchmarking semua model |
| **Evaluasi** | Compare model, F1-Score macro/weighted, accuracy per kelas |
| **Output** | `models/best_ml_model.pkl`, `models/best_emotion_model.pkl`, `models/tfidf_vectorizer.pkl` |

---

## 2. MAPPING KODE → FILE .py TUJUAN

### 📁 `src/dataloader.py` ← **BARU (Prompt 2)**
| Kode Asal | Fungsi yang Dipindahkan |
|-----------|------------------------|
| `models/data_processor.py` | `load_data()`, `process_data()`, `train_val_test_split()` |
| `models/model_dl/data_pipeline.py` | `SentimentEmotionDataset`, `create_dataloaders()` |
| `models/model_dl/run_experiments.py` (STEP 1-3) | Orchestrasi load → vocab → dataloader |
| Notebook `01_eda_preprocessing.ipynb` | Logic baca CSV mentah, validasi kolom |

Fungsi lengkap yang diusulkan:
```
load_raw_data(csv_path, ...)          → pd.DataFrame
preprocess(df, text_col, ...)         → pd.DataFrame
build_dataset(df, vocab, ...)         → SentimentEmotionDataset
build_dataloader(dataset, ...)        → DataLoader
train_val_test_split(df, ...)         → (train_df, val_df, test_df)
build_vocab_from_splits(...)          → Vocabulary
```

### 📁 `src/model.py` ← Dari `models/model_design.py`
| Kode Asal | Fungsi/Kelas |
|-----------|--------------|
| `models/model_design.py` | `SimpleBiLSTM`, `count_parameters()` |

### 📁 `src/train.py` ← Dari `models/model_dl/training_pipeline.py` & `run_experiments.py`
| Kode Asal | Fungsi |
|-----------|--------|
| `models/model_dl/training_pipeline.py` | `train_one_epoch()`, `validate()`, `train_model()`, `save_checkpoint()` |
| `models/model_dl/run_experiments.py` | `main()` / orchestrasi eksperimen |

### 📁 `src/utils.py` ← Dari berbagai modul
| Kode Asal | Fungsi |
|-----------|--------|
| `models/data_processor.py` | `_build_label_mapping()`, `save_label_encoders()` |
| `models/vocabulary_builder.py` | Class `Vocabulary` (pindah full) |
| `models/model_dl/evaluation.py` | `test_model()`, `compute_metrics()`, `plot_confusion_matrix()`, `plot_training_curves()` |
| `src/preprocessing.py` | `clean_text()`, `batch_clean()`, `SLANG_DICT` — tetap di `preprocessing.py` |

---

## 3. STRUKTUR FOLDER FINAL (USULAN)

```
pba2026-crazyrichteam/
│
├── 📂 data/
│   ├── PRDECT-ID Dataset.csv          # data mentah (jangan diubah)
│   └── clean/
│       └── cleaned_dataset.csv        # hasil preprocessing notebook 01
│
├── 📂 notebooks/
│   ├── 01_eda_preprocessing.ipynb     # EDA + preprocessing (RETAIN, kurangi kode duplikat)
│   └── 02_pycaret_automl.ipynb        # AutoML PyCaret (RETAIN)
│
├── 📂 src/                            # ← MODUL PYTHON UTAMA
│   ├── __init__.py
│   ├── preprocessing.py               # ✅ Sudah ada — 14-step clean_text
│   ├── dataloader.py                  # 🆕 Prompt 2 — semua data pipeline
│   ├── model.py                       # 🔄 Pindah dari models/model_design.py
│   ├── train.py                       # 🔄 Pindah dari models/model_dl/training_pipeline.py
│   └── utils.py                       # 🔄 Gabungkan evaluation + vocab + label helper
│
├── 📂 models/                         # ← ARTEFAK MODEL (bukan kode)
│   ├── best_ml_model.pkl
│   ├── best_emotion_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_dl/
│       ├── saved_models/              # checkpoint .pt hasil training
│       └── artifacts/                 # vocab.json, label_encoders.json
│
├── 📂 outputs/                        # ← SEMUA OUTPUT RUNTIME
│   ├── figures/                       # plot confusion matrix, training curves
│   ├── checkpoints/                   # best_model.pt, experiment results .csv
│   └── logs/                          # train_history.json, experiment logs
│
├── 📂 docs/
│   ├── STEP1_DATASET_ANALYSIS.txt
│   ├── STEP2_MODEL_DESIGN.md
│   └── AUDIT_REFACTOR_PLAN.md         # ← Dokumen ini
│
├── app.py                              # Streamlit / inference app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. RENCANA REFACTOR LANGKAH-DEMI-LANGKAH

> **Prinsip:** Refactor bertahap, setiap langkah dapat di-commit dan di-test mandiri. Tidak ada langkah yang langsung menghapus kode lama sebelum penggantinya berjalan benar.

### FASE 0 — Persiapan (Hari ini)
- [x] **0.1** Audit repository (dokumen ini)
- [ ] **0.2** Pastikan semua file di-commit ke GitHub sebelum refactor
- [ ] **0.3** Buat branch `refactor/modularisasi` agar main branch aman

### FASE 1 — Buat Modul Baru (Tidak merusak kode lama)
- [ ] **1.1** Buat `src/dataloader.py` — isi semua fungsi data pipeline (**Prompt 2**)
  - `load_raw_data()`, `preprocess()`, `build_dataset()`, `build_dataloader()`, `train_val_test_split()`
- [ ] **1.2** Buat `src/model.py` — pindahkan `SimpleBiLSTM` + `count_parameters()`
- [ ] **1.3** Buat `src/utils.py` — pindahkan `Vocabulary`, `compute_metrics()`, `plot_*()`, `save_label_encoders()`
- [ ] **1.4** Buat `src/train.py` — pindahkan fungsi training loop dan checkpoint

### FASE 2 — Buat Folder Output Baru
- [ ] **2.1** Buat `outputs/figures/`, `outputs/checkpoints/`, `outputs/logs/`
- [ ] **2.2** Update path di `config_simplified.py` agar output ke `outputs/`

### FASE 3 — Update Script Lama agar Pakai Modul Baru
- [ ] **3.1** Update `models/model_dl/run_experiments.py` → import dari `src/` bukan dari `models/`
- [ ] **3.2** Update `models/model_dl/data_pipeline.py` → gunakan `src/dataloader.py`
- [ ] **3.3** Update `models/model_dl/training_pipeline.py` → gunakan `src/train.py`
- [ ] **3.4** Test jalankan `python models/model_dl/run_experiments.py` — pastikan masih berjalan

### FASE 4 — Update Notebook
- [ ] **4.1** Update `notebooks/01_eda_preprocessing.ipynb` — ganti kode panjang dengan import dari `src/preprocessing.py` dan `src/dataloader.py`
- [ ] **4.2** Update `notebooks/02_pycaret_automl.ipynb` — ganti kode load data dengan `from src.dataloader import load_raw_data`
- [ ] **4.3** Jalankan kedua notebook dari awal (Restart & Run All) — pastikan output sama

### FASE 5 — Bersihkan Duplikasi
- [ ] **5.1** Hapus duplikasi `clean_text()` di `models/data_processor.py` → ganti dengan import dari `src/preprocessing.py`
- [ ] **5.2** Hapus `models/model_design.py` (sudah pindah ke `src/model.py`)
- [ ] **5.3** Hapus `models/vocabulary_builder.py` (sudah pindah ke `src/utils.py`)
- [ ] **5.4** Simpan `models/data_processor.py` sebagai legacy, atau redirect import saja

### FASE 6 — Finalisasi
- [ ] **6.1** Update `README.md` dengan struktur baru dan cara pakai
- [ ] **6.2** Jalankan end-to-end test: `python src/dataloader.py` → `python src/train.py`
- [ ] **6.3** Commit final ke GitHub dengan tag versi

---

## 5. RISIKO & MITIGASI

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| Import path rusak setelah pindah | Semua script error | Gunakan branch terpisah, test dulu sebelum merge |
| Notebook kehilangan output cell | Kehilangan hasil visualisasi | Simpan gambar ke `outputs/figures/` sebelum refactor |
| Path ke dataset berubah | `FileNotFoundError` | Gunakan path relatif dari `PROJECT_ROOT` di semua script |
| Model checkpoint tidak terbaca | Inference gagal | Jangan pindah/rename file `.pkl` dan `.pt` yang sudah ada |
| Dependency circular import | `ImportError` | Urutan import: `utils` ← `preprocessing` ← `dataloader` ← `model` ← `train` |

---

## 6. TEST CHECKLIST PER FASE

```bash
# Test Fase 1 — Modul baru
python src/dataloader.py          # harus print info dataset
python src/model.py               # harus print param count + shape
python src/utils.py               # harus print vocab size

# Test Fase 3 — Integrasi
python models/model_dl/run_experiments.py   # harus train dan simpan checkpoint

# Test Fase 4 — Notebook
jupyter nbconvert --to notebook --execute notebooks/01_eda_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_pycaret_automl.ipynb
```

---

*Dokumen ini dibuat otomatis saat audit pada 22 April 2026. Update setiap fase setelah selesai.*
