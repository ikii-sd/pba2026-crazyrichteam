# 🛍️ Analisis Sentimen & Emosi Ulasan E-Commerce Bahasa Indonesia

> **Mata Kuliah:** Pengolahan Bahasa Alami (PBA) — 2026 <br>
> **Tim:** Crazy Rich Team <br>
> **Dataset:** [PRDECT-ID](https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification/data) — Indonesian E-Commerce Product Reviews Dataset <br>
> **🚀 Live Demos:**
> - **[ML Model](https://huggingface.co/spaces/Hash-SD/ecommerce-sentiment-analysis)** — Scikit-learn based sentiment analyzer
> - **[Deep Learning Model](https://huggingface.co/spaces/Hash-SD/ecommerce-sentiment-emotion-dl)** — PyTorch BiLSTM sentiment & emotion analyzer

---

## 👥 Anggota Kelompok 7

| No | Nama | NIM |
|----|------|-----|
| 1  | Hermawan Manurung | 122450069 |
| 2  | Ahmad Rizqi       | 122450138 |
| 3  | Ibrahim Al-kahfi  | 122450100 |

---

## 🎯 Deskripsi Proyek

Proyek ini membangun pipeline NLP end-to-end untuk menganalisis **sentimen** (Positif / Negatif)
dan **emosi** (Bahagia, Sedih, Takut, Cinta, Marah) dari ulasan produk e-commerce berbahasa
Indonesia menggunakan dataset PRDECT-ID (5.400 ulasan, 29 kategori produk).

**Latar Belakang & Masalah Utama:**
Industri e-commerce saat ini memiliki ketergantungan yang luar biasa tinggi terhadap ulasan dan keluhan yang ditulis oleh para pelanggannya. Ulasan pembeli berisi sekumpulan simpulan opini yang dinamis, jujur, dan

---

## 📐 Arsitektur Modular

Proyek ini mengikuti prinsip **separation of concerns**:

| File | Tanggung Jawab |
|------|----------------|
| `src/preprocessing.py` | **Modul Python murni** — semua logika cleaning, normalisasi slang, stemming |
| `src/__init__.py` | Package initializer — ekspor fungsi publik |
| `notebooks/01_eda_preprocessing.ipynb` | **Notebook** — EDA, visualisasi |
| `notebooks/02_pycaret_automl.ipynb` | **Notebook** — AutoML untuk klasifikasi model |
| `notebooks/03_training_orchestrator.ipynb` | **Notebook Master** — Orkestrasi eksekusi tanpa *looping* untuk DL Pipeline |

> ⚠️ **Aturan utama:** Fungsi inti seperti komputasi `clean_text()` atau `train()` **TIDAK** lagi ditulis di dalam
> notebook. Notebook hanyalah sebuah *orchestrator* eksekutif yang mengimpor dan menjalankan modul:
> ```python
> from src.dataloader import build_full_pipeline
> from src.train import fit
> ```

---

## 🌟 Pembaruan Pipeline Deep Learning (*Production-Ready State*)

Setelah melalui sesi *refactoring* menyeluruh, basis kode kini menerapkan prinsip-prinsip operasional kelas-produksi (*Production-Ready*) dengan sorotan pembaruan poin-poin berikut:

- **Orkestrasi Pelatihan Otomatis:** Menangani logika pelatihan (*train_one_epoch*) bertingkat layaknya *Early Stopping*, *LR Scheduler*, hingga pemantauan metrik validasi langsung melalui skrip `train.py`.
- **Manajemen Eksperimen (Run Tracker):** Sistem `setup_run_env` di `utils.py` menjebak dan mengalokasikan hasil eksekusi (dinamai otomatis berdasarkan *timestamp*) ke dalam ruang terisolir, bebas dari *conflict-save*.
- **Hierarki Output Tanpa Polusi File:** Segala jenis *output* (`logs.txt`, metrik `.json`, gambar `*.png`, bobot `*.pt`) didorong masuk ke direktori `/outputs/` dan tidak membayangi berkas fundamental pada folder *root* (dipermanenkan pengabaiannya via `.gitignore`).
- **Registri Varian Model (Build Factory):** Perpindahan antara model konfigurasi ringan (Basic 1 Layer) dan konfigurasi besar secara gampang dilakukan melalui panggilan sentral tanpa redundansi script file arsitektur.
- **Standar Reprodusibilitas Penuh:** *Checkpointing* metrik akhir, pembekuan seed variabel deterministik, hingga metode evaluasi yang bisa diulang-kembali pada mesin apa pun.

---

## 📁 Struktur Proyek

```
pba2026-crazyrichteam/
│
├── 📂 src/                                ← Package preprocessing
│   ├── __init__.py                        ← Package initializer & public exports
│   └── preprocessing.py                  ← Modul utama: clean_text(), batch_clean()
│
├── 📂 notebooks/                          ← Jupyter Notebooks
│   ├── 01_eda_preprocessing.ipynb        ← EDA + eksekusi preprocessing
│   └── 02_pycaret_automl.ipynb           ← Skrip notebook melatih model
│   
├── 📂 models/                             ← Model terlatih (di-generate notebook 02)
│   ├── best_ml_model.pkl                 ← Model sentimen terbaik + TF-IDF
│   ├── best_emotion_model.pkl            ← Model emosi terbaik + TF-IDF
│   └── tfidf_vectorizer.pkl              ← TF-IDF vectorizer
│
├── 📂 data/
│   ├── clean/
│   │   └── cleaned_dataset.csv           ← Output preprocessing (di-generate notebook)
│   ├── figures/                          ← Plot EDA tersimpan (di-generate notebook)
│   └── PRDECT-ID Dataset.csv             ← Dataset mentah (separator titik koma `;`)
│
├── app.py                                ← 🚀 Gradio App
├── requirements.txt                      ← Daftar dependensi Python
├── .gitignore
└── README.md
```

---

## 📊 Tentang Dataset PRDECT-ID

| Atribut | Detail |
|---------|--------|
| Total sampel | 5.400 ulasan |
| Kategori produk | 29 kategori |
| Separator CSV | `;` (titik koma) |
| Encoding | UTF-8 |
| Label Sentimen | `Positif` (2.578) · `Negatif` (2.820) |
| Label Emosi | `Bahagia` · `Sedih` · `Takut` · `Cinta` · `Marah` |
| Kolom teks utama | `Customer Review` |

---


## 🚀 Gradio App — Demo Interaktif

File `app.py` di root proyek adalah aplikasi Gradio yang menggunakan **PyTorch BiLSTM** dan siap deploy ke **Hugging Face Spaces**.

### Fitur Aplikasi

| Fitur | Detail |
|-------|--------|
| **Input** | Teks ulasan produk e-commerce bahasa Indonesia |
| **Output 1** | Sentimen: `Positif` / `Negatif` + confidence score |
| **Output 2** | Emosi: `Bahagia` / `Sedih` / `Takut` / `Cinta` / `Marah` + confidence score |
| **Model Sentimen** | BiLSTM (Bidirectional LSTM) + embedding layer |
| **Model Emosi** | BiLSTM (shared dengan sentiment, separate output head) |
| **Vocabulary** | 6.155 tokens dari PRDECT-ID dataset |
| **Max Sequence Length** | 64 tokens (dengan padding/truncation) |
| **Framework** | PyTorch + Gradio |

### Cara Menjalankan Lokal

```bash
# 1. Install dependensi (termasuk PyTorch)
pip install -r requirements.txt

# 2. Jalankan Gradio app
python app.py
```

App akan berjalan di `http://localhost:7860`.

### Contoh Ulasan yang Bisa Dicoba

| Ulasan | Prediksi |
|--------|----------|
| `"Barang bagus, penjual ramah, pengiriman cepat!"` | Positive / Happy |
| `"Kecewa, barang rusak dan tidak sesuai deskripsi."` | Negative / Sadness |
| `"Takut beli lagi, merasa ditipu penjual."` | Negative / Fear |
| `"Alhamdulillah, produk ori sesuai gambar. Sangat puas!"` | Positive / Happy |

---

## ⚙️ Setup & Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/<org>/pba2026-crazyrichteam.git
cd pba2026-crazyrichteam
```

### 2. Buat Virtual Environment (Rekomendasi)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 5. Jalankan Notebook untuk Generate Model

```bash
cd notebooks

# Notebook 1: EDA + Preprocessing → cleaned_dataset.csv
jupyter notebook 01_eda_preprocessing.ipynb

# Notebook 2: AutoML + Benchmark → models/*.pkl
jupyter notebook 02_pycaret_automl.ipynb
```

> **Catatan:** Pastikan file `PRDECT-ID Dataset.csv` berada di direktori **root** proyek
> (sejajar dengan folder `src/` dan `notebooks/`). Setelah kedua notebook dijalankan,
> folder `models/` akan berisi file `.pkl` yang dibutuhkan oleh `app.py`.

---

## 🔬 Modul `src/preprocessing.py`

### Pipeline `clean_text()` — 14 Langkah

```
 1.  Lowercase
 2.  Hapus URL (http, https, www)
 3.  Hapus HTML/XML tags
 4.  Konversi emoji → teks deskriptif  (😊 → "smiling face")
 5.  Normalisasi harga kontekstual     (50k → 50 ribu, Rp50.000 → harga)
 6.  Hapus angka
 7.  Hapus tanda baca & karakter non-alfanumerik
 8.  Normalisasi karakter repetisi     (bagussss → baguss)
 9.  Normalisasi slang e-commerce      (bgs → bagus, gak → tidak, bgt → banget)
10.  Hapus token kosong
11.  Hapus stopword                    (Sastrawi 815 kata + tambahan manual)
12.  Stemming morfologis               (PySastrawi CachedStemmer)
13.  Filter token pendek               (min 2 karakter)
14.  Gabung token & normalisasi whitespace
```

### Kamus Slang — 140+ Entri

Mencakup kategori:
- **Negasi & modalitas:** `gak/ga/nggak → tidak`, `blm → belum`, `udah/udh → sudah`
- **Intensifier:** `bgt/bngt → banget`, `bener → benar`
- **Konjungsi & preposisi:** `yg → yang`, `dgn → dengan`, `krn → karena`, `tp → tapi`
- **Penilaian produk:** `bgs → bagus`, `mantep/mantul → mantap`, `joss → bagus`, `ori → original`
- **Transaksi & pengiriman:** `seller → penjual`, `packing → kemasan`, `ongkir → ongkos kirim`
- **Sapaan:** `makasih/thx/tq → terima kasih`
- **Ekspresi:** `wkwk/haha/lol → ""` (dihapus)

### Fungsi Publik

```python
from src.preprocessing import clean_text, batch_clean, get_stopwords, get_stemmer

# Bersihkan satu teks
teks_bersih = clean_text("Barang bagussss bgt!! seller ramah 👍")
# → 'barang baguss banget jual ramah thumbs up'

# Bersihkan seluruh kolom DataFrame
df["clean_review"] = batch_clean(df["Customer Review"], verbose=True)

# Akses stopwords & stemmer singleton
stopwords = get_stopwords()   # set of 815+ kata
stemmer   = get_stemmer()     # Sastrawi CachedStemmer
```

### Test Mandiri via CLI

```bash
python src/preprocessing.py
```

Menjalankan 8 kasus uji dari terminal tanpa perlu membuka Jupyter.

---

## 🗂️ Contoh Before → After Preprocessing

| Teks Mentah | Teks Bersih |
|-------------|-------------|
| `Barang bagussss bgt!! Penjual ramah & respon cepat 👍` | `barang baguss banget jual ramah respons cepat thumbs up` |
| `KECEWA!! gak sesuai deskripsi. Harga 50k tapi jelek bgt` | `kecewa sesuai deskripsi harga ribu buruk banget` |
| `mantep paten joss, fast delivery, packing aman` | `mantap paten bagus cepat delivery kemas aman` |
| `bgs banget, harga Rp75.000 worth it. ga ada yg rusak` | `bagus banget harga worth it kemas aman rusak` |
| `udah 3x beli krn harganya mura tp kualitas oke. Makasih!` | `beli harga murah kualitas terima kasih` |

---

## 📈 Temuan Utama EDA

| Aspek | Temuan |
|-------|--------|
| **Sentimen** | Sedikit imbalanced — Negatif 52.3% vs Positif 47.7% |
| **Emosi** | Sangat imbalanced — Happy 32.8% mendominasi, Anger 13.0% paling sedikit |
| **Panjang teks** | Median 78 karakter / 14 kata; rentang 3 – 1.058 karakter |
| **Missing values** | 2 baris di kolom `Sentiment` & `Emotion` |
| **Duplikat** | 7 baris duplikat full-row |
| **Emoji** | Mayoritas review tanpa emoji; yang ada didominasi 👍 dan 😊 |

> **Implikasi untuk modeling:** Pertimbangkan `class_weight='balanced'` atau teknik
> oversampling (SMOTE) pada Checkpoint berikutnya karena kelas Emosi sangat tidak seimbang.

---


## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — Institut Teknologi Sumatera (ITERA), 2026.
