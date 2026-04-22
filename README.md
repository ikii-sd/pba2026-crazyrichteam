# Klasifikasi Multi-Target: Sentimen & Emosi Ulasan E-Commerce Indonesia (PRDECT-ID)

Repositori ini menyajikan implementasi *machine learning* utuh untuk klasifikasi teks *multi-output* (Analisis Sentimen dan Emosi) pada *dataset* ulasan e-commerce berbahasa Indonesia (PRDECT-ID). Dibangun di atas fondasi arsitektur **Bidirectional LSTM (BiLSTM)** menggunakan **PyTorch**, repositori ini telah dirancang ulang agar menjunjung tinggi prinsip *clean code*, terstruktur secara modular, dan berada pada standar kualitas *production-ready*.

---

## ✨ Fitur Utama

- **Data Pipeline Terpusat:** Modul `dataloader.py` mengorkestrasi pemrosesan teks, prapemrosesan dinamis (*dynamic vocabulary mapping*), konversi tensor, hingga pemecahan himpunan data pelatihan, validasi, dan pengujian secara statis (`80:10:10`).
- **Orkestrasi Pelatihan Otomatis (*Hands-off Training*):** Modul `train.py` merangkum logika iterasi epoch, validasi silang, penerapan *Early Stopping*, hingga pemantauan *learning rate scheduler* secara mandiri.
- **Manajemen Lingkungan Eksperimen:** Fungsi `setup_run_env` (berada di modul `utils.py`) secara cerdas merekam dan mengalokasikan artefak setiap *run* eksperimen secara terisolasi tanpa ada risiko tumpang tindih fail *output*.
- **Sistem Registri Model Cerdas:** Dukungan pertukaran varian *hyperparameter* atau pergantian arsitektur (dari *Baseline* 1 Lapis hingga edisi komputasi penuh dengan matriks dimensi *Large*) hanya dengan mengubah sebaris identitas konfigurasi.
- **Perekaman Log Komprehensif:** Dukungan ganda *logging* yang terintegrasi di terminal (*console*) dan disimpan secara pararel pada arsip fail `outputs/logs/` bernomor presisi.

---

## 🛠 Prasyarat (*Prerequisites*)

Pastikan infrastruktur perangkat lunak Anda memenuhi prasyarat berikut sebelum memulai rilis:

- **Sistem Operasi**: Kompatibel dengan Windows, macOS, maupun distribusi Linux populer.
- **Bahasa Pemrograman**: Python versi `3.9` atau `3.10` dan ke atas.
- **Akselerasi Perangkat Keras (Opsional)**: Repositori mendeteksi ketersediaan platform CUDA (*Compute Unified Device Architecture*) secara mandiri. Bila tidak terdeteksi, program otomatis berjalan di atas CPU.

---

## 🚀 Setup Ekosistem Lokal

Lekaskan panduan persiapan berikut dari *root* repositori pada aplikasi konsol/terminal. Sangat disarankan untuk membentuk lingkungan isolasi mandiri (*virtual environment*).

```bash
# 1. Inisialisasi virtual environment
python -m venv venv

# 2.A Aktivasi environment (Bagi pengguna Linux/macOS)
source venv/bin/activate
# 2.B Aktivasi environment (Bagi pengguna Windows)
.\venv\Scripts\activate

# 3. Instalasi perpustakaan (dependencies) terkait sistem
pip install -r requirements.txt
```

---

## 📂 Peletakan Format Dataset

Dokumentasi ini mengabaikan sinkronisasi *cloud tracking version (git)* untuk data mentah sehingga kapasitas git Anda tetap ideal. Taruh pustaka data primer dalam skema folder hierarkis yang menginduk pada lokasi `data/`.

Bentuk peletakan idealnya adalah:

```text
├── data/
│   ├── PRDECT-ID Dataset.csv          # File Mentah
│   └── clean/
│       └── cleaned_dataset.csv        # Hasil eksekusi final (preprocessing)
```

> **Catatan Penting**: Identitas dan jalur baca (path) menuju `data/` telah diabadikan ketat di dalam `.gitignore`. Dataset berstatus sensitif Anda tidak akan pernah terbuka untuk dipublikasi ke dunia melalui git *pull/push* log.

---

## 💻 Panduan Eksekusi

### Metode A: Integrasi Lewat "Orchestrator Notebook" (Sangat Direkomendasikan)
1. Aktifkan platform Jupyter Server atau akses dari aplikasi editor VSCode.
2. Navigasikan struktur arsip menuju direktori `notebooks/` lalu buka `03_training_orchestrator.ipynb`.
3. Tekan mekanisme **Run All Cells** (sebagai alternatif manfaatkan instruksi `Restart Kernel & Run All`).
4. Semua progres akan dieksekusi transparan—dari transmisi impor konfigurasi *dataset* pada prapemrosesan sampai ringkasan visual *Confusion Matrix* di baris bawah lembaran—tanpa kehadiran penulisan logika fungsional panjang *looping training* di dalam antarmuka blok *notebook*.

### Metode B: Investigasi Menggunakan Ekstensi Uji Modul (*Smoke Test*)
Untuk melakukan asesmen independensi antar blok `src`, Anda disarankan melangsungkan parameter CLI internal dengan syntax argumen sebagai berikut:

```bash
# Pengujian modul registri model tanpa data set utama
python -m src.models.__init__

# Uji coba sistem environment dan penjejakan logs
python -m src.utils
```

---

## 📁 Infrastruktur Hierarki Output Terekam

Setiap *run* pelatihan mengkonstruksi struktur data penyimpanan berdasarkan waktu dan jenis iterasi. Bentuk tata letak ini dikalsifikasikan dari tag *Run Name ID* (secara harfiah berupa kompresi `TahunBulanHari_JamMenitDetik_namamodel`). Keempat variabel fundamental berikut tak akan diseminarkan pada ruang publik/server karena filter proteksi git. 

- `outputs/logs/<run_name>/log.txt` : Salinan ekstensif rekaman peringatan dan laporan riwayat performa model.
- `outputs/logs/<run_name>/history_*.json` : Konvergensi data *loss function* secara utuh sebagai rujukan komputasi plot independen.
- `outputs/checkpoints/<run_name>/` : Taksiran kompresi final parameter bobot sistem paling tajam (`best_*.pt`) dan rekam iterasi yang paling akhir (`last_*.pt`).
- `outputs/figures/<run_name>/` : Seluruh berkas pemetaan visual performa evaluasi disortir ke folder gambar tanpa mencecerkan akar sistem di file *root*.

---

## 🧪 Reprodusibilitas (*Reproducibility*)

Repositori membekukan sifat probabilistik deterministik pelatihan dengan penerapan fungsi **`set_seed(42)`**. Kontrol pengacakan ini mereplikasi distribusi angka probabilitas pada sistem basis Python, Numpy Array, kerangka PyTorch, dan interaksi inti akselerator bawaan Nvidia dari CUDNN. Proses iteratif tidak akan menceritakan inkonsistensi yang tidak diinginkan di setiap *re-run* sistem eksperimen yang sama.

---

## ❓ Panduan Analisis Kendala (*Troubleshooting*)

1. **`ModuleNotFoundError: No module named 'src'`**
   - **Solusi:** Kesalahan sering terjadi akibat eksekusi skrip CLI yang berada *di dalam* ruang modulnya sendiri (`cd src`). Pastikan Python senantiasa dijalankan di tingkat parameter direktori atas proyek atau lokasi root (*main working directory*).

2. **Kinerja Pelatihan Cukup Memakan Waktu (Tanpa GPU)**
   - **Solusi:** Sistem pada file log `[Device: cpu]` mengindikasikan ketiadaan platform paket *Compute Unified Device Architecture (CUDA)* atau instalasi versi PyTorch di mesin pengguna tidak optimal mendeteksi tipe platform CPU/GPU-nya. Upayakan instalasi ulang dari halaman paket [Get Started - Meta PyTorch](https://pytorch.org/get-started/locally/).

3. **Data Path Error / Fail Load CSV File**
   - **Solusi:** Pastikan keberadaan rekam valid modul csv di `data/clean/cleaned_dataset.csv`. Konfirmasikan presisi baris `csv_path` internal pada metode `build_full_pipeline()` jika menggunakan alokasi di tempat lain.
