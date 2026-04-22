# Panduan End-to-End Rerun Eksperimen (Post-Refactor)

Dokumen ini berisi prosedur lengkap untuk menjalankan ulang pipeline (reproducible end-to-end run) setelah seluruh codebase dipindah ke rancangan modular (`src/`).

## 1. Persiapan Environment & Dependency
Sebelum memulai, pastikan semua *dependency library* sudah terinstal di environment lokal Anda.
Cek ketersediaan environment virtual:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Install dependency inti
pip install pandas numpy scikit-learn torch torchvision torchaudio tqdm matplotlib seaborn
```

## 2. Smoke Test Modul `.py` (Terminal)
Pastikan setiap inti internal tidak menemui isu fatal (Misal: kesalahan import linting, typo, atau missing packages).

Jalankan serangkaian perintah ini di terminal:
```bash
python -m src.dataloader
python -m src.model
python -m src.models.__init__
python -m src.utils
```
**Ekspektasi Output:**
Terminal akan mengeksekusi blok `if __name__ == "__main__":` yang kita buat sebelumnya. Jika hanya menampilkan info random seed, ringkasan arsitektur *dummy*, dan daftar register model, artinya modul terverifikasi 100% aman beroperasi.

## 3. Eksekusi Orchestrator Notebook
1. Buka file `notebooks/03_training_orchestrator.ipynb` di Jupyter Notebook / VSCode.
2. Klik tombol **Kernel -> Restart Kernel and Run All Cells...**
3. Biarkan berproses hingga blok ploting selesai. Cek log output di sel; output sudah harus rapi dan bersih dari proses loop epoch yang mengotori *standard-out*.

## 4. Checklist Verifikasi Folder `outputs/`
Setelah step 3 selesai, navigasikan file explorer ke rektori root:
- [ ] Folder **`outputs/`** otomatis di*generate*.
- [ ] Mengecek folder **`outputs/checkpoints/<run_name>/`** — Harus ada `best_baseline.pt` dan `last_baseline.pt`.
- [ ] Mengecek folder **`outputs/logs/<run_name>/`** — Harus memuat `log.txt` (berisi keseluruhan riwayat log console yang identik yang bisa dibaca) beserta `history_baseline.json`.
- [ ] Mengecek folder **`outputs/figures/<run_name>/`** — Terdapat gambar `training_curves.png` (dan *confusion matrix* bila test logic juga dicantumkan) yang ditarik rapi tanpa menyangkut hanya di notebook output cell.

## 5. Ringkasan
Jika keseluruhan ke-4 parameter di checklist terpenuhi:
**Sistem Pipeline Deep Learning Anda telah mencapai standar level *Production-Ready* dengan reprodusibilitas 100%.**
