# Model Design Document

## Model Type:
BiLSTM

## Reason:
Model **BiLSTM** dipilih karena:
- Struktur sederhana dan mudah dipahami untuk implementasi multi-output.
- Lebih cepat ditraining dibanding arsitektur yang lebih kompleks.
- Efektif menangkap konteks dua arah (kiri-kanan) pada teks ulasan berbahasa Indonesia.
- Mudah dijaga agar total parameter tetap di bawah batas **2,000,000**.

## Architecture:
- **Embedding**: `nn.Embedding(vocab_size=10000, embedding_dim=128, padding_idx=0)`
- **BiLSTM**: `nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)`
- **Shared Dense + Dropout**:
  - Ambil representasi urutan dari gabungan hidden state akhir forward + backward (`2 * hidden_dim = 256`)
  - `nn.Dropout(0.4)`
  - `nn.Linear(256, 128)` + ReLU
- **Output Heads (Multi-output)**:
  - **Sentiment head**: `nn.Linear(128, 2)`
  - **Emotion head**: `nn.Linear(128, 5)`

## Parameter Count:
Dengan konfigurasi:
- `vocab_size = 10,000`
- `embedding_dim = 128`
- `hidden_dim = 128`
- `num_layers = 1`
- `sentiment_classes = 2`
- `emotion_classes = 5`

Estimasi parameter:
- **Embedding**: `10000 * 128 = 1,280,000`
- **BiLSTM**: `264,192`
- **Dense (shared)**:  
  - `Linear(256 -> 128)`: `256*128 + 128 = 32,896`
- **Output Heads**:
  - Sentiment head `Linear(128 -> 2)`: `128*2 + 2 = 258`
  - Emotion head `Linear(128 -> 5)`: `128*5 + 5 = 645`

**TOTAL**: `1,578, -ish` (tepatnya **1,577,991**)

✅ Memenuhi constraint **< 2,000,000** parameter.

## Training Speed:
Perkiraan pada dataset ~5.4k sampel, `max_seq_len=64`, batch size 64:
- **Estimate per epoch**: ~1–4 menit (GPU) atau ~5–15 menit (CPU modern)
- **Total training (10 epoch + early stopping)**: ~10–40 menit (GPU) atau ~30–150 menit (CPU)

Agar target cepat tercapai:
- Gunakan `max_seq_len=64`
- Gunakan `batch_size=64` (atau 128 jika memori cukup)
- Aktifkan early stopping (`patience=3`)

## Why Suitable for E-Commerce:
- **Menangkap konteks ulasan**: BiLSTM memahami pola kalimat seperti negasi dan opini campuran pada review produk.
- **Efisien untuk deployment**: Ukuran model ringan, cocok untuk eksperimen cepat dan iterasi tim.
- **Mendukung dua tugas sekaligus**: Satu backbone untuk prediksi **sentiment** dan **emotion**, sehingga training/inference lebih hemat.