# 🌸 Flower Classification (UAP Pembelajaran Mesin)

> Sistem klasifikasi citra bunga **5 kelas** (daisy, dandelion, rose, sunflower, tulip) menggunakan **CNN Scratch** (baseline) dan **Transfer Learning** (**EfficientNet-B0 + fine-tuning** & **MobileNetV2 + freeze**).  
> Disediakan **Streamlit** untuk prediksi gambar secara lokal (pilih model → upload → hasil prediksi + Top-3 + history).

---

## 📌 Table of Contents
1. [Project](#-deskripsi-project)  
   - [Latar Belakang](#-latar-belakang)  
   - [Tujuan](#-tujuan)  
2. [Dataset](#-dataset)  
3. [Eksperimen & Metodologi](#-eksperimen--metodologi)  
   - [EDA Singkat](#-eda-singkat)  
   - [Preprocessing](#-preprocessing)  
   - [Augmentasi](#-augmentasi)  
   - [Pemodelan](#-pemodelan)  
   - [Pemilihan Best Model](#-pemilihan-best-model)  
4. [Hasil Evaluasi & Analisis](#-hasil-evaluasi--analisis)  
   - [Perbandingan Performa](#-perbandingan-performa)  
   - [Confusion Matrix & Error Analysis](#-confusion-matrix--error-analysis)  
5. [Demo Streamlit](#-demo-streamlit)  
6. [Keterbatasan](#-keterbatasan)
7. [Link Live Demo](#-link-live-demo)  
8. [Kontributor](#-kontributor) 

---

## 🧾 Project
Project ini dibuat untuk memenuhi **UAP Mata Kuliah Pembelajaran Mesin A**.  
Tujuan utamanya adalah membangun sistem klasifikasi citra bunga 5 kelas dengan menggunakan tiga performa model (baseline vs 2 transfer learning).  
Output akhir disajikan melalui **Streamlit** agar mudah diuji coba (input gambar dari user → output prediksi).

### 🔍 Latar Belakang
Identifikasi bunga berbasis citra sering dipengaruhi variasi **pencahayaan**, **background**, **sudut pengambilan**, dan **skala objek**.  
Selain itu, sebagian bunga memiliki kemiripan visual (warna/kelopak), sehingga rawan tertukar.  
Karena itu, project ini membandingkan **CNN Scratch** dengan pendekatan **Transfer Learning** untuk mendapatkan prediksi yang lebih stabil.

### 🎯 Tujuan
1. Membuat baseline **CNN Scratch** sebagai pembanding.  
2. Menggunakan **Transfer Learning** (EfficientNet-B0 & MobileNetV2) dengan menggunakan teknik fine tuning untuk EfficientNet-B0 dan teknik freeze untuk MobileNetV2 untuk meningkatkan akurasi dan generalisasi.  
3. Membangun aplikasi Streamlit yang dapat:
   - menerima input gambar dari user (single)
   - menampilkan prediksi Top-1 sampai Top-3 probabilitas
   - menyimpan history prediksi & tombol hapus history

---

## 🗂️ Dataset
Dataset: ✅ (Dataset Bunga)  
Sumber: ✅ ([Dataset Bunga dari Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition))

Jumlah kelas: **5**
- daisy: 764
- dandelion: 1052
- rose: 784
- sunflower: 733
- tulip: 984

Total data: **4317 gambar**.

Distribusi dataset (setelah augmentasi): 5052 gambar.

Contoh struktur folder dataset:
flowers/
  daisy/
  dandelion/
  rose/
  sunflower/
  tulip/

## 🧪 Eksperimen & Metodologi

Bagian ini menjelaskan alur eksperimen dari eksplorasi data sampai pemilihan model terbaik.  
Notebook utama:
- `notebooks/01_training.ipynb` (training)
- `notebooks/02_evaluation.ipynb` (evaluasi)

---

### 📊 EDA Singkat
EDA dilakukan untuk memahami karakteristik data dan potensi sumber error pada klasifikasi.

Langkah EDA yang dilakukan:
- Mengecek **jumlah data per kelas** (distribusi kelas `dandelion` paling dominan).
- Menampilkan **contoh gambar tiap kelas** untuk melihat variasi bentuk, warna, dan tekstur.
- Mengamati variasi **pencahayaan**, **background**, **skala objek**, dan **orientasi** gambar.

---

### 🧼 Preprocessing
Tahapan preprocessing:
- **Resize input**: 224×224
- **Format warna**: RGB
- **Cast ke float32 (rentang tetap 0–255)**
- **Split data**: Train/Val/Test = 70/15/15

Catatan implementasi Streamlit:
- Untuk menjaga bentuk objek tidak “gepeng”, aplikasi Streamlit menggunakan **letterbox** (resize menjaga rasio + padding) sebelum prediksi.

---

### 🧩 Augmentasi
Augmentasi offline per kelas sampai target 1000 gambar/kelas:
- RandomFlip("horizontal")
- RandomRotation(0.10)
- RandomZoom(0.15)
- RandomContrast(0.15)

---

### 🧠 Pemodelan
Tiga model yang diuji dalam penelitian ini:

1. **CNN Scratch (Non-pretrained)**
   - Model baseline CNN dibangun dari nol.

2. **MobileNetV2 (Pretrained)**
   - Menggunakan backbone MobileNetV2 pretrained (ImageNet).
   - Strategi training: freeze dan fine-tuning.

3. **EfficientNet-B0 (Pretrained)**
   - Menggunakan backbone EfficientNet-B0 pretrained (ImageNet).
   - Strategi training: freeze dan fine-tuning.

Model tersimpan dalam format `.keras`:
- `models/scratch_cnn.keras`
- `models/mobilenetv2.keras`
- `models/effnetb0.keras`

---

### 🥇 Pemilihan Best Model
Best model dipilih berdasarkan gabungan beberapa indikator:

Kriteria pemilihan:
1. **Performa evaluasi tertinggi** pada data test (Accuracy dan Macro-F1).
2. **Stabilitas training**: gap train–validation kecil (indikasi overfitting lebih rendah).
3. **Confusion Matrix lebih baik**: error antar kelas lebih sedikit dan lebih merata.

✅ Best Model Pretrained
- Best model: **EfficientNet-B0 + fine-tuning dan MobileNetV2 + freeze**


## 🏆 Hasil Evaluasi & Analisis
### 📌 Perbandingan Performa

| Model | Akurasi (Test) | Hasil Analisis |
|------|----------------:|----------------|
| CNN Scratch | **66.62%** | Baseline. Error tinggi pada kelas mirip (khususnya **tulip ↔ rose** dan **daisy → dandelion**) |
| MobileNetV2 (Freeze) | **89.31%** | Performa tinggi dan efisien. Masih terlihat kebingungan pada **rose ↔ tulip** |
| EfficientNetB0 (FT2) | **92.35%** | **Terbaik**. Error paling sedikit dan lebih merata antar kelas, indikasi generalisasi lebih baik |

---

### 🧩 Confusion Matrix & Error Analysis

#### 1. CNN Scratch — pola error utama
Kesalahan terbesar terlihat pada pasangan kelas yang mirip:
- **tulip → rose (61)**
- **daisy → dandelion (31)**
- **sunflower → dandelion (25)** dan 
- **dandelion → sunflower (19)**
- Indikasi fitur belum cukup kuat menangkap detail spesifik tiap bunga.

#### 2. MobileNetV2 (Freeze) — pola error utama
Error dominan tetap pada:
- **rose → tulip (17)** dan **tulip → rose (14)**: masih jadi pasangan kelas tersulit.
- **daisy → dandelion (9)**: error kecil namun masih muncul.
- Secara keseluruhan performa sudah tinggi, tetapi masih sedikit kalah dari EfficientNetB0 FT2 dalam membedakan detail halus (fine-grained).

#### 3. EfficientNetB0 FT2 — pola error utama
Error jauh lebih kecil dan lebih “terkontrol”. Yang paling terlihat:
- **tulip → rose (12)** dan **rose → tulip (8)**: masih ada ambiguity antar dua kelas ini, tapi jumlahnya kecil dibanding CNN Scratch.
- **daisy → dandelion (7)**: error minor, jauh turun dibanding baseline.
- Kelas **dandelion** dan **sunflower** sangat kuat → model menangkap fitur khas lebih baik.

## 🖥️ Demo Streamlit

Aplikasi Streamlit digunakan sebagai antarmuka untuk mendemokan model klasifikasi bunga secara lokal.  
Alur penggunaan:
1. **Pilih model** (CNN Scratch / MobileNetV2 / EfficientNet-B0)  
2. **Upload gambar** bunga (`.jpg/.jpeg/.png`)  
3. Sistem menampilkan:
   - **Prediksi Top-1** (kelas terbaik)
   - **Confidence** (probabilitas Top-1)
   - **Top-3 probabilitas** (kelas kandidat teratas)
4. Prediksi otomatis tersimpan pada **history** (terbaru di atas).
5. User dapat menghapus riwayat dengan tombol **Hapus History**.

## ⚠️ Keterbatasan

Berikut beberapa keterbatasan yang ditemukan pada proyek klasifikasi bunga ini:

1. **Kemiripan visual antar kelas (fine-grained classification)**  
   Beberapa kelas memiliki ciri yang mirip (warna kelopak, bentuk bunga, dan komposisi gambar).  
   Dari confusion matrix, pasangan yang paling sering tertukar adalah **`rose` ↔ `tulip`**, terutama pada model baseline dan MobileNetV2.

2. **Sensitif terhadap kondisi foto**  
   Performa dapat menurun jika:
   - pencahayaan terlalu gelap/terang (over/under exposed)
   - gambar blur / tidak fokus
   - background terlalu ramai (daun/rumput dominan)
   - objek bunga terlalu kecil di frame

## 🔗 Link Live Demo

Aplikasi Streamlit dapat diakses publik melalui link berikut:
- **Live Demo:** https://uapmlira.streamlit.app/

## 👥 Kontributor

| Nama | NIM | Kelas |
|------|-----|------|
| **Irawana Juwita** | **202210370311446** | **Pembelajaran Mesin A** |
