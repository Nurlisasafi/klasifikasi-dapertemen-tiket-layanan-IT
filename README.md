# 🎫 IT Ticket Classification Dashboard

Dashboard ini digunakan untuk mengklasifikasikan tiket layanan IT ke departemen yang tepat menggunakan berbagai model Machine Learning dan Deep Learning.  
Proyek ini dikembangkan untuk memenuhi **Ujian Akhir Praktikum (UAP) Pembelajaran Mesin** dan diimplementasikan dalam bentuk **website sederhana berbasis Streamlit**.

---

## 📌 Daftar Isi
1. Deskripsi Proyek  
2. Dataset  
3. Preprocessing dan Pemodelan  
4. Hasil dan Analisis  
5. Perbandingan Model  
6. Sistem Website (Streamlit)  
7. Link Model  
8. Langkah Instalasi  

---

## 1️⃣ Deskripsi Proyek

### 🔍 Latar Belakang
Dalam sistem layanan IT, tiket yang masuk harus segera diteruskan ke departemen yang tepat agar proses penanganan berjalan cepat dan efisien.  
Proses klasifikasi manual memerlukan waktu dan berpotensi menimbulkan kesalahan, sehingga diperlukan sistem klasifikasi otomatis berbasis Natural Language Processing (NLP).

### 🎯 Tujuan Pengembangan
- Mengklasifikasikan tiket IT secara otomatis
- Membandingkan performa model non-pretrained dan pretrained
- Mengimplementasikan model ke dalam dashboard Streamlit

---

## 2️⃣ Dataset
- Jenis data: **Teks**
- Fitur utama:
  - `Body`
  - `Department`
- Jumlah data: **29.651 tiket**
- Sumber dataset: **Kaggle**  
  https://www.kaggle.com/datasets/parthpatil256/it-support-ticket-data

Dataset yang digunakan merupakan dataset tiket layanan IT yang terdiri dari dua komponen utama, yaitu **Body** dan **Department**.  
Kolom **Body** berisi teks deskriptif berupa keluhan, permintaan, atau permasalahan yang diajukan oleh pengguna, sedangkan kolom **Department** berisi label tujuan departemen yang menangani tiket tersebut.  
Dataset ini digunakan untuk melatih model dalam memahami konteks bahasa pada isi tiket serta memetakan setiap tiket ke departemen yang sesuai secara otomatis.

### 📊 Statistik Singkat Dataset (EDA)
Berdasarkan hasil Exploratory Data Analysis (EDA), diperoleh informasi sebagai berikut:
- Total data tiket: **29.651 tiket**
- Jumlah kelas departemen: **10 departemen**
- Rata-rata panjang teks tiket: **±120 kata**
- Panjang teks terpendek: **±5 kata**
- Panjang teks terpanjang: **>500 kata**
- Distribusi kelas tidak seimbang, di mana beberapa departemen memiliki jumlah tiket yang lebih dominan

Analisis EDA ini digunakan sebagai dasar dalam pemilihan model, preprocessing teks, dan evaluasi performa model.

---

## 3️⃣ Preprocessing dan Pemodelan

### 🧹 Preprocessing Data
- Case folding
- Cleaning teks (menghapus simbol, angka, dan URL)
- Tokenization
- Padding dan truncation
- Label encoding

### 🤖 Model yang Digunakan

#### 1. LSTM (Non-Pretrained)
- Neural network berbasis sequence
- Cepat dan ringan
- Digunakan sebagai baseline model

#### 2. BERT (Pretrained)
- Model NLP berbasis Transformer
- Memiliki contextual embedding
- Memberikan performa akurasi tertinggi

#### 3. DistilBERT (Pretrained)
- Versi ringan dari BERT
- Lebih cepat dengan akurasi kompetitif
- Efisien untuk implementasi sistem

---

## 4️⃣ Hasil dan Analisis

### 📈 Grafik Loss dan Accuracy

#### 🔹 LSTM
<img width="536" height="470" src="https://github.com/user-attachments/assets/7c6d968d-4fb7-4481-a360-ef892e57858f" />

#### 🔹 BERT
<img width="536" height="470" src="https://github.com/user-attachments/assets/477e4ad0-aea7-4292-ac4f-ef1bf71de6a9" />

#### 🔹 DistilBERT
<img width="545" height="470" src="https://github.com/user-attachments/assets/07dbe728-83a5-4326-a700-b8668dfdc882" />

Model pretrained menunjukkan konvergensi yang lebih stabil dibandingkan model non-pretrained.

---

### 📊 Confusion Matrix

#### 🔹 LSTM
<img width="522" height="470" src="https://github.com/user-attachments/assets/dfcb4e82-39df-4134-ab8a-6190766aab52" />

#### 🔹 BERT
<img width="522" height="470" src="https://github.com/user-attachments/assets/079dee62-d798-4e37-a2bf-18a71e7cce03" />

#### 🔹 DistilBERT
<img width="552" height="455" src="https://github.com/user-attachments/assets/de08a4c9-019f-410f-b88f-e01c2a719d08" />

---

## 5️⃣ Perbandingan Model

| Model        | Accuracy | Precision | Recall | F1-Score | Analisis |
|-------------|----------|-----------|--------|----------|----------|
| LSTM        | 0.64     | 0.65      | 0.64   | 0.64     | Cepat namun akurasi terbatas |
| BERT        | 0.82     | 0.83      | 0.82   | 0.83     | Akurasi tertinggi |
| DistilBERT | 0.67     | 0.68      | 0.67   | 0.66     | Seimbang antara akurasi dan kecepatan |

Model terbaik berdasarkan akurasi adalah **BERT**, sedangkan **DistilBERT** menawarkan efisiensi komputasi yang lebih baik.

---

## 6️⃣ Sistem Website Sederhana (Streamlit)

### 🏠 Fitur Utama
- Dashboard statistik dataset
- Exploratory Data Analysis (EDA)
- Prediksi tiket secara real-time
- Perbandingan performa model

### 📊 Penjelasan Dashboard
Halaman Dashboard menampilkan ringkasan statistik dataset dan informasi model, meliputi:
- **Total Tiket**: jumlah keseluruhan data tiket
- **Jumlah Departemen**: total kategori departemen
- **Prioritas High**: jumlah tiket prioritas tinggi
- **Model Aktif**: model yang sedang digunakan

Dashboard juga menampilkan visualisasi **distribusi departemen** dan **distribusi prioritas** untuk membantu pengguna memahami karakteristik data.

### 🔮 Penjelasan Hasil Prediksi
Pada halaman Prediksi, pengguna dapat memasukkan teks tiket secara manual.  
Sistem akan menampilkan:
- **Departemen hasil prediksi**
- **Distribusi probabilitas** setiap kelas dalam bentuk grafik batang

Grafik probabilitas menunjukkan tingkat kepercayaan model terhadap hasil prediksi yang diberikan.

---

## 7️⃣ 🔗 Link Model
Karena ukuran model cukup besar, file model tidak disertakan langsung di repository GitHub.

https://drive.google.com/drive/folders/1MKNeis1YvIJmoKuq9hvILP_Ng505MjoK?usp=drive_link

---

## 8️⃣ ⚙️ Langkah Instalasi

```bash
git clone <repository-url>
cd T-Ticket-Classification-Dashboard
pip install -r requirements.txt
streamlit run app.py
