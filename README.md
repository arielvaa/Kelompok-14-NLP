# 🧠 Fine-Grained Emotion Classification (20 Classes)

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan sistem klasifikasi teks yang mampu mengenali emosi dari kalimat berbahasa Inggris secara lebih rinci (fine-grained). Dataset yang digunakan memiliki 20 kategori emosi, sehingga model tidak hanya membedakan sentimen positif atau negatif, tetapi juga mampu menangkap emosi yang lebih spesifik seperti marah, sedih, cemas, dan lainnya.

Dalam proyek ini akan dilakukan perbandingan dua pendekatan utama dalam Natural Language Processing (NLP), yaitu Machine Learning (ML) dan Deep Learning (DL), untuk mengetahui metode mana yang lebih efektif dalam memahami emosi pada teks.

---

## 👥 Anggota Kelompok 14

| Nama                        | NIM        | GitHub                          |
|-----------------------------|-----------|----------------------------------|
| Arya Muda Siregar          | 123450063 | @Aryamuda                        |
| Arielva Simon Siahaan      | 123450105 | @arielvaa                        |
| Haikal Fransisko Simbolon  | 123450106 | @Haikal-Fransisko-Simbolon       |

---

## 📊 Dataset
Dataset yang digunakan adalah **20-Emotion Text Classification Dataset** yang berisi 79.595 kalimat dengan 20 label emosi yang berbeda.

- Total data: 79,595
- Jumlah kelas: 20 emosi
- Bahasa: Inggris
- Task: Multi-class text classification

Contoh label emosi:
- happiness, anger, sadness, fear, love, surprise, dan lainnya.

---

## ⚙️ Metodologi

### 1. Machine Learning (ML)
Pendekatan ML akan menggunakan framework PyCaret dengan representasi fitur berbasis TF-IDF. Beberapa algoritma yang akan dibandingkan antara lain:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

Model terbaik akan dipilih berdasarkan performa evaluasi.

### 2. Deep Learning (DL)
Pendekatan DL akan menggunakan PyTorch dengan arsitektur seperti:
- LSTM
- Transformer sederhana

Model akan dilatih untuk memahami pola emosional dalam teks dengan tetap memperhatikan batasan jumlah parameter.

---

## 🎯 Tujuan
Tujuan utama dari proyek ini adalah:
- Membandingkan performa Machine Learning dan Deep Learning dalam klasifikasi emosi
- Mengidentifikasi model terbaik untuk menangani klasifikasi emosi dengan banyak kelas
- Memberikan insight terhadap efektivitas masing-masing pendekatan dalam memahami teks

---
