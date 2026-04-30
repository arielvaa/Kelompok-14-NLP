# 🧠 Fine-Grained Emotion Classification (20 Classes)

## 🚀 Live Demo
**Try the deployed model:**  
- [SVMC (Machine Learning) Emotion Classification App on Hugging Face Spaces](https://huggingface.co/spaces/Arielva/pba2026-kelompok14)
- [BiLSTM (Deep Learning) Emotion Classification App on Hugging Face Spaces](https://huggingface.co/spaces/Arielva/dl_pba2026-kelompok14)
- [ArXiv Paper _Published_](https://arxiv.org/abs/2604.26310)

---

## 📌 Deskripsi Proyek
Proyek ini bertujuan membangun sistem klasifikasi teks yang mampu mengidentifikasi emosi dalam kalimat berbahasa Inggris secara lebih detail. Dataset yang digunakan mencakup 20 kategori emosi, sehingga model tidak hanya membedakan sentimen positif dan negatif, tetapi juga mengenali emosi spesifik seperti marah, sedih, cemas, dan lainnya.

Selain itu, proyek ini membandingkan dua pendekatan utama dalam Natural Language Processing (NLP), yaitu **Machine Learning (ML)** dan **Deep Learning (DL)**, untuk menentukan metode yang paling efektif dalam memahami emosi pada teks.

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
Pendekatan ML menggunakan scikit-learn dengan representasi fitur berbasis TF-IDF. Model yang telah dibenchmark:
- Logistic Regression - Baseline model  
- Naive Bayes (MultinomialNB) - Text classification  
- Support Vector Machine (LinearSVC) - Best ML model (selected)  

---

### 2. Deep Learning (DL)
Pendekatan Deep Learning menggunakan PyTorch dengan arsitektur:

- **BiLSTM (Bidirectional LSTM)**
- **GRU**
- **Transformer (baseline attention-based model)**

Model dilatih untuk menangkap konteks emosional dalam teks menggunakan embedding layer dan sequence modeling.

---

## 📈 Hasil Evaluasi Model

### 🧠 Deep Learning Results

#### 🔹 BiLSTM
- **Accuracy:** 0.89  
- **Macro F1:** 0.88  
- **Weighted F1:** 0.89  
- **Training Time:** 5m 53s  

#### 🔹 GRU
- **Accuracy:** 0.88  
- **Macro F1:** 0.88  
- **Weighted F1:** 0.88  
- **Training Time:** 7m 09s  

#### 🔹 Transformer
- **Accuracy:** 0.87  
- **Macro F1:** 0.86  
- **Weighted F1:** 0.87  
- **Training Time:** 11m 09s  

---

### 📊 Insight Performa Deep Learning
- **BiLSTM menjadi model terbaik** dalam eksperimen ini.
- GRU memberikan performa yang sangat dekat tetapi tidak mengungguli BiLSTM.
- Transformer memiliki performa lebih rendah pada dataset ini, kemungkinan karena:
  - Data belum cukup besar untuk memaksimalkan attention mechanism
  - Hyperparameter belum optimal

---

## 🏆 Perbandingan Model Keseluruhan

| Model        | Accuracy | F1-Score | Training Time |
|--------------|----------|----------|--------------|
| SVM (LinearSVC) | 0.8811 | 0.8808 | - |
| Logistic Regression | 0.8656 | - | - |
| Naive Bayes | 0.8381 | - | - |
| **BiLSTM** | **0.89** | **0.89** | **5m 53s** |
| GRU | 0.88 | 0.88 | 7m 09s |
| Transformer | 0.87 | 0.86 | 11m 09s |

---

## 🧠 Kesimpulan Akhir

- **BiLSTM adalah model terbaik secara keseluruhan**
- Deep Learning mengungguli Machine Learning dalam akurasi dan representasi konteks
- Namun, ML (SVM) tetap kompetitif dengan training yang lebih ringan
- Transformer membutuhkan tuning lebih lanjut untuk dataset ini

👉 Kesimpulan utama:  
**BiLSTM memberikan keseimbangan terbaik antara akurasi, stabilitas, dan efisiensi komputasi**

---
