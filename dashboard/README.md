# 📚 Data Science - Chatbot Edukatif IPA SD
**Coding Camp 2026 powered by DBS Foundation**
**ID Tim: CC26-PSU312** | Tema: Accessible & Adaptive Learning

---

## 📁 Struktur Folder

```
dashboard/
├── data/
│   ├── datasoal.csv          ← Dataset mentah (raw)
│   ├── notebook.ipynb        ← Notebook utama (wrangling → EDA → visualisasi)
│   ├── README.md             ← Dokumentasi ini
│   ├── requirements.txt      ← Daftar library Python
│   └── url.txt               ← Referensi URL sumber data
└── data_cleran/
    ├── datasoal_clean.csv    ← Dataset bersih hasil wrangling
    └── dashboard.py          ← Aplikasi Streamlit interaktif
```

---

## 🎯 Business Questions

1. Topik IPA apa yang paling banyak dibahas dalam dataset?
2. Bagaimana distribusi panjang soal dan jawaban berdasarkan topik?
3. Subtopik mana yang memiliki jumlah soal terbanyak?
4. Apakah ada ketidakkonsistenan penulisan topik yang perlu dibersihkan?
5. Bagaimana distribusi kompleksitas soal (panjang pertanyaan)?

---

## 🗂️ Data Dictionary

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `no` | int | Nomor urut soal |
| `topik` | str | Topik utama IPA (misal: alat tubuh, tumbuhan hijau) |
| `subtopik` | str | Sub-topik yang lebih spesifik |
| `soal` | str | Teks pertanyaan/soal |
| `jawaban` | str | Teks jawaban dari soal |
| `contoh` | str | Contoh konkret untuk membantu pemahaman |
| `konteks` | str | Kalimat latar belakang/konteks pertanyaan |
| `link sumber buku` | str | URL sumber buku ajar yang digunakan |

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebook
```bash
jupyter notebook data/notebook.ipynb
```

### 3. Jalankan Dashboard
```bash
streamlit run data_cleran/dashboard.py
```

---

## 👥 Anggota Tim (Data Science)

| ID | Nama | Role |
|----|------|------|
| CDCC325D6Y1436 | Matthew Russel Paul | Data Science |
| CDCC325D6Y1759 | Muhammad Farhan | Data Science |
