# 📄 Panduan & Template Penulis JIKI (`AuthorGuideline_JIKI`)

Direktori ini memuat paket resmi template dan pedoman penulisan artikel ilmiah untuk:
**Jurnal Ilmu Komputer dan Informasi (JIKI)**  
Fakultas Ilmu Komputer, Universitas Indonesia (Terakreditasi **SINTA 2** / Garuda / Dimensions).

---

## 📁 Struktur Berkas Template

| Berkas / Direktori | Tipe | Deskripsi & Fungsi |
| :--- | :---: | :--- |
| [**`mainTemplate_JIKI.tex`**](mainTemplate_JIKI.tex) | LaTeX (`.tex`) | Berkas sumber utama LaTeX siap pakai (menggunakan kelas `IEEEtran.cls`, header formal JIKI, margin 3 cm, gaya sitasi numerik IEEE). |
| [**`AuthorGuideline_JIKI.docx`**](AuthorGuideline_JIKI.docx) | Word (`.docx`) | Pedoman penulisan lengkap versi Microsoft Word beserta petunjuk gaya penulisan (*author guidelines*), batas halaman, dan format tabel/gambar. |
| [**`bare_conf.pdf`**](bare_conf.pdf) | PDF | Contoh hasil kompilasi naskah yang sudah terformat sesuai tata letak dua kolom standar. |
| **`IEEEtran.cls`** | Class (`.cls`) | Berkas kelas dokumen IEEE Transactions / Conference resmi (V1.8b). |
| **`*.sty`** | Style (`.sty`) | Paket dependensi lokal: `algorithm2e.sty`, `balance.sty`, `caption.sty`, `cite.sty`, `import.sty`. |
| [**`bibtex/`**](bibtex/) | Folder | Utilitas bibliografi BibTeX (`IEEEtran.bst`, `IEEEexample.bib`, `IEEEabrv.bib`). |
| [**`pics/`**](pics/) | Folder | Direktori penyimpanan gambar dan visualisasi naskah. |
| [**`tools/`**](tools/) | Folder | Alat bantu makro tambahan (`IEEEtrantools.sty`). |

---

## ⚙️ Petunjuk Penggunaan Template

### 1. Menggunakan Overleaf
1. Buka [Overleaf](https://www.overleaf.com/) $\rightarrow$ **New Project** $\rightarrow$ **Upload Project**.
2. Kompresi seluruh folder `AuthorGuideline_JIKI/` menjadi zip lalu unggah, atau unggah langsung berkas-berkasnya.
3. Atur berkas utama (*Main document*) ke `mainTemplate_JIKI.tex`.
4. Pilih compiler: **pdfLaTeX** (disarankan) dengan TeX Live terbaru.

### 2. Kompilasi Lokal (CLI / TeX Live / MiKTeX)
```powershell
pdflatex mainTemplate_JIKI.tex
bibtex mainTemplate_JIKI
pdflatex mainTemplate_JIKI.tex
pdflatex mainTemplate_JIKI.tex
```

---

## 🎯 Penyesuaian Naskah Penelitian Skripsi ke JIKI

Saat memindahkan draf naskah dari [`docs/manuscript/draft/`](../../draft/README.md) ke template JIKI ini:
1. **Judul & Penulis:** Sesuaikan blok `\title{...}` dan `\author{...}` pada `mainTemplate_JIKI.tex`.
2. **Abstrak Bilingual:** Sediakan Abstrak dalam Bahasa Indonesia dan *Abstract* dalam Bahasa Inggris beserta kata kunci (*keywords*).
3. **Struktur Bagian IMRaD:**
   - **I. Pendahuluan (Introduction):** Latar belakang klinis, disparitas bahasa, paradoks ketergantungan teks, dan rumusan masalah.
   - **II. Metode Penelitian (Proposed Method):** Rumusan metode Bi-ViS ($M_1, \Delta P_{\text{dir}}, D_{\text{lang}}$) dan estimator risiko logistik teratur.
   - **III. Hasil dan Pembahasan (Results and Discussion):** Validasi empiris rev05 (RQ1 reliabilitas bilingual, RQ2 sensitivitas intervensi visual, RQ3 prediksi selektif & kurva AURC).
   - **IV. Kesimpulan dan Saran (Conclusion):** Demarkasi kontribusi, batasan latensi klinis ($3\times$), dan rekomendasi penerapan.
4. **Sitasi & Referensi:** Gunakan format numerik `\cite{...}` berbasis file BibTeX dengan gaya `IEEEtran.bst`.
