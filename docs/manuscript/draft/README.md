# 📄 Draft Naskah Ilmiah (ITB Journal Publisher / JICT)

Folder ini dikhususkan untuk seluruh berkas draf naskah publikasi:
**"Prediksi Selektif Berbasis Sensitivitas Visual pada Model Vision-Language Medis Bilingual dalam Pergeseran Bahasa Inggris–Indonesia"**  
Penulis: **Imam Dzaqhoir** (Universitas Hasanuddin)

---

## 📁 Struktur Berkas

| Berkas | Keterangan |
| :--- | :--- |
| [`main.tex`](main.tex) | Berkas sumber utama LaTeX lengkap (Title, Abstract, Pendahuluan, Metode, Tabel 1, Persamaan 1--9, Nomenclature, References). |
| [`references.bib`](references.bib) | Berkas bibliografi BibTeX (12 entri sitasi lengkap standar IEEE / ITB). |
| [`itb_logo.png`](itb_logo.png) | Aset logo resmi ITB untuk header halaman pertama. |
| [`Imam Dzaqhoir - Prediksi Selektif VLM Medis Bilingual.pdf`](<Imam Dzaqhoir - Prediksi Selektif VLM Medis Bilingual.pdf>) | Draf naskah versi PDF asli (14 halaman). |

---

## ⚙️ Petunjuk Kompilasi

### 1. Kompilasi di Overleaf
1. Buat **New Project** $\rightarrow$ **Upload Project** (atau upload seluruh isi folder `draft/` ini).
2. Pastikan `main.tex` diset sebagai *Main document*.
3. Pilih compiler **pdfLaTeX** (default) atau **XeLaTeX**.
4. Klik **Recompile**.

### 2. Pengaturan Mode Review vs Camera-Ready
Pada berkas [`main.tex`](main.tex):
* **Mode Review (Aktif saat ini):** Baris `\linenumbers` diaktifkan agar nomor baris telaah (1 s.d. 349) muncul di sisi kiri naskah.
* **Mode Camera-Ready (Final):** Berikan tanda komentar `% \linenumbers` untuk mematikan penomoran baris.
