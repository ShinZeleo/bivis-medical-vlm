# 📑 Master Direktori Template Naskah Publikasi (`docs/manuscript/templates/`)

Direktori ini mengelola seluruh paket template jurnal dan konferensi target publikasi untuk mendiseminasikan hasil penelitian:
> **"Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"**

---

## 🗂️ Daftar Template Tersedia

| Folder Template | Target Publikasi | Akreditasi / Indeksasi | Format | Status |
| :--- | :--- | :---: | :---: | :---: |
| [**`AuthorGuideline_JIKI/`**](AuthorGuideline_JIKI/README.md) | **Jurnal Ilmu Komputer dan Informasi (JIKI)** — Fasilkom UI | **SINTA 2** / Garuda / Dimensions | LaTeX (`IEEEtran`) & Word (`.docx`) | `READY` |

---

## 🧭 Panduan Alur Penulisan & Porting Naskah

1. **Draf Naskah Aktif:** Berkas draf naskah penelitian lengkap (format ITB Journal / review version) berada di [`docs/manuscript/draft/`](../draft/README.md).
2. **Porting ke Template JIKI:**
   - Gunakan [**`mainTemplate_JIKI.tex`**](AuthorGuideline_JIKI/mainTemplate_JIKI.tex) sebagai kerangka kerja baru.
   - Pindahkan teks abstrak, pendahuluan, metode Bi-ViS, dan hasil eksperimen rev05 dari [`docs/manuscript/draft/main.tex`](../draft/main.tex) atau [`docs/manuscript/MANUSCRIPT_BLUEPRINT.md`](../MANUSCRIPT_BLUEPRINT.md).
   - Pastikan tabel dan gambar resolusi tinggi (300 DPI) merujuk ke visualisasi terverifikasi di [`artifacts/figures/`](../../../artifacts/figures/) dan [`reports/figures/`](../../../reports/figures/).
