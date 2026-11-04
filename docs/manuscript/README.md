# 📝 Scientific Manuscript & Proposed Method Foundation (`docs/manuscript/`)

Direktori ini memuat dokumen fondasi penulisan naskah artikel ilmiah (publikasi jurnal/konferensi) dan formalisasi metodologis metode **Bi-ViS** (*Bilingual Visual-Sensitivity-Aware Selective Prediction*).

---

## 📑 Urutan Pembacaan yang Direkomendasikan (Reading Order)

Untuk memahami kerangka narasi dan formalisasi metode secara koheren, dokumen disusun dalam urutan logis berikut:

| Urutan | Berkas | Fokus Utama |
|:---:|---|---|
| 1 | [**`RESEARCH_NARRATIVE.md`**](RESEARCH_NARRATIVE.md) | **Narasi Penelitian Utuh:** Masalah klinis di negara berkembang, fenomena *cross-lingual flip*, paradoks ketergantungan teks, dan rasionalisasi arsitektur multimodal. |
| 2 | [**`MANUSCRIPT_BLUEPRINT.md`**](MANUSCRIPT_BLUEPRINT.md) | **Cetak Biru Naskah:** Kerangka struktur publikasi standar IMRaD (Title, Abstract, Intro, Related Work, Method, Experiments, Discussion, Conclusion). |
| 3 | [**`PROPOSED_METHOD_BIVIS.md`**](PROPOSED_METHOD_BIVIS.md) | **Formalisasi Metode Bi-ViS:** Perumusan matematis estimator risiko logistik berbasis tiga sinyal ($M_1, \Delta P_{\text{dir}}, D_{\text{lang}}$), standardisasi z-score, dan optimasi loss cross-entropy. |
| 4 | [**`METHOD_ALGORITHM.md`**](METHOD_ALGORITHM.md) | **Algoritma & Implementasi:** Pseudocode operasional runtime inference, kalibrasi ambang batas offline via OOF GroupKFold, dan peta fungsi pada kode produksi. |
| 5 | [**`METHOD_COMPARISON.md`**](METHOD_COMPARISON.md) | **Matriks Komparasi:** Perbandingan Bi-ViS vs baseline ($M_1, M_2, M_{\text{dir}}, M_{\text{norm}}$), analisis kegagalan formulasi rasio heuristik, dan pembuktian nilai inkremental fitur. |
| 6 | [**`BIVIS_DEPLOYMENT_LIMITATIONS.md`**](BIVIS_DEPLOYMENT_LIMITATIONS.md) | **Batasan Klinis & Deployment:** Analisis sifat OOD kanvas hitam ($P_1$), latensi inferensi ($3\times$ forward passes, ~21.7s/kasus), batas komputasi edge klinis, dan mitigasi bias. |
| 7 | [**`CONTRIBUTION_STATEMENT.md`**](CONTRIBUTION_STATEMENT.md) | **Pernyataan Kontribusi Ilmiah:** Demarkasi tegas anti-HARKing antara temuan primer ($M_{\text{norm}}$) dan metode learned multi-sinyal (Bi-ViS) yang dikembangkan pasca-analisis ablasi. |
| 8 | [**`FINAL_METHODOLOGICAL_ADJUDICATION.md`**](FINAL_METHODOLOGICAL_ADJUDICATION.md) | **Putusan Metodologi:** Adjudikasi final atas 5 isu kritis audit statistik: posisi $M_{\text{norm}}$, perhitungan EPV, dependensi klaster citra, koreksi Holm-Bonferroni, dan presisi angka. |

---

## 🔬 Parameter Kunci Metode Bi-ViS (Ringkasan)

- **Fungsi Estimasi Risiko:**
  $$P(\text{error} \mid x) = \sigma\left(\beta_0 + \beta_1 \tilde{M}_1 + \beta_2 \widetilde{\Delta P}_{\text{dir}} + \beta_3 \tilde{D}_{\text{lang}}\right)$$
- **Skor Reliabilitas Selektif:**
  $$S(x) = 1 - P(\text{error} \mid x)$$
- **Kebijakan Keputusan Selektif:**
  $$\hat{y}(x) = \begin{cases} \hat{y}_{\text{ID}}(x), & \text{jika } S(x) \ge \tau \\ \text{ABSTAIN (Rujuk ke Radiolog)}, & \text{jika } S(x) < \tau \end{cases}$$
- **Kinerja Empiris Utama (Primary Test Set, $N=799$):**
  - **Qwen2.5-VL-7B:** AURC $0.2053$ (turun dari $0.2422$, $\Delta = -0.0369, p = 0.0310$), Risk@80% = $28.48\%$.
  - **HuatuoGPT-Vision-7B:** AURC $0.2179$ (turun dari $0.2884$, $\Delta = -0.0705, p = 0.0010$), Risk@80% = $35.68\%$.

---

## 📄 Naskah Publikasi LaTeX & Template Jurnal

1. **Draf Naskah Aktif (`draft/`):**
   - [**`draft/`**](draft/README.md): Memuat berkas sumber `main.tex`, `references.bib`, logo ITB, dan draf naskah PDF asli (Overleaf ready).

2. **Template Publikasi Jurnal (`templates/`):**
   - [**`templates/`**](templates/README.md): Direktori induk template naskah publikasi ilmiah.
   - [**`templates/AuthorGuideline_JIKI/`**](templates/AuthorGuideline_JIKI/README.md): Paket resmi pedoman penulis dan template LaTeX/Word untuk **Jurnal Ilmu Komputer dan Informasi (JIKI) Fasilkom UI** (Terakreditasi SINTA 2).


