<div align="center">

# 🏥 Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift

**Judul Bahasa Indonesia:**  
*Prediksi Selektif Berbasis Sensitivitas Visual pada Model Vision-Language Medis Bilingual dalam Pergeseran Bahasa Inggris–Indonesia*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HF-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Kaggle Dual T4](https://img.shields.io/badge/Compute-Kaggle%20Dual%20T4-20BEFF.svg?logo=kaggle&logoColor=white)](https://kaggle.com/)
[![Protocol rev05](https://img.shields.io/badge/Protocol-rev05--v10%20(Frozen)-4CAF50.svg)](#-protokol-dan-metodologi-ilmiah)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Repositori Resmi Penelitian Skripsi Sarjana (S1) Ilmu Komputer / Teknik Informatika*  
*Bidang Keahlian: Medical Multimodal Machine Learning, Trustworthy AI, & Selective Prediction*

</div>

---

## 📌 Daftar Isi
- [🗺️ Navigasi Cepat: Mau Cari Apa?](#-navigasi-cepat-mau-cari-apa)
- [🛡️ Audit Suite & Verifikasi Ilmiah (Supervisor Ready)](#-audit-suite--verifikasi-ilmiah-supervisor-ready)
- [Ringkasan Eksekutif](#-ringkasan-eksekutif)
- [Arsitektur & Pipeline Evaluasi](#-arsitektur--pipeline-evaluasi)
- [Pertanyaan Penelitian (Research Questions)](#-pertanyaan-penelitian-research-questions)
- [Ringkasan Hasil Empiris Utama (`rev05`)](#-ringkasan-hasil-empiris-utama-rev05)
- [Peta Struktur Direktori Repositori](#-peta-struktur-direktori-repositori)
- [Panduan Instalasi & Reproduksi](#-panduan-instalasi--reproduksi)
- [Batasan Epistemik & Integritas Ilmiah](#-batasan-epistemik--integritas-ilmiah)
- [Sitasi & Referensi Utama](#-sitasi--referensi-utama)

---

## 🗺️ Navigasi Cepat: Mau Cari Apa?

Gunakan tabel kompas ini untuk langsung menuju file atau folder yang Anda butuhkan tanpa perlu membongkar seluruh direktori:

| Kebutuhan Anda | File / Folder Sasaran | Deskripsi Singkat |
| :--- | :--- | :--- |
| 🛡️ **Laporan Audit Dosen Pembimbing** | [`docs/audit/SUPERVISOR_VERIFICATION_REPORT.md`](docs/audit/SUPERVISOR_VERIFICATION_REPORT.md) | **Master Synthesis Report**: Ringkasan kesiapan sidang, status validasi RQ1–RQ5, dan tabel verifikasi klaim riset. |
| ❓ **Bank 35 Soal Sidang & Jawaban Ilmiah** | [`docs/audit/SUPERVISOR_QUESTION_BANK.md`](docs/audit/SUPERVISOR_QUESTION_BANK.md) | 35 pertanyaan kritis penguji dari 11 dimensi metodologi beserta jawaban dan sitasi bukti. |
| 🗂️ **Suite Dokumen Audit Lengkap (17 Laporan)** | [`docs/audit/`](docs/audit/) | Dokumentasi audit formal: formula, kode, integritas data, eksperimen, literatur, dan reproduksibilitas. |
| 📖 **Membaca Teori & Latar Belakang Skripsi** | [`docs/research/RESEARCH_MASTER.md`](docs/research/RESEARCH_MASTER.md) | **Single Source of Truth (Versi 10.0)** lengkap berisi rumusan matematika, latar belakang klinis, dan batasan klaim. |
| 📽️ **Melihat Slide Ujian Proposal / Bimbingan** | [`presentations/proposal/Medical_VLM_Proposal.pptx`](presentations/proposal/Medical_VLM_Proposal.pptx) | Naskah slide resmi seminar proposal skripsi. |
| 🚀 **Menjalankan Eksperimen Utama di Kaggle** | [`notebooks/kaggle/08_qwen_decisive_validation.ipynb`](notebooks/kaggle/08_qwen_decisive_validation.ipynb) | Notebook utama siap eksekusi di GPU Dual Tesla T4 (Internet ON). |
| 📊 **Melihat Angka Hasil & Metrik Final** | [`artifacts/experiments/decisive_validation/rev05/`](artifacts/experiments/decisive_validation/rev05/) | Tabel metrik resmi RQ1 (Akurasi/Flip), RQ2 (Visual), RQ3 (AURC), dan kurva PNG. |
| ⚙️ **Melihat Kode Inferensi Inti** | [`scripts/run_experiment.py`](scripts/run_experiment.py) | Engine kalkulasi sequence log-likelihood & intervensi kanvas hitam. |
| 🧪 **Menjalankan Unit Test (46/46 Pass)** | [`tests/`](tests/) | Suite pengujian lokal deterministik verifikasi logit kausal, data, stats, dan metrik. |
| 🔍 **Audit Data Bebas Kebocoran (Zero Leakage)** | [`docs/audit/DATASET_INTEGRITY_REPORT.md`](docs/audit/DATASET_INTEGRITY_REPORT.md) | Pembuktian bebas kebocoran citra (0.00% overlap antara 88 cluster dev & 204 test). |
| 📄 **Draf Naskah Publikasi LaTeX** | [`docs/manuscript/draft/`](docs/manuscript/draft/) | Berkas draf naskah publikasi ilmiah (Overleaf ready, `main.tex`, `references.bib`, template ITB/JICT). |
| 📑 **Membaca Paper Literatur Acuan** | [`references/papers/`](references/papers/) | PDF artikel rujukan: Yudhistira et al. (2026) & Sadanandan (2026). |
| 📦 **Melihat Arsip Draf Historis (v9–v10)** | [`docs/archive/`](docs/archive/) | Riwayat proses eksplorasi sebelum mencapai protokol rev05 dan Bi-ViS. |

---

## 🛡️ Audit Suite & Verifikasi Ilmiah (Supervisor Ready)

Untuk memastikan bahwa seluruh penelitian dapat dipertanggungjawabkan secara ilmiah dan teknis saat diuji oleh dosen pembimbing, repositori ini telah diaudit secara formal mencakup seluruh rantai riset:
$$\text{Literature} \longrightarrow \text{Research Gap} \longrightarrow \text{Research Questions} \longrightarrow \text{Formulation} \longrightarrow \text{Code} \longrightarrow \text{Outputs} \longrightarrow \text{Statistical Tests} \longrightarrow \text{Conclusion}$$

Seluruh laporan audit formal tersedia di folder [`docs/audit/`](docs/audit/):

1. **[SUPERVISOR_VERIFICATION_REPORT.md](docs/audit/SUPERVISOR_VERIFICATION_REPORT.md)**: Dokumen rujukan utama pembimbing memetakan status validasi seluruh pertanyaan penelitian (RQ1–RQ5) dan tabel verifikasi klaim ilmiah.
2. **[SUPERVISOR_QUESTION_BANK.md](docs/audit/SUPERVISOR_QUESTION_BANK.md)**: Bank 35 pertanyaan kritis ujian skripsi dari 11 dimensi metodologi beserta jawaban ilmiah berbasis bukti.
3. **[RESEARCH_ACTION_PLAN.md](docs/audit/RESEARCH_ACTION_PLAN.md)**: Rencana aksi berprioritas (P0–P3) mencakup pembingkaian temuan negatif heuristik dan roadmap publikasi konferensi (CHIL / MedNeurIPS).
4. **[LITERATURE_VERIFICATION_REPORT.md](docs/audit/LITERATURE_VERIFICATION_REPORT.md)**: Audit 11 paper primer (DOI & arXiv resmi), taksonomi 7 klaster, dan verifikasi research gap.
5. **[DATASET_INTEGRITY_REPORT.md](docs/audit/DATASET_INTEGRITY_REPORT.md)**: Rekonsiliasi lengkap 2.244 baris data mentah ke 1.122 pasangan bersih, serta pembuktian zero cluster leakage.
6. **[FORMULA_VERIFICATION_REPORT.md](docs/audit/FORMULA_VERIFICATION_REPORT.md)**: Verifikasi matematis token sequence log-likelihood, normalisasi panjang, orientasi skor, dan manual calculation.
7. **[CODE_AUDIT_REPORT.md](docs/audit/CODE_AUDIT_REPORT.md)**: Audit implementasi PyTorch/Transformers, slicing logit kausal $[L_p-1:L_p-1+L_t]$, penanganan OOM, dan unit test.
8. **[EXPERIMENT_OUTPUT_AUDIT.md](docs/audit/EXPERIMENT_OUTPUT_AUDIT.md)**: Silsilah artefak checkpoint Kaggle GPU Dual T4, rekalkulasi metrik 100% bit-identical, dan signifikansi statistik.
9. **[REPRODUCIBILITY_CHECKLIST.md](docs/audit/REPRODUCIBILITY_CHECKLIST.md)**: Protokol reproduksi 3-tier (7 detik CPU unit test, 1 menit CPU metric recompute, 45 menit GPU re-inference).

---

## 🔬 Ringkasan Eksekutif

Dalam penerapan kecerdasan artifisial radiologi di negara berkembang seperti Indonesia, **Medical Vision-Language Models (Med-VLMs)** dituntut menjawab pertanyaan diagnostik dalam bahasa lokal (Bahasa Indonesia). Namun, mayoritas model fondasi multimodal dilatih menggunakan data biomedis berbahasa Inggris (*language bias*). Ketika dihadapkan pada translasi bahasa lokal, model mengalami fenomena:
1. **Pergeseran Kinerja Lintas-Bahasa (*Cross-Lingual Flip*):** Pasien dan citra rontgen/CT yang sama dapat menerima kesimpulan diagnostik bertolak belakang hanya karena bahasa pertanyaan dialihkan dari Inggris ke Indonesia ($\hat{y}_{\text{ID}} \ne \hat{y}_{\text{EN}}$).
2. **Ketergantungan Prior Teks (*Visual Ungroundedness*):** Model berisiko memberikan jawaban dengan tingkat keyakinan (*confidence*) sangat tinggi tanpa memeriksa bukti radiografis secara memadai (*Dangerous Quadrant*).

Penelitian ini memformulasikan sistem **Selective Prediction (Prediksi Selektif)** yang memungkinkan model **mengabstraksi diri (*abstain/referral*)** dan merujuk kasus berisiko tinggi ke dokter spesialis radiologi manusia. Sistem mengombinasikan dua sinyal perilaku tanpa memerlukan pelatihan ulang (*inference-time evaluation*):
- **Sensitivitas Penghilangan Citra (*Visual Deletion Sensitivity*, $\Delta P_{\text{dir}}$):** Mengukur penurunan probabilitas ketika citra radiologi asli ($P_0$) digantikan oleh kanvas hitam murni berdimensi identik ($P_1$).
- **Ketidaksepakatan Lintas-Bahasa (*Cross-Lingual Disagreement*, $D_{\text{lang}}$):** Sinyal biner $\mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$ sebagai penanda instabilitas representasi multimodal.

---

## 🏗 Arsitektur & Pipeline Evaluasi

```
                                 [ Citra Radiologi Asli (P0) ]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ Query Bahasa Inggris (EN) ]                                [ Query Bahasa Indonesia (ID) ]
                 │                                                             │
                 ▼                                                             ▼
   ┌───────────────────────────┐                                 ┌───────────────────────────┐
   │ Medical VLM (Zero-Shot)   │                                 │ Medical VLM (Zero-Shot)   │
   │ Sequence Log-Likelihood   │                                 │ Sequence Log-Likelihood   │
   │ Scoring: ["Yes", "No"]    │                                 │ Scoring: ["Ya", "Tidak"]  │
   └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                 │                                                             │
                 ▼                                                             ▼
           Prediksi ŷ_EN                                                 Prediksi ŷ_ID
                 │                                                             │
                 └───────────────────────┬─────────────────────────────────────┘
                                         ▼
                 [ Cross-Lingual Disagreement Signal: D_lang = 𝕀(ŷ_EN ≠ ŷ_ID) ]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                                                                 │
        ▼                                                                 ▼
[ Citra Asli P0: p_orig ]                                    [ Citra Blank Hitam P1: p_blank ]
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
            [ Visual Deletion Sensitivity: ΔP_dir = p_orig - p_blank ]
                                         │
                                         ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │        Learned Selective Risk Estimator (Logistic Regression)   │
        │        Risk Score = σ( β0 + β1·Confidence + β2·ΔP_dir + β3·D_lang )
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │      Selective Prediction & Risk-Coverage Curve (AURC Optimization)
        │      ✅ High Confidence & Visually Grounded  ──> Otomatis Diterima
        │      ⚠️ Inconsistent / Blind Text Prior      ──> Rujuk ke Radiolog
        └─────────────────────────────────────────────────────────────────┘
```

---

## ❓ Pertanyaan Penelitian (Research Questions)

* **RQ1 (Bilingual Reliability):** Seberapa besar degradasi akurasi diagnostik dan laju pergeseran kesimpulan (*flip rate*) model ketika pertanyaan beralih dari bahasa Inggris ke Indonesia?
* **RQ2 (Visual Deletion Responsiveness):** Apakah model medis mempertahankan overconfidence pada masukan teks murni saat informasi visual dieliminasi secara total ($P_1$ Kanvas Hitam)?
* **RQ3 (Selective Risk Estimation & AURC):** Apakah penggabungan sensitivitas visual ($\Delta P_{\text{dir}}$) dan ketidaksepakatan bilingual ($D_{\text{lang}}$) mampu memperbaiki perankingan risiko dan menurunkan metrik **Area Under the Risk-Coverage Curve (AURC)** dibandingkan baseline confidence tunggal?

---

## 📊 Ringkasan Hasil Empiris Utama (`rev05`)

Evaluasi dieksekusi secara ketat pada **Primary Test Set** ($N = 799$ pasangan QA, 204 klaster citra fisik unik tanpa kebocoran data / *zero-leakage*):

### 1. Evaluasi Reliabilitas Bilingual (RQ1)
| Model Fondasi Multimodal | Akurasi EN | Akurasi ID | Kesenjangan (Gap) | Laju Balik (*Flip Rate*) | Akurasi Gabungan (*Joint Acc*) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-7B-Instruct** | **75.84%** | **62.83%** | 13.02% | **28.79%** | **54.94%** |
| **HuatuoGPT-Vision-7B** (Medical) | 74.84% | 57.57% | 17.27% | **50.81%** | 40.80% |

> *Catatan Klinis:* HuatuoGPT-Vision-7B mengalami *flip rate* sebesar **50.81%** (setiap 2 kasus, 1 mengalami perubahan kesimpulan saat diterjemahkan ke bahasa Indonesia), membuktikan kerentanan model spesialis medis terhadap *language shift*.

### 2. Evaluasi Prediksi Selektif & Optimasi AURC (RQ3)
| Model | Estimator Risiko | Fitur yang Digunakan | AURC (Primary Test) ↓ | Risiko @ Coverage 80% ↓ | Risiko @ Coverage 50% ↓ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Qwen2.5-VL-7B** | `LR_conf` (Baseline) | $M_1$ (Confidence) | 0.2422 | 33.65% | 24.56% |
| | `LR_visual` | $M_1$ + $\Delta P_{\text{dir}}$ | 0.2434 | 36.15% | 31.83% |
| | **`LR_full` (Bi-ViS)** | **$M_1$ + $\Delta P_{\text{dir}}$ + $D_{\text{lang}}$** | **0.2053** | **28.48%** | **18.00%** |
| **HuatuoGPT-Vision-7B** | `LR_conf` (Baseline) | $M_1$ (Confidence) | 0.2884 | 40.38% | 31.08% |
| | `M_norm` (Heuristic) | $p_{\text{orig}} \cdot \frac{1 + \Delta_{\text{dir}}}{2}$ | 0.2552 | 37.40% | 25.81% |
| | **`LR_full` (Bi-ViS)** | **$M_1$ + $\Delta P_{\text{dir}}$ + $D_{\text{lang}}$** | **0.2179** | **35.68%** | **26.32%** |

> *Temuan Kunci:* Penggabungan sinyal ketidaksepakatan lintas-bahasa ($D_{\text{lang}}$) bersama sensitivitas visual melalui metode **Bi-ViS** menghasilkan penurunan AURC paling dramatis:  
> - **Qwen2.5-VL-7B:** AURC terpangkas dari **0.2422 $\to$ 0.2053** ($p = 0.0310$, penurunan risiko selektif @ 80% coverage dari 33.65% ke 28.48%).  
> - **HuatuoGPT-Vision-7B:** AURC terpangkas dari **0.2884 $\to$ 0.2179** ($p = 0.0010$, penurunan risiko selektif @ 80% coverage dari 40.38% ke 35.68%).

---

## 🗂 Peta Struktur Direktori Repositori

```text
VLM/
├── README.md                               # Kompas utama & panduan repositori
├── LICENSE                                 # Lisensi terbuka (MIT)
├── requirements.txt                        # Dependensi pustaka Python global
├── pyproject.toml                          # Konfigurasi standar paket vlm-research
├── .gitignore                              # Aturan eksklusi Git (cache, weights, zip > 100MB)
│
├── docs/                                   # DOKUMENTASI AKADEMIK & METODOLOGI
│   ├── README.md                           # Indeks dokumentasi master
│   ├── manuscript/                         # Fondasi artikel ilmiah, formalisasi Bi-ViS, dan blueprint
│   │   └── draft/                          # Naskah publikasi LaTeX lengkap (Overleaf Ready, main.tex, references.bib)
│   ├── audit/                              # SUITE LAPORAN AUDIT ILMIAH & REPRODUKSIBILITAS
│   ├── protocols/                          # Protokol eksperimen, tata kelola data, dan checklist
│   ├── research/                           # Blueprint SSoT, tinjauan pustaka, dan log perubahan
│   └── archive/                            # Draf historis naskah versi terdahulu (v9, v10)
│
├── configs/                                # KONFIGURASI PARAMETER & MODEL
│   ├── README.md
│   └── evaluation_config.yaml              # Hyperparameter inferensi & decoding (YAML)
│
├── notebooks/                              # ALUR JUPYTER NOTEBOOKS
│   ├── README.md
│   ├── 01_dataset_audit_eda.ipynb          # Audit integritas data & SHA-256 citra fisik
│   └── kaggle/                             # Notebook eksekusi GPU Dual-T4 Kaggle
│       ├── 08_qwen_decisive_validation.ipynb
│       ├── 09_huatuo_decisive_validation.ipynb
│       └── 10_standalone_aggregation_evaluation.ipynb
│
├── src/                                    # MODUL PUSTAKA UTAMA (PACKAGE)
│   ├── stats_engine.py                     # Sentralisasi fungsi statistik (permutasi, bootstrap, CI)
│   ├── metrics.py                          # Metrik keandalan lanjutan (ECE, Brier, Confidently-Wrong)
│   └── vlm_research/
│       ├── config.py                       # Resolusi path absolut & konstanta global
│       ├── data/                           # Parser jawaban, data loader, dan audit kebocoran
│       ├── inference/                      # Engine evaluasi inferensi logit kausal
│       ├── evaluation/                     # Scoring normalisasi log-likelihood & estimasi AURC
│       ├── models/                         # Loader arsitektur multimodal
│       └── visualization/                  # Pustaka pembuatan grafik Risk-Coverage
│
├── scripts/                                # SKRIP EKSEKUSI & CLI ENGINE
│   ├── README.md
│   ├── run_experiment.py                   # CLI runner terpadu (--model, --dry-run)
│   ├── run_decisive_validation.py          # Engine inferensi kausal rev05 lengkap
│   ├── adaptive_abstention_engine.py       # Engine penentuan threshold operasional
│   └── visual_perturbation.py              # Generator citra kontrol kanvas hitam (P1)
│
├── tests/                                  # SUITE PENGUJIAN OTOMATIS (46/46 PASSING)
│   ├── README.md
│   ├── test_decisive_validation.py         # Master Preflight Suite (9 assertions lolos)
│   ├── test_scoring.py                     # Uji sequence log-likelihood & sensitivitas visual
│   ├── test_data_quality.py                # Invarian kohort 1.122 pasangan & label kanonikal
│   ├── test_bilingual_pairing.py           # Validasi pasangan pertanyaan ID-EN
│   ├── test_leakage_audit.py               # Pembuktian bebas kebocoran (0.00% cluster leakage)
│   ├── test_statistical_analysis.py        # Verifikasi hand-calculation AURC & permutasi
│   ├── test_stats_engine.py                # Pengujian fungsi sentral stats_engine
│   ├── test_metrics.py                     # Pengujian kalkulasi ECE, Brier, dan Confidently-Wrong
│   └── test_pilot_gate.py                  # Validasi gate evaluasi pilot
│
├── data/                                   # DATASET RADIOLOGI & MANIFEST RESMI
│   ├── README.md
│   ├── raw/indorad_vqa/images/             # 314 citra radiologi fisik asli (.png)
│   ├── manifests/                          # Manifest resmi split (1.122 pasangan bersih)
│   └── processed/                          # Data ringkasan dan kamus skema data
│
├── artifacts/                              # ARTEFAK HASIL EMPIRIS RESMI
│   ├── README.md
│   └── experiments/
│       ├── decisive_validation/rev05/      # HASIL RESMI PROTOKOL BEKU REV05
│       │   ├── metrics/                    # CSV metrik resmi RQ1, RQ2, RQ3, dan risk ablation
│       │   ├── predictions/                # Checkpoint prediksi per-sampel (Huatuo & Qwen)
│       │   ├── figures/                    # Kurva Risk-Coverage resolusi tinggi (.png)
│       │   ├── logs/                       # Log SHA-256 artefak dan dekompresi
│       │   ├── perturbed_images/           # 314 citra kontrol kanvas hitam (P1)
│       │   └── audit_recomputed/           # Rekalkulasi statistik independen & matriks bukti
│       └── decisive_validation/historical/ # Arsip hasil iterasi eksperimen rev02-rev04
│
├── references/                             # ARTIKEL ILMIAH & PAPER RUJUKAN (PDF)
│   ├── README.md
│   ├── papers/                             # PDF IndoRad-VQA (2026) & Sadanandan (CHIL 2026)
│   └── bibliography/references.bib         # Sitasi BibTeX lengkap 11 paper fondasional
│
└── presentations/                          # SLIDE PRESENTASI UJIAN
    ├── README.md
    ├── proposal/Medical_VLM_Proposal.pptx   # Slide seminar proposal skripsi
    └── progress/Bimbingan_Progres.pptx     # Slide bahan bimbingan progres berkala
```

---

## 🚀 Panduan Instalasi & Reproduksi

### 1. Persiapan Lingkungan Lokal
Clone repositori dan pasang dependensi virtual environment:
```bash
git clone https://github.com/ShinZeleo/Skripshit.git
cd Skripshit
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# atau: .venv\Scripts\activate   # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                # Pasang paket vlm-research dalam mode editable
```

### 2. Menjalankan Unit Test Mandiri (Preflight Audit)
Sebelum inferensi GPU, verifikasi kalkulasi logit kausal, invariansi prefix tokenisasi, dan integritas metrik dalam 7 detik di CPU:
```bash
python scripts/run_experiment.py --dry-run
```
*Atau jalankan seluruh pengujian unit modular (46/46 Tests):*
```bash
pytest tests/ -q
```
*Ekspektasi: Seluruh 46 unit test lolos (`46 passed in ~14s`).*

### 3. Eksekusi Inferensi Skala Penuh (Kaggle Dual-T4 GPU)
1. Unggah notebook dari `notebooks/kaggle/08_qwen_decisive_validation.ipynb` (atau `09_huatuo_decisive_validation.ipynb`) ke Kaggle.
2. Pilih Accelerator: **GPU T4 x 2** dan aktifkan **Internet ON**.
3. Eksekusi seluruh sel untuk memproduksi kembali tabel metrik `rq1_bilingual_metrics.csv`, `rq3_selective_prediction_primary_test.csv`, dan grafik kurva risk-coverage.

---

## 🛡 Batasan Epistemik & Integritas Ilmiah

Sesuai standar kejujuran akademik skripsi:
1. **Sifat Out-of-Distribution (OOD) Kanvas Hitam:** Manipulasi citra hitam polos ($P_1$) adalah kondisi ekstrem OOD bagi vision encoder. Penurunan probabilitas ($\Delta P_{\text{dir}}$) mencerminkan **sensitivitas perilaku terhadap ketiadaan sinyal visual**, dan **bukan** bukti pemahaman kausal anatomi atau lokalisasi patologi klinis.
2. **Evaluasi Retrospektif vs Uji Klinis Prospektif:** Optimasi AURC mengurutkan ketidakpastian model secara statistik pada dataset uji retrospektif. Hasil ini tidak dapat menggantikan validasi prospektif dalam alur kerja klinis rumah sakit nyata.
3. **Penyajian Temuan Empiris Utuh:** Analisis mencakup pelaporan jujur atas kondisi di mana estimator visual tunggal ($M_2$) tidak mengungguli baseline kepercayaan teks pada model tertentu, membuktikan pentingnya perpaduan sinyal multimodal ($LR_{\text{full}}$).

---

## 📚 Sitasi & Referensi Utama

Jika Anda menggunakan kode, manifest, atau temuan dalam repositori ini, silakan sitasi rujukan skripsi:

```bibtex
@thesis{imam2026medicalvlm,
  author       = {Imam},
  title        = {Visual Sensitivity--Aware Selective Prediction for Bilingual Medical Vision-Language Models under English--Indonesian Language Shift},
  school       = {Universitas / Program Studi Teknik Informatika},
  year         = {2026},
  type         = {Undergraduate CS Thesis},
  note         = {Protocol rev05-v10-master}
}
```

**Referensi Ilmiah Fondasional:**
- Yudhistira et al. (2026). *IndoRad-VQA: Bilingual Indonesian-English Medical Vision Question Answering Dataset*. arXiv:2606.03693.
- Sadanandan & Behzadan (2026). *Evaluating Paraphrase Consistency and Visual Sensitivity in Medical Vision-Language Models*. ACM Conference on Health, Inference, and Learning (CHIL).
- Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural Networks*. Advances in Neural Information Processing Systems (NeurIPS).

---
<div align="center">
  <b>Departemen Ilmu Komputer / Teknik Informatika &bull; 2026</b>
</div>
