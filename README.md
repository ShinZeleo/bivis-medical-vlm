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
| 🛡️ **Laporan Audit Dosen Pembimbing** | [`SUPERVISOR_VERIFICATION_REPORT.md`](SUPERVISOR_VERIFICATION_REPORT.md) | **Master Synthesis Report**: Ringkasan kesiapan sidang, status validasi RQ1–RQ5, dan tabel verifikasi klaim riset. |
| ❓ **Bank 35 Soal Sidang & Jawaban Ilmiah** | [`docs/audit/SUPERVISOR_QUESTION_BANK.md`](docs/audit/SUPERVISOR_QUESTION_BANK.md) | 35 pertanyaan kritis penguji dari 11 dimensi metodologi beserta jawaban dan sitasi bukti. |
| 🗂️ **Suite Dokumen Audit Lengkap (9 Laporan)** | [`docs/audit/`](docs/audit/) | Dokumentasi audit formal: formula, kode, integritas data, eksperimen, literatur, dan reproduksibilitas. |
| 📖 **Membaca Teori & Latar Belakang Skripsi** | [`docs/RESEARCH_MASTER.md`](docs/RESEARCH_MASTER.md) | **Single Source of Truth (Versi 10.0)** lengkap berisi rumusan matematika, latar belakang klinis, dan batasan klaim. |
| 📽️ **Melihat Slide Ujian Proposal / Bimbingan** | [`presentations/Medical_VLM_Proposal.pptx`](presentations/Medical_VLM_Proposal.pptx) | Naskah slide resmi seminar proposal skripsi. |
| 🚀 **Menjalankan Eksperimen Utama di Kaggle** | [`main/notebooks/decisive_validation_primary_rev05.ipynb`](main/notebooks/decisive_validation_primary_rev05.ipynb) | Notebook utama siap eksekusi di GPU Dual Tesla T4 (Internet ON). |
| 📊 **Melihat Angka Hasil & Metrik Final** | [`main/results/decisive_validation/rev05/`](main/results/decisive_validation/rev05/) | Tabel metrik resmi RQ1 (Akurasi/Flip), RQ2 (Visual), RQ3 (AURC), dan kurva PNG. |
| ⚙️ **Melihat Kode Inferensi Inti** | [`main/scripts/decisive_validation_experiment_v5.py`](main/scripts/decisive_validation_experiment_v5.py) | Engine kalkulasi sequence log-likelihood & intervensi kanvas hitam. |
| 🧪 **Menjalankan Unit Test (9/9 Pass)** | [`main/scripts/test_decisive_validation_v5.py`](main/scripts/test_decisive_validation_v5.py) | Suite pengujian lokal deterministik verifikasi logit kausal dan metrik. |
| 🔍 **Audit Data Bebas Kebocoran (Zero Leakage)** | [`docs/audit/DATASET_INTEGRITY_REPORT.md`](docs/audit/DATASET_INTEGRITY_REPORT.md) | Pembuktian bebas kebocoran citra (0.00% overlap antara 88 cluster dev & 204 test). |
| 📑 **Membaca Paper Literatur Acuan** | [`paper/`](paper/) | PDF artikel rujukan: Yudhistira et al. (2026) & Sadanandan (2026). |
| 📦 **Melihat Arsip Iterasi Lama (v1–v4)** | [`archive/`](archive/) & [`main/reports/`](main/reports/) | Riwayat proses eksplorasi sebelum mencapai protokol rev05. |

---

## 🛡️ Audit Suite & Verifikasi Ilmiah (Supervisor Ready)

Untuk memastikan bahwa seluruh penelitian dapat dipertanggungjawabkan secara ilmiah dan teknis saat diuji oleh dosen pembimbing, repositori ini telah diaudit secara formal mencakup seluruh rantai riset:
$$\text{Literature} \longrightarrow \text{Research Gap} \longrightarrow \text{Research Questions} \longrightarrow \text{Formulation} \longrightarrow \text{Code} \longrightarrow \text{Outputs} \longrightarrow \text{Statistical Tests} \longrightarrow \text{Conclusion}$$

Seluruh laporan audit formal tersedia di folder [`docs/audit/`](docs/audit/):

1. **[SUPERVISOR_VERIFICATION_REPORT.md](SUPERVISOR_VERIFICATION_REPORT.md)**: Dokumen rujukan utama pembimbing memetakan status validasi seluruh pertanyaan penelitian (RQ1–RQ5) dan tabel verifikasi klaim ilmiah.
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
| | **`LR_full` (Ours)** | **$M_1$ + $\Delta P_{\text{dir}}$ + $D_{\text{lang}}$** | **0.2053** | **—** | **—** |
| **HuatuoGPT-Vision-7B** | `LR_conf` (Baseline) | $M_1$ (Confidence) | 0.2884 | 40.38% | 31.08% |
| | `M_norm` (Heuristic) | $p_{\text{orig}} \cdot \frac{1 + \Delta_{\text{dir}}}{2}$ | 0.2552 | 37.40% | 25.81% |
| | **`LR_full` (Ours)** | **$M_1$ + $\Delta P_{\text{dir}}$ + $D_{\text{lang}}$** | **0.2179** | **—** | **—** |

> *Temuan Kunci:* Penggabungan sinyal ketidaksepakatan lintas-bahasa ($D_{\text{lang}}$) bersama sensitivitas visual menghasilkan penurunan AURC paling dramatis:  
> - **Qwen2.5-VL-7B:** AURC terpangkas dari **0.2422 $\to$ 0.2053** (penurunan risiko relatif signifikan).  
> - **HuatuoGPT-Vision-7B:** AURC terpangkas dari **0.2884 $\to$ 0.2179**.

---

## 🗂 Peta Struktur Direktori Repositori

```
VLM/
├── README.md                               # Dokumentasi utama repositori (SSoT Ringkas)
├── LICENSE                                 # Lisensi lisensi terbuka (MIT)
├── requirements.txt                        # Dependensi pustaka Python global
├── .gitignore                              # Aturan eksklusi Git (cache, zip > 100MB)
│
├── docs/                                   # Dokumentasi Akademik & Master Blueprint Skripsi
│   ├── RESEARCH_MASTER.md                  # Blueprint Otoritatif Versi 10.0 (Single Source of Truth)
│   ├── LITERATURE_AUDIT.md                 # Matriks audit literatur komprehensif & preseden riset
│   ├── REVISION_CHANGELOG.md               # Log perubahan teknis & metodologis lintas revisi
│   ├── REVISION_NOTES.md                   # Catatan arahan dosen pembimbing & resolusi metodologis
│   └── EXPERIMENT_UPDATE_CHECKLIST.md      # Lembar verifikasi kepatuhan pra-eksekusi
│
├── presentations/                          # Slide Presentasi Ujian & Bimbingan
│   ├── Medical_VLM_Proposal.pptx           # Naskah slide seminar proposal skripsi resmi
│   └── Bimbingan_Progres.pptx              # Slide bimbingan progres berkala dengan dosen
│
├── paper/                                  # Literatur Inti Rujukan Penelitian (PDF)
│   ├── 2606.03693v1.pdf                    # Yudhistira et al. (2026) - Baseline IndoRad-VQA
│   └── sadanandan26a.pdf                   # Sadanandan & Behzadan (2026, CHIL) - Visual Grounding
│
├── archive/                                # Artefak Historis & Eksperimen Terdahulu
│   ├── kaggle_legacy/                      # Notebook dan visualisasi pilot iterasi lama (Run 01–03)
│   └── scratch_logs/                       # Log inspeksi teks dan tokenisasi sementara
│
└── main/                                   # Kode Sumber & Pipeline Eksperimen Utama
    ├── configs/                            # Konfigurasi hyperparameter & evaluasi YAML
    ├── data/                               # Data manifest dan subset citra benchmark
    │   ├── images/                         # 314 citra radiologi fisik terverifikasi (PNG)
    │   └── manifests/                      # Manifest klaster identitas & split zero-leakage
    ├── notebooks/                          # Notebook Jupyter Reproduksi Eksperimen
    │   ├── 01_indorad_vqa_dataset_audit_eda.ipynb # Audit kebocoran & eksplorasi data
    │   ├── 02_model_pilot_experiments.ipynb       # Uji coba inferensi pilot lokal
    │   └── decisive_validation_primary_rev05.ipynb# Master notebook validasi rev05 (Kaggle/Colab)
    ├── scripts/                            # Skrip otomatisasi pengujian, inferensi & evaluasi
    │   ├── decisive_validation_experiment_v5.py   # Mesin inferensi utama rev05
    │   ├── test_decisive_validation_v5.py        # Suite unit-test otomatis lokal (9/9 PASS)
    │   └── resilient_validation_engine.py         # Engine toleransi gangguan jaringan/OOM
    ├── src/                                # Modul fungsional Python (data, models, metrics)
    │   ├── data/                           # Loader citra, manifest, dan parser jawaban biner
    │   ├── models/                         # Wrapper Qwen2.5-VL & HuatuoGPT-Vision
    │   ├── image_audit.py                  # Hashing piksel SHA-256 & canonical grouping
    │   ├── leakage_audit.py                # Pemeriksaan kebocoran antar split
    │   └── statistical_analysis.py         # Kalkulasi AURC, bootstrap CI 95%, dan uji permutasi
    ├── reports/                            # Laporan teknis lengkap & audit per revisi (rev02–rev05)
    └── results/                            # Luaran artefak eksperimen (CSV metrik, grafik AURC)
        └── decisive_validation/rev05/      # Hasil final tervalidasi rev05
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
```

### 2. Menjalankan Unit Test Mandiri (Preflight Audit)
Sebelum inferensi GPU, verifikasi kalkulasi logit kausal, invariansi prefix tokenisasi, dan integritas metrik:
```bash
python main/scripts/test_decisive_validation_v5.py
```
*Ekspektasi: 9/9 unit tests lolos dengan status `ALL TESTS PASSED`.*

### 3. Eksekusi Inferensi Skala Penuh (Kaggle Dual-T4 GPU)
1. Unggah direktori `main/` atau notebook `main/notebooks/decisive_validation_primary_rev05.ipynb` ke Kaggle.
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
