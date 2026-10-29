# ⚙️ IndoRad-VQA: Core Codebase & Pipeline (`main/`)

Direktori ini memuat seluruh implementasi kode sumber, skrip evaluasi, modul pemrosesan data, dan notebook eksperimen untuk penelitian skripsi:
> **"Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"**

Untuk dokumentasi komprehensif, arsitektur, dan ringkasan temuan empiris, silakan merujuk pada dokumentasi utama di [**`README.md` di Root Repositori**](../README.md) dan cetak biru otoritatif di [**`docs/RESEARCH_MASTER.md`**](../docs/RESEARCH_MASTER.md).

---

## 📂 Struktur Direktori `main/`

| Direktori / File | Deskripsi Fungsional |
| :--- | :--- |
| `configs/` | File konfigurasi evaluasi (`evaluation_config.yaml`). |
| `data/images/` | 314 citra radiologi fisik unik berformat PNG yang telah diaudit bebas kebocoran. |
| `data/manifests/` | Manifest pemisahan split zero-leakage (`image_identity_manifest.csv`, `image_level_split_manifest.csv`). |
| `notebooks/` | Notebook eksplorasi data (`01_indorad_vqa_dataset_audit_eda.ipynb`) dan eksekusi utama (`decisive_validation_primary_rev05.ipynb`). |
| `scripts/` | Skrip eksekusi inferensi (`decisive_validation_experiment_v5.py`), pengujian unit lokal (`test_decisive_validation_v5.py`), dan generator notebook. |
| `src/` | Modul inti Python (data loader, parser jawaban, ekstraksi metrik selektif, dan utilitas citra). |
| `reports/` | Laporan teknis lengkap per siklus revisi (`DECISIVE_VALIDATION_REV05_AUDIT.md`, `1_DATASET_AND_BASELINE_AUDIT.md`, dll.). |
| `results/` | Luaran hasil inferensi tervalidasi `rev05` (tabel metrik RQ1, RQ2, RQ3, kurva AURC, dan artefak verifikasi SHA-256). |

---

## 🧪 Pengujian Unit Lokal

Untuk menjalankan pengujian unit otomatis sebelum melakukan inferensi pada GPU cloud:

```bash
# Dari root repositori:
python main/scripts/test_decisive_validation_v5.py
```
Skrip akan menguji 9 komponen kritis secara deterministik:
1. Invariansi prefix tokenisasi kandidat
2. Keselarasan pemotongan logit kausal (*causal logit slicing alignment*)
3. Sifat non-negatif dan penalti directional $\Delta P_{\text{dir}}$
4. Penanganan zero-division pada metrik selektif
5. Monotonisitas kurva risk-coverage
6. Perhitungan AURC empiris
7. Uji permutasi dua sampel & koreksi Holm-Bonferroni
8. Validitas format manifest dataset
9. Integritas hash kriptografis SHA-256
