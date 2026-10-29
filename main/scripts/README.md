# 📜 Panduan Skrip Otomasi & Evaluasi (`main/scripts/`)

Direktori ini berisi seluruh skrip eksekusi inferensi, pengujian unit mandiri (*preflight audit*), generator notebook, dan generator manifest.

---

## 🟢 1. Skrip Aktif & Utama (`rev05` - Standar Skripsi Saat Ini)

Gunakan skrip di bawah ini untuk seluruh kebutuhan pengujian dan replikasi eksperimen terbaru:

| File Skrip | Peran / Deskripsi | Cara Menjalankan |
| :--- | :--- | :--- |
| **`decisive_validation_experiment_v5.py`** | **Mesin Inferensi Utama**: Mengeksekusi evaluasi zero-shot bilingual, sequence log-likelihood scoring, manipulasi kanvas hitam ($P_1$), kalkulasi metrik AURC, dan bootstrap confidence interval 95%. | `python main/scripts/decisive_validation_experiment_v5.py --help` |
| **`test_decisive_validation_v5.py`** | **Suite Pengujian Unit (Preflight)**: 9 unit-test otomatis lokal untuk memverifikasi indexing logit kausal, invariansi tokenisasi, perhitungan AURC, dan integritas hash SHA-256. | `python main/scripts/test_decisive_validation_v5.py` |
| **`generate_resilient_notebooks.py`** | **Generator Notebook Kaggle**: Membangun notebook modular yang tahan gangguan koneksi (*network-resilient*) dan checkpoint otomatis. | `python main/scripts/generate_resilient_notebooks.py` |
| **`resilient_validation_engine.py`** | **Engine Eksekusi Resilien**: Logika checkpoint per batch untuk mencegah kehilangan hasil inferensi jika terjadi OOM atau timeout di GPU cloud. | Digunakan oleh notebook Kaggle rev05 |
| **`audit_and_generate_rev05_manifest.py`** | **Generator Manifest rev05**: Membangun `manifest_rev05.csv` dengan split zero-leakage berbasis klaster hash piksel. | `python main/scripts/audit_and_generate_rev05_manifest.py` |
| **`evaluate_rev05_huatuo_audit.py`** | **Skrip Audit HuatuoGPT**: Mengaudit performa khusus model HuatuoGPT-Vision-7B pada manifest rev05. | `python main/scripts/evaluate_rev05_huatuo_audit.py` |

---

## 🟡 2. Skrip Pengujian Tambahan
- `test_resilient_pipeline.py`: Pengujian integritas checkpoint dan mekanisme pemulihan inferensi.

---

## ⚪ 3. Skrip Arsip / Riwayat Iterasi Terdahulu (Preserved for Audit Trail)
*Skrip-skrip di bawah ini dipertahankan sebagai bukti riwayat akademik iterasi sebelumnya, namun telah disempurnakan oleh versi `v5`:*
- `decisive_validation_experiment_v2.py` / `_v3.py` / `_v4.py`: Versi eksperimen awal sebelum standarisasi exact sequence log-likelihood.
- `test_decisive_validation_v2.py` / `_v3.py` / `_v4.py`: Suite pengujian untuk versi terdahulu.
- `generate_kaggle_notebook_v2.py` s/d `_v5.py`: Generator notebook lama sebelum pipeline modular.
- `audit_and_generate_rev04_manifest.py`: Generator manifest versi rev04.
- `run_baseline.py`, `run_pilot_full.py`, `evaluate_candidate_model.py`: Eksperimen screening model awal.
