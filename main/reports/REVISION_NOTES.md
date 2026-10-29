# REVISION_NOTES.md — Decisive Validation Experiment (Revision rev02)

**Research Project:** Evaluasi Reliabilitas dan Selective Prediction pada Medical Vision-Language Model di Bawah Pergeseran Bahasa (IndoRad-VQA)  
**Author:** Imam (Undergraduate CS Thesis)  
**Auditor:** Research Engineer & Reproducibility Auditor  
**Revision Identifier:** `rev02`  
**Date:** 2026-10-19  

---

## 1. Executive Summary & Rationale

Eksperimen validasi empiris utama pada Kaggle Dual Tesla T4 telah dijalankan pada revisi pertama (`04-decisive-validation-live.ipynb`, durasi runtime ~3.5 jam). Namun, audit menyeluruh terhadap artefak hasil dan log eksekusi menemukan anomali metodologis kritis pada *data resolution* yang menyebabkan split evaluasi tidak sesuai dengan protokol otoritatif.

Sesuai dengan **Critical Requirement: Create a New Revision — Never Edit Original Files**, seluruh file historis, skrip lama, checkpoint, CSV, dan notebook rev01 dipertahankan dalam status **read-only** (diverifikasi melalui hash SHA-256 sebelum dan sesudah rekonstruksi). Revisi independen baru (**rev02**) dibuat untuk memperbaiki seluruh anomali dengan jaminan matematika 100% reproduktif.

---

## 2. Root Cause Analysis (Bugs in rev01)

### Bug 1: Fallback Manifest Tidak Sah (Critical Methodological Violation)
- **Gejala:** Pada runtime Kaggle `04-decisive-validation-live.ipynb`, Cell 4 gagal menemukan path lokal manifest karena dataset Kaggle belum diekstrak ke path yang diantisipasi.
- **Dampak Fatal:** Kode melakukan fallback otomatis mengunduh dataset Hugging Face `Lab-IS/IndoRad-VQA` dan memotong 150 sampel pertama (`head(150)`). Hal ini menghasilkan:
  - Distribusi split yang salah: `{'dev': 103, 'test': 47}` (seharusnya `{'test': 92, 'dev': 58}`).
  - Klaster gambar yang salah: 150 citra unik 1:1 tanpa pengelompokan pertanyaan (seharusnya 58 klaster citra fisik unik dengan rata-rata 2.59 pertanyaan per citra).
  - Label pada Cell 13 "Test-Only (N=92)" menjadi *misleading*, karena kenyataannya hanya mengevaluasi $N=47$ sampel test.

### Bug 2: Resolusi Citra Gagal Karena Konvensi Penamaan Ganda
- **Gejala:** Citra di lingkungan lokal tersimpan sebagai `{canonical_image_id}.png` (contoh: `6185dbde43bcbc0c.png`). Di dalam arsip `indorad-vqa-research.zip`, citra tersimpan sebagai `data/images/img_{qa_id}.png` (contoh: `img_47.png`).
- **Dampak:** Skrip rev01 hanya mencari `{canonical_image_id}.png` di Kaggle, gagal menemukan file citra yang sebenarnya ada di zip, lalu memicu fallback Hugging Face.

### Bug 3: Validasi Checkpoint Kurang Ketat
- Skrip rev01 memverifikasi kecocokan kolom dan nama model, tetapi belum memvalidasi hash SHA-256 dari manifest atau identifier revisi. Hal ini membuka risiko checkpoint lama dari manifest yang salah termuat kembali.

### Bug 4: Deprecation Warning
- Penggunaan `datetime.utcnow()` memicu warning `DeprecationWarning: datetime.datetime.utcnow() is deprecated`.

---

## 3. Systematic Fixes in Revision rev02

| Komponen | Implementasi rev01 | Perbaikan rev02 |
| :--- | :--- | :--- |
| **Manifest Loading** | Mencari path disk; jika gagal, download dari HF dan potong 150 sampel sembarang | **Bit-Identical Embedded Store**: Manifest CSV dikompresi (base64+zlib) di dalam kode skrip/notebook. Wajib cocok dengan SHA256 otoritatif `988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52`. Dilarang keras fallback ke Hugging Face. |
| **Split Verification** | Pasif (menerima split apa pun yang terbentuk) | **Hard Invariant Assertions**: Menegaskan secara programatik $N=150$, split `{'test': 92, 'dev': 58}`, klaster test $= 35$, klaster dev $= 23$, dan *cross-split leakage* $= 0$. Abort jika ada selisih 1 sampel pun. |
| **Image Resolver** | Hanya mencari `{canonical_image_id}.png` | **Multi-Strategy Resolver**: Mencari `{cid}.png`, `{img_id}.png`, `{image_name}`, `img_{qa_id}.png`, dan `img_{qa_id:05d}.png` di seluruh direktori input Kaggle dan lokal. Memverifikasi integritas file dengan PIL. |
| **Checkpoint Resume** | Hanya cek kolom dasar & nama model | **Strict Revision & Hash Verification**: Memvalidasi `revision_id == "rev02"` dan `manifest_sha256 == AUTHORITATIVE_MANIFEST_SHA256`. Checkpoint rev01 atau manifest yang tidak cocok otomatis dibackup dan diisolasi. |
| **Datetime Standard** | `datetime.utcnow()` | `datetime.now(timezone.utc)` (bebas warning). |
| **Direktori Output** | `main/outputs/results/` | Terisolasi penuh pada `results/decisive_validation/rev02/{model_name}/`. |
| **Download Archive** | `decisive_validation_live_results_*.zip` | `decisive_validation_rev02_results_{timestamp}.zip` memuat hanya artefak rev02. |

---

## 4. Source Preservation & SHA-256 Hash Audit

Seluruh 14 file historis diverifikasi **tidak mengalami modifikasi apa pun** (bit-identical):

```
988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52  main/results/decisive_validation_manifest.csv
1f8af04e0df567cefbb6e132a2f47fad7ca4540ee77314cfc391088c9a814c32  main/notebooks/04-decisive-validation-live.ipynb
dc675a4300be67a615ec73d3d772c09ecc306f931feb139d812bc6b7cdc87efc  main/scripts/generate_kaggle_notebook.py
c3ea3d95440dfcc3f64cf5e8ac084292f009a7e229f8d82a0cd3594541e403e9  main/scripts/decisive_validation_experiment.py
2d83bb98e32a4931262b8ee5199678015f632dbb21d32a08a10da367a1f84230  main/scripts/confidence.py
f1f683f06a249f58a0d6a7d88d75eee6eee6465a7c6db5c690f7c92dbf752829  main/scripts/test_visual_intervention.py
e4cffd89de5c1b23613b5979b22f5b7a6d295508a86b5692023443c29730f22e  main/reports/3_DECISIVE_VALIDATION_PROTOCOL.md
8316d409370ff201e89ce91e4a8289b3ed90714a231500440a13fdd26634cac6  main/outputs/results/final_cross_model_comparison_test_only.csv
02f8937a4fdf1878d14f4d6eefd004429c330b9019d5fc81a4e92f41e7e517fe  main/outputs/results/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/estimator_comparison_huatuo7b.csv
e59f0ebdf4a2b61fea5b4385d81d373e88be520971ee4fc2b7b80ec493688f5f  main/outputs/results/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv
4e3d364979304c5b8f04636b44d288483306989fde1224f2c9f7b0dc49ea41c6  main/outputs/results/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/risk_coverage_curve_huatuo7b_test.png
47b1f398dd61c803e47da17fcae3b89e1b08f3d5c0716e6d89a407aa1250a139  main/outputs/results/Qwen_Qwen2.5_VL_7B_Instruct/estimator_comparison_qwen7b.csv
82d229641a42feb646d497f06ec969045555550f161336067bb6f48627e5690b  main/outputs/results/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv
a2a44190747412e2f21d66b9c071ecbeedb190e67129883aa046b9288c998ee3  main/outputs/results/Qwen_Qwen2.5_VL_7B_Instruct/risk_coverage_curve_qwen7b_test.png
```

---

## 5. New File Deliverables Created for rev02

1. **`main/Kaggle/05_decisive_validation_revised.ipynb`**
   - Self-contained publication-grade notebook for Kaggle Dual-T4 GPU.
   - Total cells: 15 (1 Markdown directive cell + 14 Code cells).
   - Validated: 100% of code cells compile with 0 syntax errors.
2. **`main/scripts/generate_kaggle_notebook_v2.py`**
   - Deterministic generator for `05_decisive_validation_revised.ipynb`.
   - Embeds authoritative frozen manifest as bit-identical base64 store.
3. **`main/scripts/decisive_validation_experiment_v2.py`**
   - Standalone, modular, independently executable Python script for rev02.
   - Provides CLI with `--manifest`, `--out-dir`, and `--dry-run-check`.
4. **`main/scripts/test_decisive_validation_v2.py`**
   - New automated unit test suite covering manifest invariants, embedded decompression, image resolution, and checkpoint compatibility.
   - Result: 5/5 tests PASS in 0.105s.
5. **`main/reports/REVISION_NOTES.md`** (this document)
   - Comprehensive audit trail, root-cause analysis, and change log.
6. **`main/results/decisive_validation/rev02/`**
   - Dedicated output directory for rev02 experiment runs.

---

## 6. Preflight Verification Results

- **Authoritative Manifest Invariants:**
  - Total QA pairs: 150 (Test: 92, Dev: 58)
  - Total Unique Image Clusters: 58 (Test: 35, Dev: 23)
  - Cross-Split Cluster Leakage: **ZERO** (verified by set intersection)
  - Question Format: 100% binary Yes/No (`is_yes_no == True`)
  - Target Reference Answers: NO: 79, YES: 71
- **Disk Image Resolution:** 150 / 150 images verified and readable.
- **Unit Test Regression:**
  - `main/scripts/test_visual_intervention.py`: 5/5 PASS.
  - `main/scripts/test_decisive_validation_v2.py`: 5/5 PASS.

---

## 7. Execution Instructions for Kaggle GPU Dual-T4

1. **Upload Notebook ke Kaggle:**
   - Unggah file `main/Kaggle/05_decisive_validation_revised.ipynb` ke Kaggle.
2. **Konfigurasi Lingkungan:**
   - Accelerator: **GPU T4 x2**.
   - Persistence: **Files only** (atau None).
   - Internet: **On** (untuk mengunduh checkpoint model dari Hugging Face: `Qwen/Qwen2.5-VL-7B-Instruct` dan `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`).
3. **Data Input (Opsional tapi Direkomendasikan):**
   - Lampirkan dataset Kaggle yang memuat `indorad-vqa-research.zip` (atau gambar IndoRad-VQA).
   - Jika dataset dilampirkan, notebook akan mengekstrak citra secara otomatis.
   - Manifest evaluasi tidak bergantung pada upload manual karena telah tertanam secara *bit-identical* di dalam Cell 4.
4. **Jalankan Notebook ("Run All"):**
   - Cell 1-5 akan melakukan audit sistem, memvalidasi GPU ganda, memverifikasi hash manifest, dan mengecek seluruh 150 citra.
   - Cell 8 akan menjalankan smoke test empiris pada sampel pertama untuk mengukur runtime per sampel.
   - Cell 9-10 mengevaluasi Model 1 (Qwen2.5-VL-7B-Instruct).
   - Cell 11-12 mengevaluasi Model 2 (HuatuoGPT-Vision-7B-Qwen2.5VL).
   - Cell 13 menghasilkan tabel komparasi final $N=92$ test-only.
   - Cell 14 memaketkan seluruh artefak ke dalam `decisive_validation_rev02_results_{timestamp}.zip`.
5. **Download Hasil:**
   - Unduh file zip dari panel output Kaggle dan ekstrak ke `main/results/decisive_validation/rev02/`.
