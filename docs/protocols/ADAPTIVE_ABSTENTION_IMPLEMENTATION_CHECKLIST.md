# ADAPTIVE ABSTENTION IMPLEMENTATION CHECKLIST (V11.0)
## Medical Vision-Language Models | Operational Execution & Audit Checklist

**Metadata Dokumen:**  
- **Versi:** 1.0.0 (Master Checklist)  
- **Tanggal:** 31 Oktober 2026  
- **Peneliti / Penulis:** Imam (Undergraduate CS Thesis Researcher)  
- **Standar Integritas:** Reproducible Research & Clean Architecture  

---

## 1. Kode dan Artefak yang Sudah Tersedia (Verified Available)

- [x] **Authoritative Manifest (`manifest_rev05.csv`):**  
  - SHA-256: `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`.
  - 2.244 baris QA total, 1.122 primary cohort biner bersih, 314 citra fisik kanonis.
  - Zero-leakage: 88 klaster citra Dev Set vs 204 klaster citra Test Set.
- [x] **Inferensi GPU Penuh Kaggle Dual-T4 (100% Selesai):**  
  - `Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv` (1.133 baris $\times$ 62 kolom).
  - `FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv` (1.133 baris $\times$ 62 kolom).
  - Berkas metadata eksekusi: `run_metadata.json` (timestamp `2026-10-25T02:48:53 UTC`).
- [x] **Modul Pemodelan Risiko Terpelajar (`resilient_validation_engine.py`):**  
  - Fungsi `train_and_evaluate_risk_ablation` yang mengimplementasikan GroupKFold pada Dev Set untuk `LR-conf`, `LR-visual`, dan `LR-full`.
  - Perhitungan Discrete Riemann AURC via `calculate_discrete_aurc`.
  - Uji permutasi klaster berpasangan via `compute_paired_cluster_permutation_test`.
  - Bootstrap klaster berpasangan via `compute_paired_cluster_bootstrap`.
  - Koreksi Holm-Bonferroni via `apply_holm_bonferroni_correction`.
- [x] **Suite Pengujian Preflight rev05 (`test_decisive_validation_v5.py`):**  
  - 9/9 unit test lolos 100% (`PASS_UNIT_TEST`).

---

## 2. Kode yang Perlu Dimodifikasi (Refactoring / Adaptation)

- [x] **`main/scripts/resilient_validation_engine.py`:**  
  - Menambahkan modul ekstraksi skor risiko eksplisit $R(x)$ dan $S_{\text{LR}}(x)$ untuk pengujian abstention diskrit.
  - Memastikan kompatibilitas tipe data numpy array antara pandas Series dan scikit-learn.
- [x] **Struktur Pelaporan Metrik:**  
  - Mengintegrasikan metrik penahanan operasional (Coverage Terwujud, Abstention Rate, Selective Risk, Selective Accuracy) di samping metrik ranking pasif (AURC).

---

## 3. Kode yang Perlu Ditambahkan (New Modules)

- [x] **`main/scripts/adaptive_abstention_engine.py`:**  
  - Engine komputasi CPU mandiri untuk mengeksekusi kalibrasi threshold Dev Set dan evaluasi Test Set.
  - Menyediakan dua fungsi utama:
    1. `calibrate_and_evaluate_coverage_targeted(df_dev, df_test, model_name, target_grid)`
    2. `calibrate_and_evaluate_risk_targeted(df_dev, df_test, model_name, risk_grid)`
  - Menyimpan hasil ke direktori: `main/results/decisive_validation/rev05/`
- [x] **`main/scripts/test_adaptive_abstention.py`:**  
  - Uji unit otomatis untuk memastikan:
    1. Threshold monotonic invariance (threshold coverage lebih tinggi $\implies$ threshold risiko $\tau$ lebih longgar).
    2. Zero data leakage (parameter kalibrasi tidak pernah menyentuh Test Set).
    3. Formula selective risk dan selective accuracy menghasilkan nilai dalam rentang valid $[0, 1]$.
    4. Keselarasan denominator (sampel abstained tidak dihitung sebagai denominator akurasi).

---

## 4. Pengujian yang Harus Dilakukan (Verification Plan)

| Pengujian | Prosedur Uji | Kriteria Kelulusan | Status |
| :--- | :--- | :--- | :---: |
| **U1: Non-Leakage Assertion** | Verifikasi bahwa fungsi kalibrasi threshold hanya menerima `df_dev`. | Tidak ada akses ke kolom atau indeks `df_test` saat menghitung $\tau$. | `PASSED` |
| **U2: Denominator Invariant** | $N_{\text{answered}} + N_{\text{abstained}} == N_{\text{total}}$. | Selisih $= 0$ pada seluruh iterasi. | `PASSED` |
| **U3: Range Invariant** | $0.0 \le \text{Selective Risk} \le 1.0$ dan $0.0 \le \Phi \le 1.0$. | Tidak ada nilai negatif atau $> 1.0$. | `PASSED` |
| **U4: Monotonic Selective Risk Trade-Off** | Evaluasi arah kurva: cakupan menurun $\implies$ risiko selektif menurun. | Risiko selektif pada coverage 50% $<$ coverage 90%. | `PASSED` |
| **U5: Statistical Reproducibility** | Rekomputasi metrik dari checkpoint mentah. | 100% desimal identik terhadap berkas luaran resmi. | `PASSED` |

---

## 5. Output yang Harus Dihasilkan (Deliverable Artifacts)

- [x] **`adaptive_abstention_coverage_targeted.csv`**:  
  Tabel evaluasi target coverage 50%, 60%, 70%, 80%, 90% untuk 4 estimator (`M1_direct`, `LR_conf`, `LR_visual`, `LR_full`) pada Qwen dan Huatuo.
- [x] **`adaptive_abstention_risk_targeted.csv`**:  
  Tabel evaluasi batas risiko target 15%, 20%, 25% untuk mengamati coverage yang dipertahankan dan fenomena *risk slippage*.
- [x] **`RESEARCH_MASTER.md`**:  
  Dokumen master penelitian lengkap versi 11.0 (Single Source of Truth).
- [x] **`REVISION_CHANGELOG_11.md`**:  
  Laporan komparasi perubahan v10.0 $\to$ v11.0.
- [x] **`EXPERIMENT_PROTOCOL.md`**:  
  Rencana metodologi eksperimen adaptif.
- [x] **`LITERATURE_AUDIT_V11.md`**:  
  Matriks literatur mutakhir 11+ paper terverifikasi.
- [x] **`SUPERVISOR_DECISION_REPORT_V11.md`**:  
  Laporan eksekutif ringkas untuk dosen pembimbing.

---
*ADAPTIVE_ABSTENTION_IMPLEMENTATION_CHECKLIST.md disahkan sebagai panduan verifikasi eksekusi v11.0.*
