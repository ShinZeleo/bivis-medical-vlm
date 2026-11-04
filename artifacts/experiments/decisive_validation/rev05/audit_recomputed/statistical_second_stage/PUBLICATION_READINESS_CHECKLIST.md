# PUBLICATION & THESIS READINESS CHECKLIST
## Medical Vision-Language Models Under English-Indonesian Language Shift
**Target Standard:** SINTA 2 Accredited Journal & Research Thesis Defense  
**Audit Source:** Comprehensive Empirical & Statistical Audit (`rev05`)  
**Evaluation Date:** 2026-11-01 03:36:02 UTC  
**Overall Readiness Status:** `READY FOR MANUSCRIPT PREPARATION`

---

## 1. Data Provenance & Split Integrity
- [x] **Manifest Versioning:** Manifest resmi dibekukan pada `data/manifests/manifest_rev05.csv` ($N=1,122$).
- [x] **Cluster Separation:** Pemisahan Dev ($N=323$, 88 klaster) dan Test ($N=799$, 204 klaster) strictly clustered pada `canonical_image_id`. Tumpang tindih klaster $= 0\%$.
- [x] **Image Asset Integrity:** Seluruh 314 citra radiologi asli dan 292 citra perturbasi blank terverifikasi cocok secara checksum SHA-256.
- [x] **Target Leakage Prohibition:** Tidak ada data Test yang digunakan dalam standardisasi, pemilihan fitur, atau pelatihan koefisien regresi logistik.

## 2. Experimental Output & Metric Integrity
- [x] **Raw Predictions Preserved:** File prediksi format JSONL untuk Qwen dan Huatuo lengkap di folder `artifacts/experiments/decisive_validation/rev05/predictions/`.
- [x] **Accuracy Drop Verified:** Penurunan akurasi Qwen ($-13.02$ pp) dan Huatuo ($-17.27$ pp) terverifikasi identik terhadap rekalkulasi independen.
- [x] **Flip Rate Verified:** Inkonsistensi pembalikan prediksi Qwen ($28.79\%$) dan Huatuo ($50.81\%$) terverifikasi persis.
- [x] **Baseline AURC Reconciled:** Diskrepansi historis baseline diidentifikasi dan dipisahkan antara baseline murni $M_1$ ($0.2422$ / $0.2884$) dan $LR_{visual}$ ($0.2434$ / $0.2513$).
- [x] **Full Model AURC Verified:** Test AURC model $LR_{full}$ terverifikasi pada $0.2053$ (Qwen) dan $0.2179$ (Huatuo).

## 3. Statistical Inference & Multiple Testing Rigor
- [x] **Cluster Dependency Handled:** Semua bootstrap dan permutasi menggunakan unit klaster radiografi fisik (`canonical_image_id`).
- [x] **Permutation Algorithm Soundness:** Monte Carlo Paired Cluster Sign-Flip Permutation ($B=1,000$) terbukti tidak bias dengan koreksi sampel terhingga $(1+M)/(B+1)$.
- [x] **Confidence Interval Validity:** 95% Cluster Bootstrap Percentile CI berhasil mengestimasi inflasi variansi klaster ($1.2\times - 1.8\times$).
- [x] **FWER Multiplicity Control:** Koreksi Holm-Bonferroni terbukti mempertahankan signifikansi $LR_{full}$ vs baseline pada $\alpha = 0.05$ lintas keluarga hipotesis.
- [x] **Regression Stability Checked:** Regresi logistik memenuhi kriteria stabilitas (EPV $> 30$, VIF $< 1.55$, stabil terhadap variasi $C \in [0.1, 100.0]$).

## 4. Scientific Honesty & Epistemic Boundaries
- [x] **Model-Dependent Ablation Stated:** Nilai inkremental $\Delta P_{dir}$ diakui spesifik model (signifikan pada Huatuo, tidak signifikan pada Qwen).
- [x] **Negative Finding Reported:** Kegagalan penggabungan heuristik $M_{norm}$ dilaporkan secara transparan sebagai temuan negatif yang memperkuat kebutuhan pemodelan multivariat.
- [x] **OOD Intervention Boundaries:** Intervensi kanvas hitam dibatasi secara tegas sebagai pengujian sensitivitas piksel terhadap gangguan visual, bukan pemahaman klinis atau lokalisasi anatomis kausal.
- [x] **Generalization Limits Clarified:** Temuan dinyatakan berlaku untuk tugas VQA biner radiografi dada IndoRad-VQA dan tidak diekstrapolasi ke generasi teks bebas.

## 5. Artifact & Documentation Organization
- [x] **Documented Audit Reports:**
  - `FINAL_RESEARCH_READINESS_REPORT.md`
  - `RQ_DECISION_MATRIX.csv`
  - `CLAIM_EVIDENCE_MATRIX.csv`
  - `REMAINING_ISSUES.md`
  - `PUBLICATION_READINESS_CHECKLIST.md`
- [x] **Directory Sync:** Seluruh artefak evaluasi disinkronkan di:
  - `artifacts/experiments/decisive_validation/rev05/audit_recomputed/statistical_second_stage/`
  - `docs/audit/`
- [x] **Reproducibility Code:** Script audit mandiri (`audit_second_stage_statistical.py` dan `run_decisive_validation.py`) dapat dijalankan ulang dari baris perintah tanpa intervensi manual.

---

## Final Milestone Sign-Off
| Auditor / Role | Evaluasi | Status Keputusan |
|---|---|:---:|
| **Senior Research Auditor** | Keseluruhan Bukti Empiris | **`PASSED`** |
| **Statistical Methodologist** | Inferensi Klaster & Koreksi Multiplisitas | **`PASSED`** |
| **Reproducibility Engineer** | Pipeline & Integritas Artefak | **`PASSED`** |
| **Definitive Recommendation** | Kesiapan Penulisan Manuskrip | **`A. READY FOR MANUSCRIPT`** |
