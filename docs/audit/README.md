# 🛡️ Scientific Audit & Reproducibility Suite (`docs/audit/`)

Direktori ini memuat seluruh laporan audit saintifik independen, verifikasi matematis, integritas data, validasi statistik tahap lanjut, dan reproduksibilitas eksperimen:
$$\text{Literature} \longrightarrow \text{Research Gap} \longrightarrow \text{Research Questions} \longrightarrow \text{Mathematical Formulation} \longrightarrow \text{Code Implementation} \longrightarrow \text{Experimental Output} \longrightarrow \text{Statistical Analysis} \longrightarrow \text{Scientific Conclusion}$$

---

## 📑 Indeks Dokumen Audit

### 1. Master Reports & Defense Preparation
| Berkas | Ruang Lingkup & Fungsi Utama | Status |
|---|---|:---:|
| [**`SUPERVISOR_VERIFICATION_REPORT.md`**](SUPERVISOR_VERIFICATION_REPORT.md) | **Master Synthesis Report:** Ringkasan komprehensif kesiapan sidang skripsi, status validasi RQ1–RQ5, dan matriks pembuktian klaim ilmiah. | `VERIFIED` |
| [**`SUPERVISOR_QUESTION_BANK.md`**](SUPERVISOR_QUESTION_BANK.md) | **Bank Soal Sidang Skripsi:** 35 pertanyaan kritis dewan penguji dari 11 dimensi metodologi beserta jawaban ilmiah berbasis bukti numerik. | `VERIFIED` |
| [**`RESEARCH_ACTION_PLAN.md`**](RESEARCH_ACTION_PLAN.md) | **Rencana Aksi Berprioritas (P0–P3):** Pembingkaian temuan heuristik ($M_{\text{norm}}$) dan roadmap publikasi ke konferensi/jurnal (CHIL / MedNeurIPS / SINTA 2). | `VERIFIED` |
| [**`PUBLICATION_READINESS_CHECKLIST.md`**](PUBLICATION_READINESS_CHECKLIST.md) | **Checklist Kesiapan Naskah:** Evaluasi komprehensif kepatuhan standar publikasi ilmiah, keterlacakan bukti, dan integritas data. | `VERIFIED` |

### 2. Method Verification & Implementation-to-Evidence
| Berkas | Ruang Lingkup & Fungsi Utama | Status |
|---|---|:---:|
| [**`BIVIS_IMPLEMENTATION_AUDIT.md`**](BIVIS_IMPLEMENTATION_AUDIT.md) | **Audit Implementasi Bi-ViS:** Verifikasi kesesuaian formula, urutan fitur, standardisasi dev-only, dan putusan `A. IMPLEMENTATION VERIFIED`. | `VERIFIED` |
| [**`BIVIS_RUNTIME_REPRODUCIBILITY.csv`**](BIVIS_RUNTIME_REPRODUCIBILITY.csv) | **Data Replikasi Bit-Identical:** Bukti replikasi numerik test AURC dan Risk@80% dengan selisih persis $0.00\text{e}+00$ pada Qwen dan Huatuo. | `EXACT 0.00` |
| [**`DIAGNOSTICS_REPORT.md`**](DIAGNOSTICS_REPORT.md) | **Laporan Diagnostik Validitas Ilmiah:** Investigasi keruntuhan sensitivitas bahasa lokal, tie clipping $\Delta P_{\text{dir}}$, dan kalibrasi uji permutasi $B=10.000$. | `VERIFIED` |
| [**`FINAL_RESEARCH_READINESS_REPORT.md`**](FINAL_RESEARCH_READINESS_REPORT.md) | **Laporan Kesiapan Riset Final:** Evaluasi hasil audit statistik tahap kedua, dependensi klaster, VIF, dan koreksi Holm-Bonferroni. | `VERIFIED` |
| [**`RQ_DECISION_MATRIX.csv`**](RQ_DECISION_MATRIX.csv) | **Matriks Keputusan RQ (RQ1–RQ5):** Status pengujian hipotesis, bukti kuantitatif, interval kepercayaan 95%, dan p-value Holm-adjusted. | `VERIFIED` |
| [**`CLAIM_EVIDENCE_MATRIX.csv`**](CLAIM_EVIDENCE_MATRIX.csv) | **Matriks Bukti Klaim Ilmiah:** Pemetaan setiap klaim artikel terhadap berkas bukti empiris dan baris kode penjamin. | `VERIFIED` |

### 3. Core Scientific Audits (Foundation Suite)
| Berkas | Ruang Lingkup & Fungsi Utama | Status |
|---|---|:---:|
| [**`LITERATURE_VERIFICATION_REPORT.md`**](LITERATURE_VERIFICATION_REPORT.md) | **Audit Literatur Primer:** Verifikasi 11 paper rujukan utama (DOI & venue resmi) serta rekonsiliasi sitasi ganda Sadanandan (2026). | `VERIFIED` |
| [**`DATASET_INTEGRITY_REPORT.md`**](DATASET_INTEGRITY_REPORT.md) | **Audit Integritas Data:** Rekonsiliasi 2.244 baris ke 1.122 pasangan bersih dan bukti **0.00% image cluster leakage** (88 dev vs 204 test). | `VERIFIED` |
| [**`FORMULA_VERIFICATION_REPORT.md`**](FORMULA_VERIFICATION_REPORT.md) | **Audit Rumusan Matematis:** Analisis log-likelihood kausal, normalisasi panjang, orientasi metrik, dan kalkulasi manual terverifikasi. | `VERIFIED` |
| [**`CODE_AUDIT_REPORT.md`**](CODE_AUDIT_REPORT.md) | **Audit Kode Produksi:** Verifikasi slicing token logit kausal $[L_p-1 : L_p-1+L_t]$, pemisahan prefix tokenisasi, dan unit test passing. | `VERIFIED` |
| [**`EXPERIMENT_OUTPUT_AUDIT.md`**](EXPERIMENT_OUTPUT_AUDIT.md) | **Silsilah Artefak Eksperimen:** Verifikasi checkpoint GPU Kaggle Dual-T4 (`2026-10-25T02:48:53 UTC`) dengan 100% paritas desimal. | `VERIFIED` |
| [**`REPRODUCIBILITY_CHECKLIST.md`**](REPRODUCIBILITY_CHECKLIST.md) | **Protokol Reproduksi 3-Tier:** Panduan eksekusi CPU test (7s), CPU recompute (1m), dan GPU re-inference (45m). | `VERIFIED` |

### 4. Repository Governance & Migration Audits
| Berkas | Ruang Lingkup & Fungsi Utama | Status |
|---|---|:---:|
| [**`FINAL_POST_MIGRATION_AUDIT.md`**](FINAL_POST_MIGRATION_AUDIT.md) | **Audit Pasca-Migrasi:** Verifikasi keutuhan 898 file baseline, integritas checksum 314 citra, dan dry-run pipeline setelah migrasi non-destruktif. | `PASS` |
| [**`REPOSITORY_RESTRUCTURING_REPORT.md`**](REPOSITORY_RESTRUCTURING_REPORT.md) | **Laporan Rekayasa Direktori:** Dokumentasi restrukturisasi repositori menuju 10 direktori standar industri. | `COMPLETED` |
| [**`REPOSITORY_INVENTORY.md`**](REPOSITORY_INVENTORY.md) | **Inventaris Aset Repositori:** Rekonsiliasi lengkap seluruh file di disk dan pemetaan direktori kerja. | `VERIFIED` |
| [**`FILE_MIGRATION_MAP.csv`**](FILE_MIGRATION_MAP.csv) | **Peta Migrasi Berkas:** Tabel pemetaan relokasi 109 berkas historis ke struktur repositori kanonikal. | `VERIFIED` |
| [**`REMAINING_ISSUES.md`**](REMAINING_ISSUES.md) | **Log Monitoring Isu Metodologis:** Daftar isu metodologis/teknis yang berhasil diselesaikan dan status monitoringnya. | `RESOLVED` |
