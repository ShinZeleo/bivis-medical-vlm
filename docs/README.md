# 📚 Dokumentasi Akademik & Cetak Biru Skripsi (`docs/`)

Direktori ini memuat seluruh naskah dokumentasi formal, landasan teori, audit literatur, protokol eksperimen, dan penulisan naskah publikasi:
> **"Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"**

---

## 🗂 Indeks Subdirektori

```text
docs/
├── manuscript/  # Naskah artikel ilmiah, formalisasi metode Bi-ViS, blueprint, dan limitasi klinis
├── audit/       # Suite laporan audit verifikasi ilmiah, integritas data, dan reproduksibilitas
├── protocols/   # Protokol formal eksperimen, tata kelola data, dan checklist pra-eksekusi
├── research/    # Cetak biru SSoT penelitian, tinjauan pustaka, dan log evolusi metodologi
└── archive/     # Draf dokumentasi historis versi terdahulu (v9, v10)
```

---

## 📑 Rincian Dokumen per Subdirektori

### 1. Manuscript & Proposed Method ([`manuscript/`](manuscript/))
Dokumentasi inti persiapan artikel ilmiah dan formalisasi metode yang diusulkan:
- [**`MANUSCRIPT_BLUEPRINT.md`**](manuscript/MANUSCRIPT_BLUEPRINT.md): Cetak biru struktur artikel jurnal/konferensi (6 bagian standar IMRaD).
- [**`RESEARCH_NARRATIVE.md`**](manuscript/RESEARCH_NARRATIVE.md): Narasi utuh penelitian: gap klinis, paradoks sensitivitas visual, dan solusi multimodal.
- [**`PROPOSED_METHOD_BIVIS.md`**](manuscript/PROPOSED_METHOD_BIVIS.md): Definisi formal metode **Bi-ViS** (*Bilingual Visual-Sensitivity-Aware Selective Prediction*), landasan teoretis, dan arsitektur logistik.
- [**`METHOD_ALGORITHM.md`**](manuscript/METHOD_ALGORITHM.md): Pseudocode algoritma runtime & kalibrasi offline serta pemetaan modul kode.
- [**`METHOD_COMPARISON.md`**](manuscript/METHOD_COMPARISON.md): Matriks komparasi empiris, analisis kegagalan rasio $M_{\text{norm}}$, dan nilai inkremental fitur.
- [**`CONTRIBUTION_STATEMENT.md`**](manuscript/CONTRIBUTION_STATEMENT.md): Pernyataan kontribusi ilmiah formal bebas HARKing (*Honest Demarcation*).
- [**`FINAL_METHODOLOGICAL_ADJUDICATION.md`**](manuscript/FINAL_METHODOLOGICAL_ADJUDICATION.md): Putusan final metodologi atas 5 temuan kritis audit statistik.
- [**`BIVIS_DEPLOYMENT_LIMITATIONS.md`**](manuscript/BIVIS_DEPLOYMENT_LIMITATIONS.md): Batasan implementasi klinis, overhead komputasi ($3\times$), dan cakupan OOD kanvas hitam.
- [**Draf Naskah Publikasi LaTeX (`draft/`)**](manuscript/draft/README.md): Berkas draf naskah publikasi format LaTeX standar ITB/JICT Overleaf ready (`main.tex`, `references.bib`, `itb_logo.png`, PDF).
- [**Template Jurnal Target JIKI UI (`templates/AuthorGuideline_JIKI/`)**](manuscript/templates/AuthorGuideline_JIKI/README.md): Paket resmi pedoman penulisan dan template LaTeX (`IEEEtran.cls`) / Word untuk Jurnal Ilmu Komputer dan Informasi (JIKI Fasilkom UI, SINTA 2).

### 2. Scientific Audit Suite ([`audit/`](audit/))
Suite laporan audit saintifik independen untuk pertanggungjawaban di hadapan dosen pembimbing dan penguji:
- [**`SUPERVISOR_VERIFICATION_REPORT.md`**](audit/SUPERVISOR_VERIFICATION_REPORT.md): Laporan sintesis master status validasi RQ1–RQ5 dan kesiapan sidang.
- [**`SUPERVISOR_QUESTION_BANK.md`**](audit/SUPERVISOR_QUESTION_BANK.md): Bank 35 pertanyaan kritis ujian skripsi beserta jawaban berbasis bukti.
- [**`BIVIS_IMPLEMENTATION_AUDIT.md`**](audit/BIVIS_IMPLEMENTATION_AUDIT.md): Laporan audit kesesuaian implementasi-ke-bukti Bi-ViS (`VERIFIED`).
- [**`BIVIS_RUNTIME_REPRODUCIBILITY.csv`**](audit/BIVIS_RUNTIME_REPRODUCIBILITY.csv): Tabel replikasi numerik bit-identical test AURC dan Risk@80%.
- [**`DIAGNOSTICS_REPORT.md`**](audit/DIAGNOSTICS_REPORT.md): Laporan diagnostik validitas ilmiah pipeline evaluasi, bias asimetri, dan kalibrasi uji permutasi.
- [**`FINAL_RESEARCH_READINESS_REPORT.md`**](audit/FINAL_RESEARCH_READINESS_REPORT.md): Penilaian kesiapan riset tahap akhir pasca audit inferensi statistik.
- [**`FINAL_POST_MIGRATION_AUDIT.md`**](audit/FINAL_POST_MIGRATION_AUDIT.md): Laporan audit integritas pasca restrukturisasi repositori.
- [**`DATASET_INTEGRITY_REPORT.md`**](audit/DATASET_INTEGRITY_REPORT.md): Pembuktian zero leakage dan rekonsiliasi kohort 1.122 data bersih.
- [**`FORMULA_VERIFICATION_REPORT.md`**](audit/FORMULA_VERIFICATION_REPORT.md): Verifikasi matematis log-likelihood, $\Delta P_{\text{dir}}$, dan estimasi risiko.
- [**`CODE_AUDIT_REPORT.md`**](audit/CODE_AUDIT_REPORT.md): Audit implementasi kode inferensi logit kausal dan unit test.
- [**`EXPERIMENT_OUTPUT_AUDIT.md`**](audit/EXPERIMENT_OUTPUT_AUDIT.md): Silsilah artefak checkpoint Kaggle Dual-T4.
- [**`LITERATURE_VERIFICATION_REPORT.md`**](audit/LITERATURE_VERIFICATION_REPORT.md): Audit 11 paper primer dengan DOI/arXiv terverifikasi.
- [**`REPRODUCIBILITY_CHECKLIST.md`**](audit/REPRODUCIBILITY_CHECKLIST.md): Protokol reproduksi 3-tier.
- [**`RESEARCH_ACTION_PLAN.md`**](audit/RESEARCH_ACTION_PLAN.md): Rencana aksi berprioritas (P0–P3) dan roadmap publikasi.
- [**`PUBLICATION_READINESS_CHECKLIST.md`**](audit/PUBLICATION_READINESS_CHECKLIST.md): Daftar periksa kelayakan publikasi ilmiah.
- [**`REMAINING_ISSUES.md`**](audit/REMAINING_ISSUES.md): Log monitoring isu teknis dan metodologis yang terselesaikan.
- [**`REPOSITORY_INVENTORY.md`**](audit/REPOSITORY_INVENTORY.md): Inventaris aset dan file seluruh repositori.
- [**`REPOSITORY_RESTRUCTURING_REPORT.md`**](audit/REPOSITORY_RESTRUCTURING_REPORT.md): Laporan proses restrukturisasi struktur direktori.

### 3. Protocols & Checklists ([`protocols/`](protocols/))
- [**`EXPERIMENT_PROTOCOL.md`**](protocols/EXPERIMENT_PROTOCOL.md): Protokol formal eksekusi decisive validation rev05, skema abstensi adaptif, dan kalibrasi threshold $\tau$.
- [**`DATA_PROCESSING_PROTOCOL.md`**](protocols/DATA_PROCESSING_PROTOCOL.md): Protokol pembersihan dan pemisahan klaster citra bebas kebocoran.
- [**`ADAPTIVE_ABSTENTION_IMPLEMENTATION_CHECKLIST.md`**](protocols/ADAPTIVE_ABSTENTION_IMPLEMENTATION_CHECKLIST.md): Checklist implementasi modul abstensi adaptif.
- [**`EXPERIMENT_UPDATE_CHECKLIST.md`**](protocols/EXPERIMENT_UPDATE_CHECKLIST.md): Daftar periksa metodologis pra-eksekusi eksperimen.

### 4. Research Foundation & SSoT ([`research/`](research/))
- [**`RESEARCH_MASTER.md`**](research/RESEARCH_MASTER.md): **Single Source of Truth (SSoT)** otoritatif berisi rumusan matematika, hipotesis, dan desain eksperimen.
- [**`RESEARCH_OVERVIEW.md`**](research/RESEARCH_OVERVIEW.md): Ringkasan eksekutif latar belakang klinis dan status validasi empiris per Research Question.
- [**`LITERATURE_AUDIT.md`**](research/LITERATURE_AUDIT.md): Matriks pemetaan literatur internasional dan analisis gap riset.
- [**`REVISION_CHANGELOG.md`**](research/REVISION_CHANGELOG.md): Catatan evolusi metodologi dari LoRA ke *test-time selective prediction*.
- [**`CHANGELOG_REV06.md`**](research/CHANGELOG_REV06.md): Log perubahan audit, refactoring modular, dan spesifikasi protokol rev06.
- [**`REVISION_NOTES.md`**](research/REVISION_NOTES.md): Tanggapan atas arahan dan masukan dosen pembimbing.
- [**`SUPERVISOR_DECISION_REPORT_V11.md`**](research/SUPERVISOR_DECISION_REPORT_V11.md): Laporan putusan bimbingan atas kalibrasi OOF dan hierarki ablasi.

### 5. Historical Archive ([`archive/`](archive/))
Arsip draf naskah versi 9 dan versi 10 untuk menjaga jejak audit historis:
- [**`RESEARCH_MASTER_old.md`**](archive/RESEARCH_MASTER_old.md): Draf awal SSoT penelitian.
- [**`RESEARCH_MASTER_v10.md`**](archive/RESEARCH_MASTER_v10.md): Draf SSoT versi 10 sebelum formalisasi Bi-ViS.
- [**`LITERATURE_AUDIT_v10.md`**](archive/LITERATURE_AUDIT_v10.md): Audit literatur versi 10 terdahulu.
- [**`REVISION_CHANGELOG_v10.md`**](archive/REVISION_CHANGELOG_v10.md): Log revisi metodologi versi 10 terdahulu.
