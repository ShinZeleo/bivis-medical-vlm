# Changelog Revisi Protokol & Metodologi Penelitian Pasca-Persetujuan Dosen

**Dokumen:** `reports/changelog_revisi_dosen.md`  
**Tanggal:** 18 Oktober 2026  
**Auditor/Engineer:** Tim Auditor Metodologi & Reproducibility Medical VLM  
**Status:** Sinkronisasi Selesai & Terverifikasi  

---

## 1. Latar Belakang & Arahan Dosen Pembimbing

Dosen pembimbing telah menelaah dan menyetujui arah penelitian skripsi dengan instruksi operasional yang tegas:
> *"Blur tidak perlu digunakan karena kualitas gambar sudah baik dan blur diperkirakan tidak memberikan pengaruh berarti. Intervensi gambar blank hitam diperbolehkan."*

Tujuan dari revisi ini adalah menyinkronkan seluruh artefak kode, proposal, dokumen induk, dan protokol eksperimen agar selaras 100% dengan keputusan dosen, mengeliminasi sisa-sisa konsep blur, memvalidasi pengukuran confidence bilingual, dan memastikan seluruh klaim ilmiah berada dalam koridor bukti empiris yang dapat dipertanggungjawabkan.

---

## 2. Rincian Perubahan per Komponen

### A. Proposal Presentasi (`Medical_VLM_Proposal.pptx` $\to$ `Medical_VLM_Proposal_Revisi_Dosen.pptx`)
* **Tindakan:** File asli `Medical_VLM_Proposal.pptx` **TIDAK ditimpa**, melainkan dibuat salinan revisi transparan `Medical_VLM_Proposal_Revisi_Dosen.pptx` beserta log perbedaan teks di [`main/reports/proposal_slide17_diff.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/proposal_slide17_diff.md).
* **Perubahan Slide 17:**
  1. Judul slide disinkronkan menjadi: *"Sinyal Visual: Dua Kondisi Utama & Evaluasi Berarah"*.
  2. Kondisi utama dikunci menjadi dua:
     * $P_0$: Citra asli (radiologi tanpa manipulasi).
     * $P_1$: Citra blank hitam (kanvas deterministik $(0,0,0)$ yang disetujui dosen).
  3. Kondisi $P_2$ (pixel-shuffled) dipindahkan menjadi eksperimen opsional sekunder.
  4. Formula sensitivitas visual diperbarui dari nilai mutlak tak berarah ($|\Delta P|$) menjadi **sensitivitas berarah**: $\Delta_{\text{dir}} = p_{\text{orig}}(\hat{y}) - p_{\text{blank}}(\hat{y})$.
  5. Evaluasi AURC diperbarui dari grid 50 ambang kasar menjadi **discrete sample-level AURC** dengan paired cluster bootstrap.
  6. Catatan kaki menambahkan batasan epistemik: citra hitam berstatus *out-of-distribution* (OOD) dan penurunan AURC tidak sama dengan jaminan keselamatan klinis.

### B. Dokumen Induk Penelitian (`RESEARCH_MASTER.md`)
* Menambahkan **Sub-bab 1.1: Sinkronisasi Metodologis Pasca-Persetujuan Dosen**:
  * Mencatat penghapusan permanen blur dari protokol eksperimen utama.
  * Mendokumentasikan rasional kanvas hitam deterministik sebagai uji dominasi prior teks.
  * Menegaskan status OOD dan batasan klaim klinis.
* Memperbarui Bab 4 (Desain Intervensi & Formula Estimator):
  * Menstandardisasi notasi matematika: $p_{\text{orig}}$, $p_{\text{blank}}$, $\Delta_{\text{dir}}$.
  * Menjadikan 5 estimator ($M_1, M_2, M_{\text{dir}}, M_{\text{norm}}, M_3^{\text{old}}$) dinilai secara sejajar (*fair comparison*).

### C. Protokol Pra-Registrasi (`main/reports/3_DECISIVE_VALIDATION_PROTOCOL.md`)
* Mengonsolidasikan 4 file lama (`decisive_validation_protocol.md`, `statistical_analysis_plan.md`, `final_research_protocol_gate.md`, `primary_prompt_template.md`) menjadi satu protokol definitif.
* Mengunci aturan scoring sekuens biner:
  * Bahasa Inggris: `"Yes"` vs `"No"`
  * Bahasa Indonesia: `"Ya"` vs `"Tidak"`
* Mengunci prosedur paired image-cluster bootstrap (1.000 iterasi pada 58 kluster citra independen).

### D. Skrip Eksperimen Inti (`main/scripts/decisive_validation_experiment.py`)
* **Eliminasi Total Blur:** Menghapus seluruh dependensi atau opsi manipulasi Gaussian blur.
* **Target Candidate Sequence Scoring:** Mengimplementasikan conditional log-likelihood sekuens target via `extract_target_loglikelihood`, menyelesaikan masalah *first-token logit truncation* dan kalimat pembuka.
* **Logging Metadata Lengkap:** Setiap baris hasil mencatat `canonical_image_id`, `orig_raw_sha256`, `orig_pixel_sha256`, `image_width`, `image_height`, `image_mode`, `blank_generation_method`, `model_id`, `language`, dan `prompt_template`.
* **Dukungan Dua Mode Eksekusi:**
  * Mode GPU: inferensi multimodal penuh bfloat16/4-bit dengan Qwen2.5-VL / HuatuoGPT.
  * Mode Dry-Run / Simulasi Offline (`--dry-run`): verifikasi komputasi discrete AURC, pengujian seluruh 5 estimator, plotting otomatis, dan bootstrap 1.000 iterasi pada lingkungan non-GPU (Windows).
* **Output Artefak Otomatis:** Menghasilkan tabel CSV perbandingan estimator, distribusi JSON bootstrap, dan kurva visualisasi PNG resolusi publikasi.

### E. Rangkaian Unit Test Baru (`main/scripts/test_visual_intervention.py`)
* Dibuat modul pengujian otomatis independen yang memverifikasi 5 aspek kritis:
  1. `test_01`: Verifikasi determinisme kanvas hitam P1 ($\min=0, \max=0, \text{mean}=0, \text{std}=0$) dan integritas dimensi RGB terhadap P0.
  2. `test_02`: Verifikasi normalisasi softmax biner dan simetri scoring sekuens target.
  3. `test_03`: Verifikasi perhitungan discrete AURC dan risiko selektif terhadap data sintetis dengan *analytic ground truth*.
  4. `test_04`: Verifikasi perilaku berarah seluruh 5 estimator pada kasus $\Delta_{\text{dir}} > 0$ vs. $\Delta_{\text{dir}} < 0$, membuktikan cacat matematis formula nilai mutlak $M_3^{\text{old}}$.
  5. `test_05`: Verifikasi determinisme dan stabilitas paired image-cluster bootstrap.
* **Hasil Uji:** 5/5 unit test **LOLOS (100% OK)**.

### F. Konsolidasi & Perapihan File Markdown Repositori
* Sesuai instruksi *"jangan terlalu banyak file md, gabung saja, jangan redundan"*, lebih dari 40 file markdown pecahan di root dan subdirektori telah dipangkas dan dikonsolidasikan menjadi 5 dokumen inti terstruktur:
  1. [`RESEARCH_MASTER.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/RESEARCH_MASTER.md): Master rancangan skripsi.
  2. [`main/reports/audit_sebelum_revisi.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/audit_sebelum_revisi.md): Audit teknis awal.
  3. [`main/reports/3_DECISIVE_VALIDATION_PROTOCOL.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/3_DECISIVE_VALIDATION_PROTOCOL.md): Protokol pra-registrasi final.
  4. [`main/reports/decisive_validation_results_report.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/decisive_validation_results_report.md): Laporan hasil empiris & keterbatasan.
  5. [`main/reports/changelog_revisi_dosen.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/changelog_revisi_dosen.md): Catatan perubahan (dokumen ini).

---

## 3. Matriks Perbandingan Status Komponen

| Komponen | Status Sebelum Revisi | Status Pasca-Revisi |
| :--- | :--- | :--- |
| **Manipulasi Blur** | Masih tercantum di beberapa dokumen rancangan | **DIHAPUS TOTAL** sesuai arahan dosen |
| **Kondisi Intervensi** | 3 kondisi eksploratif (asli, blank, shuffled) | **2 kondisi utama terkunci** ($P_0$ Asli vs $P_1$ Kanvas Hitam) |
| **Proposal Slide 17** | Menampilkan 3 kondisi & formula lama | **Disinkronkan** ke salinan revisi baru |
| **Scoring Confidence** | Pembacaan logit token pertama (rentan kalimat pembuka) | **Target candidate sequence conditional log-likelihood** |
| **Sensitivitas Visual** | Nilai mutlak unsigned $|\Delta P|$ | **Sensitivitas berarah** $\Delta_{\text{dir}} = p_{\text{orig}} - p_{\text{blank}}$ |
| **Estimator Utama** | $M_3$ lama diunggulkan tanpa kontrol | **5 estimator dinilai sejajar** ($M_1, M_2, M_{\text{dir}}, M_{\text{norm}}, M_3^{\text{old}}$) |
| **Evaluasi AURC** | 50 grid linear kasar | **Discrete sample-level sorting** (Geifman et al.) |
| **Uji Signifikansi** | Normal approximation / row-level | **Paired image-cluster bootstrap** (58 kluster citra) |
| **Unit Test** | Belum ada unit test intervensi khusus | **5/5 Unit test lolos 100%** |
| **Klaim Klinis** | Klaim potensi "safety" berlebihan | **Epistemic humility**: batas OOD & error ranking retrospektif |
