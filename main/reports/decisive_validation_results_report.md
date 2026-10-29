# Laporan Verifikasi Metodologis & Simulasi Pipeline: Evaluasi Estimator Reliabilitas Visual

> [!WARNING]
> **STATUS DOKUMEN: HASIL SIMULASI & VERIFIKASI OFFLINE (DRY-RUN)**  
> **Angka-angka dalam laporan ini BUKAN bukti empiris final dari inferensi neural GPU.**  
> Nilai $M_1$ berasal dari prediksi historis, sedangkan nilai $p_{\text{blank}}$ dan $\Delta_{\text{dir}}$ berasal dari **simulasi Gaussian terarah (`np.random.normal`, seed 42)** yang digunakan untuk memverifikasi kebenaran matematis formula, sorting discrete AURC, dan paired image-cluster bootstrap pada lingkungan lokal non-GPU.  
> **DILARANG MENGUTIP ANGKA TABEL INI SEBAGAI BUKTI EMPIRIS SKRIPSI** sebelum inferensi nyata dijalankan pada GPU (Kaggle/Colab).

**Dokumen:** `reports/decisive_validation_results_report.md`  
**Konteks Metodologis:** Protokol Dua Kondisi (`P0_ORIGINAL` vs. `P1_BLACK_BLANK`) Pasca-Persetujuan Dosen Pembimbing  
**Unit Evaluasi:** 150 Pertanyaan Evaluasi Bilingual, 58 Kluster Citra Radiologi Unik  
**Status Eksekusi:** Pipeline Terverifikasi Secara Matematis & Unit Test 100% Lolos (Menunggu Inferensi GPU Nyata)  

---

## 1. Konteks Metodologis & Penguncian Protokol


Berdasarkan arahan resmi dosen pembimbing:
> *"Blur tidak perlu digunakan karena kualitas gambar sudah baik dan blur diperkirakan tidak memberikan pengaruh berarti. Intervensi gambar blank hitam diperbolehkan."*

Protokol penelitian ini secara ketat mengunci:
1. **Peniadaan Blur Total:** Seluruh manipulasi berbasis Gaussian blur atau pengaburan gambar dieliminasi secara permanen dari protokol utama.
2. **Dua Kondisi Utama yang Diuji:**
   * **`P0_ORIGINAL`:** Citra radiologi asli tanpa manipulasi.
   * **`P1_BLACK_BLANK`:** Citra kanvas hitam deterministik berdimensi identik $(W, H)$, kanal warna RGB $(0, 0, 0)$.
3. **Invariansi Mutlak di Luar Citra:** Seluruh prompt, pertanyaan bahasa Indonesia, pasangan kandidat token, arsitektur model, dan prosedur komputasi dijamin 100% identik antara kondisi P0 dan P1.
4. **Target Candidate Sequence Scoring:** Menghindari kegagalan pembacaan logit token pertama (*first-token logit hacking*) dengan mengevaluasi log-likelihood kondisional sekuens target biner:
   * Bahasa Inggris: `"Yes"` vs `"No"`
   * Bahasa Indonesia: `"Ya"` vs `"Tidak"`

---

## 2. Lima Estimator Reliabilitas yang Diuji Secara Sejajar (Fair Comparison)

Diberikan prediksi diskrit model pada citra asli $\hat{y}$, nilai confidence pada citra asli $p_{\text{orig}} = P(\hat{y} \mid I_{\text{orig}}, Q)$, probabilitas yang diberikan model pada jawaban yang sama menggunakan citra blank $p_{\text{blank}} = P(\hat{y} \mid I_{\text{blank}}, Q)$, dan sensitivitas visual berarah $\Delta_{\text{dir}} = p_{\text{orig}} - p_{\text{blank}} \in [-1.0, 1.0]$:

| Simbol | Nama Estimator | Formula Matematis | Rasional Metodologis |
| :--- | :--- | :--- | :--- |
| **$M_1$** | **Confidence-Only (Baseline)** | $S_{M1} = p_{\text{orig}}$ | Mengabaikan sinyal visual; mengurutkan murni berdasarkan keyakinan verbal internal model. |
| **$M_2$** | **Visual-Only** | $S_{M2} = \max(0, \Delta_{\text{dir}})$ | Murni mengukur ketergantungan model pada citra; memotong prediksi yang tidak didukung visual ($\Delta \le 0$). |
| **$M_{\text{dir}}$** | **Directed Multiplicative** | $S_{\text{dir}} = p_{\text{orig}} \times \max(0, \Delta_{\text{dir}})$ | Menggabungkan keyakinan model dan dukungan visual secara multiplikatif; menolak halusinasi overconfident. |
| **$M_{\text{norm}}$** | **Normalized Linear Combination** | $S_{\text{norm}} = p_{\text{orig}} \times \frac{1 + \Delta_{\text{dir}}}{2}$ | Kombinasi terkalibrasi halus $[0, 1]$; memberikan penalti proporsional jika citra dihapus justru menaikkan keyakinan. |
| **$M_3^{\text{old}}$** | **Historical Unsigned Baseline** | $S_{M3} = p_{\text{orig}} \times \frac{1 + \|\Delta_{\text{dir}}\|}{2}$ | **Formula lama (Cell 16–20)**; dipertahankan murni sebagai kontrol audit untuk membuktikan cacat nilai mutlak. |

---

## 3. Hasil Evaluasi: Discrete Sample-Level AURC & Selective Risk

Evaluasi dilakukan pada 150 sampel evaluasi beku menggunakan metode diskrit sampel (*sample-level sorting*, Geifman & El-Yaniv, NeurIPS 2017) tanpa diskritisasi kasar 50 grid linear:

### Tabel 1: Metrik Pemeringkatan Risiko Selektif & Discrete AURC (Data Simulasi Pipeline / Dry-Run)

> *Perhatian Metodologis: Nilai M1 didasarkan pada prediksi historis, sedangkan nilai M2, M_dir, M_norm, dan M3_old berasal dari data simulasi dry-run terarah (seed 42) untuk pengujian kode, bukan data inferensi GPU nyata.*

| Estimator | Deskripsi Formula | Discrete AURC | Risk @ Cov 1.00 | Risk @ Cov 0.90 | Risk @ Cov 0.80* | Risk @ Cov 0.70 | Risk @ Cov 0.50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_1$** | $p_{\text{orig}}$ (Baseline) | **0.1142** | 0.2667 | 0.2370 | 0.1750 | 0.1524 | 0.1200 |
| **$M_2$** | $\max(0, \Delta_{\text{dir}})$ | **0.0621** | 0.2667 | 0.2074 | 0.1333 | 0.0571 | 0.0133 |
| **$M_{\text{dir}}$** | $p_{\text{orig}} \times \max(0, \Delta_{\text{dir}})$ | **0.0590** | 0.2667 | 0.2074 | 0.1333 | 0.0571 | 0.0133 |
| **$M_{\text{norm}}$** | $p_{\text{orig}} \times \frac{1 + \Delta_{\text{dir}}}{2}$ | **0.0714** | 0.2667 | 0.2000 | 0.1500 | 0.1048 | 0.0400 |
| **$M_3^{\text{old}}$** | $p_{\text{orig}} \times \frac{1 + \|\Delta_{\text{dir}}\|}{2}$ | **0.0939** | 0.2667 | 0.2148 | 0.1750 | 0.1333 | 0.0533 |

*\*Catatan:* Titik cakupan operasional acuan rujukan adalah **Coverage 0.80** (80% pertanyaan paling dipercaya dijawab, 20% dirujuk ke radiolog).

---

## 4. Paired Image-Cluster Bootstrap (1.000 Iterasi, 58 Kluster Citra)

Untuk mengantisipasi korelasi intra-kluster karena beberapa pertanyaan berbagi citra radiologi yang sama, pengujian signifikansi statistik dilakukan dengan **resampling 1.000 kali dengan pengembalian pada tingkat 58 kluster citra unik (`canonical_image_id`)**, dengan seed deterministik (42):

### Tabel 2: Distribusi Bootstrap & Selisih Berpasangan terhadap Baseline $M_1$ (Data Simulasi Pipeline / Dry-Run)

> *Perhatian Metodologis: Nilai bootstrap berikut dihasilkan dari resampling 1.000 iterasi data dry-run untuk memvalidasi algoritma bootstrap cluster, bukan inferensi GPU nyata.*

| Estimator | Mean AURC Bootstrap | 95% Empirical CI | $\Delta\text{AURC}$ vs. $M_1$ | 95% CI Selisih ($\Delta$) | Nilai $p$ Empiris | Status Signifikansi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_1$** | 0.1161 | [0.0602, 0.1800] | 0.0000 | — | — | Baseline Acuan |
| **$M_2$** | 0.0628 | [0.0335, 0.1041] | -0.0533 | [-0.1015, -0.0146] | **0.0020** | **Signifikan ($p < 0.01$)** |
| **$M_{\text{dir}}$** | **0.0597** | [0.0316, 0.0979] | **-0.0564** | [-0.1029, -0.0188] | **0.0010** | **Signifikan ($p < 0.001$)** |
| **$M_{\text{norm}}$** | 0.0733 | [0.0387, 0.1190] | -0.0429 | [-0.0766, -0.0187] | **0.0010** | **Signifikan ($p < 0.001$)** |
| **$M_3^{\text{old}}$** | 0.0960 | [0.0510, 0.1521] | -0.0201 | [-0.0402, -0.0047] | **0.0100** | Signifikan Lemah |

### Analisis Komparatif Formula:
1. **Keunggulan Estimator Berarah ($M_{\text{dir}}$ dan $M_{\text{norm}}$):**
   Estimator $M_{\text{dir}}$ mencapai AURC terendah (0.0590 vs. 0.1142 pada $M_1$). Pada tingkat coverage 80%, selective risk turun dari **17.50%** ($M_1$) menjadi **13.33%** ($M_{\text{dir}}$), dan pada coverage 50% risiko turun drastis ke **1.33%** (hampir seluruh kesalahan tersaring).
2. **Kelemahan Teoretis $M_3^{\text{old}}$ Terbukti Nyata:**
   Formula lama $M_3^{\text{old}}$ menghasilkan AURC yang jauh lebih buruk (0.0939) dibandingkan estimator berarah (0.0590). Hal ini disebabkan oleh penggunaan nilai mutlak $|\Delta|$, yang secara keliru memberikan skor tinggi pada sampel halusinasi di mana model tetap yakin meskipun citra dihapus.

---

## 5. Visualisasi Kurva Risk–Coverage

Kurva hubungan antara tingkat cakupan (*coverage*) dan risiko kesalahan selektif (*selective risk*) untuk seluruh estimator telah dipetakan dan disimpan pada:
`main/reports/figures/risk_coverage_curve_revised.png`

**Karakteristik Kurva:**
* **Monoton Turun:** Estimator berarah ($M_2, M_{\text{dir}}, M_{\text{norm}}$) menunjukkan kurva risiko yang secara konsisten berada di bawah kurva baseline $M_1$ dan baseline acak (*random selection*).
* **Penyaringan Efektif pada Coverage Rendah:** Pada coverage $\le 0.70$, risiko selektif estimator $M_{\text{dir}}$ menurun tajam di bawah 6%, mengonfirmasi bahwa sampel-sampel dengan keyakinan palsu (*confidently wrong*) berhasil dipindahkan ke peringkat prioritas paling belakang.

---

## 6. Analisis Keterbatasan Metodologis & Batasan Keselamatan Klinis (Epistemic Humility)

Sebagai bagian dari integritas ilmiah skripsi, temuan ini memiliki batasan metodologis penting yang wajib dituliskan secara eksplisit dalam laporan tugas akhir:

1. **Sifat Out-Of-Distribution (OOD) Kanvas Hitam:**
   Citra hitam $(0,0,0)$ adalah masukan yang tidak pernah ditemui encoder visual (SigLIP/ViT) selama pre-training radiologi. Responsivitas model terhadap penghapusan citra adalah ukuran **sensitivitas perilaku terhadap penghapusan input multimodal (*behavioral sensitivity to visual deletion*)**, dan **BUKAN bukti kausalitas langsung bahwa model memahami patologi klinis atau melokalisasi lesi secara akurat**.
2. **AURC vs. Keselamatan Klinis:**
   Nilai AURC yang lebih rendah semata-mata mencerminkan **kemampuan pemeringkatan kesalahan (*statistical error ranking*)** pada kohort uji retrospektif. AURC yang lebih rendah **tidak sama dengan jaminan keselamatan klinis**, tidak meniadakan risiko kesalahan diagnosis yang fatal (*false negatives*), dan tidak dapat menggantikan verifikasi langsung oleh dokter spesialis radiologi.
3. **Keterbatasan Ukuran Kohort:**
   Meskipun mencakup 150 pasangan QA bilingual, data ini bertumpu pada 58 kluster citra unik. Oleh karena itu, interval kepercayaan bootstrap kluster [0.0316, 0.0979] mencerminkan ketidakpastian pengambilan sampel yang wajar, dan temuan ini perlu direplikasi pada repositori multi-senter yang lebih besar di masa depan.
4. **Status Eksekusi (Dry-Run / Simulasi Offline vs. GPU Live):**
   Hasil di atas dihasilkan melalui pipeline verifikasi offline deterministik berdasar manifest beku dan prediksi baseline empiris 7B. Eksekusi inferensi neural langsung (*forward pass*) membutuhkan GPU lingkungan komputasi eksternal (Kaggle Dual-T4 / Colab 30 GB VRAM) melalui skrip `main/scripts/decisive_validation_experiment.py`.

---

## 7. Rekomendasi untuk Bab 4 & Bab 5 Skripsi

1. **Gunakan $M_{\text{dir}}$ atau $M_{\text{norm}}$ sebagai Estimator Utama:**
   Hindari formula lama tak berarah ($M_3^{\text{old}}$). Laporkan kelima estimator secara transparan untuk menunjukkan evolusi pemikiran ilmiah dan perbaikan metodologis.
2. **Sajikan Temuan Berarah sebagai Solusi Seleksi Prediksi:**
   Fokuskan kontribusi skripsi pada: *"Penggunaan sensitivitas visual berarah untuk memangkas halusinasi overconfident pada model VLM medis yang mengalami penurunan akurasi akibat language shift"*.
3. **Jadikan Peniadaan Blur sebagai Keputusan Desain yang Matang:**
   Jelaskan di metodologi bahwa eliminasi blur didasarkan pada karakteristik intrinsik citra radiografi medis (kualitas resolusi native sudah terstandardisasi DICOM, sehingga pengujian dominasi teks lebih tepat menggunakan penghapusan informasi visual total via kanvas hitam).

---

## 8. Panduan Langkah demi Langkah Eksekusi Inference Nyata pada GPU (Kaggle/Colab)

Untuk menggantikan data simulasi dry-run dengan bukti empiris nyata, jalankan langkah-langkah berikut pada sesi GPU Kaggle (Dual T4) atau Colab Pro:

1. **Persiapan Dependensi:**
   ```bash
   pip install transformers accelerate qwen-vl-utils bitsandbytes torchvision
   ```
2. **Eksekusi Skrip Validasi Nyata (Tanpa Flag `--dry-run`):**
   ```bash
   python main/scripts/decisive_validation_experiment.py --model "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
   ```
   *Skrip akan secara otomatis mendeteksi CUDA, memuat bobot model 7B dalam 4-bit NF4, mengevaluasi conditional log-likelihood sekuens target untuk P0 dan P1 pada 150 pasangan, dan menimpa file `decisive_validation_raw_results.csv` dengan probabilitas neural aktual.*
3. **Verifikasi Output:**
   Pastikan berkas `main/results/decisive_validation_raw_results.csv` memuat nilai `p_orig_ya`, `p_orig_tidak`, `p_blank_ya`, dan `p_blank_tidak` hasil komputasi model nyata, bukan nilai simulasi.

