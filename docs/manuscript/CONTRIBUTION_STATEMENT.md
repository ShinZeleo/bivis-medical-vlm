# AUTHORITATIVE CONTRIBUTION STATEMENT
## Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift
**Target Milestone:** Formal Manuscript Contribution Specification  
**Target Path:** `docs/manuscript/CONTRIBUTION_STATEMENT.md`  
**Date Logged:** 2026-11-01 10:52:09 UTC  
**Review Standard:** Maximum 3 Specific, Evidence-Grounded, Non-Overclaiming Contributions

---

Berdasarkan audit empiris menyeluruh dan verifikasi literatur bebas halusinasi, penelitian ini menetapkan **tiga kontribusi ilmiah utama yang spesifik, terukur, dan dapat dipertahankan secara metodologis**:

### Kontribusi 1: Karakterisasi Empiris Komparatif Kerentanan Diagnostik dan Patologi Ketergantungan Visual Negatif pada VLM Medis di Bawah Pergeseran Bahasa
Menyajikan evaluasi empiris terkontrol pertama mengenai dampak *English–Indonesian language shift* pada dua kelas arsitektur VLM (model generalis `Qwen2.5-VL-7B` vs model spesialis medis `HuatuoGPT-Vision-7B`). Penelitian ini membuktikan bahwa:
1. Pergeseran bahasa memicu penurunan akurasi yang signifikan secara statistik pada kedua model ($-13.02$ pp pada Qwen, $-17.27$ pp pada Huatuo, $p < 0.0001$).
2. Model spesialis medis mengalami ketidakstabilan prediksi yang jauh lebih parah (*flip rate* $50.81\%$ vs $28.79\%$, $OR_{flip} = 2.54$) dan mengidap patologi *negative visual reliance* yang ekstrem ($80.0\%$ prediksi memiliki keyakinan lebih tinggi saat citra dihilangkan), membuktikan bahwa adaptasi medis unimodal bahasa Inggris memperburuk kerentanan terhadap transfer bahasa non-dominan.

### Kontribusi 2: Pengembangan Framework Selective Prediction Waktu-Inferensi Multi-Sinyal Tanpa Pelatihan Ulang Model
Merumuskan dan memvalidasi sistem klasifikasi selektif waktu-inferensi (*zero-retraining runtime selective prediction*) yang memadukan skor keyakinan internal ($M_1$), sensitivitas penghapusan visual kontrafaktual ($\Delta P_{dir}$), dan indikator ketidaksepakatan terjemahan bilingual ($D_{lang}$).
Melalui pemodelan regresi logistik multivariat teratur yang dievaluasi pada kurva Risk–Coverage berbasis klaster citra fisik radiografi independen ($K=204$ klaster, $N=799$ sampel uji), framework yang diusulkan ($LR_{full}$) terbukti secara statistik mereduksi Area Under the Risk–Coverage Curve (AURC) diskrit secara signifikan pada kedua model:
- **Qwen2.5-VL-7B:** $\Delta\text{AURC} = -0.0369$ ($95\%$ Cluster Bootstrap CI $[-0.0589, -0.0159]$, Holm $p_{adj} = 0.0310$), menurunkan selective risk sebesar $-5.16$ pp pada target coverage $80\%$.
- **HuatuoGPT-Vision-7B:** $\Delta\text{AURC} = -0.0705$ ($95\%$ Cluster Bootstrap CI $[-0.0984, -0.0415]$, Holm $p_{adj} = 0.0010$), menurunkan selective risk sebesar $-4.69$ pp pada target coverage $80\%$.

### Kontribusi 3: Demarkasi Metodologis antara Kegagalan Heuristik Penggabungan Sederhana dan Kebutuhan Pembobotan Multivariat Terpelajar
Membuktikan secara teoretis dan empiris bahwa aturan kombinasi heuristik berbasis normalisasi min-max aritmatika seragam ($M_{norm} = \tilde{M}_1 + \Delta \tilde{P}_{dir}$) **gagal mencapai signifikansi statistik** ($p = 0.2298$ pada Qwen, $p = 0.3846$ pada Huatuo) akibat asimetri variansi dan karakteristik distribusi antar-modalitas.
Sebaliknya, penelitian ini membuktikan bahwa pemodelan multivariat terpelajar dengan kalibrasi *cluster-robust* mampu mengekstraksi sinyal ortogonal secara stabil (bebas multikolinearitas, $VIF < 1.55$; rasio kejadian per variabel $EPV > 30$; stabil terhadap variasi regularisasi $C \in [0.1, 100.0]$), menegaskan bahwa integrasi ketidaksepakatan lintas-bahasa ($D_{lang}$, $OR > 2.7$) adalah komponen esensial untuk mengoptimalkan penolakan risiko kesalahan pada model visi-bahasa bilingual.

---

## Pernyataan Integritas Ilmiah (Non-Overclaiming Demarcation)
Penulis secara eksplisit **TIDAK mengklaim**:
1. Menemukan formulasi dasar selective classification (diakui sebagai fondasi Geifman & El-Yaniv, 2017).
2. Menemukan metode kanvas hitam kontrafaktual (diakui sebagai adaptasi dari Zafar et al., 2026 dan Sadanandan & Behzadan, 2026).
3. Bahwa sensitivitas kanvas hitam membuktikan pemahaman klinis atau lokalisasi patologi anatomis kausal.
4. Bahwa temuan ini otomatis berlaku untuk tugas pembuatan laporan radiologi teks bebas (*free-text reporting*).
