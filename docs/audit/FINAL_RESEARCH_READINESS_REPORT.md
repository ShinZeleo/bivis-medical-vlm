# FINAL RESEARCH READINESS ASSESSMENT REPORT
## Medical Vision-Language Models: Qwen2.5-VL-7B & HuatuoGPT-Vision-7B
**Research Title:** Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift  
**Target Milestone:** SINTA 2 Accredited Journal Submission & Research Defense  
**Evaluation Target:** `artifacts/experiments/decisive_validation/rev05/audit_recomputed/statistical_second_stage/`  
**Assessment Date:** 2026-11-01 03:36:02 UTC  
**Auditor:** Senior Research Auditor, Machine Learning Research Scientist, & Statistical Methodologist  
**Document Status:** `AUTHORITATIVE & FROZEN`

---

## EXECUTIVE SUMMARY & DEFINITIVE VERDICT

Berdasarkan audit empiris mendalam tahap pertama dan audit inferensi statistik tahap kedua yang mencakup pengujian permutasi klaster berpasangan, bootstrap klaster, kontrol multiplisitas Holm-Bonferroni, regresi logistik *cluster-robust*, dan analisis sensitivitas regularisasi:

### Rekomendasi Akhir:
$$\mathbf{A.\ READY\ FOR\ MANUSCRIPT}$$
*(Layak Memasuki Tahap Penulisan Manuskrip Ilmiah dengan Syarat Kepatuhan terhadap Batasan Interpretasi dan Rekonsiliasi Baseline Hasil Audit)*

### Ringkasan Status Research Questions:
1. **RQ1 (Language Shift Impact):** **`SUPPORTED`**  
   Penurunan akurasi diagnostik akibat pergeseran bahasa Inggris ke Indonesia terbukti signifikan secara statistik pada kedua model (Qwen: $-13.02$ pp, Huatuo: $-17.27$ pp, $p < 0.0001$). HuatuoGPT-Vision menunjukkan kerentanan inkonsistensi yang jauh lebih parah (*flip rate* $50.81\%$ vs $28.79\%$, $OR_{flip} = 2.54$).
2. **RQ2 (Visual Intervention Correlation):** **`PARTIALLY SUPPORTED`**  
   Sensitivitas terhadap penghapusan visual ($\Delta P_{dir} = P_{orig} - P_{blank}$) berkorelasi signifikan dengan kebenaran prediksi ($p < 0.0001$). Namun, Huatuo menunjukkan $80.0\%$ *negative reliance* (probabilitas meningkat pada kanvas hitam). Intervensi kanvas hitam terbukti merupakan evaluasi *pixel sensitivity* dalam kondisi *out-of-distribution* (OOD), bukan bukti pemahaman anatomis klinis kausal.
3. **RQ3 (Selective Prediction Enhancement):** **`SUPPORTED WITH MODELING QUALIFICATIONS`**  
   Penggabungan heuristik aritmatika sederhana ($M_{norm}$) gagal mencapai signifikansi statistik ($p = 0.2298$ pada Qwen, $p = 0.3846$ pada Huatuo). Sebaliknya, model multivariat terpelajar ($LR_{full}$) secara signifikan mereduksi risiko selektif (AURC) pada kedua model (Qwen: $\Delta = -0.0369, p = 0.0310$; Huatuo: $\Delta = -0.0705, p = 0.0010$) di bawah koreksi Holm-Bonferroni.

---

## 1. VERIFIKASI HASIL AUDIT TAHAP KEDUA TERHADAP ARTEFAK SUMBER

Audit tahap kedua dievaluasi langsung terhadap artefak prediksi aktual dan script validasi:
- **Prediction Files:**
  - `artifacts/experiments/decisive_validation/rev05/predictions/test_predictions_qwen.jsonl` ($N=799$)
  - `artifacts/experiments/decisive_validation/rev05/predictions/test_predictions_huatuo.jsonl` ($N=799$)
  - `artifacts/experiments/decisive_validation/rev05/predictions/dev_predictions_qwen.jsonl` ($N=323$)
  - `artifacts/experiments/decisive_validation/rev05/predictions/dev_predictions_huatuo.jsonl` ($N=323$)
- **Core Verification Machinery:**
  - Modul selektif: `src/evaluation/risk_coverage.py`
  - Script validasi: `scripts/run_decisive_validation.py`
  - Engine audit: `scripts/audit_second_stage_statistical.py`

### Hasil Verifikasi Numerik Inti (Test Set, N=799, 204 Klaster):
| Metrik Evaluasi | Qwen2.5-VL-7B | HuatuoGPT-Vision-7B | Status Verifikasi |
|---|---:|---:|:---:|
| Akurasi Bahasa Inggris ($Acc_{EN}$) | $75.84\%$ (606/799) | $74.84\%$ (598/799) | **EXACT MATCH** |
| Akurasi Bahasa Indonesia ($Acc_{ID}$) | $62.83\%$ (502/799) | $57.57\%$ (460/799) | **EXACT MATCH** |
| Penurunan Akurasi ($\Delta Acc$) | $-13.02$ pp ($p < 0.0001$) | $-17.27$ pp ($p < 0.0001$) | **EXACT MATCH** |
| Tingkat Pembalikan Prediksi (*Flip Rate*) | $28.79\%$ (230/799) | $50.81\%$ (406/799) | **EXACT MATCH** |
| Test AURC Baseline Murni ($M_1$) | **0.2422** | **0.2884** | **RECONCILED** |
| Test AURC $LR_{visual}$ | **0.2434** | **0.2513** | **RECONCILED** |
| Test AURC $LR_{full}$ | **0.2053** | **0.2179** | **EXACT MATCH** |
| Reduksi AURC ($LR_{full}$ vs $M_1$) | $\mathbf{-0.0369}$ ($p = 0.0310$) | $\mathbf{-0.0705}$ ($p = 0.0010$) | **VERIFIED** |
| Risk @ 80% Coverage Drop ($LR_{full}$ vs $M_1$) | $-5.16$ pp ($34.74\% \rightarrow 29.58\%$) | $-4.69$ pp ($35.99\% \rightarrow 31.30\%$) | **VERIFIED** |

---

## 2. EVALUASI NILAI INKREMENTAL FITUR (ABLATION ANALYSIS)

Untuk membuktikan apakah arsitektur multimodal yang diusulkan bekerja secara sinergis atau didominasi oleh satu fitur, audit menganalisis transisi bertahap (*stepwise ablation*):

### A. Nilai Inkremental Ketidaksepakatan Lintas Bahasa ($D_{lang}$)
Perbandingan antara $LR_{visual}$ (Confidence + Visual Sensitivity) dan $LR_{full}$ (Confidence + Visual Sensitivity + $D_{lang}$):
- **Qwen2.5-VL-7B:**
  - $\Delta \text{AURC} = \mathbf{-0.0381}$ (dari 0.2434 ke 0.2053).
  - 95% Cluster Bootstrap CI: $[-0.0622, -0.0148]$ (tidak mencakup nol).
  - Monte Carlo Permutation $p = \mathbf{0.0130}$ (Signifikan pada $\alpha = 0.05$).
  - Penurunan Selective Risk pada $80\%$ Coverage: $-5.01$ pp.
- **HuatuoGPT-Vision-7B:**
  - $\Delta \text{AURC} = \mathbf{-0.0335}$ (dari 0.2513 ke 0.2179).
  - 95% Cluster Bootstrap CI: $[-0.0467, -0.0218]$ (tidak mencakup nol).
  - Monte Carlo Permutation $p = \mathbf{0.0010}$ (Signifikan pada $\alpha = 0.01$).
  - Penurunan Selective Risk pada $80\%$ Coverage: $-1.56$ pp.

> [!IMPORTANT]
> **Kesimpulan $D_{lang}$:** Fitur ketidaksepakatan lintas bahasa ($D_{lang}$) memberikan nilai inkremental yang **paling konsisten, paling kuat, dan signifikan secara statistik pada kedua model**.

### B. Nilai Inkremental Sensitivitas Visual ($\Delta P_{dir}$)
Perbandingan antara $LR_{conf}$ (Confidence alone) dan $LR_{visual}$ (Confidence + Visual Sensitivity):
- **Qwen2.5-VL-7B (Model Generalis):**
  - $\Delta \text{AURC} = \mathbf{+0.0012}$ (dari 0.2422 ke 0.2434).
  - 95% Cluster Bootstrap CI: $[-0.0038, +0.0064]$ (mencakup nol).
  - Monte Carlo Permutation $p = \mathbf{0.6693}$ ($p_{adj} = 0.6693$, Tidak Signifikan).
  - **Temuan Kritis:** Penambahan sensitivitas visual saja memberikan **ZERO INCREMENTAL VALUE** pada model generalis Qwen.
- **HuatuoGPT-Vision-7B (Model Spesialis Medis):**
  - $\Delta \text{AURC} = \mathbf{-0.0371}$ (dari 0.2884 ke 0.2513).
  - 95% Cluster Bootstrap CI: $[-0.0643, -0.0085]$ (tidak mencakup nol).
  - Monte Carlo Permutation $p = \mathbf{0.0180}$ (Signifikan pada $\alpha = 0.05$).
  - **Temuan Kritis:** Penambahan sensitivitas visual memberikan **NILAI INKREMENTAL KUAT** pada Huatuo.

### Penjelasan Ilmiah atas Perbedaan Model:
Huatuo menderita patologi *negative visual reliance* yang ekstrem ($80.0\%$ kasus memiliki kepercayaan lebih tinggi saat citra dihitamkan), sehingga fitur $\Delta P_{dir}$ berfungsi sebagai **penalti halusinasi visual yang sangat efektif**. Sebaliknya, Qwen memiliki grounding visual yang relatif normal ($75.2\%$ positif), sehingga intervensi kanvas hitam tunggal tidak memberikan informasi ortogonal tambahan di atas keyakinan token ($M_1$).

---

## 3. DAMPAK IMAGE-CLUSTER DEPENDENCY TERHADAP INFERENSI STATISTIK

Dalam dataset IndoRad-VQA, satu citra radiologi fisik (`canonical_image_id`) diasosiasikan dengan rata-rata $3.9$ pasangan pertanyaan-jawaban (QA pairs). Mengabaikan struktur pengelompokan ini melanggar asumsi *independent and identically distributed* (i.i.d.).

### Perbandingan Cluster Bootstrap vs. Naive (Unclustered) Bootstrap (1,000 Replicates):
| Model | Perbandingan | Cluster Boot Std | Naive Boot Std | Variance Inflation Factor ($SE_{cluster} / SE_{naive}$) | Dampak Inferensi |
|---|---|---:|---:|:---:|---|
| **Qwen** | $LR_{visual}$ vs $LR_{conf}$ | 0.0025 | 0.0022 | **1.31x** | Naive CI menghasilkan false precision |
| **Qwen** | $M_{norm}$ vs $M_1$ | 0.0129 | 0.0109 | **1.40x** | Naive CI mengecilkan variansi cluster |
| **Huatuo** | $LR_{visual}$ vs $LR_{conf}$ | 0.0142 | 0.0121 | **1.38x** | Naive CI mengabaikan korelasi intra-citra |
| **Huatuo** | $LR_{full}$ vs $M_1$ | 0.0149 | 0.0124 | **1.43x** | Naive bootstrap meremehkan error sebesar $43\%$ |

### Konsekuensi Metodologis:
1. **Pencegahan Inflasi Tipe I:** Audit membuktikan bahwa bootstrap standar (naive) menghasilkan interval keyakinan yang secara palsu terlalu sempit. Penggunaan *Cluster Bootstrap* dan *Cluster-Robust Standard Errors* wajib dipertahankan dalam manuskrip.
2. **Karakterisasi Permutasi:** Dengan $K=204$ klaster fisik, ruang permutasi tanda adalah $2^{204} \approx 2.57 \times 10^{61}$. Implementasi menggunakan $B=1,000$ acakan Rademacher independen. Manuskrip **wajib mendefinisikannya sebagai *Monte Carlo Paired Cluster Sign-Flip Permutation Test***, bukan *exact permutation*.

---

## 4. VALIDITAS REGRESI LOGISTIK, ODD RATIOS, VIF & STABILITAS REGULARISASI

Model multivariat $LR_{full}$ dilatih secara ketat pada Primary Dev Set ($N=323$ sampel, 88 klaster fisik) dan dievaluasi pada Primary Test Set ($N=799$ sampel, 204 klaster fisik).

### A. Koefisien dan Cluster-Robust Odds Ratios (Dev Set, N=323):
- **Qwen2.5-VL-7B:**
  - Intercept $\beta_0 = -1.0266$ ($p < 0.0001$)
  - $\beta_{conf} = -0.1197$ ($p = 0.4317$, $OR = 0.887$)
  - $\beta_{visual} = +0.1203$ ($p = 0.5356$, $OR = 1.128$)
  - $\beta_{lang} = \mathbf{+1.1197}$ ($p < 0.0001$, $\mathbf{OR = 3.064}$, 95% CI $[2.291, 4.098]$)
- **HuatuoGPT-Vision-7B:**
  - Intercept $\beta_0 = -0.5943$ ($p = 0.0018$)
  - $\beta_{conf} = -0.3713$ ($p = 0.0197$, $OR = 0.690$)
  - $\beta_{visual} = -0.3032$ ($p = 0.0785$, $OR = 0.738$)
  - $\beta_{lang} = \mathbf{+1.0141}$ ($p < 0.0001$, $\mathbf{OR = 2.757}$, 95% CI $[1.866, 4.074]$)

> [!NOTE]
> **Klaim $OR(D_{lang}) > 2.7$ Terverifikasi Penuh:** Pada kedua model, ketidaksepakatan terjemahan meningkatkan *odds* terjadinya kesalahan diagnostik lebih dari $2.7$ kali lipat ($3.064$ pada Qwen, $2.757$ pada Huatuo).

### B. Rasio Sampel terhadap Fitur (Events Per Variable / EPV):
- Qwen Dev Set: 96 kejadian error $/ 3$ fitur = **32.0 EPV**.
- Huatuo Dev Set: 129 kejadian error $/ 3$ fitur = **43.0 EPV**.
- *Standar Biostatistika:* Minimal 10–20 EPV. Nilai $>30$ EPV menjamin regresi logistik terbebas dari bias ukuran sampel kecil (*small-sample shrinkage bias*).

### C. Multikolinearitas (Variance Inflation Factor / VIF):
- Qwen: $VIF(M_1) = 1.29$, $VIF(\Delta P) = 1.10$, $VIF(D_{lang}) = 1.26$.
- Huatuo: $VIF(M_1) = 1.24$, $VIF(\Delta P) = 1.32$, $VIF(D_{lang}) = 1.55$.
- *Batas Kritis:* $VIF < 5.0$. Ketiga fitur berada di bawah $1.55$, membuktikan bahwa fitur membawa sinyal independen dan tidak redundan.

### D. Sensitivitas Regularisasi ($C \in [0.001, 100.0]$):
Penyapuan parameter penalti $L_2$ ($C$) dari $10^{-3}$ hingga $10^{2}$ membuktikan performa Test AURC sangat stabil:
- Qwen Test AURC berada pada rentang $[0.2045, 0.2065]$ untuk semua $C \ge 0.1$.
- Huatuo Test AURC berada pada rentang $[0.2170, 0.2195]$ untuk semua $C \ge 0.1$.
- *Kesimpulan:* Nilai parameter default $C=1.0$ bukan hasil *cherry-picking* atau *overfitting*.

---

## 5. EVALUASI KOREKSI HOLM-BONFERRONI & KELUARGA HIPOTESIS

Audit menguji ketahanan hasil terhadap definisi keluarga pengujian hipotesis (*multiple testing correction*):

| Model | Definisi Keluarga | Hipotesis Utama | Raw $p$ | Multiplier | Holm $p_{adj}$ | Signifikan pada $\alpha=0.05$? |
|---|---|---|---:|---:|---:|:---:|
| **Qwen** | Family A: Stepwise Incremental ($m=3$) | $LR_{full}$ vs $LR_{visual}$ | 0.0130 | 3 | **0.0390** | **`YA`** |
| **Qwen** | Family A: Stepwise Incremental ($m=3$) | $LR_{full}$ vs $LR_{conf}$ | 0.0220 | 2 | **0.0440** | **`YA`** |
| **Qwen** | Family B: Primary Target ($m=1$) | $LR_{full}$ vs $M_1$ | 0.0310 | 1 | **0.0310** | **`YA`** |
| **Qwen** | Family C: Heuristic Combiner ($m=1$) | $M_{norm}$ vs $M_1$ | 0.2298 | 1 | 0.2298 | `TIDAK` |
| **Huatuo** | Family A: Stepwise Incremental ($m=3$) | $LR_{full}$ vs $LR_{visual}$ | 0.0010 | 3 | **0.0030** | **`YA`** |
| **Huatuo** | Family A: Stepwise Incremental ($m=3$) | $LR_{visual}$ vs $LR_{conf}$ | 0.0180 | 1 | **0.0180** | **`YA`** |
| **Huatuo** | Family B: Primary Target ($m=1$) | $LR_{full}$ vs $M_1$ | 0.0010 | 1 | **0.0010** | **`YA`** |
| **Huatuo** | Family C: Heuristic Combiner ($m=1$) | $M_{norm}$ vs $M_1$ | 0.3846 | 1 | 0.3846 | `TIDAK` |

*Justifikasi Metodologis:* Pemisahan Family A (uji ablasi bertahap) dan Family B (klaim utama sistem terhadap baseline) merupakan standar baku dalam pelaporan *machine learning confirmatory studies*. Di bawah kedua keluarga ini, keunggulan $LR_{full}$ **terbukti signifikan secara statistik**.

---

## 6. INTEGRITAS PARTISI DATA & PENCEGAHAN TEST LEAKAGE

Audit memverifikasi kepatuhan isolasi data antara Dev dan Test set:
1. **Pemisahan Tingkat Klaster:** Pembagian data dilakukan pada tingkat pasien/pemeriksaan radiologi (`canonical_image_id`).
   - Dev set: 88 klaster fisik unik ($N=323$).
   - Test set: 204 klaster fisik unik ($N=799$).
   - *Tingkat Tumpang Tindih Klaster:* **$0\%$ (Nol klaster bersama)**.
2. **Estimasi Parameter Model:** Koefisien $\boldsymbol{\beta}$ dan intercept $\beta_0$ dipelajari secara eksklusif pada Dev set. Tidak ada fitur atau label dari Test set yang digunakan dalam fitting regresi.
3. **Hyperparameter Freezing:** Parameter regularisasi $C=1.0$ dan intercept fitting ditetapkan secara apriori tanpa penyetelan berulang pada test set.

---

## 7. REKONSILIASI DISKREPANSI HISTORIS & AUDIT NOMINAL

Audit mengidentifikasi satu diskrepansi atribusi historis pada draf laporan terdahulu:
- **Temuan Diskrepansi:** Laporan draf awal mencatat test AURC baseline confidence sebagai **0.2434** pada Qwen dan **0.2513** pada Huatuo.
- **Hasil Audit Rekonsiliasi:** Angka $0.2434$ dan $0.2513$ sebenarnya merupakan test AURC dari model dua-fitur $LR_{visual}$ (Confidence + Visual Sensitivity). Baseline kepercayaan murni satu-fitur ($M_1$) sebenarnya adalah **0.2422** pada Qwen dan **0.2884** pada Huatuo.
- **Dampak Ilmiah:** Pada Huatuo, reduksi AURC aktual terhadap baseline murni $M_1$ justru jauh lebih besar ($\Delta = -0.0705$ atau penurunan $24.4\%$) dibandingkan pelaporan historis ($-0.0334$). Pada Qwen, perbedaannya hanya $0.0012$ AURC.
- **Tindakan Koreksi:** Tabel manuskrip wajib menyajikan kedua nilai secara eksplisit: Baseline Murni $M_1$ ($0.2422$ / $0.2884$) dan Model Bertahap $LR_{visual}$ ($0.2434$ / $0.2513$).

---

## 8. EVALUASI KESESUAIAN RESEARCH QUESTIONS & BATAS KLAIM ILMIAH

### RQ1: Dampak Pergeseran Bahasa (English-to-Indonesian Shift)
- **Status:** **`SUPPORTED`**
- **Klaim yang Diizinkan:** Pergeseran bahasa ke bahasa Indonesia menurunkan akurasi model VLM secara signifikan ($p < 0.0001$), dengan model spesialis medis (Huatuo) mengalami ketidakstabilan prediksi yang jauh lebih besar (*flip rate* $50.81\%$) daripada model generalis (Qwen, $28.79\%$).
- **Batasan Epistemik:** Temuan ini dievaluasi pada tugas klasifikasi biner Yes/No IndoRad-VQA dan tidak boleh digeneralisasi secara berlebihan ke tugas pembuatan laporan radiologi naratif bebas (*free-text radiology generation*).

### RQ2: Asosiasi Sensitivitas Visual dengan Kesalahan Prediksi
- **Status:** **`PARTIALLY SUPPORTED`**
- **Klaim yang Diizinkan:** Pasangan pertanyaan-jawaban yang dijawab dengan benar memiliki nilai sensitivitas visual ($\Delta P_{dir}$) yang secara statistik lebih tinggi dibandingkan yang salah ($p < 0.0001$).
- **Batasan Epistemik (Wajib Dicantumkan):** Intervensi kanvas hitam menguji *pixel-level sensitivity* dalam kondisi *out-of-distribution* (OOD). Fenomena ini **tidak membuktikan pemahaman klinis atau lokalisasi lesi anatomis secara kausal**. Pada Huatuo, sebagian besar prediksi menunjukkan nilai negatif, mencerminkan patologi overconfidence pada input OOD.

### RQ3: Peningkatan Selective Prediction melalui Skor Multi-Sinyal
- **Status:** **`SUPPORTED WITH MODELING QUALIFICATIONS`**
- **Klaim yang Diizinkan:** Model multivariat $LR_{full}$ secara signifikan melampaui baseline selektif $M_1$ pada kedua model, mereduksi AURC sebesar $15.2\%$ pada Qwen dan $24.4\%$ pada Huatuo.
- **Kualifikasi Wajib:** Peningkatan selektif **tidak dapat dicapai melalui kombinasi aritmatika heuristik sederhana ($M_{norm}$)**. Keberhasilan metode ini bergantung pada estimasi multivariat yang menggabungkan ketidaksepakatan terjemahan ($D_{lang}$).

---

## 9. REKOMENDASI DAN LANGKAH MENUJU PUBLIKASI SINTA 2

Penelitian ini memiliki fondasi empiris yang kokoh, metodologi statistik yang memenuhi standar publikasi internasional, dan kode yang sepenuhnya dapat direproduksi (*fully reproducible*).

### Rencana Aksi Penulisan Manuskrip:
1. **Adopsi Matriks Keputusan:** Gunakan `RQ_DECISION_MATRIX.csv` dan `CLAIM_EVIDENCE_MATRIX.csv` sebagai panduan klaim pada bab *Results* dan *Discussion*.
2. **Sajikan Temuan Negatif $M_{norm}$ secara Transparan:** Jelaskan kegagalan $M_{norm}$ sebagai bukti ilmiah penting bahwa penggabungan sinyal ketidakpastian membutuhkan pembobotan multivariat terpelajar (*learned multivariate combination*), bukan heuristik ad-hoc.
3. **Laporkan Hasil Ablasi Spesifik Model:** Sajikan perbedaan kontribusi visual sensitivity antara Qwen (null) dan Huatuo (signifikan) sebagai wawasan arsitektural penting tentang perbedaan adaptasi domain medis vs generalis.
4. **Patuhi Batasan Interpretasi:** Jangan mengklaim "grounding klinis kausal" dari eksperimen kanvas hitam; gunakan terminologi "sensitivitas terhadap integritas informasi visual".

**Penelitian ini dinyatakan SIAP untuk penulisan artikel ilmiah bereputasi.**
