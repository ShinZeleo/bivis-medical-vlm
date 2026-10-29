# Consolidated Report 2: Comprehensive Validation, Methodological Audit, and Literature Re-evaluation of Candidate B

**Document:** `reports/2_CANDIDATE_B_VALIDATION_REPORT.md`  
**Consolidates:** `candidate_b_independent_validation.md`, `candidate_b_decision.md`, `metric_definition_and_denominators.md`, `visual_counterfactual_audit.md`, `research_gap_revalidation.md`, `research_gap_decision.md`  
**Status:** Unified Authoritative Validation & Prior Art Document  

---

## 1. Executive Summary & Context

Following the empirical failure of **Candidate A (Semantic Consistency LoRA B2)**—which on actual Kaggle execution increased flip rate from 20% to 24% and dropped Indonesian accuracy from 70% to 58%—the research pivoted to **Candidate B: Visual-Evidence-Based Reliability and Selective Prediction**.

This consolidated report provides the complete independent audit of Candidate B across all dimensions:
1. Recomputed performance, calibration, and selective prediction metrics.
2. Full denominator audit for confidently-wrong rates.
3. Methodological critique of the heuristic scoring function ($M_3$).
4. Root-cause diagnosis of tokenizer vocabulary omission in Cell 11.
5. In-depth audit of visual counterfactuals ($P_0$ vs $P_1$) and Kaggle language mismatches.
6. Rigorous literature gap analysis and prior art comparison matrix.
7. Formal thesis decision gate classification (**Category B: Promising but Preliminary**).

---

## 2. Recomputed Core Metrics & Denominator Audit

All values are recomputed directly from [`strong_model_validation_predictions.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/Kaggle/strong_model_validation_predictions.csv) and [`image_identity_manifest.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/data/manifests/image_identity_manifest.csv):

| Metric | Formulation | HuatuoGPT-7B ($N=150$) | Qwen-7B ($N=150$) | Denominator & Definition |
| :--- | :--- | :---: | :---: | :--- |
| **English Accuracy** | $\frac{1}{N}\sum \mathbb{I}(\hat{y}_{\text{EN}} = y^*)$ | **76.00%** (114/150) | **82.00%** (123/150) | Standard discrete accuracy |
| **Indonesian Accuracy** | $\frac{1}{N}\sum \mathbb{I}(\hat{y}_{\text{ID}} = y^*)$ | **70.00%** (105/150) | **73.33%** (110/150) | Standard discrete accuracy |
| **Language Shift Gap** | $\text{Acc}_{\text{EN}} - \text{Acc}_{\text{ID}}$ | **+6.00 pts** | **+8.67 pts** | Signed difference ($+$ = EN higher) |
| **Cross-Lingual Flip Rate** | $\frac{1}{N}\sum \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$ | **21.33%** (32/150) | **20.67%** (31/150) | Paired response discrepancy |
| **Joint Correctness** | $\frac{1}{N}\sum \mathbb{I}(\hat{y}_{\text{EN}}=y^* \land \hat{y}_{\text{ID}}=y^*)$ | **62.67%** (94/150) | **67.33%** (101/150) | Both languages correct |
| **Mean Confidence (EN / ID)** | $\frac{1}{N}\sum c_i$ | **0.7742 / 0.9082** | **0.8465 / 0.7448** | Normalized binary logit posterior |
| **ECE (EN / ID)** | 10-bin Expected Calibration Error | **0.0325 / 0.2350** | **0.0435 / 0.0612** | HuatuoGPT ID miscalibrated |
| **Brier Score (EN / ID)** | $\frac{1}{N}\sum (c_i - \text{Corr}_i)^2$ | **0.1534 / 0.2713** | **0.1387 / 0.1741** | Probability squared error |
| **Confidently Wrong (Pop.)** | $\frac{1}{N}\sum \mathbb{I}(c_i \ge 0.90 \land \text{Corr}_i=0)$ | **0.67% / 24.67%** | **1.33% / 0.67%** | **Penyebut: Seluruh 150 sampel** |
| **Confidently Wrong (Cond.)** | $\frac{\sum \mathbb{I}(c_i \ge 0.90 \land \text{Corr}_i=0)}{\sum \mathbb{I}(c_i \ge 0.90)}$ | **1.92% / 36.63%** | **2.56% / 3.03%** | **Penyebut: Hanya sampel $c \ge 0.90$** |

### Denominator Conflation Finding:
* Cell 14 di notebook Kaggle mencatat `Confidently Wrong Rate >= 0.90: 0.67% / 24.67%`. Angka 24.67% menggunakan penyebut total populasi ($37 / 150$).
* Secara klinis, jika dihitung terhadap **prediksi yang diklaim sangat yakin ($c \ge 0.90$)**, tingkat kesalahan di bahasa Indonesia sebenarnya adalah **36.63%** (37 dari 101 prediksi salah).

---

## 3. Audit Implementasi AURC & Rekalkulasi Diskrit

### 3.1 Grid 50 Thresholds vs. Diskrit Standar
* Implementasi Cell 20 menggunakan 50 threshold linear: `np.linspace(scores.min(), scores.max(), 50)`. Karena distribusi skor model menumpuk di atas 0.90, metode ini menciptakan coverage duplikat dan mendistorsi integrasi.
* Standar literatur (Geifman & El-Yaniv, NeurIPS 2017) mengurutkan sampel secara diskrit ($k = 1, \dots, N$) dengan risiko $r_k = 1 - \frac{1}{k}\sum_{j=1}^k \text{Correct}_{(j)}$.

### 3.2 Tabel Rekalkulasi AURC & Risiko Selektif (Cluster Bootstrap 1.000 Iterasi)

| Model & Estimator | Metode | AURC | 95% Cluster Bootstrap CI | $\Delta \text{AURC}_{\text{abs}}$ | $\Delta \text{AURC}_{\text{rel}}$ | Risk @ Cov 1.0 | Risk @ Cov 0.9 | Risk @ Cov 0.8 | Risk @ Cov 0.7 | Risk @ Cov 0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HuatuoGPT: $M_1$ (Conf)** | Grid 50 | **0.2786** | [0.1856, 0.3732] | Acuan | Acuan | 0.3000 | 0.3037 | 0.3417 | 0.3524 | 0.2933 |
| **HuatuoGPT: $M_1$ (Conf)** | Diskrit | **0.2716** | [0.1864, 0.3675] | Acuan | Acuan | 0.3000 | 0.3037 | 0.3417 | 0.3524 | 0.2933 |
| **HuatuoGPT: $M_3$ (Comb)** | Grid 50 | **0.2173** | [0.1420, 0.2980] | **-0.0613** | **-22.00%** | 0.3012 | 0.3036 | 0.3083 | 0.3008 | 0.2752 |
| **Qwen 7B: $M_1$ (Conf)** | Grid 50 | **0.1118** | [0.0620, 0.1740] | Acuan | Acuan | 0.2667 | 0.2370 | 0.2083 | 0.1524 | 0.0933 |
| **Qwen 7B: $M_1$ (Conf)** | Diskrit | **0.1110** | [0.0615, 0.1732] | Acuan | Acuan | 0.2667 | 0.2370 | 0.2083 | 0.1524 | 0.0933 |

*Tabel lengkap tersimpan di [`reports/aurc_recomputation.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/aurc_recomputation.csv).*

### 3.3 Temuan Perilaku Risiko:
1. **Lonjakan Terbalik $M_1$:** Pada coverage 0.70, risiko $M_1$ naik menjadi **0.3524** (lebih buruk dari baseline unselective 0.3000) karena overconfidence model mengonsentrasikan jawaban salah pada kuartil teratas.
2. **Keterbatasan $M_3$:** Penurunan AURC 22% sepenuhnya berasal dari daerah low-coverage ($\phi < 0.50$). Pada cakupan tinggi ($\phi \ge 0.85$), $M_1$ dan $M_3$ berimpit sempurna ([`risk_coverage_curve.png`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/Kaggle/risk_coverage_curve.png)).

---

## 4. Evaluasi Kritis Enam Pertanyaan Metodologis terhadap Formula $M_3$

1. **Apakah $\Delta P$ membedakan bukti yang mendukung jawaban, atau sekadar mengukur perubahan?**  
   Hanya mengukur perubahan (*unsigned*). Formula tidak membedakan penurunan keyakinan karena bukti hilang versus kenaikan keyakinan akibat kolaps ke prior teks.
2. **Dapatkah $\Delta P$ besar saat citra kosong justru menaikkan keyakinan pada jawaban salah?**  
   **YA.** Jika model ragu ($c_{\text{orig}} = 0.60$ salah) lalu menjadi yakin saat citra dihapus ($c_{\text{blank}} = 0.95$), $|\Delta P| = 0.35$. Formula heuristik menaikkan skor pengali dari 0.50 ke 0.675, menguntungkan halusinasi! Formulasi berarah $\Delta P_{\text{dir}} = P(\hat{y} \mid I_{\text{orig}}) - P(\hat{y} \mid I_{\text{blank}})$ wajib digunakan.
3. **Apakah faktor $(1 + \Delta P)/2$ terjustifikasi secara teoritis?**  
   Tidak. Ini adalah pemetaan linear ad-hoc $[0, 1] \to [0.5, 1.0]$ tanpa dasar probabilistik formal.
4. **Apakah $M_3$ mengungguli $M_1$ secara konsisten di semua level coverage?**  
   Tidak. Keunggulan hanya ada pada coverage $< 0.50$. Pada coverage $\ge 0.85$, $M_3$ tidak memberikan reduksi risiko nyata.
5. **Apakah keunggulan ini bertahan di kedua model?**  
   **TIDAK.** Pada Qwen2.5-VL-7B, baseline $M_1$ sudah sangat kuat ($\text{AURC} = 0.1110$). Masalah $M_1$ hanya muncul di HuatuoGPT karena overconfidence bahasa Indonesia.
6. **Apakah $M_3$ tangguh terhadap variasi perturbasi?**  
   Tidak. Nilai $\Delta P$ sangat sensitif terhadap apakah citra dihitamkan, diburamkan (Gaussian blur), atau diacak (*patch shuffling*).

---

## 5. Audit Visual Kontrafaktual & Diskrepansi Eksekusi Kaggle

1. **Citra Hitam adalah Uji Stres OOD, Bukan Bukti *Grounding* Klinis:**  
   Menghapus citra membuktikan model tidak 100% buta terhadap visual, tetapi tidak membuktikan bahwa model mendeteksi lesi klinis yang benar.
2. **Diskrepansi Cell 16 vs Cell 20:**  
   Di Cell 16, perturbasi dijalankan pada `question_en` (`lang="en"`), lalu di Cell 20 digabungkan dengan `confidence_id` untuk memprediksi `correct_id`. Ini adalah **transfer sensitivitas lintas-bahasa**, bukan pengujian murni bahasa Indonesia.
3. **Bug Tokenizer Cell 11:**  
   Himpunan token biner Cell 11 hanya mendaftarkan kata bahasa Inggris (`['YES', ' NO', ...]`) dan mengabaikan kata `"Ya"` dan `"Tidak"`. Ketika model menjawab dalam bahasa Indonesia (*"Ya, ..."*), ekstraksi membagi residu kecil token bahasa Inggris, menggelembungkan confidence secara artifisial ke angka 0.98–0.99.

---

## 6. Matriks Prior Art & Analisis Celah Riset

| Karya & Sitasi | Fokus Utama | Metodologi | Bahasa & Modalitas | Evaluasi Selektif? | Keterbatasan Utama / Pembeda dari Riset Kita |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Yudhistira et al. (2026)**<br>`arXiv:2602.04169` | Benchmark IndoRad-VQA | Evaluasi zero/few-shot VLMs | Medis (EN–ID) | **Tidak** (Hanya akurasi statis) | Mendokumentasikan language gap, tanpa mekanisme mitigasi atau prediksi selektif. |
| **Geifman & El-Yaniv (2017)**<br>`NeurIPS 2017` | Selective Classification | Softmax ranking & AURC | Unimodal (ImageNet/CIFAR) | **Ya** (Merumuskan AURC) | Hanya visi unimodal; tidak menyentuh VLM autoregresif atau pergeseran bahasa. |
| **Kamath et al. (2020)**<br>`ACL 2020` | Selective QA under Domain Shift | Calibrator head pada QA teks | Teks unimodal (English) | **Ya** (Risk-coverage QA) | Hanya teks; tidak melibatkan modalitas citra radiologi. |
| **Agrawal et al. (2018)**<br>`CVPR 2018` (VQA-CP) | Mengatasi Priors Teks di VQA | Branch adversarial pertanyaan | Umum (English VQA) | **Tidak** (Akurasi robust) | Uji citra kosong sebagai benchmark diagnostik/loss debiasing latih, bukan aturan seleksi waktu-inferensi. |
| **Chen et al. (2024)**<br>`arXiv:2406.19280` (HuatuoGPT) | Foundation Model Medis VLM | Visual instruction tuning | Medis (EN–ZH) | **Tidak** | Menunjukkan overconfidence parah pada bahasa non-latih (ID ECE = 0.2350). |
| **Riset Kita (Kandidat B)** | Prediksi Selektif VLM Medis Bilingual | Sensitivitas visual berarah + kalibrasi sekuens target | Medis (EN–ID) | **Ya** (Diskrit AURC, cluster bootstrap) | Aplikasi pertama sensitivitas multimodal untuk memitigasi overconfidence lintas-bahasa pada inferensi. |

---

## 7. Keputusan Akhir: Decision Gate

### 🟡 **KATEGORI B: PROMISING BUT PRELIMINARY**

* **Alasan:** Penurunan AURC 22% pada model HuatuoGPT nyata secara empiris, namun masih terikat pada tiga cacat metodologis di kode lama (diskrepansi bahasa Cell 16/20, bug tokenizer, dan formula heuristik unsigned).
* **Tindakan:** Jalankan **Smallest Decisive Validation Experiment** yang telah dipra-registrasi pada [`3_DECISIVE_VALIDATION_PROTOCOL.md`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/reports/3_DECISIVE_VALIDATION_PROTOCOL.md) sebelum penelitian dikunci resmi.
