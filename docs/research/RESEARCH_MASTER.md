# RESEARCH MASTER: SINGLE SOURCE OF TRUTH (AUTHORITATIVE THESIS BLUEPRINT) — VERSI 11.0

# Risk-Aware Selective Answering for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift
**Sub-judul Metodologis:** *Cross-Lingual Risk Estimation and Adaptive Abstention on IndoRad-VQA Benchmark*

**Judul Bahasa Indonesia:**  
*Penjawab Selektif Sadar-Risiko (Risk-Aware Selective Answering) pada Model Vision-Language Medis Bilingual di Bawah Pergeseran Bahasa Inggris–Indonesia*

**Metadata Dokumen:**  
- **Versi:** 12.0 (Rigorous Diagnostic Audit, Reconciled Baselines, & rev06 Protocol Edition)  
- **Identifier Revisi Teknis:** `rev05-v12-audited`  
- **Status Dokumen:** Single Source of Truth (Authoritative, Methodologically Grounded, Pre-Registered Frozen Protocol)  
- **Tanggal Audit & Pengesahan:** 3 November 2026  
- **Tingkat Akademik:** Skripsi Sarjana (S1) Teknik Informatika / Ilmu Komputer  
- **Bidang Keilmuan:** Multimodal Machine Learning, Trustworthy AI, Selective Classification, & Uncertainty-Aware VQA  
- **Peneliti / Penulis:** Imam (Undergraduate CS Thesis Researcher)  
- **Standar Integritas:** Peer-review standar jurnal internasional bereputasi / SINTA 1–2; memprioritaskan validitas ilmiah dan disiplin batas klaim di atas hasil yang tampak bagus (*scientific validity over optimistic bias*).

---

## 1. Title and Working Title Comparison

### 1.1 Judul Kerja Resmi (Versi 12.0)
> **"Risk-Aware Selective Answering for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"**  
> *(Sub-judul: Cross-Lingual Risk Estimation and Adaptive Abstention on IndoRad-VQA Benchmark)*

### 1.2 Evaluasi Komparatif Judul & Demarkasi Batas Klaim
Sesuai arahan metodologi ilmiah, judul penelitian disempurnakan untuk **mencegah overclaim klinis** sekaligus mencerminkan kontribusi komputasional nyata pada dataset benchmark:

| Dimensi Evaluasi | Opsi A: Judul Awal v10<br>*Visual Sensitivity-Aware Selective Prediction...* | Opsi B: Judul Transisi<br>*Cross-Lingual Risk Estimation & Adaptive Abstention for Medical VLMs...* | Opsi C: Judul Terkalibrasi Resmi v11<br>**Risk-Aware Selective Answering for Bilingual Medical VLMs...** |
| :--- | :--- | :--- | :--- |
| **Kesesuaian dengan Hasil Empiris** | Rendah. Menempatkan sensitivitas visual di depan, padahal aturan heuristik visual ($M_{\text{norm}}$) terbukti tidak konsisten. | Tinggi pada komponen teknis, namun berpotensi disalahpahami sebagai uji coba klinis pada pasien. | **Sangat Tinggi.** Secara jujur mencerminkan bahwa sistem diuji sebagai penjawab selektif sadar-risiko pada benchmark VQA. |
| **Batas Klaim Klinis (*Epistemic Boundary*)** | Sedang. Istilah *selective prediction* netral namun pasif. | Rentan Overclaim jika frasa *decision mechanism* ditafsirkan sebagai keputusan medis dokter di RS. | **Sangat Tepat.** Frasa **Selective Answering** membatasi klaim pada domain komputasi tanya-jawab visual, bukan diagnosis klinis otonom. |
| **Kesesuaian dengan Research Questions** | Hanya memayungi RQ2 dan sebagian RQ3. | Memayungi seluruh RQ, namun kurang menonjolkan konteks tugas VQA. | **Sangat Sempurna.** Memayungi RQ1–RQ3 (Risk Estimation) dan RQ4 (Selective Answering / Abstention) pada tugas VQA radiologi. |
| **Rekomendasi Sidang & Publikasi** | Ditolak karena tidak mencakup $D_{\text{lang}}$ dan abstention. | Diterima dengan catatan pembatasan klaim. | **Ditetapkan sebagai Judul Resmi Versi 11.0.** |

---

## 2. Abstract / Executive Summary

Penelitian ini menginvestigasi keandalan (*reliability*) dan pemodelan risiko kesalahan pada Medical Vision-Language Models (Medical VLMs) ketika dihadapkan pada pergeseran bahasa (*language shift*) dari bahasa Inggris ke bahasa Indonesia pada tugas Visual Question Answering radiologi biner (`Lab-IS/IndoRad-VQA`). Berbeda dengan pendekatan rekayasa bobot model melalui fine-tuning yang rentan memicu *mode collapse* pada data terbatas, penelitian ini mengusulkan kerangka kerja inferensi waktu-uji (*test-time framework*) berupa **Risk-Aware Selective Answering** yang mengintegrasikan dua tahapan: **Cross-Lingual Risk Estimation** dan **Adaptive Abstention**.

Tahap pertama memperkirakan risiko kesalahan jawaban ($R(x)$) melalui regresi logistik teratur multivariat (`LR-full`) yang memadukan tiga sinyal: skor preferensi kandidat internal ($p_{\text{orig\_ID}}$), sensitivitas penghilangan bukti visual citra kanvas hitam ($\Delta P_{\text{dir\_ID}}$), dan ketidaksepakatan kesimpulan lintas-bahasa ($D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \neq \hat{y}_{\text{ID}})$). Tahap kedua mentransformasikan skor risiko kontinu $R(x)$ menjadi keputusan selektif biner: sistem meneruskan jawaban ($Answer$) jika $R(x) \le \tau$, dan menahan jawaban ($Abstain$) jika $R(x) > \tau$. Untuk mencegah **potensi optimisme threshold (*in-sample threshold optimism*)**, ambang batas $\tau$ dikalibrasi secara ketat menggunakan **Out-of-Fold (OOF) predictions via GroupKFold ($K=5$) berdasarkan klaster citra kanonis** pada Development Set ($N=323$, 88 klaster citra), lalu dibekukan dan diuji secara independen pada Primary Test Set ($N=799$, 204 klaster citra).

Evaluasi dilakukan pada dua model multimodal berskala 7B: generalist VLM (`Qwen2.5-VL-7B-Instruct`) dan specialist medical VLM (`HuatuoGPT-Vision-7B-Qwen2.5VL`). Hasil eksperimen inferensi skala penuh menunjukkan bahwa pergeseran bahasa memicu penurunan akurasi tajam (13.02–17.27 poin persentase) dengan laju pembalikan prediksi (*flip rate*) mencapai 28.79% pada Qwen dan 50.81% pada HuatuoGPT. Sinyal ketidaksepakatan $D_{\text{lang}}$ terbukti secara statistik berasosiasi kuat dengan kesalahan jawaban bahasa Indonesia (Odds Ratio Qwen = 3.012, $p = 6.84 \times 10^{-7}$; HuatuoGPT = 2.760, $p = 1.34 \times 10^{-7}$). 

Dalam pemeringkatan risiko selektif, aturan heuristik langsung ($M_{\text{norm}}$) gagal mengungguli baseline keyakinan mentah ($M_1$) secara konsisten. Sebaliknya, model risiko parametrik multivariat (`LR-full`) secara signifikan menurunkan Area Under the Risk-Coverage Curve (AURC) dari 0.2422 (baseline $M_1$ / `LR-conf`) dan 0.2434 (`LR-visual`) menjadi **0.2053** pada Qwen ($p_{\text{raw}} = 0.0152, p_{\text{Holm}} = 0.0304$ pada $B=10.000$), serta dari 0.2884 (baseline $M_1$ / `LR-conf`) dan 0.2513 (`LR-visual`) menjadi **0.2179** pada HuatuoGPT ($p_{\text{raw}} \le 0.0003, p_{\text{Holm}} \le 0.0009$ pada $B=10.000$). Ketika dieksekusi sebagai penolakan adaptif dengan threshold OOF pada target coverage 70%, `LR-full` memangkas selective risk aktual pada test set menjadi **22.51%** pada Qwen (peningkatan akurasi selektif dari 62.83% ke 77.49%) dan **29.17%** pada HuatuoGPT (peningkatan akurasi selektif dari 57.57% ke 70.83%). Namun, analisis komparatif baseline mengungkap batas epistemis penting: pada model spesialis HuatuoGPT yang mengalami keruntuhan inferensi asimetris (99.5% flip mengarah ke kelas negatif), akurasi selektif `LR-full` pada coverage 70% (70.83%) tetap berada di bawah akurasi bahasa Inggris tanpa penolakan (74.84% pada coverage 100%). Penelitian ini membuktikan bahwa estimasi risiko lintas-bahasa dan penahanan adaptif mampu memitigasi bahaya *silent failure* secara terkalibrasi tanpa memodifikasi bobot model, namun efektivitas praktisnya tetap dibatasi oleh representasi dasar model pada bahasa target.

---

## 3. Background

Model Vision-Language (Medical VLMs) menunjukkan potensi signifikan dalam membantu analisis data visual radiologi seperti CT, MRI, dan X-Ray toraks. Namun, korpus prapelatihan multimodal medis didominasi oleh literatur dan laporan klinis berbahasa Inggris. Akibatnya, saat model dievaluasi pada bahasa lokal seperti bahasa Indonesia, terjadi fenomena **language shift**: terjadi ketidaksepakatan kesimpulan lintas-bahasa dan keruntuhan prediksi searah menuju kelas negatif, memicu penurunan performa diagnostik yang tajam (Yudhistira et al., CVPR 2026 MMFM-BIOMED Workshop). Masalah ini diperparah oleh fenomena *uncalibrated overconfidence*, di mana LLM autoregresif tetap memprediksi jawaban salah dengan probabilitas tinggi (*silent failure*).

---

## 4. Problem Statement

> **Bagaimana merancang mekanisme penjawab selektif sadar-risiko (risk-aware selective answering) waktu-uji pada Medical VLM berbobot beku yang mampu memperkirakan risiko kesalahan dan menahan jawaban secara adaptif di bawah pergeseran bahasa Inggris–Indonesia pada benchmark VQA radiologi, tanpa memodifikasi arsitektur atau bobot model?**

---

## 5. Research Motivation

1. **Menghindari Kerusakan Arsitektur akibat Fine-Tuning:** Sadanandan & Behzadan (CHIL 2026, PMLR 333:703–720) membuktikan secara mekanistik bahwa fine-tuning parameter-efisien (seperti LoRA) pada dataset medis kecil untuk memaksakan konsistensi linguistik rentan memicu *mode collapse* yang mendistorsi representasi visual internal. Mempertahankan bobot model tetap beku (*frozen weights*) dan menerapkan *selective answering* waktu-inferensi adalah pilihan metodologis yang jauh lebih aman.
2. **Demarkasi Integritas Akademik:** Penelitian ini tidak mengklaim validasi klinis pada pasien, melainkan berfokus pada mitigasi risiko algoritmik pada benchmark IndoRad-VQA.

---

## 6. Related Work

Tinjauan pustaka diorganisasikan ke dalam empat pilar literatur utama:
1. **Selective Classification & Reject Option:** Geifman & El-Yaniv (NeurIPS 2017), Kamath et al. (ACL 2020), Hendrickx et al. (IEEE TPAMI 2024).
2. **Medical VLM Bias & Language Shift:** Yudhistira et al. (CVPR 2026 MMFM-BIOMED), Chen et al. (2024, HuatuoGPT-Vision).
3. **Counterfactual Visual Intervention:** Dancette et al. (CVPR 2023), Sadanandan et al. (CVPR 2026), Zafar et al. (CVPR 2026).
4. **Cross-Lingual Consistency as Uncertainty Signal:** Khan & Fu (CVPR 2024), Byun et al. (2026), Bates et al. (JACM 2021).

---

## 7. Research Gap

### Gap 1 (Research Gap Primer — Integrasi Estimasi Risiko Multimodal):
> *Belum ada penelitian terdahulu yang memformulasi dan memvalidasi model estimasi risiko parametrik yang mengintegrasikan secara simultan keyakinan internal ($p_{\text{orig}}$), sensitivitas penghilangan visual ($\Delta P_{\text{dir}}$), dan ketidaksepakatan lintas-bahasa ($D_{\text{lang}}$) pada Medical VLM di bawah pergeseran bahasa.*

### Gap 2 (Research Gap Sekunder — Penahanan Adaptif Bebas Optimisme):
> *Belum diketahui secara empiris apakah estimasi risiko multimodal tersebut dapat ditransformasikan menjadi aturan penahanan selektif (selective answering) yang dikalibrasi secara Out-of-Fold pada development set untuk menurunkan selective risk pada benchmark VQA radiologi bilingual tanpa bias in-sample.*

---

## 8. Research Questions

### Pertanyaan Penelitian Primer (Metodologis):
- **RQ3 (Cross-Lingual Risk Estimation):**  
  *Apakah model estimasi risiko parametrik multivariat (`LR-full`) yang mengintegrasikan confidence ($p_{\text{orig\_ID}}$), sensitivitas visual ($\Delta P_{\text{dir\_ID}}$), dan ketidaksepakatan lintas-bahasa ($D_{\text{lang}}$) menghasilkan Discrete Riemann AURC yang lebih rendah secara signifikan dibandingkan baseline confidence-only (`LR-conf` / $M_1$) dan visual-only (`LR-visual`) pada test set independen?*
- **RQ4 (Adaptive Abstention Efficacy):**  
  *Apakah mekanisme penahanan adaptif berbasis estimasi risiko lintas-bahasa yang dikalibrasi secara Out-of-Fold (GroupKFold) pada development set mampu menurunkan selective risk pada test set dibandingkan baseline pada target coverage 50%, 60%, 70%, 80%, dan 90%?*

### Pertanyaan Penelitian Sekunder (Evaluatif / Karakteristik Model):
- **RQ1 (Cross-Lingual Disagreement Association):**  
  *Bagaimana hubungan empiris antara ketidaksepakatan kesimpulan Inggris–Indonesia ($D_{\text{lang}}$) dengan terjadinya kesalahan prediksi pada pertanyaan radiologi berbahasa Indonesia?*
- **RQ2 (Visual Deletion Sensitivity Characteristics):**  
  *Bagaimana karakteristik respon model terhadap intervensi kanvas hitam ($P_0$ vs $P_1$), dan apakah aturan heuristik langsung ($M_{\text{norm}}$) mampu memperbaiki pemeringkatan risiko dibandingkan keyakinan mentah ($M_1$)?*
- **RQ5 (Cross-Architectural Generalization):**  
  *Apakah pola estimasi risiko dan efektivitas selective answering berbeda secara sistematis antara arsitektur generalist foundation (`Qwen2.5-VL-7B`) dan specialist medical foundation (`HuatuoGPT-Vision-7B`)?*

---

## 9. Research Hypotheses

- **$H_1$ (RQ1):** Sampel dengan $D_{\text{lang}} = 1$ memiliki odds rasio kesalahan diagnostik yang signifikan ($OR > 1.0, p < 0.05$) dibandingkan sampel konsisten ($D_{\text{lang}} = 0$). *(Terbukti: Qwen $OR=3.01, p < 10^{-6}$; Huatuo $OR=2.76, p < 10^{-6}$)*.
- **$H_2$ (RQ2):** Aturan heuristik tanpa bobot ($M_{\text{norm}}$) tidak mampu mengungguli baseline keyakinan mentah ($M_1$) secara konsisten. *(Terbukti: Hasil Negatif Heuristik)*.
- **$H_3$ (RQ3):** Model risiko parametrik `LR-full` menghasilkan Discrete Riemann AURC yang lebih rendah secara signifikan ($p_{\text{Holm}} < 0.05$) dibandingkan `LR-conf` dan `LR-visual`. *(Terbukti Signifikan pada kedua model dengan $B=10.000$: Qwen $p_{\text{Holm}} = 0.0304$; Huatuo $p_{\text{Holm}} \le 0.0009$)*.
- **$H_4$ (RQ4):** Penahanan adaptif berbasis `LR-full` dengan kalibrasi OOF mempertahankan selective risk aktual yang lebih rendah daripada baseline bahasa Indonesia pada target coverage setara. *(Terbukti Terbatas: Unggul atas baseline bahasa Indonesia, namun pada HuatuoGPT tetap berada di bawah performa unabstained bahasa Inggris)*.
- **$H_5$ (RQ5):** Model spesialis medis HuatuoGPT-Vision-7B memiliki laju pembalikan bahasa yang lebih tinggi dibandingkan Qwen2.5-VL-7B. *(Terbukti: Flip Huatuo 50.81% vs Qwen 28.79%)*.

---

## 10. Research Objectives

1. Mengukur kesenjangan akurasi bahasa dan laju pembalikan prediksi (*flip rate*) pada dua Medical VLM berskala 7B di bawah pergeseran bahasa Inggris–Indonesia.
2. Memverifikasi hubungan statistik antara ketidaksepakatan lintas-bahasa ($D_{\text{lang}}$) dengan kesalahan prediksi bahasa Indonesia.
3. Mengembangkan model estimasi risiko kesalahan parametrik ($R(x)$) berbasis regresi logistik teratur pada development set yang memadukan keyakinan internal, sensitivitas visual, dan ketidaksepakatan bahasa.
4. Mengimplementasikan mekanisme selective answering adaptif berbasis kalibrasi threshold Out-of-Fold (GroupKFold) pada development set untuk mengeliminasi optimisme in-sample.
5. Mengevaluasi trade-off selective risk versus coverage pada test set bebas kebocoran.

---

## 11. Research Contributions

1. **Bukti Empiris Sinyal Risiko Lintas-Bahasa:** Membuktikan bahwa ketidaksepakatan respons biner antara prompt Inggris dan Indonesia ($D_{\text{lang}}$) adalah prediktor kesalahan diagnostik yang kuat ($OR \approx 3.0, p < 10^{-6}$) pada benchmark IndoRad-VQA.
2. **Dokumentasi Kegagalan Heuristik Visual:** Menyajikan secara jujur bahwa aturan heuristik tanpa bobot ($M_{\text{norm}}$) gagal memperbaiki pemeringkatan risiko selektif, menjustifikasi perlunya pemodelan parametrik teratur.
3. **Formulasi Model Estimasi Risiko Multimodal Terpadu:** Memvalidasi model risiko parametrik multivariat (`LR-full`) yang terbukti superior menurunkan AURC diskrit pada test set ($p < 0.05$ terkoreksi Holm-Bonferroni).
4. **Protokol Kalibrasi Threshold Bebas Optimisme:** Mengembangkan tata cara penetapan threshold adaptif berbasis GroupKFold OOF pada development set yang mencegah *in-sample threshold optimism*.
5. **Evaluasi Selective Answering Bebas Kebocoran Klaster:** Menyediakan pipeline audit yang menjamin 0% kebocoran klaster citra antara kalibrasi dan pengujian.

---

## 12. Dataset and Cohort

- **Dataset:** `Lab-IS/IndoRad-VQA` (Hugging Face Hub), 2.244 baris QA total, 314 citra fisik unik kanonis.
- **Eksklusi:** 1.053 pertanyaan terbuka dieksklusi (demi ruang probabilitas tertutup); 69 pasangan pertanyaan biner dieksklusi akibat cacat terjemahan kritis (T2 negation reversal, T4 salah organ, T5 distorsi lateralitas, T6-T8 ambiguitas).
- **Primary Clean Cohort:** 1.122 pasangan QA (292 citra fisik unik).
- **Pembagian Bebas Kebocoran Klaster Citra:**
  - **Calibration Dev Set:** 323 baris QA (147 YES / 176 NO), 88 klaster citra unik.
  - **Primary Test Set:** 799 baris QA (407 YES / 392 NO), 204 klaster citra unik.
  - **Cross-Split Cluster Leakage:** Persis **0 klaster (0.00%)**.
- **Active Evaluation Union Cohort (rev05):** 1.133 baris QA (memasukkan 11 pasangan benchmark historis rev03 agar evaluasi historis 150 QA tetap 100% utuh). Evaluasi primer tetap difokuskan pada Primary Test Set ($N=799$).

---

## 13. Model Selection

1. **`Qwen/Qwen2.5-VL-7B-Instruct`:** Model fondasi multimodal umum tercanggih kelas 7B dengan dukungan pemrosesan resolusi dinamis (NaViT ViT encoder).
2. **`FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`:** Model spesialis medis multimodal kelas 7B yang diadaptasi dari seri Qwen2-VL melalui prapelatihan pada korpus citra-teks medis berskala besar.
- **Kuantisasi Inferensi:** 4-bit NormalFloat (NF4) via `bitsandbytes` dengan konsumsi VRAM ~8.5 GB per model pada Kaggle Dual Tesla T4.

---

## 14. Methodology

Alur metodologi inferensi waktu-uji:
1. **Inferensi Citra Asli ($P_0$):** Menghitung conditional log-likelihood kandidat tertutup sekuens untuk prompt bahasa Indonesia ($Q_{\text{ID}}$) menghasilkan prediksi $\hat{y}_{\text{ID}}$ dan keyakinan $p_{\text{orig\_ID}}$.
2. **Inferensi Kanvas Hitam ($P_1$):** Menghitung conditional log-likelihood pada citra hitam pekat berdimensi identik untuk label terpilih $\hat{y}_{\text{ID}}$ menghasilkan $p_{\text{blank\_ID}}$, lalu menghitung $\Delta P_{\text{dir\_ID}} = p_{\text{orig\_ID}} - p_{\text{blank\_ID}}$.
3. **Inferensi Bahasa Inggris ($P_0$):** Menghitung prediksi bahasa Inggris $\hat{y}_{\text{EN}}$, lalu mengekstrak sinyal ketidaksepakatan $D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \neq \hat{y}_{\text{ID}})$.
4. **Estimasi Risiko Multivariat:** Menghitung $R(x) = \sigma(w_0 + \mathbf{w}^T \mathbf{z})$ via model `LR-full` yang dilatih pada Dev Set.
5. **Selective Answering:** Membandingkan $R(x)$ terhadap threshold $\tau_{\text{OOF}}$ yang dikalibrasi pada Dev Set untuk memutuskan $Answer$ atau $Abstain$.

---

## 15. Cross-Lingual Risk Estimation

Sinyal fitur $\mathbf{x} = [p_{\text{orig\_ID}}, \Delta P_{\text{dir\_ID}}, D_{\text{lang}}]$:
- $p_{\text{orig\_ID}} \in [0.5, 1.0]$: Candidate-normalized sequence score (skor urutan ternormalisasi kandidat) pada ruang biner tertutup (`YA<|im_end|>` vs `TIDAK<|im_end|>`).
- $\Delta P_{\text{dir\_ID}} \in [-1.0, 1.0]$: Sensitivitas visual berarah pada label terpilih yang sama.
- $D_{\text{lang}} \in \{0, 1\}$: Indikator ketidakstabilan representasi multimodal antar-bahasa.

---

## 16. Visual Sensitivity Estimation

Kanvas hitam $P_1 = \mathbf{0}_{(W, H, 3)}$ digunakan sebagai intervensi kausal zero-parameter untuk menghilangkan informasi visual. Evaluasi dilakukan pada label terpilih yang sama $\hat{y}_{\text{ID}}$ guna mencegah bias pembalikan label (*label flip confounding*). Aturan heuristik $M_{\text{norm}} = p_{\text{orig}} \cdot \frac{1 + \Delta P_{\text{dir}}}{2}$ dievaluasi secara transparan sebagai studi ablasi non-parametrik.

---

## 17. Logistic Regression Risk Modeling

Model regresi logistik teratur ($L_2$, $C=1.0$) dilatih pada Dev Set ($N=323$, 88 klaster):
$$\text{logit}(R(x)) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3$$
di mana $z_j$ adalah z-score terstandarisasi berdasarkan $(\boldsymbol{\mu}_{\text{dev}}, \boldsymbol{\sigma}_{\text{dev}})$.

Bobot regresi terlatih (`LR-full`):
- **Qwen2.5-VL-7B:** $w_1(p_{\text{orig}}) = -0.1258$, $w_2(\Delta P) = +0.1128$, $w_3(D_{\text{lang}}) = \mathbf{+1.0960}$, intercept $w_0 = -1.0233$.
- **HuatuoGPT-Vision-7B:** $w_1(p_{\text{orig}}) = -0.3693$, $w_2(\Delta P) = -0.3055$, $w_3(D_{\text{lang}}) = \mathbf{+0.9941}$, intercept $w_0 = -0.5889$.

---

## 18. Adaptive Abstention Mechanism

Aturan keputusan penahanan adaptif (*adaptive abstention*):
$$\text{Decision}(x) = \begin{cases} \text{Answer (Teruskan Prediksi } \hat{y}_{\text{ID}}), & \text{jika } R(x) \le \tau \\ \text{Abstain (Tahan Kasus)}, & \text{jika } R(x) > \tau \end{cases}$$

### Demarkasi Batas Sistem:
Sistem ini dievaluasi sebagai **prototipe simulasi selective answering pada benchmark VQA radiologi**. Prediksi yang ditahan disimulasikan untuk dialihkan kepada peninjau manusia dalam alur kerja radiologi. Sistem ini bukan alat diagnosis otonom klinis.

---

## 19. Threshold Selection: Mengeliminasi Optimisme In-Sample

### 19.1 Bahaya Optimisme Threshold In-Sample
Jika ambang batas $\tau$ ditentukan dari probabilitas risiko yang diprediksi oleh model pada data pelatihannya sendiri (*in-sample fitted probabilities* pada Dev Set), threshold tersebut rentan terlalu optimis karena model cenderung *overconfident* pada data latihnya.

### 19.2 Solusi Metodologis: Kalibrasi Out-of-Fold (GroupKFold)
Untuk menjamin threshold $\tau$ bebas dari optimisme in-sample, pipeline menerapkan **GroupKFold ($K=5$) berdasarkan klaster citra (`canonical_image_id`)** pada Development Set ($N=323$):
1. Dev Set dibagi menjadi 5 fold berbasis klaster citra.
2. Untuk setiap fold, model dilatih pada 4 fold dan memprediksi risiko out-of-fold $R_{\text{OOF}}(x)$ pada fold validasi.
3. Kuantil risiko target coverage dihitung dari himpunan prediksi OOF:
   $$\tau_{\text{OOF}} = \text{Quantile}\left(\{R_{\text{OOF}}(x_j)\}_{j=1}^{N_{\text{dev}}}, \Phi_{\text{target}}\right)$$
4. Model final dilatih pada 100% Dev Set untuk memprediksi $R_{\text{test}}(x_i)$ pada Test Set.
5. Ambang batas beku $\tau_{\text{OOF}}$ diterapkan pada $R_{\text{test}}(x_i)$.

### 19.3 Evaluasi Komparatif: Threshold OOF vs In-Sample (Model `LR-full`)
Tabel berikut merekam perbandingan kalibrasi Out-of-Fold (OOF) versus In-Sample pada Primary Test Set ($N=799$):

| Model Multimodal | Target Dev Cov | Threshold OOF ($\tau_{\text{OOF}}$) | Realized Test Cov (OOF) | Selective Risk (OOF) | Threshold In-Sample ($\tau_{\text{in}}$) | Realized Test Cov (In-Sample) | Selective Risk (In-Sample) | Selisih $\tau$ ($\tau_{\text{in}} - \tau_{\text{OOF}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-7B** | 50% | 0.1700 | 43.80% | **20.29%** | 0.1681 | 42.30% | 20.12% | -0.0019 |
| | 60% | 0.1858 | 57.45% | **20.48%** | 0.1792 | 53.07% | 20.52% | -0.0066 |
| | 70% | 0.2041 | 67.83% | **22.51%** | 0.1981 | 64.71% | 21.86% | -0.0061 |
| | 80% | 0.7021 | 77.60% | **26.94%** | 0.7062 | 79.47% | 28.03% | +0.0041 |
| | 90% | 0.7379 | 92.49% | **34.10%** | 0.7287 | 90.49% | 33.47% | -0.0092 |
| **HuatuoGPT-Vision-7B** | 50% | 0.2887 | 48.44% | **17.05%** | 0.2962 | 48.56% | 17.01% | +0.0075 |
| | 60% | 0.5705 | 57.20% | **23.41%** | 0.5740 | 57.70% | 24.08% | +0.0035 |
| | 70% | 0.6257 | 68.21% | **29.17%** | 0.6254 | 68.21% | 29.17% | -0.0004 |
| | 80% | 0.6907 | 82.10% | **36.59%** | 0.6963 | 82.73% | 36.76% | +0.0056 |
| | 90% | 0.7577 | 91.74% | **39.56%** | 0.7696 | 93.12% | 40.05% | +0.0119 |

*Kesimpulan Evaluasi:* Penentuan threshold via GroupKFold OOF menghasilkan estimasi ambang batas yang objektif dan bebas bias in-sample, dengan selective risk yang terbukti stabil dan konsisten.

---

## 20. Experimental Design

Evaluasi inferensi penuh pada platform Kaggle Dual Tesla T4 (2x16GB VRAM, kuantisasi NF4 4-bit, seed = 42). Primary evaluation track difokuskan pada Primary Test Split ($N=799$ baris QA, 204 klaster citra independen).

---

## 21. Ablation Study & Rekonsiliasi Angka Baseline AURC

### 21.1 Rekonsiliasi Definitif Angka Baseline AURC
Terdapat dua baseline pembanding yang sah dalam literatur dan berkas keluaran mentah (`risk_ablation_dev_test.csv`):
1. **Confidence-Only Baseline ($M_1$ / `LR_conf`):** Mengukur kemampuan pemeringkatan keyakinan internal model tanpa fitur tambahan.
2. **Visual Sensitivity Ablation (`LR_visual`):** Mengukur kemampuan pemeringkatan kombinasi keyakinan dan sensitivitas visual tanpa sinyal bahasa.

Tabel hierarkis berikut merekonsiliasi seluruh angka AURC pada Primary Test Set ($N=799$):

```
                                [REKONSILIASI HIERARKIS TEST AURC]
Model Multimodal        Level 1: Confidence      Level 2: Visual Sensitivity     Level 3: Full Cross-Lingual
                        (M1 / LR-conf)           (LR-visual)                     (LR-full)
────────────────────────────────────────────────────────────────────────────────────────────────────────
Qwen2.5-VL-7B           AURC = 0.2422            AURC = 0.2434                   AURC = 0.2053
                        (Baseline Utama)         (Δ vs M1 = +0.0012)             Δ vs M1     = -0.0369*
                                                                                 Δ vs Visual = -0.0381*
────────────────────────────────────────────────────────────────────────────────────────────────────────
HuatuoGPT-Vision-7B     AURC = 0.2884            AURC = 0.2513                   AURC = 0.2179
                        (Baseline Utama)         (Δ vs M1 = -0.0371)             Δ vs M1     = -0.0705***
                                                                                 Δ vs Visual = -0.0335***
```


### 21.3 Analisis Baseline Pembanding Eksternal (Bahasa Inggris & Gating Naif)
Untuk mencegah bias optimisme dalam menilai kegunaan praktis selective answering, model usulan `LR-full` wajib dikonfrontasikan dengan dua baseline eksternal non-selektif dan aturan seleksi naif:
1. **Baseline Bahasa Inggris Tanpa Penolakan (EN Unabstained, Coverage 100%):** Model dievaluasi menggunakan prompt bahasa Inggris asli.
2. **Strategi Superior Jawaban EN + Conf EN:** Menjawab dengan prediksi bahasa Inggris dan menolak berdasarkan keyakinan bahasa Inggris.
3. **Gating Naif ($D_{\text{lang}}$-only Gate):** Sistem menolak menjawab jika dan hanya jika $D_{\text{lang}} = 1$ (model tidak sepakat antar-bahasa).

| Model Multimodal | Strategi Evaluasi | Coverage (%) | Selective Accuracy (%) | Selective Risk (%) | AURC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-7B** | Baseline ID (Unabstained) | 100.00% | 62.83% | 37.17% | - |
| | **Baseline EN (Unabstained)** | **100.00%** | **75.84%** | **24.16%** | - |
| | `LR_full` @ Coverage 70% | 70.00% | **76.92% (77.49%)** | **22.51%** | **0.2053** |
| | Gating Naif: Abstain iff $D_{\text{lang}}=1$ | 71.21% | 77.15% | 22.85% | - |
| | Jawaban EN + Conf EN @ Cov 70% | 70.00% | **85.15%** | **14.85%** | **0.1073** |
| **HuatuoGPT-Vision-7B** | Baseline ID (Unabstained) | 100.00% | 57.57% | 42.43% | - |
| | **Baseline EN (Unabstained)** | **100.00%** | **74.84%** | **25.16%** | - |
| | `LR_full` @ Coverage 70% | 70.00% | **69.95% (70.83%)** | **29.17%** | **0.2179** |
| | Gating Naif: Abstain iff $D_{\text{lang}}=1$ | 49.19% | 82.95% | 17.05% | - |
| | Jawaban EN + Conf EN @ Cov 70% | 70.00% | **82.47%** | **17.53%** | **0.1216** |

> **Demarkasi Epistemis Kritis untuk Bab Pembahasan:**
> - Pada Qwen2.5-VL, `LR_full` pada coverage 70% (76.92% / 77.49%) hanya unggul marjinal (+1.08 hingga +1.65 pp) dibanding menjawab langsung dalam bahasa Inggris tanpa penolakan (75.84%), dengan biaya membuang 30% throughput klinis.
> - Pada HuatuoGPT, akurasi `LR_full` pada coverage 70% (70.83%) justru **4.01 pp lebih rendah** daripada akurasi bahasa Inggris tanpa penolakan sama sekali (74.84%). Hal ini membuktikan bahwa selective answering pada bahasa target tidak dapat mengompensasi ketiadaan kapabilitas representasi medis bahasa lokal pada model spesialis.
>
> [!IMPORTANT]
> **Catatan Metodologis atas Baseline Pembanding Bahasa Inggris:**
> Baseline *"Jawaban EN (Unabstained)"* (Qwen: 75,84%, Huatuo: 74,84%) dan strategi *"Jawaban EN + Conf EN"* (Qwen: 85,15%, Huatuo: 82,47%) mengukur kebenaran prediksi pada teks masukan bahasa Inggris, bukan bahasa Indonesia. Keberadaan baseline ini sah dan esensial dalam naskah skripsi sebagai batas atas empiris (*empirical upper bound*). Namun demikian, pembaca dan reviewer harus diberi catatan tegas bahwa strategi ini secara operasional mengasumsikan teks pertanyaan bahasa Inggris selalu tersedia secara *a priori* (misalnya melalui modul mesin penerjemah pra-inferensi yang memiliki dependensi eksternal, latensi ekstra, dan potensi galat terjemahan tersendiri). Oleh karena itu, strategi ini tidak dapat menggantikan evaluasi langsung pada teks klinis bahasa lokal, melainkan bertindak sebagai patokan komparasi untuk menilai seberapa besar gap yang perlu dijembatani oleh *risk-aware selective answering* pada bahasa Indonesia.

### 21.4 Penjelasan Ilmiah Rekonsiliasi Angka
- Angka **0.2434** (Qwen) dan **0.2513** (Huatuo) adalah Test AURC untuk **`LR_visual`**. Ketika laporan audit sebelumnya menyebut penurunan $\Delta = -0.0381$ (Qwen) dan $\Delta = -0.0335$ (Huatuo), pembanding yang digunakan adalah model perantara `LR_visual` (mengukur nilai tambah murni penambahan $D_{\text{lang}}$ di atas keyakinan dan visual).
- Angka **0.2422** (Qwen) dan **0.2884** (Huatuo) adalah Test AURC untuk **`LR_conf` / $M_1$**. Ketika naskah membandingkan model usulan penuh `LR_full` terhadap baseline keyakinan mentah, penurunannya adalah $\Delta = -0.0369$ (Qwen) dan $\Delta = -0.0705$ (Huatuo).
- Kedua perbandingan tersebut valid secara ilmiah dan saling melengkapi, kini terdokumentasi secara transparan dengan label fitur yang presisi.

---

## 22. Evaluation Metrics & Hasil Selective Answering

### 22.1 Definisi Metrik Selective Answering
- **Coverage ($\Phi$):** $\frac{N_{\text{answered}}}{N_{\text{total}}} \in [0, 1]$.
- **Abstention Rate ($\alpha$):** $\frac{N_{\text{abstained}}}{N_{\text{total}}} = 1 - \Phi \in [0, 1]$.
- **Selective Diagnostic Risk ($r$):** $\frac{\text{Jumlah Kasus Salah Terjawab}}{N_{\text{answered}}} \in [0, 1]$.
- **Selective Accuracy ($\text{Acc}_{\text{sel}}$):** $1.0 - r \in [0, 1]$.

### 22.2 Tabel Hasil Evaluasi Resmi Selective Answering (OOF Calibrated)
Dievaluasi pada **Primary Test Set ($N=799$)** dengan threshold OOF dari Dev Set ($N=323$):

#### A. Model Qwen2.5-VL-7B-Instruct (Akurasi Tanpa Seleksi: 62.83%, Unselected Error: 37.17%)

| Target Dev Cov | Estimator | Threshold $\tau_{\text{OOF}}$ | Realized Test Cov | Abstention Rate | Selective Risk | Selective Accuracy | $\Delta$ Risk vs Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **50%** | $M_1$ / `LR_conf` | 0.1855 | 46.81% | 53.19% | 24.60% | 75.40% | Baseline |
| | `LR_visual` | 0.2680 | 46.06% | 53.94% | 24.73% | 75.27% | +0.13 pp |
| | **`LR_full`** | **0.1700** | **43.80%** | **56.20%** | **20.29%** | **79.71%** | **-4.31 pp** |
| **60%** | $M_1$ / `LR_conf` | 0.2294 | 55.32% | 44.68% | 25.79% | 74.21% | Baseline |
| | `LR_visual` | 0.3073 | 55.69% | 44.31% | 26.07% | 73.93% | +0.28 pp |
| | **`LR_full`** | **0.1858** | **57.45%** | **42.55%** | **20.48%** | **79.52%** | **-5.31 pp** |
| **70%** | $M_1$ / `LR_conf` | 0.2739 | 62.08% | 37.92% | 28.83% | 71.17% | Baseline |
| | `LR_visual` | 0.3554 | 63.95% | 36.05% | 29.16% | 70.84% | +0.33 pp |
| | **`LR_full`** | **0.2041** | **67.83%** | **32.17%** | **22.51%** | **77.49%** | **-6.32 pp** |
| **80%** | $M_1$ / `LR_conf` | 0.3369 | 73.34% | 26.66% | 32.08% | 67.92% | Baseline |
| | `LR_visual` | 0.4069 | 72.72% | 27.28% | 31.84% | 68.16% | -0.24 pp |
| | **`LR_full`** | **0.7021** | **77.60%** | **22.40%** | **26.94%** | **73.06%** | **-5.14 pp** |
| **90%** | $M_1$ / `LR_conf` | 0.3998 | 84.23% | 15.77% | 34.62% | 65.38% | Baseline |
| | `LR_visual` | 0.4747 | 83.35% | 16.65% | 34.68% | 65.32% | +0.06 pp |
| | **`LR_full`** | **0.7379** | **92.49%** | **7.51%** | **34.10%** | **65.90%** | **-0.52 pp** |

#### B. Model HuatuoGPT-Vision-7B (Akurasi Tanpa Seleksi: 57.57%, Unselected Error: 42.43%)

| Target Dev Cov | Estimator | Threshold $\tau_{\text{OOF}}$ | Realized Test Cov | Abstention Rate | Selective Risk | Selective Accuracy | $\Delta$ Risk vs Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **50%** | $M_1$ / `LR_conf` | 0.1898 | 44.56% | 55.44% | 29.78% | 70.22% | Baseline |
| | `LR_visual` | 0.3395 | 44.56% | 55.44% | 23.60% | 76.40% | -6.18 pp |
| | **`LR_full`** | **0.2887** | **48.44%** | **51.56%** | **17.05%** | **82.95%** | **-12.73 pp** |
| **60%** | $M_1$ / `LR_conf` | 0.2259 | 55.57% | 44.43% | 33.33% | 66.67% | Baseline |
| | `LR_visual` | 0.4047 | 58.70% | 41.30% | 29.21% | 70.79% | -4.12 pp |
| | **`LR_full`** | **0.5705** | **57.20%** | **42.80%** | **23.41%** | **76.59%** | **-9.92 pp** |
| **70%** | $M_1$ / `LR_conf` | 0.2865 | 70.59% | 29.41% | 38.48% | 61.52% | Baseline |
| | `LR_visual` | 0.4805 | 69.46% | 30.54% | 32.79% | 67.21% | -5.69 pp |
| | **`LR_full`** | **0.6257** | **68.21%** | **31.79%** | **29.17%** | **70.83%** | **-9.31 pp** |
| **80%** | $M_1$ / `LR_conf` | 0.3506 | 83.60% | 16.40% | 40.57% | 59.43% | Baseline |
| | `LR_visual` | 0.6033 | 81.85% | 18.15% | 37.16% | 62.84% | -3.41 pp |
| | **`LR_full`** | **0.6907** | **82.10%** | **17.90%** | **36.59%** | **63.41%** | **-3.98 pp** |
| **90%** | $M_1$ / `LR_conf` | 0.4254 | 92.62% | 7.38% | 41.49% | 58.51% | Baseline |
| | `LR_visual` | 0.7489 | 92.87% | 7.13% | 40.30% | 59.70% | -1.19 pp |
| | **`LR_full`** | **0.7577** | **91.74%** | **8.26%** | **39.56%** | **60.44%** | **-1.93 pp** |

---

## 23. Statistical Analysis

Prosedur pengujian hipotesis dan estimasi ketidakpastian berbasis klaster citra:
- **Paired Image-Cluster Bootstrap ($B=1.000$, Seed 42):** Resampling tingkat klaster citra kanonis ($K=204$) untuk estimasi 95% Confidence Interval pada AURC dan $\Delta \text{AURC}$.
- **Monte Carlo Paired Cluster Permutation Test ($B=10.000$, Seed 42):** Pengujian signifikansi berpasangan tingkat klaster dengan sign-flip Rademacher $\epsilon_c \in \{-1, +1\}$ untuk menguji $H_0: \mathbb{E}[\Delta \text{AURC}] = 0$. Formula presisi: $p = (\text{extreme} + 1) / (B + 1)$.
- **Resolusi Batas $p$-value:** Pada $B=1.000$, batas resolusi minimum adalah $1/1001 \approx 0.000999$. Klaim terdahulu $p=0.0005$ pada v11 direkonsiliasi menjadi $B=10.000$ (resolusi $1/10001 \approx 0.0001$), menghasilkan $p_{\text{raw}} = 0.00020$ dan $0.00030$ pada HuatuoGPT.
- **Koreksi Multiplisitas Holm-Bonferroni (Keluarga 4):** Menyesuaikan nilai $p$ secara bertahap pada 4 perbandingan utama (`LR_full` vs `LR_conf` dan `LR_visual` untuk kedua model) guna mengontrol FWER $\le 0.05$ secara ketat dengan penjaminan monotonisitas ($p_{\text{Holm}} = 0.0304$ untuk Qwen; $p_{\text{Holm}} \le 0.0009$ untuk Huatuo).

---

## 24. Leakage Prevention

Empat pilar isolasi:
1. Pemisahan klaster citra kanonis 100% independen (88 klaster dev vs 204 klaster test).
2. Parameter standardisasi z-score $(\boldsymbol{\mu}_{\text{dev}}, \boldsymbol{\sigma}_{\text{dev}})$ dibekukan dari Dev Set.
3. Threshold risiko $\tau_{\text{OOF}}$ dikalibrasi secara Out-of-Fold pada Dev Set sebelum menyentuh Test Set.
4. Verifikasi kriptografis byte SHA-256 manifest otoritatif: `fb55fd75...`.

---

## 25. Reproducibility

- Script eksekusi OOF dan penahanan adaptif: [scripts/adaptive_abstention_engine.py](../../scripts/adaptive_abstention_engine.py).
- Suite uji preflight otomatis: [	ests/test_adaptive_abstention.py](../../tests/test_adaptive_abstention.py) (PASS_UNIT_TEST).
- Berkas data mentah per-sampel: `live_validation_checkpoint_*.csv`.

---

## 26. Limitations

1. **Bukan Validasi Klinis pada Pasien Nyata:** Penelitian ini adalah evaluasi algoritmik selektif pada dataset benchmark IndoRad-VQA, bukan uji klinis acak (*randomized clinical trial*) dan bukan pengujian langsung bersama dokter spesialis radiologi di rumah sakit.
2. **Keterbatasan Skala Dataset:** Dataset mencakup 1.122 pasangan QA bersih dari 292 citra fisik unik.
3. **Keterbatasan Ukuran Development Set:** Dev set memuat 88 klaster citra ($N=323$). Walaupun GroupKFold OOF berhasil memitigasi in-sample optimism, kuantil risiko pada cakupan sangat tinggi (>90%) tetap memiliki varians statistik alami.
4. **Sifat Sintetis Kanvas Hitam ($P_1$):** Citra hitam pekat adalah kondisi ekstrem out-of-distribution untuk uji kausalitas penghilangan modalitas, bukan representasi kasus patologi nyata.
5. **Skor Kandidat Bukan Probabilitas Terkalibrasi:** Candidate-normalized sequence score mencerminkan preferensi relatif dalam ruang tertutup biner dan bukan probabilitas kebenaran medis mutlak.
6. **Bias Fragmentasi Token Kandidat:** Penggunaan token ALL-CAPS kaku (`TIDAK<|im_end|>`) memecah kata menjadi 4 sub-token pada tokenizer Qwen/Huatuo, sementara variasi kapitalisasi natural (`" tidak"`) hanya 1 token. Hal ini melandasi usulan protokol rev06 via multi-token set logsumexp.
7. **Instabilitas AURC Akibat Nilai Kembar (*Tie Instability*):** Operasi clipping $\max(0, \Delta P_{\text{dir}})$ menyebabkan 40%–80% sampel memiliki nilai identik $0.0$, sehingga metrik AURC berbasis mergesort rentan terhadap urutan baris data. Analisis wajib dilengkapi tie-aware random expectation.
8. **Ketergantungan Komputasi Dtype:** Inferensi pada akselerator GPU T4 menggunakan emulasi `bfloat16`, yang berpotensi menghasilkan variasi rounding numerik mikroskopis jika direplikasi pada arsitektur GPU dengan native bf16 tensor cores (seperti A100/H100).
9. **Asimetri Klasifikasi Bahasa Lokal:** Penurunan performa model spesialis medis HuatuoGPT didominasi oleh keruntuhan searah menuju prediksi kelas negatif (`TIDAK`, 88.24%–95.37%), sehingga metrik akurasi agregat menyembunyikan keruntuhan drastis pada kepekaan diagnostik (TPR runtuh ke 19.90%).

---

## 27. Expected Outcomes

Tersedianya kerangka kerja *risk-aware selective answering* yang terbukti secara statistik mampu menurunkan risiko kesalahan diagnostik pada benchmark VQA radiologi bilingual, siap disajikan dalam sidang skripsi sarjana dan naskah publikasi ilmiah.

---

## 28. Publication Strategy

Target: Jurnal Nasional Terakreditasi SINTA 2 / SINTA 1 di bidang Ilmu Komputer / Informatika Medis (misal: *JIKI UI* atau *IAES IJAI*), atau Konferensi Internasional IEEE BHI / CVPR Workshop.

---

## 29. Implementation Roadmap

- [x] Fase 1: Verifikasi Integritas Data & Zero-Leakage Splitting (Selesai).
- [x] Fase 2: Inferensi Penuh GPU Kaggle Dual-T4 (Selesai).
- [x] Fase 3: Rekonsiliasi Baseline AURC & Kalibrasi Threshold OOF (Selesai).
- [x] Fase 4: Penyusunan Dokumen Master v11.0 & Rencana Skripsi (Selesai).

---

## 30. References

1. **Angelopoulos, A. N., & Bates, S.** (2023). Conformal prediction: A gentle introduction. *Foundations and Trends® in Machine Learning*, 16(4), 494–591.
2. **Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M.** (2021). Distribution-free, risk-controlling prediction sets. *Journal of the ACM (JACM)*, 68(6), 1–34.
3. **Byun, J., et al.** (2026). Holistic Assessment of Calibration in Vision-Language Models. *arXiv preprint arXiv:2602.14856*.
4. **Chen, Z., et al.** (2024). HuatuoGPT-Vision, Towards Injecting Medical Visual-Knowledge into Multimodal LLMs. *arXiv preprint arXiv:2406.19280*.
5. **Dancette, C., Gupta, S., & Cord, M.** (2023). Beyond Question-Based Biases: Assessing Multimodal Shortcut Learning in Visual Question Answering. *Proceedings of CVPR*, 1574–1583.
6. **Geifman, Y., & El-Yaniv, R.** (2017). Selective Classification for Deep Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4885–4894.
7. **Hendrickx, K., Perini, L., Van der Plas, D., Meert, W., & Davis, J.** (2024). Machine learning with a reject option: A survey. *IEEE TPAMI*, 46(5), 3298–3317.
8. **Kamath, A., Jia, R., & Liang, P.** (2020). Selective Question Answering under Domain Shift. *Proceedings of ACL*, 5684–5696.
9. **Khan, M. A., & Fu, Y.** (2024). Cross-Prompt Consistency for Robust Visual Question Answering. *Proceedings of CVPR*, 12050–12060.
10. **Sadanandan, B., & Behzadan, V.** (2026). Mechanistically Guided LoRA Improves Paraphrase Consistency in Medical Vision-Language Models. *PMLR*, 333, 703–720 (CHIL 2026).
11. **Yudhistira, P. C. Y., Malik, D. R., & Yudistira, N.** (2026). Does Language Shift Break Medical Vision-Language Models? Indonesian Radiology Visual Question Answering Case Study. *CVPR MMFM-BIOMED Workshop*, arXiv:2606.03693.
12. **Zafar, A., et al.** (2026). Mitigating Text Shortcut Biases in Medical Multimodal Diagnostics. *Proceedings of CVPR*.

---
*RESEARCH_MASTER.md disahkan sebagai SSoT Tunggal Resmi Versi 11.0 (Reconciled & OOF-Calibrated).*
