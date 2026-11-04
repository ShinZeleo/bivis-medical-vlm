# ADAPTIVE ABSTENTION EXPERIMENT PLAN (V11.0)
## Risk-Controlled Decision Making for Bilingual Medical Vision-Language Models

**Metadata Dokumen:**  
- **Versi:** 1.0.0 (Master Experiment Protocol)  
- **Tanggal:** 31 Oktober 2026  
- **Identifier:** `exp-plan-v11-adaptive-abstention`  
- **Status:** Approved Pre-Registered Protocol  
- **Protokol Inferensi Terkait:** `rev05` (`decisive_validation_experiment_v5.py`, `resilient_validation_engine.py`)  
- **Dataset Acuan:** `Lab-IS/IndoRad-VQA` via `manifest_rev05.csv` (SHA-256: `fb55fd75...`)  

---

## 1. Pendahuluan dan Tujuan Eksperimen

Eksperimen ini dirancang untuk memvalidasi secara empiris mekanisme keputusan operasional **Adaptive Abstention** pada Medical VLM. Jika selective prediction konvensional hanya menghasilkan pemeringkatan pasif (kurva Risk–Coverage dan metrik AURC), rencana eksperimen ini menetapkan protokol formal untuk:
1. Menentukan nilai ambang batas risiko ($\tau$) secara objektif dan bebas kebocoran pada Development Set.
2. Mengubah estimasi probabilitas risiko kontinu $R(x)$ menjadi keputusan biner diskrit: **Answer (Teruskan Jawaban)** versus **Abstain (Tahan untuk Radiolog)**.
3. Mengukur efektivitas penurunan risiko diagnostik (*selective risk mitigation*) pada Test Set independen di berbagai tingkat cakupan operasional (*coverage targets*).

---

## 2. Definisi Matematis Input, Sinyal Perantara, dan Output

### 2.1 Ruang Input Multimodal ($X$)
Setiap instansi evaluasi $i$ terdiri dari tripel data:
$$X_i = (I_i, Q_{\text{ID}, i}, Q_{\text{EN}, i})$$
di mana $I_i$ adalah citra radiologi kanonis, $Q_{\text{ID}, i}$ adalah pertanyaan bahasa Indonesia, dan $Q_{\text{EN}, i}$ adalah pertanyaan bahasa Inggris yang semantik ekuivalen.

### 2.2 Sinyal Fitur Perantara ($\mathbf{x}_i \in \mathbb{R}^3$)
Dari eksekusi inferensi model pada kondisi citra asli ($P_0$) dan citra kanvas hitam ($P_1$), diekstraksi tiga fitur perantara:
1. **Confidence / Candidate Preference ($p_{\text{orig\_ID}}$):**
   $$p_{\text{orig\_ID}, i} = P_{\text{cand}}(\hat{y}_{\text{ID}, i} \mid P_0, Q_{\text{ID}, i}) \in [0.5, 1.0]$$
2. **Directed Visual Deletion Sensitivity ($\Delta P_{\text{dir\_ID}}$):**
   $$\Delta P_{\text{dir\_ID}, i} = p_{\text{orig\_ID}, i} - P_{\text{cand}}(\hat{y}_{\text{ID}, i} \mid P_1, Q_{\text{ID}, i}) \in [-1.0, 1.0]$$
3. **Cross-Lingual Disagreement ($D_{\text{lang}}$):**
   $$D_{\text{lang}, i} = \mathbb{I}(\hat{y}_{\text{EN}, i} \neq \hat{y}_{\text{ID}, i}) \in \{0, 1\}$$

### 2.3 Estimasi Risiko Kontinu ($R(x) \in [0, 1]$)
Model regresi logistik teratur ($L_2$, $C=1.0$) yang dilatih pada Dev Set memetakan fitur terstandardisasi $\mathbf{z}_i$ menjadi estimasi risiko kesalahan:
$$R(x_i) = \sigma(w_0 + \mathbf{w}^T \mathbf{z}_i) = \frac{1}{1 + \exp(-(w_0 + w_1 z_{1, i} + w_2 z_{2, i} + w_3 z_{3, i}))}$$

### 2.4 Aturan Keputusan Biner ($\text{Decision}(x)$)
Diberikan threshold risiko $\tau \in [0, 1]$:
$$\text{Decision}(x_i) = \begin{cases} \text{Answer}, & \text{jika } R(x_i) \le \tau \\ \text{Abstain}, & \text{jika } R(x_i) > \tau \end{cases}$$

### 2.5 Luaran Akhir Sistem ($\hat{Y}_{\text{final}}$)
$$\hat{Y}_{\text{final}, i} = \begin{cases} \hat{y}_{\text{ID}, i}, & \text{jika } \text{Decision}(x_i) = \text{Answer} \\ \text{DEFERRED\_TO\_HUMAN\_RADIOLOGIST}, & \text{jika } \text{Decision}(x_i) = \text{Abstain} \end{cases}$$

---

## 3. Protokol Pemisahan Data & Pencegahan Kebocoran (Zero-Leakage Guarantee)

Kalibrasi dan pengujian dilakukan secara terisolasi tanpa tumpang-tindih:

```
                            PRIMARY COHORT (N = 1.122)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   CALIBRATION DEV SET                                   PRIMARY TEST SET
 323 Baris QA | 88 Klaster Citra                       799 Baris QA | 204 Klaster Citra
            │                                                     │
            ▼                                                     ▼
 1. Standarisasi Fitur z-score                          1. Terapkan (μ_dev, σ_dev)
    (μ_dev, σ_dev)                                      2. Evaluasi Estimasi Risiko R(x)
 2. Latih Model Risiko:                                 3. Terapkan Threshold Beku τ
    LR-conf, LR-visual, LR-full                         4. Hitung Realized Coverage,
 3. Tentukan Ambang Batas τ:                               Abstention Rate, Selective Risk,
    - Coverage-Targeted (50%-90%)                          dan Selective Accuracy.
    - Risk-Targeted (15%-25%)                                     │
            │                                                     ▼
            └──────────────── BEKUKAN THRESHOLD τ ───────────────┘
```

**Aturan Mutlak:**
- Parameter skalar $\boldsymbol{\mu}_{\text{dev}}$ dan $\boldsymbol{\sigma}_{\text{dev}}$ dibekukan murni dari Dev Set.
- Threshold $\tau$ dipilih 100% pada Dev Set. Test Set tidak boleh dilihat atau digunakan dalam menentukan $\tau$.
- Tidak ada klaster citra fisik kanonis (`canonical_image_id`) yang muncul di kedua split ($\mathcal{C}_{\text{dev}} \cap \mathcal{C}_{\text{test}} = \emptyset$).

---

## 4. Dua Strategi Penentuan Threshold

### 4.1 Strategi A: Coverage-Targeted Threshold Selection (Strategi Utama)
Menargetkan fraksi kasus yang dapat dijawab secara otomatis dalam alur kerja radiologi:
$$\Phi_{\text{target}} \in \{0.50, 0.60, 0.70, 0.80, 0.90\}$$

**Algoritma Kalibrasi Dev Set:**
1. Hitung $R(x_j)$ untuk seluruh $j \in \mathcal{D}_{\text{dev}}$ ($N_{\text{dev}} = 323$).
2. Tentukan $\tau(\Phi_{\text{target}})$ sebagai persentil risiko empiris:
   $$\tau = \text{Quantile}\left(\{R(x_j)\}_{j=1}^{N_{\text{dev}}}, \Phi_{\text{target}}\right)$$
3. Pada Test Set ($N_{\text{test}} = 799$), hitung himpunan kasus yang dijawab:
   $$\mathcal{A}_{\text{test}} = \{i \in \mathcal{D}_{\text{test}} \mid R(x_i) \le \tau\}$$
4. Hitung cakupan terwujud (*realized coverage*):
   $$\Phi_{\text{realized}} = \frac{|\mathcal{A}_{\text{test}}|}{N_{\text{test}}}$$
5. Hitung laju penahanan (*abstention rate*):
   $$\alpha_{\text{realized}} = 1.0 - \Phi_{\text{realized}}$$
6. Hitung risiko selektif aktual (*actual selective risk*):
   $$r_{\text{selective}} = \frac{1}{|\mathcal{A}_{\text{test}}|} \sum_{i \in \mathcal{A}_{\text{test}}} \mathbb{I}(\hat{y}_{\text{ID}, i} \neq y_{\text{ID}, i}^*)$$

### 4.2 Strategi B: Risk-Targeted Threshold Selection (Strategi Sekunder)
Menargetkan batas atas toleransi kesalahan medis pada kasus yang diteruskan:
$$r_{\text{target}} \in \{0.15, 0.20, 0.25\} \quad (15\%, 20\%, 25\%)$$

**Algoritma Kalibrasi Dev Set:**
1. Urutkan sampel Dev Set berdasarkan $R(x_j)$ secara menaik: $R_{(1)} \le R_{(2)} \le \dots \le R_{(N_{\text{dev}})}$.
2. Hitung risiko kumulatif dev: $\text{risk}_{\text{dev}}(k) = \frac{1}{k} \sum_{m=1}^k \mathbb{I}(\hat{y}_{(m)} \neq y_{(m)}^*)$.
3. Temukan indeks terbesar $k^* = \max \{k \in \{1, \dots, N_{\text{dev}}\} \mid \text{risk}_{\text{dev}}(k) \le r_{\text{target}}\}$.
4. Tetapkan threshold $\tau = R_{(k^*)}$.
5. Evaluasi pada Test Set: amati cakupan yang dapat dipertahankan ($\Phi_{\text{test}}$) dan evaluasi apakah risiko aktual test set ($r_{\text{test}}$) berada di sekitar batas yang diharapkan.
6. Catat fenomena *empirical risk slippage* secara objektif.

---

## 5. Estimator yang Dibandingkan

Seluruh evaluasi dijalankan secara paralel pada 4 estimator:

1. **`M1_direct` (Confidence Baseline Langsung):**  
   Skor risiko: $R(x) = 1.0 - p_{\text{orig\_ID}}$.
2. **`LR_conf` (Confidence Baseline Terlatih):**  
   Model logistik 1-fitur: $\mathbf{x} = [p_{\text{orig\_ID}}]$.
3. **`LR_visual` (Multimodal Visual Sensitivity):**  
   Model logistik 2-fitur: $\mathbf{x} = [p_{\text{orig\_ID}}, \Delta P_{\text{dir\_ID}}]$.
4. **`LR_full` (Proposed Multimodal Cross-Lingual Model):**  
   Model logistik 3-fitur: $\mathbf{x} = [p_{\text{orig\_ID}}, \Delta P_{\text{dir\_ID}}, D_{\text{lang}}]$.

---

## 6. Protokol Validasi Statistik

### 6.1 Uji Signifikansi Perbedaan Selective Risk
Untuk membandingkan apakah selective risk `LR_full` lebih rendah secara signifikan daripada `M1_direct` pada tingkat coverage yang sama:
- **Paired Cluster Permutation Test ($B=1.000$):**  
  Pembalikan tanda Rademacher $\epsilon_c \in \{-1, +1\}$ pada tingkat klaster citra fisik kanonis ($K=204$). Menghitung empirical two-sided p-value atas selisih selective risk:
  $$\Delta r = r(\text{LR\_full}) - r(\text{M1})$$
- **Holm-Bonferroni Step-Down Procedure:**  
  Mengontrol Family-Wise Error Rate (FWER $\le 0.05$) pada grid coverage yang dievaluasi.

### 6.2 Interval Kepercayaan 95%
Menggunakan **Paired Cluster Bootstrap ($B=1.000$)** untuk menghitung persentil $[2.5\%, 97.5\%]$ bagi realized coverage dan selective risk.

---

## 7. Rencana Modifikasi Kode dan Modul Baru

Eksperimen ini memanfaatkan berkas checkpoint inferensi yang telah selesai dieksekusi 100% pada Kaggle Dual-T4 (`live_validation_checkpoint_*.csv`). **Tidak diperlukan inferensi GPU tambahan.**

### Berkas Baru yang Akan Ditambahkan:
1. **`main/scripts/adaptive_abstention_engine.py`**:
   - Memuat data Dev ($N=323$) dan Test ($N=799$).
   - Mengalibrasi threshold $\tau$ (Coverage-targeted & Risk-targeted).
   - Mengeksekusi penahanan biner dan menghitung seluruh metrik.
   - Menghasilkan tabel luaran CSV dan grafik perbandingan.
2. **`tests/test_adaptive_abstention.py`**:
   - Unit test otomatis untuk verifikasi integritas formula, orientasi skor, dan jaminan non-kebocoran data.

---
*EXPERIMENT_PROTOCOL.md disahkan sebagai panduan teknis operasional penahanan adaptif v11.0.*
