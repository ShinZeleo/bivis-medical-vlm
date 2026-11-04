# Algorithmic Specification of Bilingual Visual-Sensitivity-Aware Selective Prediction (Bi-ViS)

**Target Document:** `docs/manuscript/METHOD_ALGORITHM.md`  
**Target Publication Scope:** SINTA 2 / International Journal Reproducibility & Methodology Section  
**Status:** `VERIFIED_ALGORITHM_SPECIFICATION`  
**Date:** 2026-11-01  

---

## 1. Overview & System Architecture

Dokumen ini mendokumentasikan spesifikasi prosedural dan algoritma formal dari metode **Bi-ViS** (*Bilingual Visual-Sensitivity-Aware Selective Prediction*). 

Bi-ViS beroperasi dalam dua fase:
1. **Fase Kalibrasi Offline (Development Phase):** Melatih *standardizer* dan regresi logistik teratur pada Development Set ($N=323$, 88 klaster citra independen) menggunakan 5-Fold Group Cross-Validation untuk mengestimasi parameter bobot $\boldsymbol{\beta}$ dan ambang batas seleksi $\tau$.
2. **Fase Inferensi & Penolakan Online (Runtime Inference Phase):** Menerima kueri radiologi multimodal baru, mengekstraksi tiga sinyal informasi, menghitung skor keandalan selektif $S_{\text{Bi-ViS}}$, dan menerapkan aturan penolakan selektif (*selective abstention*).

---

## 2. Pseudocode Lengkap Algoritma Bi-ViS

### Algoritma 1: End-to-End Runtime Selective Prediction (Bi-ViS)

```text
=========================================================================================================
ALGORITMA 1: Bi-ViS Runtime Inference and Selective Prediction Protocol
=========================================================================================================
INPUT:
  - I_orig            : Citra radiologi asli dalam format RGB (H x W x 3)
  - Q_ID              : Pertanyaan visual klinis dalam Bahasa Indonesia
  - Q_EN              : Pertanyaan visual klinis padanan dalam Bahasa Inggris
  - f_theta           : Vision-Language Model autoregresif yang telah dibekukan (frozen)
  - Scaler (mu, sigma): Parameter standardisasi fitur yang difit pada Dev Set
  - clf (beta, beta_0): Model regresi logistik teratur L2 yang dilatih pada Dev Set
  - tau               : Ambang batas seleksi (ditetapkan via target coverage kappa atau target risk rho)

OUTPUT:
  - y_hat_ID          : Jawaban diagnostik Bahasa Indonesia terpilih ("Ya" atau "Tidak")
  - S_BiViS           : Skor keandalan selektif kontinu in [0, 1]
  - decision          : Keputusan klinis in {ACCEPT, ABSTAIN}

PROSES:
  -------------------------------------------------------------------------------------------------------
  Langkah 1: Inferensi Asli Bahasa Indonesia (Primary Diagnostic Inference)
  -------------------------------------------------------------------------------------------------------
  1.1. Bentuk prompt multimodal Bahasa Indonesia:
         prompt_ID = ConstructPrompt(I_orig, Q_ID, lang="ID")
  1.2. Hitung exact sequence log-likelihood untuk setiap kandidat c in {"Ya", "Tidak"}:
         P(c | I_orig, Q_ID) = exp( (1 / |c|) * sum_{t=1}^{|c|} log P(w_t | w_<t, prompt_ID) )
  1.3. Tentukan prediksi sekuens terpilih:
         y_hat_ID = argmax_{c in {"Ya", "Tidak"}} P(c | I_orig, Q_ID)
  1.4. Ekstraksi Fitur 1 (Indonesian Original Confidence):
         M_1 = P(y_hat_ID | I_orig, Q_ID)

  -------------------------------------------------------------------------------------------------------
  Langkah 2: Inferensi Kontrafaktual Kanvas Hitam (Visual Deletion Sensitivity)
  -------------------------------------------------------------------------------------------------------
  2.1. Bangkitkan kanvas hitam kosong berdimensi identik:
         I_blank = ZerosLike(I_orig)  // Larik nol piksel RGB
  2.2. Bentuk prompt kontrafaktual Bahasa Indonesia:
         prompt_blank = ConstructPrompt(I_blank, Q_ID, lang="ID")
  2.3. Evaluasi probabilitas sekuens kandidat terpilih y_hat_ID pada kanvas kosong:
         P(y_hat_ID | I_blank, Q_ID) = exp( (1 / |y_hat_ID|) * sum_{t=1}^{|y_hat_ID|} log P(w_t | w_<t, prompt_blank) )
  2.4. Ekstraksi Fitur 2 (Directional Visual Deletion Sensitivity):
         Delta_P_dir = M_1 - P(y_hat_ID | I_blank, Q_ID)

  -------------------------------------------------------------------------------------------------------
  Langkah 3: Inferensi Bahasa Inggris & Ekstraksi Ketidaksepakatan (Cross-Lingual Disagreement)
  -------------------------------------------------------------------------------------------------------
  3.1. Bentuk prompt multimodal Bahasa Inggris:
         prompt_EN = ConstructPrompt(I_orig, Q_EN, lang="EN")
  3.2. Evaluasi exact sequence log-likelihood untuk kandidat c in {"Yes", "No"}:
         P(c | I_orig, Q_EN) = exp( (1 / |c|) * sum_{t=1}^{|c|} log P(w_t | w_<t, prompt_EN) )
  3.3. Tentukan prediksi sekuens Bahasa Inggris terpilih:
         y_hat_EN = argmax_{c in {"Yes", "No"}} P(c | I_orig, Q_EN)
  3.4. Kanonikalisasi label biner:
         label_ID = 1 if y_hat_ID == "Ya" else 0
         label_EN = 1 if y_hat_EN == "Yes" else 0
  3.5. Ekstraksi Fitur 3 (Bilingual Disagreement Indicator):
         D_lang = 1 if label_EN != label_ID else 0

  -------------------------------------------------------------------------------------------------------
  Langkah 4: Standardisasi Fitur & Estimasi Risiko Multivariat
  -------------------------------------------------------------------------------------------------------
  4.1. Susun vektor fitur mentah:
         x_raw = [M_1, Delta_P_dir, D_lang]
  4.2. Terapkan z-score transformation menggunakan parameter Dev Set:
         z_1 = (M_1 - mu_1) / sigma_1
         z_2 = (Delta_P_dir - mu_2) / sigma_2
         z_3 = (D_lang - mu_3) / sigma_3
         z = [z_1, z_2, z_3]
  4.3. Hitung estimasi probabilitas kesalahan diagnostik:
         logit = beta_0 + (beta_1 * z_1) + (beta_2 * z_2) + (beta_3 * z_3)
         P_error = 1.0 / (1.0 + exp(-logit))

  -------------------------------------------------------------------------------------------------------
  Langkah 5: Penentuan Skor Keandalan Selektif & Keputusan Klinis
  -------------------------------------------------------------------------------------------------------
  5.1. Hitung skor keandalan selektif:
         S_BiViS = 1.0 - P_error
  5.2. Terapkan aturan seleksi (Selective Decision Gate):
         if S_BiViS >= tau:
             decision = "ACCEPT"  // Prediksi reliabel; keluarkan y_hat_ID ke klinisi
         else:
             decision = "ABSTAIN" // Prediksi berisiko tinggi; teruskan ke radiolog manusia

  RETURN y_hat_ID, S_BiViS, decision
=========================================================================================================
```

---

### Algoritma 2: Prosedur Kalibrasi Ambang Batas Selektif Offline (Dev Set)

```text
=========================================================================================================
ALGORITMA 2: Offline Dev-Set Training & Adaptive Abstention Calibration Protocol
=========================================================================================================
INPUT:
  - D_dev             : Dataset Development (N=323, terbagi dalam K=88 klaster canonical_image_id)
  - target_kappa      : Target cakupan operasional (misal: kappa in [0.70, 0.95])
  - target_rho        : Target toleransi risiko kesalahan klinis (misal: rho in [0.15, 0.30])

OUTPUT:
  - Scaler            : Fitted StandardScaler
  - clf               : Fitted LogisticRegression (C=1.0, L2 penalty)
  - tau_kappa         : Ambang batas berbasis cakupan (Coverage-targeted threshold)
  - tau_rho           : Ambang batas berbasis toleransi risiko (Risk-targeted threshold)

PROSES:
  1. Ekstraksi fitur mentah untuk seluruh i in D_dev:
       X_dev_raw = [[M_1^(i), Delta_P_dir^(i), D_lang^(i)] for i in D_dev]
       y_dev_err = [1 if y_hat_ID^(i) != y_true^(i) else 0 for i in D_dev]
       groups    = [canonical_image_id^(i) for i in D_dev]

  2. Kalibrasi Out-of-Fold (OOF) via 5-Fold GroupKFold:
       Inisialisasi array OOF_scores berukuran |D_dev|
       Bagi D_dev menjadi 5 fold berbasis groups (tumpang-tindih klaster antar fold = 0%)
       Untuk setiap fold k in {1, ..., 5}:
         - Fit Scaler_k pada fold latih
         - Fit LogisticRegression_k (C=1.0) pada fold latih
         - Transformasi fold validasi menggunakan Scaler_k
         - Prediksi risiko OOF: P_err_val = LogisticRegression_k.predict_proba(val)[:, 1]
         - Simpan skor keandalan OOF: OOF_scores[val] = 1.0 - P_err_val

  3. Fit Model Final pada 100% Data Dev:
       Scaler.fit(X_dev_raw)
       X_dev_scaled = Scaler.transform(X_dev_raw)
       clf.fit(X_dev_scaled, y_dev_err)

  4. Kalibrasi Ambang Batas Berbasis Target Cakupan (Coverage-Targeted):
       Urutkan sampel Dev berdasarkan OOF_scores secara menurun:
         idx_sorted = argsort(OOF_scores)[::-1]
       Pilih indeks batas:
         k_cutoff = floor(target_kappa * |D_dev|)
       Tetapkan ambang batas:
         tau_kappa = OOF_scores[idx_sorted[k_cutoff]]

  5. Kalibrasi Ambang Batas Berbasis Target Risiko (Risk-Targeted):
       Cari nilai tau tertinggi sedemikian rupa sehingga selective risk pada sampel terjawab <= target_rho:
         tau_rho = min_{tau} { tau | (sum_{i: OOF_scores[i] >= tau} y_dev_err[i]) / (sum_{i: OOF_scores[i] >= tau} 1) <= target_rho }

  RETURN Scaler, clf, tau_kappa, tau_rho
=========================================================================================================
```

---

## 3. Pemetaan Kode Program & Ketertelusuran Repositori (Traceability)

Seluruh logika matematika dan algoritma di atas terimplementasi secara presisi pada skrip produksi repositori:

| Modul Algoritma | File Sumber Aktual | Fungsi & Lokasi Baris |
| :--- | :--- | :--- |
| **Exact Sequence Log-Likelihood** | [`src/vlm_research/evaluation/scoring.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/src/vlm_research/evaluation/scoring.py) | `compute_exact_sequence_log_likelihood()` (Baris 35–85) |
| **Pembangkitan Kanvas Hitam** | [`src/vlm_research/data/image_utils.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/src/vlm_research/data/image_utils.py) | `generate_blank_canvas_image()` (Baris 20–45) |
| **Ekstraksi Sinyal $\Delta P_{\text{dir}}$ & $D_{\text{lang}}$** | [`scripts/resilient_validation_engine.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/scripts/resilient_validation_engine.py) | `train_and_evaluate_risk_ablation()` (Baris 638–649) |
| **Training & GroupKFold OOF Dev** | [`scripts/resilient_validation_engine.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/scripts/resilient_validation_engine.py) | `train_and_evaluate_risk_ablation()` (Baris 654–699) |
| **Discrete Riemann AURC** | [`scripts/resilient_validation_engine.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/scripts/resilient_validation_engine.py) | `calculate_discrete_aurc()` (Baris 190–240) |
| **Coverage & Risk Abstention** | [`scripts/adaptive_abstention_engine.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/scripts/adaptive_abstention_engine.py) | `calibrate_and_evaluate_coverage_targeted()` & `calibrate_and_evaluate_risk_targeted()` (Baris 70–210) |
| **Unit Test Governance** | [`tests/test_adaptive_abstention.py`](file:///c:/.PUSAT/Project/Kuliah/VLM/tests/test_adaptive_abstention.py) | `test_coverage_targeted_invariants()` & `test_risk_targeted_invariants()` |

---

## 4. Analisis Kompleksitas Komputasi & Latensi Inferensi

### 4.1. Jumlah Forward Passes per Kueri
Untuk setiap kueri VQA, Bi-ViS memerlukan tepat **3 kali evaluasi forward pass** model VLM:
1. $1\times$ forward pass untuk pasangan `(I_orig, Q_ID)` $\rightarrow$ menghasilkan $\hat{y}_{\text{ID}}$ dan $M_1$.
2. $1\times$ forward pass untuk pasangan `(I_blank, Q_ID)` $\rightarrow$ menghasilkan $P(y_{\text{ID}} \mid I_{\text{blank}})$ dan $\Delta P_{\text{dir}}$.
3. $1\times$ forward pass untuk pasangan `(I_orig, Q_EN)` $\rightarrow$ menghasilkan $\hat{y}_{\text{EN}}$ dan $D_{\text{lang}}$.

### 4.2. Overhead Ekstraksi & Estimasi Risiko
- **Komputasi Logistik:** Vektor fitur hanya berdimensi 3 ($d=3$). Operasi perkalian titik $\mathbf{w}^T \mathbf{z} + b_0$ dan evaluasi fungsi sigmoid membutuhkan waktu komputasi **$< 0.05 \text{ milidetik}$** pada CPU standar.
- **Biaya Retraining:** **Nol (Zero Training Cost).** Model dasar VLM ($7\text{B}$ parameter) dibekukan sepenuhnya (*frozen*), mengeliminasi kebutuhan GPU ratusan jam untuk fine-tuning LoRA.

### 4.3. Justifikasi Trade-Off Klinis (Safety over Latency)
Meskipun 3 kali forward pass melipatgandakan waktu inferensi sebesar $3\times$ (sekitar $1.5\text{--}3.0$ detik tambahan per kueri pada 1 GPU RTX A6000), kompensasi tersebut sangat dapat diterima dalam alur kerja radiologi medis (*asynchronous radiology workflow*):
- Mencegah kesalahan diagnosis fatal akibat halusinasi atau degradasi pergeseran bahasa.
- Menjamin penurunan risiko diagnostik hingga $5.16$ persentase poin pada cakupan $80\%$, memberikan jaminan keselamatan pasien yang jauh lebih tinggi daripada inferensi cepat namun rentan salah.
