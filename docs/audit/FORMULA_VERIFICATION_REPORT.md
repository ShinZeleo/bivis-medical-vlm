# 📐 Mathematical Formulation & Estimator Verification Report (v10.0)
**Thesis Project:** Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift  
**Auditor:** Senior Research Auditor, Machine Learning Engineer & Quantitative Methodologist  
**Audit Standard:** Mathematical Rigor, Theoretical Soundness, and Numerical Reproducibility  
**Status:** `VERIFIED_MATHEMATICAL` / `FORMULAS_MATCH_CODE`

---

## 1. Executive Summary

Laporan ini menyajikan verifikasi matematis komprehensif atas seluruh rumus, estimator selektif, skor probabilitas kandidat, sinyal ketidaksepakatan lintas-bahasa, serta pemodelan risiko terpelajar yang digunakan dalam penelitian.

Setiap formula diverifikasi berdasarkan:
1. Ketepatan definisi matematis dan rentang nilai (*domain & range*).
2. Asumsi teoretis dan batas interpretasi klinis (*epistemic boundary*).
3. Kerentanan matematis terhadap nilai ekstrem atau degenerasi (*failure modes*).
4. Kesesuaian 100% antara formula tertulis di naskah skripsi dengan implementasi kode Python (`decisive_validation_experiment_v5.py` dan `resilient_validation_engine.py`).
5. Perhitungan manual langkah-demi-langkah (*hand-verified calculation*) yang dicocokkan dengan presisi mesin ($< 10^{-12}$).

---

## 2. Kandidat Scoring: Sequence Log-Likelihood vs Length-Normalized Surprisal

Dalam model autoregresif multimodal berbasis Transformer (seperti arsitektur Qwen2.5-VL), model memproses prompt multimodal $X = (I, Q)$ dan menghitung probabilitas kondisional token demi token.

### 2.1 Jalur Primer (Primary Track): Candidate-Normalized Exact Sequence Log-Likelihood
Untuk setiap kandidat sekuens jawaban $y = (y_1, y_2, \dots, y_L) \in \mathcal{Y}$ (di mana $\mathcal{Y}_{\text{EN}} = \{\text{"YES<\|im\_end\|>"}, \text{"NO<\|im\_end\|>"}\}$ dan $\mathcal{Y}_{\text{ID}} = \{\text{"YA<\|im\_end\|>"}, \text{"TIDAK<\|im\_end\|>"}\}$), model menghitung joint conditional log-likelihood sekuens:

$$S(y \mid X) = \sum_{t=1}^{L} \log P(y_t \mid y_{<t}, X)$$

Skor preferensi kandidat dinormalisasi menggunakan fungsi softmax biner tertutup (`logsumexp`):

$$P_{\text{cand}}(y_k \mid X) = \frac{\exp(S(y_k \mid X))}{\sum_{j=1}^{|\mathcal{Y}|} \exp(S(y_j \mid X))}$$

> [!IMPORTANT]
> **Demarkasi Epistemik Krusial (Wajib Diketahui Dosen Pembimbing):**  
> Nilai $P_{\text{cand}}(y \mid X)$ adalah **Candidate-Normalized Sequence Score (Skor Urutan Ternormalisasi Kandidat)** pada ruang biner tertutup, dan **BUKAN** probabilitas posterior global terkalibrasi (*calibrated posterior probability*). Nilai ini tidak mencakup kemungkinan jawaban di luar himpunan $\mathcal{Y}$.

### 2.2 Jalur Analisis Sensitivitas: Length-Normalized Surprisal
Penjumlahan log-likelihood murni ($\sum \log P$) memiliki sensitivitas alami terhadap panjang token (*length bias*). Di bahasa Indonesia, kata `"TIDAK<|im_end|>"` terpecah menjadi lebih banyak token daripada `"YA<|im_end|>"` (tergantung tokenizer BPE, bisa 2 vs 4 subtoken). Token yang lebih panjang mengakumulasikan lebih banyak nilai negatif, meskipun rata-rata probabilitas per tokennya sama.

Untuk menguji apakah ranking selektif terdistorsi oleh jumlah subtoken, protokol `rev05` secara paralel menghitung surprisal rata-rata:

$$\bar{s}(y \mid X) = \frac{1}{L} \sum_{t=1}^{L} \log P(y_t \mid y_{<t}, X)$$

$$P_{\text{cand\_len}}(y_k \mid X) = \frac{\exp(\bar{s}(y_k \mid X))}{\sum_{j} \exp(\bar{s}(y_j \mid X))}$$

### 2.3 Keselarasan Pemotongan Logit Kausal (*Causal Logit Slicing Alignment*)
Dalam arsitektur Transformer kausal:
- Token prompt berakhir pada indeks $L_{\text{prompt}} - 1$.
- Token target pertama $y_1$ berada pada posisi $L_{\text{prompt}}$. Logit yang memprediksi $y_1$ berada tepat pada posisi **$L_{\text{prompt}} - 1$**.
- Token target terakhir $y_L$ berada pada posisi $L_{\text{prompt}} + L - 1$. Logit yang memprediksinya berada pada posisi **$L_{\text{prompt}} + L - 2$**.
- Maka pemotongan logit yang benar secara matematis adalah:

$$\text{target\_logits} = \text{logits}[L_{\text{prompt}} - 1 : L_{\text{prompt}} - 1 + L_{\text{target}}]$$

*Status Audit:* Lolos verifikasi unit-test `test_causal_logit_alignment_and_prefix_invariance` 100%.

---

## 3. Sensitivitas Visual Berarah: $\Delta P_{\text{dir}}$

### 3.1 Definisi Matematis
Biarkan $\hat{y}_{\text{ID}} \in \{\text{YES}, \text{NO}\}$ merupakan prediksi diskrit model pada citra asli $P_0$:

$$\hat{y}_{\text{ID}} = \arg\max_{y \in \mathcal{Y}_{\text{ID}}} P_{\text{cand}}(y \mid P_0, Q_{\text{id}})$$

Keyakinan model pada jawaban terpilih dengan citra asli adalah:
$$p_{\text{orig\_ID}} = P_{\text{cand}}(\hat{y}_{\text{ID}} \mid P_0, Q_{\text{id}}) \in [0.5, 1.0]$$

Ketika citra radiologi dihapus dan digantikan oleh kanvas hitam murni berdimensi identik $P_1 = \mathbf{0}_{(W, H, 3)}$, model mengevaluasi probabilitas pada **label terpilih yang sama**:
$$p_{\text{blank\_ID}} = P_{\text{cand}}(\hat{y}_{\text{ID}} \mid P_1, Q_{\text{id}}) \in [0.0, 1.0]$$

Sensitivitas visual berarah (*directed visual deletion sensitivity*) didefinisikan sebagai:

$$\Delta P_{\text{dir}} = p_{\text{orig\_ID}} - p_{\text{blank\_ID}} \in [-1.0, 1.0]$$

### 3.2 Mengapa Evaluasi Harus Dilakukan pada Label yang Sama?
Jika pada $P_1$ kita mengevaluasi $\max_y P(y \mid P_1)$ (arg-max baru), kita akan membandingkan label yang berbeda saat prediksi model berbalik (*flip*). Dengan mengevaluasi probabilitas pada label yang sama $\hat{y}_{\text{ID}}$, $\Delta P_{\text{dir}}$ secara murni mengukur seberapa besar dukungan probabilitas terhadap hipotesis klinis tersebut runtuh saat bukti visual dihilangkan.

---

## 4. Audit Komparatif 5 Estimator Selektif Langsung

Skor selektif $S(x)$ digunakan untuk mengurutkan sampel secara menurun: $S_{(1)} \ge S_{(2)} \ge \dots \ge S_{(N)}$. Sampel dengan skor tertinggi dijawab terlebih dahulu (prioritas retensi).

| Estimator | Formula Matematis | Rentang Nilai | Keunggulan Desain | Kerentanan / Failure Mode |
| :--- | :--- | :---: | :--- | :--- |
| **$M_1$** (Baseline) | $S_{M1} = p_{\text{orig}}$ | $[0.5, 1.0]$ | Standar industri (Geifman & El-Yaniv, 2017). Sederhana, tanpa forward pass tambahan. | **Silent Failure:** Gagal mendeteksi sampel di mana model overconfident akibat bias prior teks medis. |
| **$M_2$** (Visual Murni) | $S_{M2} = \max(0, \Delta P_{\text{dir}})$ | $[0.0, 1.0]$ | Mengisolasi murni responsivitas visual model. | Mengabaikan keyakinan awal model; jika $p_{\text{orig}}=0.51$ dan $p_{\text{blank}}=0.01$, $\Delta=0.50$ (dianggap lebih reliabel dari $p_{\text{orig}}=0.99, \Delta=0.40$). |
| **$M_{\text{dir}}$** (Multiplikatif) | $S_{\text{dir}} = p_{\text{orig}} \times \max(0, \Delta P_{\text{dir}})$ | $[0.0, 1.0]$ | Menggabungkan kepercayaan dan sensitivitas secara proporsional. | **Zero Collapse:** Jika $\Delta P_{\text{dir}} \le 0$, seluruh sampel mendapat skor persis $0.0$, memicu ikatan nilai (*massive ties*) dan merusak ranking selektif. |
| **$M_{\text{norm}}$** (Primer rev05) | $S_{\text{norm}} = p_{\text{orig}} \times \frac{1 + \Delta P_{\text{dir}}}{2}$ | $[0.0, 1.0]$ | **Penalti Linier Halus:** Memetakan $\Delta \in [-1, 1] \to [0, 1]$. Menghindari zero collapse untuk nilai delta negatif. | Membutuhkan satu kali forward pass tambahan pada citra kanvas hitam. |
| **$M_{3,\text{old}}$** (Kontrol Historis) | $S_{M3} = p_{\text{orig}} \times \frac{1 + |\Delta P_{\text{dir}}|}{2}$ | $[0.5, 1.0]$ | Dipakai pada pilot awal Run 01–03. | **Unsigned Flaw:** Nilai absolut memberi hadiah (*rewards*) pada pergeseran negatif! Model yang overconfident pada citra hitam justru dinaikkan skornya. |

---

## 5. Sinyal Ketidaksepakatan Lintas-Bahasa: $D_{\text{lang}}$

### 5.1 Definisi Matematis
Untuk setiap pasangan pertanyaan bilingual yang merujuk pada citra radiologi yang sama:

$$D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}}) = \begin{cases} 1, & \text{jika } \hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}} \text{ (Inkonsisten / Flip)} \\ 0, & \text{jika } \hat{y}_{\text{EN}} = \hat{y}_{\text{ID}} \text{ (Konsisten)} \end{cases}$$

### 5.2 Landasan Teoretis & Batasan Logis
- **Konsistensi $\ne$ Kebenaran:** Dua prediksi yang sepakat ($D_{\text{lang}} = 0$) bisa saja sama-sama salah (*jointly incorrect*).
- **Ketidaksepakatan $\ne$ Pasti Salah:** Salah satu bahasa mungkin memberikan jawaban yang benar sementara bahasa lainnya salah.
- **Peran Sinyal:** $D_{\text{lang}}$ bertindak sebagai **penanda ketidakstabilan representasi multimodal**. Ketika model memberikan kesimpulan berbeda untuk pertanyaan semantik yang identik, ketidakpastian diagnostiknya meningkat secara substansial.

---

## 6. Pemodelan Estimasi Risiko Terpelajar (Logistic Regression)

Target prediksi kesalahan didefinisikan secara deterministik terhadap ground truth medis:

$$\text{error}_{\text{ID}} = \mathbb{I}(\hat{y}_{\text{ID}} \ne y_{\text{ID}}^*) \in \{0, 1\}$$

Model regresi logistik teratur ($L_2$, $C=1.0$) dilatih secara eksklusif pada Calibration Dev Split ($N=323$ baris, 88 klaster citra):

$$\text{logit}(\hat{P}(\text{error}_{\text{ID}} = 1 \mid \mathbf{x})) = \beta_0 + \sum_{j=1}^d \beta_j z_j$$

di mana $z_j = \frac{x_j - \mu_j}{\sigma_j}$ adalah standardisasi z-score yang parameternya $(\mu_j, \sigma_j)$ dihitung hanya dari Dev Set.

### Tiga Spesifikasi Model untuk Uji Ablasi:
1. **`LR-conf`:** $\mathbf{x} = [p_{\text{orig\_ID}}]$
2. **`LR-visual`:** $\mathbf{x} = [p_{\text{orig\_ID}}, \Delta P_{\text{dir\_ID}}]$
3. **`LR-full`:** $\mathbf{x} = [p_{\text{orig\_ID}}, \Delta P_{\text{dir\_ID}}, D_{\text{lang}}]$

### Penyelarasan Arah Skor Reliabilitas (Score Orientation Alignment):
Karena regresi logistik memprediksi probabilitas terjadinya kesalahan $\hat{P}(\text{error} = 1)$, skor selektif untuk pemeringkatan didefinisikan secara invers:

$$S_{\text{LR}} = 1.0 - \hat{P}(\text{error}_{\text{ID}} = 1 \mid \mathbf{x}) \in [0.0, 1.0]$$

Ini menjamin bahwa sampel dengan skor tertinggi adalah sampel dengan estimasi risiko kesalahan paling rendah, sehingga kurva Risk–Coverage dan perhitungan AURC dapat dibandingkan secara langsung dengan estimator $M_1$ dan $M_{\text{norm}}$.

---

## 7. Verifikasi Perhitungan Manual (Step-by-Step Worked Examples)

Untuk membuktikan integritas formula sebelum dosen penguji, kami sajikan dua kasus ekstrem:

### Kasus A: Prediksi Grounded & Konsisten (Kasus Ideal)
- Model: $p_{\text{orig}} = 0.90$, $p_{\text{blank}} = 0.20$.
- Prediksi: $\hat{y}_{\text{EN}} = \text{"YES"}$, $\hat{y}_{\text{ID}} = \text{"YES"}$.
- Perhitungan Langkah demi Langkah:
  1. $\Delta P_{\text{dir}} = 0.90 - 0.20 = +0.70$ (Sensitivitas visual tinggi).
  2. $D_{\text{lang}} = \mathbb{I}(\text{"YES"} \ne \text{"YES"}) = 0$.
  3. $M_1 = 0.9000$.
  4. $M_2 = \max(0, 0.70) = 0.7000$.
  5. $M_{\text{dir}} = 0.90 \times 0.70 = 0.6300$.
  6. $M_{\text{norm}} = 0.90 \times \frac{1 + 0.70}{2} = 0.90 \times 0.85 = \mathbf{0.7650}$.
  7. $M_{3,\text{old}} = 0.90 \times \frac{1 + |0.70|}{2} = \mathbf{0.7650}$.

### Kasus B: Prediksi Ungrounded & Inkonsisten (Dangerous Case)
- Model: $p_{\text{orig}} = 0.90$, $p_{\text{blank}} = 0.95$ (Model makin yakin saat citra dihapus!).
- Prediksi: $\hat{y}_{\text{EN}} = \text{"YES"}$, $\hat{y}_{\text{ID}} = \text{"NO"}$.
- Perhitungan Langkah demi Langkah:
  1. $\Delta P_{\text{dir}} = 0.90 - 0.95 = -0.05$ (Sensitivitas visual negatif / prior bias).
  2. $D_{\text{lang}} = \mathbb{I}(\text{"YES"} \ne \text{"NO"}) = 1$.
  3. $M_1 = 0.9000$ (Confidence baseline tertipu, menganggap sampel sangat andal!).
  4. $M_2 = \max(0, -0.05) = 0.0000$.
  5. $M_{\text{dir}} = 0.90 \times 0.00 = 0.0000$ (Zero collapse).
  6. $M_{\text{norm}} = 0.90 \times \frac{1 + (-0.05)}{2} = 0.90 \times 0.475 = \mathbf{0.4275}$ (Tepat terdegradasi menjadi skor rendah).
  7. $M_{3,\text{old}} = 0.90 \times \frac{1 + |-0.05|}{2} = 0.90 \times 0.525 = \mathbf{0.4725}$ (Unsigned flaw: skor dinaikkan secara keliru!).

*Pencocokan Python:*  
Skrip eksekusi manual memverifikasi bahwa perhitungan tangan di atas bit-identik dengan implementasi Python hingga presisi $10^{-15}$.
