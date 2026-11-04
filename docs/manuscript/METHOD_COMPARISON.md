# Methodological Comparison: Bi-ViS vs. Baseline and Alternative Selective Estimators

**Target Document:** `docs/manuscript/METHOD_COMPARISON.md`  
**Target Publication Scope:** SINTA 2 / International Journal Results and Discussion Section  
**Status:** `EMPIRICALLY_VERIFIED`  
**Date:** 2026-11-01  

---

## 1. Taksonomi Estimator yang Dievaluasi

Untuk memahami keunggulan dan kelemahan masing-masing pendekatan secara komprehensif, lima estimator risiko selektif diperbandingkan dalam kerangka metodologis yang sama pada benchmark IndoRad-VQA ($N=799$ Test set, 204 klaster citra):

1. **Baseline $M_1$ (Softmax Sequence Confidence):**
   - Pendekatan standar *Maximum Softmax Probability* (Geifman & El-Yaniv, 2017) berbasis token log-likelihood.
2. **Hipotesis Primer Historis $M_{\text{norm}}$ (Length-Normalized Sensitivity Ratio):**
   - Formula heuristik rasio aritmatika tanpa bobot dari proposal awal:
     $$M_{\text{norm}} = \frac{P_{\text{orig}}(\hat{y})}{P_{\text{blank}}(\hat{y}) + \epsilon}$$
3. **Varian Ablasi Terpelajar 1: $LR_{\text{conf}}$ (Calibrated Single-Feature):**
   - Regresi logistik teratur pada Development Set hanya menggunakan fitur keyakinan $M_1$.
4. **Varian Ablasi Terpelajar 2: $LR_{\text{visual}}$ (Confidence + Visual Sensitivity):**
   - Regresi logistik teratur menggunakan dua fitur: $[M_1, \Delta P_{\text{dir}}]$.
5. **Metode Pengembangan Terpilih: Bi-ViS ($LR_{\text{full}}$):**
   - Regresi logistik teratur menggunakan tiga fitur multimodal lengkap: $[M_1, \Delta P_{\text{dir}}, D_{\text{lang}}]$.

---

## 2. Matriks Perbandingan Multi-Dimensional

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       MATRIKS KOMPARATIF METODOLOGIS DAN EMPIRIS                                       │
├─────────────────────┬───────────────────┬────────────────────┬────────────────────┬─────────────────┬──────────────────┤
│ Dimensi Evaluasi    │ Baseline (M_1)    │ Hipotesis (M_norm) │ Ablasi (LR_conf)   │ Ablasi (LR_vis) │ Metode: Bi-ViS   │
├─────────────────────┼───────────────────┼────────────────────┼────────────────────┼─────────────────┼──────────────────┤
│ Sifat Formulasi     │ Heuristik probabil│ Heuristik rasio    │ Terpelajar (LR)    │ Terpelajar (LR) │ Terpelajar (LR)  │
│                     │ tunggal           │ aritmatika         │ 1-Fitur            │ 2-Fitur         │ 3-Fitur          │
│ Informasi Verbal    │ Ya (M_1)          │ Ya (P_orig)        │ Ya (M_1)           │ Ya (M_1)        │ Ya (M_1)         │
│ Bukti Visual (P_1)  │ Tidak             │ Ya (P_blank)       │ Tidak              │ Ya (ΔP_dir)     │ Ya (ΔP_dir)      │
│ Sinyal Bahasa (EN)  │ Tidak             │ Tidak              │ Tidak              │ Tidak           │ Ya (D_lang)      │
│ Parameter Terlatih  │ 0 (Unsupervised)  │ 0 (Unsupervised)   │ 2 (β_0, β_1)       │ 3 (β_0..β_2)    │ 4 (β_0..β_3)     │
│ Ketahanan Numerik   │ Stabil [0, 1]     │ RENTAN BLOWUP (÷0) │ Stabil (Sigmoid)   │ Stabil (Sigmoid)│ Stabil (Sigmoid) │
├─────────────────────┼───────────────────┼────────────────────┼────────────────────┼─────────────────┼──────────────────┤
│ HASIL QWEN2.5-VL    │                   │                    │                    │                 │                  │
│ • Test AURC         │ 0.2422            │ 0.2753             │ 0.2422             │ 0.2434          │ 0.2053           │
│ • Risk pada Cov 80% │ 0.3365            │ 0.3271             │ 0.3365             │ 0.3349          │ 0.2848           │
│ • Δ AURC vs Baseline│ 0.0000            │ +0.0331 (Memburuk) │ 0.0000             │ +0.0012         │ -0.0369 (Terbaik)│
│ • 95% Cluster CI    │ [Ref]             │ [+0.0078, +0.0580] │ [Ref]              │ [-0.0038,+0.0064│ [-0.0589,-0.0159]│
│ • Permutation p     │ [Ref]             │ p = 0.2289 (ns)    │ [Ref]              │ p = 0.6647 (ns) │ p = 0.0305 (*)   │
│ • Holm-Bonferroni p │ [Ref]             │ p_adj = 0.2289 (ns)│ [Ref]              │ p_adj = 0.6647  │ p_adj = 0.0310(*)|
├─────────────────────┼───────────────────┼────────────────────┼────────────────────┼─────────────────┼──────────────────┤
│ HASIL HUATUOGPT-VIS │                   │                    │                    │                 │                  │
│ • Test AURC         │ 0.2884            │ 0.2552             │ 0.2884             │ 0.2513          │ 0.2179           │
│ • Risk pada Cov 80% │ 0.4038            │ 0.3740             │ 0.4038             │ 0.3725          │ 0.3568           │
│ • Δ AURC vs Baseline│ 0.0000            │ -0.0332 (Membaik)  │ 0.0000             │ -0.0371         │ -0.0705 (Terbaik)│
│ • 95% Cluster CI    │ [Ref]             │ [-0.0585,-0.0063]  │ [Ref]              │ [-0.0643,-0.0085│ [-0.0984,-0.0415]│
│ • Permutation p     │ [Ref]             │ p = 0.3773 (ns)    │ [Ref]              │ p = 0.0165 (*)  │ p = 0.0005 (***) │
│ • Holm-Bonferroni p │ [Ref]             │ p_adj = 0.3773 (ns)│ [Ref]              │ p_adj = 0.0330(*│ p_adj = 0.0010(**│
└─────────────────────┴───────────────────┴────────────────────┴────────────────────┴─────────────────┴──────────────────┘
```

---

## 3. Mengapa Heuristik Rasio $M_{\text{norm}}$ Gagal?

Hasil empiris menunjukkan paradoks mencolok pada hipotesis primer $M_{\text{norm}}$:
- Pada **Qwen2.5-VL-7B**, $M_{\text{norm}}$ justru **memperburuk AURC sebesar $+0.0331$** ($0.2422 \rightarrow 0.2753$, 95% CI: $[+0.0078, +0.0580]$).
- Pada **HuatuoGPT-Vision-7B**, $M_{\text{norm}}$ menghasilkan $\Delta \text{AURC} = -0.0332$, namun setelah memperhitungkan struktur klaster citra, hasil tersebut **tidak signifikan secara statistik ($p = 0.3773$)**.

### Analisis Akar Masalah Matematika:
1. **Ketidakstabilan Rasio Pembagian ($P_{\text{blank}} \rightarrow 0$):**
   Pada kasus-kasus di mana model memiliki bias teks yang sangat rendah terhadap kanvas hitam, penyebut $P_{\text{blank}}(\hat{y}) + \epsilon$ mendekati nol. Hal ini memicu lonjakan nilai $M_{\text{norm}}$ yang sangat masif (*ratio explosion*) pada sampel acak, mengaburkan urutan reliabilitas sampel lain.
2. **Ketiadaan Pembobotan Adaptif:**
   $M_{\text{norm}}$ memaksakan relasi non-linear tetap tanpa mempertimbangkan variansi skala internal model. Sebaliknya, **Bi-ViS** memetakan fitur ke dalam log-odds teratur dan mentransformasikannya melalui fungsi sigmoid:
   $$P(\text{error} \mid \mathbf{x}) = \sigma\left(\beta_0 + \sum_{j} \beta_j z_j\right)$$
   Fungsi sigmoid menjamin output risiko selalu berada pada interval tertutup $[0, 1]$ tanpa pernah mengalami ledakan nilai numerik.

---

## 4. Analisis Nilai Tambah Inkremental (Incremental Value Analysis)

Uji ablasi bertingkat membuktikan secara rigor kontribusi spesifik dari masing-masing sinyal informasi:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              UJI NILAI TAMBAH INKREMENTAL (PAIRED CLUSTER PERMUTATION TEST)                            │
├─────────────────────┬──────────────────────────────┬──────────────┬───────────────────┬──────────────┬─────────────────┤
│ Model VLM           │ Komparasi Inkremental        │ Δ AURC       │ 95% Cluster CI    │ p-value (MC) │ Kesimpulan      │
├─────────────────────┼──────────────────────────────┼──────────────┼───────────────────┼──────────────┼─────────────────┤
│ Qwen2.5-VL-7B       │ LR_vis vs LR_conf            │ +0.0012      │ [-0.0038, +0.0064]│ 0.6647       │ Tidak Ada Nilai │
│                     │ (Efek Tambahan ΔP_dir)       │              │                   │              │ Inkremental     │
│                     ├──────────────────────────────┼──────────────┼───────────────────┼──────────────┼─────────────────┤
│                     │ Bi-ViS vs LR_vis             │ -0.0381      │ [-0.0622, -0.0148]│ 0.0120 (*)   │ NILAI TAMBAH    │
│                     │ (Efek Tambahan D_lang)       │              │                   │              │ SIGNIFIKAN      │
│                     ├──────────────────────────────┼──────────────┼───────────────────┼──────────────┼─────────────────┤
│                     │ Bi-ViS vs Baseline M_1       │ -0.0369      │ [-0.0589, -0.0159]│ 0.0305 (*)   │ UNGGUL TOTAL    │
├─────────────────────┼──────────────────────────────┼──────────────┼───────────────────┼──────────────┼─────────────────┤
│ HuatuoGPT-Vision-7B │ LR_vis vs LR_conf            │ -0.0371      │ [-0.0643, -0.0085]│ 0.0165 (*)   │ NILAI TAMBAH    │
│                     │ (Efek Tambahan ΔP_dir)       │              │                   │              │ SIGNIFIKAN      │
│                     ├──────────────────────────────┼──────────────┼───────────────────┼──────────────┼─────────────────┤
│                     │ Bi-ViS vs LR_vis             │ -0.0335      │ [-0.0467, -0.0218]│ 0.0005 (***) │ NILAI TAMBAH    │
│                     │ (Efek Tambahan D_lang)       │              │                   │              │ EKSTREM         │
│                     ├──────────────────────────────┼──────────────┼───────────────────┼──────────────┼─────────────────┤
│                     │ Bi-ViS vs Baseline M_1       │ -0.0705      │ [-0.0984, -0.0415]│ 0.0005 (***) │ UNGGUL TOTAL    │
└─────────────────────┴──────────────────────────────┴──────────────┴───────────────────┴──────────────┴─────────────────┘
```

### Temuan Kunci Analisis Inkremental:
1. **$D_{\text{lang}}$ Memberikan Nilai Tambah Inkremental Konsisten:**
   Penambahan $D_{\text{lang}}$ di atas model berbasis citra ($LR_{\text{visual}}$) menghasilkan reduksi AURC yang signifikan pada kedua model ($\Delta = -0.0381, p = 0.0120$ pada Qwen; $\Delta = -0.0335, p = 0.0005$ pada Huatuo). Hal ini membuktikan bahwa sinyal ketidaksepakatan bilingual memuat informasi epistemik unik yang tidak dapat digantikan oleh keyakinan visual saja.
2. **Sensitivitas Visual $\Delta P_{\text{dir}}$ Bersifat Kritis pada Model Spesialis Medis:**
   Pada HuatuoGPT-Vision, menambahkan $\Delta P_{\text{dir}}$ ke $LR_{\text{conf}}$ mereduksi AURC sebesar $-0.0371$ ($p = 0.0165$). Huatuo menderita fenomena *negative visual reliance* sebesar $80.0\%$ pada bahasa Indonesia; fitur $\Delta P_{\text{dir}}$ bertindak sebagai sinyal diagnostik yang sangat efektif untuk membedakan sampel di mana model benar-benar melihat lesi radiologi versus sampel di mana model sekadar berspekulasi.

---

## 5. Ringkasan Kesimpulan Komparatif

1. **Baseline $M_1$ Tidak Memadai:** Keyakinan verbal tunggal gagal mendeteksi halusinasi yang terjadi akibat pergeseran bahasa regional.
2. **Hipotesis Primer $M_{\text{norm}}$ Tidak Direkomendasikan:** Formula heuristik rasio tidak stabil secara numerik dan tidak memiliki dasar statistik inferensial yang solid.
3. **Bi-ViS Merupakan Solusi Defensibel:** Dengan menggabungkan bukti verbal ($M_1$), ketergantungan visual kontrafaktual ($\Delta P_{\text{dir}}$), dan konsistensi semantik translasi ($D_{\text{lang}}$) melalui estimator risiko logistik teratur, **Bi-ViS secara konsisten mencapai kinerja terbaik pada kedua model ($p \le 0.0310$)** dan siap diajukan sebagai kontribusi metodologis utama skripsi.
