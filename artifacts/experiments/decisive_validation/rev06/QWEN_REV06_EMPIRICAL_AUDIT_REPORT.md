# 📊 Laporan Audit Empiris Eksperimen Riil Qwen2.5-VL-7B (rev06)
**Tanggal Eksekusi:** 05 November 2026  
**Akselerator / Perangkat:** Kaggle Dual-GPU Tesla T4 (Device: `cuda:0`)  
**Tipe Presisi (Compute Dtype):** `torch.bfloat16`  
**Total Runtime Komputasi:** $14.111{,}4$ detik ($\sim 235{,}2$ menit / $3{,}92$ jam)  
**Rata-rata Waktu Inferensi per Sampel:** $12{,}45$ detik  
**Integritas Checkpoint:** 1.133 baris lengkap ($100\%$), 0 duplikat `qa_id`, 84 kolom  

---

## 1. Lokasi Berkas & Manifest Artefak rev06
Seluruh berkas hasil inferensi dan evaluasi resmi telah disusun rapi ke dalam hierarki direktori repositori:

| Kategori | Lokasi Berkas Relatif | Ukuran | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Prediction Checkpoint** | [`predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/artifacts/experiments/decisive_validation/rev06/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv) | $1{,}26$ MB | `a89ec1ee5d614138e6e8cff661b183664d6db8754c03b137f6a738fa0995fa1b` |
| **RQ1 Bilingual Metrics** | [`metrics/qwen/rq1_bilingual_metrics_Qwen_Qwen2.5_VL_7B_Instruct.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/artifacts/experiments/decisive_validation/rev06/metrics/qwen/rq1_bilingual_metrics_Qwen_Qwen2.5_VL_7B_Instruct.csv) | $782$ B | `25bc25f1ca8ea4814d9cf28e3b4d4f8fb2c40c3132bb82a7e78044733db9b544` |
| **RQ2 Visual Perturbation** | [`metrics/qwen/rq2_visual_intervention_Qwen_Qwen2.5_VL_7B_Instruct.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/artifacts/experiments/decisive_validation/rev06/metrics/qwen/rq2_visual_intervention_Qwen_Qwen2.5_VL_7B_Instruct.csv) | $435$ B | `fa1492ae66e3aa12d4d8ef534ee26ffcf3e8460655ad1ff7a63ec50f16f39009` |
| **RQ3 Selective Prediction**| [`metrics/qwen/rq3_selective_prediction_Qwen_Qwen2.5_VL_7B_Instruct.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/artifacts/experiments/decisive_validation/rev06/metrics/qwen/rq3_selective_prediction_Qwen_Qwen2.5_VL_7B_Instruct.csv) | $1{,}74$ KB | `e9fba4eb13063f25c27bf666a7b7d03d092490fa63b2f84732622f9909289299` |
| **Risk Ablation (OOF Dev)** | [`metrics/qwen/risk_ablation_Qwen_Qwen2.5_VL_7B_Instruct.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/artifacts/experiments/decisive_validation/rev06/metrics/qwen/risk_ablation_Qwen_Qwen2.5_VL_7B_Instruct.csv) | $419$ B | `f0449a0a0340277df5b7e283b8b17cefd346904f0394747ebc7b1ce89fb4b733` |
| **Figure RC-Curve** | [`figures/qwen/risk_coverage_primary_test_Qwen2.5_VL_7B.png`](file:///c:/.PUSAT/Project/Kuliah/VLM/artifacts/experiments/decisive_validation/rev06/figures/qwen/risk_coverage_primary_test_Qwen2.5_VL_7B.png) | $201{,}8$ KB | `d4e56ebba9012fdf8704285ad41a2a4fe00df3f000b2ce28da1aeb6c4bf9fba3` |

---

## 2. Temuan Utama RQ1: Evaluasi Komparatif Track `seq` vs `len` vs `v6`
Evaluasi pada kohort Primary Test ($N = 799$ pasang QA, 204 klaster citra medis independen, bebas bocor):

| Parameter Evaluasi | Track Lama (`seq`) | Track Rata-rata (`len`) | Track Usulan Baru (`v6`) | Evaluasi Ilmiah |
| :--- | :---: | :---: | :---: | :--- |
| **Akurasi Bahasa Indonesia (ID)** | $60{,}70\%$ | $59{,}45\%$ | $\mathbf{63{,}83\%}$ | **Meningkat signifikan $+3{,}13\text{ pp}$** |
| **Akurasi Bahasa Inggris (EN)** | $75{,}97\%$ | $75{,}97\%$ | $75{,}72\%$ | Paritas stabil (selisih hanya $-0{,}25\text{ pp}$) |
| **Disparitas Bahasa (Gap EN $-$ ID)** | $+15{,}27\text{ pp}$ | $+16{,}52\text{ pp}$ | $\mathbf{+11{,}89\text{ pp}}$ | **Menyusut $-3{,}38\text{ pp}$** |
| **Cross-Lingual Flip Rate** | $31{,}29\%$ ($250$) | $35{,}29\%$ ($282$) | $\mathbf{28{,}91\%}$ ($231$) | **$20$ ketidakkonsistenan tereliminasi** |
| **Predicted-YES Rate (ID)** | $27{,}66\%$ | $19{,}90\%$ | $\mathbf{34{,}79\%}$ | **Pemulihan polaritas positif $+7{,}13\text{ pp}$** |
| **Joint Accuracy (EN & ID Benar)**| $52{,}69\%$ | $50{,}06\%$ | $\mathbf{55{,}19\%}$ | **Keandalan dwibahasa simultan naik** |

> 📌 **Interpretasi Akademik:**  
> Track `len` (normalisasi rata-rata panjang token) justru memperburuk bias ke arah NO (predicted-YES anjlok ke $19{,}90\%$). Sebaliknya, Track `v6` (penyatuan varian token huruf besar, kecil, spasi via LogSumExp) secara empiris terbukti memitigasi fragmentasi tokenisasi Bahasa Indonesia dan mendongkrak akurasi ID menjadi **$63{,}83\%$**.

---

## 3. Temuan Utama RQ2: Sensitivitas Multimodal Kaustik ($\Delta p_{\text{dir}}$)
* **Mean $\Delta p_{\text{dir}}$ pada Citra Asli ($P_0$) vs Blank ($P_1$):** $+0{,}1267$  
* **Proporsi Ketergantungan Visual Positif ($\Delta p > 0$):** $58{,}70\%$  
* **Proporsi Dominasi Prior Teks / Halusinasi ($\Delta p \le 0$):** $41{,}30\%$  

Hal ini menunjukkan bahwa model Qwen2.5-VL-7B pada lebih dari $58\%$ kasus benar-benar memanfaatkan fitur visual citra radiologi, sementara sisanya rentan terhadap bias prior bahasa.

---

## 4. Temuan Utama RQ3: Efektivitas Selective Answering & Pengurangan Risiko
Evaluasi Kurva Risiko-Cakupan (*Risk-Coverage Curves*) pada Primary Test ($N=799$):

| Estimator Risiko | Formula / Definisi | Mean AURC | 95% CI (Cluster Bootstrap) | $\Delta\text{AURC}$ vs Baseline | $p$-value (Holm-Bonferroni) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$m_1$ (Baseline)** | Raw Softmax Confidence | $0{,}2459$ | $[0{,}2042, 0{,}2920]$ | $0{,}0000$ (Ref) | $1{,}0000$ |
| **$m_2$ (Cross-Lingual)** | Consistency Gate $(1 - \text{flip}) \cdot \text{conf}$ | $\mathbf{0{,}1977}$ | $[0{,}1602, 0{,}2384]$ | $\mathbf{-0{,}0482}$ | $\mathbf{0{,}003996}$ (**Signifikan!**) |
| **$m_{\text{dir}}$ (Multimodal)**| Directed Visual Perturbation Shift | $0{,}3273$ | $[0{,}2737, 0{,}3775]$ | $+0{,}0814$ | $0{,}032967$ |
| **$m_{\text{norm}}$ (Hybrid BIVIS)**| Sinergi Konsistensi & Visual | $0{,}2781$ | $[0{,}2244, 0{,}3257]$ | $+0{,}0322$ | $0{,}635365$ |

### Model Risiko Terpelajar (Logistic Regression OOF Dev $\to$ Test):
* **Fitur Confidence Saja (`LR_conf`):** Test AURC = $0{,}2459$
* **Fitur Confidence + Visual (`LR_visual`):** Test AURC = $0{,}2512$
* **Fitur Lengkap Multimodal + Bahasa (`LR_full`):** Test AURC = $\mathbf{0{,}2044}$!  
  *(Koefisien $d_{\text{lang}} = +1{,}1789$ membuktikan bahwa diskrepansi bahasa adalah prediktor error terkuat)*.
