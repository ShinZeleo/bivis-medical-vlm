# MANUSCRIPT BLUEPRINT FOR SINTA 2 SUBMISSION
## Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift
**Target Journal Scope:** Jurnal Nasional Bereputasi Terakreditasi SINTA 2 (Bidang Ilmu Komputer, Sistem Informasi, atau Kecerdasan Buatan) / Target Internasional Q2/Q3  
**Target Path:** `docs/manuscript/MANUSCRIPT_BLUEPRINT.md`  
**Date Logged:** 2026-11-01  
**Architecture:** 9-Section Scientific Structure with Empirical Evidence Mapping and Epistemic Demarcation (Anti-HARKing)

---

## 1. Abstract
- **Tujuan Subbagian:** Memberikan ringkasan padat dan mandiri mengenai latar belakang, kesenjangan, metodologi, temuan kunci kuantitatif, dan signifikansi praktis riset.
- **Argumen Utama:** Evaluasi unselective pada Medical VLM di bawah pergeseran bahasa Inggris–Indonesia berbahaya bagi keselamatan klinis; integrasi sinyal sensitivitas visual kontrafaktual dan ketidaksepakatan bilingual dalam metode **Bi-ViS** (*Bilingual Visual-Sensitivity-Aware Selective Prediction*) memungkinkan penyaringan kesalahan diagnostik secara efektif tanpa pelatihan ulang model.
- **Bukti Empiris Wajib:**
  - Akurasi degradasi: Qwen $-13.02$ pp ($75.84\% \rightarrow 62.83\%$), Huatuo $-17.27$ pp ($74.84\% \rightarrow 57.57\%$), $p < 0.0001$.
  - Flip rate: Huatuo $50.81\%$ vs Qwen $28.79\%$.
  - Kegagalan hipotesis primer $M_{\text{norm}}$: degradasi AURC pada Qwen $+0.0331$ ($p = 0.2289$, tidak signifikan).
  - Keunggulan metode pengembangan Bi-ViS ($LR_{\text{full}}$): reduksi AURC konsisten pada Qwen $-0.0369$ ($p = 0.0310$) dan Huatuo $-0.0705$ ($p = 0.0010$).
  - Penurunan selective risk pada coverage $80\%$: hingga $-5.16$ pp.
- **Referensi Kunci:** Geifman & El-Yaniv (2017), Yudhistira et al. (2026), Sadanandan & Behzadan (2026).
- **Potensi Kritik Reviewer:** Abstrak terkesan menjanjikan implementasi klinis nyata.
  - *Mitigasi:* Nyatakan secara eksplisit bahwa evaluasi dilakukan pada benchmark tertutup IndoRad-VQA untuk pertanyaan biner.

---

## 2. Introduction
- **Tujuan Subbagian:** Membangun konteks disparitas layanan radiologi di Indonesia, memetakan keterbatasan adaptasi VLM, dan merumuskan urgensi selective prediction.
- **Argumen Utama:**
  1. Kebutuhan asisten radiologi AI berbahasa Indonesia tinggi, namun model fondasi dilatih pada korpus medis bahasa Inggris.
  2. Pergeseran bahasa memicu kegagalan diagnostik yang tidak terdeteksi oleh skor keyakinan internal tunggal.
  3. Diperlukan mekanisme penolakan (*reject option*) berbasis bukti multimodal waktu-inferensi.
- **Bukti Empiris Wajib:** Fakta demografi radiolog Indonesia (< 2.000 radiolog untuk 280 juta jiwa), kutipan angka penurunan akurasi awal dari Yudhistira et al. (2026).
- **Referensi Kunci:** Yudhistira et al. (2026), Kamath et al. (2020), Dancette et al. (2023).
- **Potensi Kritik Reviewer:** Mengapa tidak melakukan fine-tuning model dengan dataset bahasa Indonesia?
  - *Mitigasi:* Rujuk temuan Sadanandan & Behzadan (CHIL 2026) yang membuktikan fine-tuning konsistensi pada dataset kecil rentan memicu *mode collapse*, serta tekankan efisiensi komputasi *zero-retraining runtime selection*.

---

## 3. Related Work
- **Tujuan Subbagian:** Menempatkan penelitian secara presisi di antara 4 rumpun literatur dan menegaskan batas kebaruan (*novelty demarcation*).
- **Argumen Utama:**
  - *Selective VQA & Abstention:* Konsep selektif telah ada (Geifman & El-Yaniv, 2017; Dancette et al., 2023), namun terbatas pada domain umum bahasa Inggris.
  - *Cross-Lingual Consistency:* Yudhistira et al. (2026) mendokumentasikan fenomena pergeseran bahasa pada IndoRad-VQA, tetapi hanya dalam rezim unselective statis.
  - *Confidence Calibration:* Kalibrasi pasca-proses monotonik standar tidak mengubah peringkat AURC (Byun et al., 2026).
  - *Visual Grounding & Counterfactual Probes:* Uji kanvas hitam (Zafar et al., 2026; Sadanandan & Behzadan, 2026) membuktikan fenomena ungroundedness, namun belum pernah diformulasikan sebagai fungsi seleksi risiko kontinu.
- **Referensi Kunci:** Geifman & El-Yaniv (2017), Yudhistira et al. (2026), Byun et al. (2026), Zafar et al. (2026), Khan & Fu (2024).
- **Potensi Kritik Reviewer:** Mengklaim "novelty" yang berlebihan pada penggunaan citra hitam atau kombinasi fitur generik.
  - *Mitigasi:* Akui secara eksplisit bahwa citra hitam diadaptasi dari Zafar et al. (2026); kebaruan terletak pada integrasinya bersama ketidaksepakatan translasi bilingual ($D_{\text{lang}}$) ke dalam estimator risiko logistik teratur tanpa pelatihan ulang model.

---

## 4. Materials and Methods
- **Tujuan Subbagian:** Menjelaskan secara formal formulasi matematika selective prediction, ekstraksi fitur, pemisahan epistemik antara hipotesis primer dan metode pengembangan, serta estimator risiko.
- **Argumen Utama:**
  1. *Selective Prediction Formulation:* Pasangan $(f, g)$, fungsi risiko $r(f, g)$, coverage $\phi(g)$, dan Discrete Riemann AURC.
  2. *Dua Tingkatan Metode yang Dievaluasi (Anti-HARKing Demarcation):*
     - **Hipotesis Primer Historis ($M_{\text{norm}}$):** Rasio sensitivitas visual ternormalisasi tanpa bobot dari pre-registrasi awal (RQ3): $M_{\text{norm}} = P_{\text{orig}}(\hat{y}) / (P_{\text{blank}}(\hat{y}) + \epsilon)$.
     - **Metode Pengembangan Terpilih (Bi-ViS):** Kerangka pemodelan risiko terpelajar multivariat *Bilingual Visual-Sensitivity-Aware Selective Prediction* (Bi-ViS) yang menggabungkan tiga sinyal ortogonal $[M_1, \Delta P_{\text{dir}}, D_{\text{lang}}]$.
  3. *Feature Extraction Pipeline Bi-ViS:*
     - Keyakinan token murni Bahasa Indonesia: $M_1 = P(y = \hat{y}_{\text{ID}} \mid I_{\text{orig}}, Q_{\text{ID}})$.
     - Sensitivitas visual berarah kontrafaktual: $\Delta P_{\text{dir}} = P_{\text{orig}}(\hat{y}_{\text{ID}}) - P_{\text{blank}}(\hat{y}_{\text{ID}})$.
     - Indikator ketidaksepakatan lintas-bahasa: $D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$.
  4. *Risk Modeling & Score Alignment:*
     - Standardisasi z-score berbasis parameter Dev set ($N=323$, 88 klaster citra).
     - Estimator risiko logistik teratur $L_2$ ($C=1.0$): $P(\text{error} \mid \mathbf{x}) = \sigma(\beta_0 + \beta_1 z_{M_1} + \beta_2 z_{\Delta P_{\text{dir}}} + \beta_3 z_{D_{\text{lang}}})$.
     - Penyelarasan skor keandalan selektif: $S_{\text{Bi-ViS}}(\mathbf{x}) = 1 - P(\text{error} \mid \mathbf{x})$.
- **Bukti Empiris Wajib:** Rumusan matematis diskrit Riemann AURC, spesifikasi standardisasi scaler pada Dev set, diagram alur arsitektur Bi-ViS.
- **Referensi Kunci:** Geifman & El-Yaniv (2017), Kamath et al. (2020), Dancette et al. (2023).
- **Potensi Kritik Reviewer:** Risiko data leakage antara fitur dan model seleksi, serta kecurigaan HARKing.
  - *Mitigasi:* Tampilkan diagram isolasi partisi klaster citra fisik (Dev Split 88 klaster vs Test Split 204 klaster; tumpang tindih 0%) dan deklarasikan secara transparan bahwa Bi-ViS dikembangkan setelah kegagalan rasio $M_{\text{norm}}$.

---

## 5. Experimental Setup
- **Tujuan Subbagian:** Merinci konfigurasi teknis, dataset benchmark, model yang dievaluasi, dan protokol inferensi statistik inferensial.
- **Argumen Utama:**
  - *Dataset:* IndoRad-VQA ($N=1.122$ pertanyaan biner *Clean*, terbagi atas Dev $N=323$ dan Test $N=799$).
  - *Models:* Qwen2.5-VL-7B-Instruct (generalis) dan HuatuoGPT-Vision-7B-Qwen2.5VL (spesialis medis).
  - *Inferential Statistical Protocol:*
    - Unit independen: `canonical_image_id` ($K=204$ klaster pada test set).
    - Paired Cluster Bootstrap ($B=1.000$ replikasi) untuk mengestimasi 95% Percentile CI.
    - Monte Carlo Paired Cluster Sign-Flip Permutation Test ($B=1.000$, koreksi Davison & Hinkley $(1+M)/(B+1)$).
    - FWER Multiplicity Control via Holm-Bonferroni step-down.
- **Referensi Kunci:** Davison & Hinkley (1997), Phipson & Smyth (2010).
- **Potensi Kritik Reviewer:** Jumlah ukuran sampel $N=1.122$ dianggap relatif kecil untuk deep learning.
  - *Mitigasi:* Tunjukkan bukti bahwa rasio Events Per Variable (EPV) pada Dev set mencapai $32.0$ dan $43.0$ ($> 10-20$), menjamin validitas estimasi regresi logistik.

---

## 6. Results
- **Tujuan Subbagian:** Menyajikan temuan empiris secara netral, terstruktur per research question, dan didukung tabel serta grafik resmi.
- **Argumen & Bukti Empiris Wajib:**
  1. *Hasil RQ1 (Language Shift):* Tabel akurasi EN vs ID, penurunan akurasi, dan flip rate (Qwen $28.79\%$ vs Huatuo $50.81\%$).
  2. *Hasil RQ2 (Visual Sensitivity):* Pemisahan nilai $\Delta P_{\text{dir}}$ antara kelompok benar vs salah; korelasi tingkat klaster ($\rho = +0.287$ Qwen, $+0.246$ Huatuo); dokumentasi $80.0\%$ negative reliance pada Huatuo.
  3. *Hasil RQ3 (Evaluasi Hipotesis Primer $M_{\text{norm}}$ vs Baseline $M_1$):*
     - Hasil empiris kegagalan $M_{\text{norm}}$ pada Qwen ($\Delta = +0.0331$, memburuk) dan nonsignifikansi pada Huatuo ($\Delta = -0.0332, p = 0.3773$).
  4. *Hasil RQ3 Sekunder & Validasi Metode Pengembangan (Bi-ViS / $LR_{\text{full}}$):*
     - Tabel Ablasi 4-Tingkat (Baseline $M_1$, $LR_{\text{conf}}$, $LR_{\text{visual}}$, dan Bi-ViS / $LR_{\text{full}}$).
     - Grafik Risk–Coverage Curve yang memperlihatkan pemisahan kurva Bi-ViS secara konsisten di bawah baseline $M_1$.
     - Tabel koefisien regresi logistik, cluster-robust standard errors, odds ratios ($OR(D_{\text{lang}}) = 3.064$ Qwen, $2.757$ Huatuo), dan nilai VIF ($< 1.55$).
- **Potensi Kritik Reviewer:** Baseline confidence AURC pada laporan draf terdahulu berbeda dengan tabel.
  - *Mitigasi:* Sajikan tabel rekonsiliasi yang membedakan secara eksplisit antara baseline murni satu-fitur ($M_1$) dan model dua-fitur ($LR_{\text{visual}}$).

---

## 7. Discussion
- **Tujuan Subbagian:** Menginterpretasikan signifikansi temuan, menghubungkannya dengan teori yang ada, dan mendiskusikan implikasi arsitektural.
- **Argumen Utama:**
  1. *Mengapa Huatuo Lebih Rapuh?* Fine-tuning medis pada teks biomedis bahasa Inggris mengunci representasi linguistik model, sehingga saat diuji pada bahasa Indonesia, kemampuan visual grounding-nya terdistorsi parah (*negative reliance*).
  2. *Mengapa Heuristik $M_{\text{norm}}$ Gagal?* Penjumlahan seragam dan rasio pembagian langsung mengalami ledakan numerik (*ratio blowup*) saat penyebut mendekati nol; sebaliknya, Bi-ViS memberikan pemetaan probabilitas teratur via fungsi sigmoid yang stabil.
  3. *Peran Kritis $D_{\text{lang}}$:* Ketidaksepakatan terjemahan bertindak sebagai proksi ketidakpastian epistemik yang ortogonal terhadap keyakinan visual ($OR > 2.7\times$).
- **Referensi Kunci:** Byun et al. (2026), Sadanandan & Behzadan (2026), Dancette et al. (2023).
- **Potensi Kritik Reviewer:** Sistem membutuhkan inferensi paralel (EN dan ID serta Blank), sehingga meningkatkan latensi.
  - *Mitigasi:* Bahas trade-off klinis: dalam diagnosis medis berisiko tinggi, latensi komputasi tambahan beberapa detik jauh lebih dapat diterima dibandingkan membiarkan kesalahan diagnostik fatal lolos tanpa peringatan.

---

## 8. Limitations
- **Tujuan Subbagian:** Menyatakan batasan metodologis secara jujur untuk menjaga kerendahan hati ilmiah (*epistemic modesty*).
- **Argumen Batasan Wajib:**
  1. *Batasan Format Jawaban:* Evaluasi terbatas pada pertanyaan biner (*closed-set Yes/No*); belum mencakup generasi naratif bebas (*free-text radiology report*).
  2. *Sifat Intervensi Kanvas Hitam:* Kanvas hitam adalah stimulus *out-of-distribution*; sensitivitas terhadap kanvas hitam mengukur integritas informasi visual, bukan pemahaman kausal klinis.
  3. *Ukuran Dataset:* IndoRad-VQA diadaptasi dari VQA-RAD dengan ukuran sampel terbatas, meskipun kecukupan EPV telah diverifikasi.
  4. *Spesifisitas Pasangan Bahasa:* Penelitian mengevaluasi transfer bahasa Inggris–Indonesia; generalisasi ke bahasa daerah lain (misalnya bahasa Jawa atau Sunda) memerlukan penelitian lanjutan dan berstatus `NOT VERIFIED`.

---

## 9. Conclusion
- **Tujuan Subbagian:** Menutup naskah dengan menegaskan pesan inti dan implikasi riset.
- **Argumen Utama:** Pergeseran bahasa adalah ancaman nyata bagi keselamatan penerapan VLM medis; metode pengembangan **Bi-ViS** (*Bilingual Visual-Sensitivity-Aware Selective Prediction*) terbukti secara statistik mampu menyaring kesalahan diagnostik dan mereduksi risiko klinis secara substansial tanpa biaya pelatihan ulang model (*zero-retraining*).
