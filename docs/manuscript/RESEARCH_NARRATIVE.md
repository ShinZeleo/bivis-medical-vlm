# AUTHORITATIVE RESEARCH NARRATIVE
## Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift
**Target Milestone:** Foundation Narrative for SINTA 2 Journal & Academic Thesis Defense  
**Target Path:** `docs/manuscript/RESEARCH_NARRATIVE.md`  
**Date Logged:** 2026-11-01 10:52:09 UTC  
**Epistemic Standard:** Evidence-Grounded, Statistically Audited, Anti-Overclaiming

---

## 1. Masalah Ilmiah yang Diteliti (*The Scientific Problem*)

Penerapan *Vision-Language Models* (VLMs) dalam domain medis menjanjikan demokratisasi akses keahlian diagnostik radiologi. Namun, ketika model fondasi visi-bahasa yang dilatih secara dominan menggunakan korpus medis bahasa Inggris (seperti MIMIC-CXR dan ChestX-ray14) dioperasikan di negara berkembang seperti Indonesia, model menghadapi **pergeseran distribusi bahasa (*English–Indonesian language shift*)**.

Masalah kritisnya bukan sekadar penurunan akurasi rata-rata, melainkan **ketiadaan mekanisme penolakan terpercaya (*unreliable abstention mechanism*)**. Model bahasa autoregresif cenderung menghasilkan jawaban yang salah dengan keyakinan verbal (*verbalized confidence*) yang sangat tinggi, memicu kegagalan hening (*silent diagnostic failures*). Di lingkungan klinis berisiko tinggi (*high-stakes clinical environments*), sistem AI tidak boleh dipaksa menjawab seluruh pertanyaan (*forced-choice regime*); sistem harus memiliki kemampuan klasifikasi selektif (*selective classification*) untuk merujuk kasus-kasus yang tidak pasti kepada dokter spesialis radiologi manusia.

---

## 2. Mengapa English–Indonesian Language Shift Penting untuk Medical VLM?

1. **Kesenjangan Layanan Radiologi di Indonesia:**  
   Indonesia memiliki populasi lebih dari 280 juta jiwa yang tersebar di ribuan pulau, namun rasio dokter spesialis radiologi sangat terbatas (kurang dari 2.000 radiolog aktif, terkonsentrasi di kota-kota besar Pulau Jawa). Kebutuhan asisten triase diagnostik multimodal berbahasa Indonesia sangat mendesak.
2. **Asimetri Pelatihan Biomedis:**  
   Korpus biomedis teranotasi skala besar hampir seluruhnya berbahasa Inggris. Bahasa Indonesia tergolong bahasa dengan sumber daya medis rendah (*low-resource medical language*).
3. **Kerentanan Inkonsistensi Diagnostik (*Diagnostic Fragility*):**  
   Penelitian ini membuktikan bahwa pergeseran instruksi dari bahasa Inggris ke bahasa Indonesia memicu degradasi performa yang masif:
   - Akurasi Qwen2.5-VL-7B anjlok sebesar **$-13.02$ persentase poin** ($75.84\% \rightarrow 62.83\%$, $p < 0.0001$).
   - Akurasi HuatuoGPT-Vision-7B anjlok sebesar **$-17.27$ persentase poin** ($74.84\% \rightarrow 57.57\%$, $p < 0.0001$).
   - Yang paling mengkhawatirkan, model spesialis medis Huatuo mengalami tingkat pembalikan diagnosis (*flip rate*) hingga **$50.81\%$** ($OR_{flip} = 2.54$ terhadap Qwen), membuktikan bahwa spesialisasi medis pada domain bahasa Inggris justru meningkatkan kerentanan model terhadap transfer bahasa non-dominan.

---

## 3. Mengapa Confidence Score Tunggal Tidak Cukup?

Pada pendekatan selective prediction konvensional (Geifman & El-Yaniv, 2017), model menolak menjawab jika skor preferensi kandidat internal ($M_1 = \max P(y \mid x)$) berada di bawah ambang batas tertentu. Namun, pendekatan ini memiliki kelemahan mendasar dalam VLM bilingual:
1. **Endemisme Overconfidence:** Model autoregresif mengalami overconfidence sistemik akibat *temperature scaling* dan bias distribusi kata token target.
2. **Kegagalan Monotonik Kalibrasi:** Metode kalibrasi pasca-proses standar (seperti Platt scaling atau Temperature scaling) bersifat monotonik ketat. Transformasi monotonik **sama sekali tidak mengubah urutan pemeringkatan (*ranking order*)**, sehingga metrik AUROC dan Area Under the Risk–Coverage Curve (AURC) tetap tidak berubah (Byun et al., 2026).
3. **Ketidakmampuan Mengidentifikasi Halusinasi:** Model dapat sangat yakin pada jawaban salah semata-mata karena frekuensi kata prior dalam korpus teks pelatihan, tanpa memeriksa apakah jawaban tersebut didukung oleh citra radiologi.

---

## 4. Mengapa Sensitivitas Penghapusan Visual ($\Delta P_{dir}$) Digunakan?

Model visi-bahasa medis rentan mengalami bias ketergantungan pada teks (*textual shortcut learning* atau "Dangerous Quadrant", Sadanandan & Behzadan, 2026; Zafar et al., 2026). Model menjawab pertanyaan radiologi bukan berdasarkan interpretasi bayangan opak atau lesi paru, melainkan menebak berdasarkan korelasi statistik teks pertanyaan.

Untuk menguji apakah jawaban model benar-benar bersandar pada citra fisik, penelitian ini menerapkan intervensi kontrafaktual waktu-inferensi: menghapus citra radiologi dan menggantinya dengan kanvas hitam pekat ($P_1$).
- Sensitivitas visual berarah didefinisikan sebagai:
  $$\Delta P_{dir} = P_{orig}(\hat{y}) - P_{blank}(\hat{y})$$
- **Rasionalitas:** Jika prediksi model didasarkan pada bukti radiologis aktual, probabilitas jawaban tersebut harus turun secara signifikan ketika citra dihapus ($\Delta P_{dir} > 0$). Sebaliknya, jika probabilitas tidak berubah atau bahkan meningkat saat citra dihitamkan ($\Delta P_{dir} \le 0$), prediksi tersebut mengindikasikan halusinasi visual atau tebakan berbasis prior teks.

---

## 5. Mengapa Cross-Lingual Disagreement ($D_{lang}$) Dapat Menjadi Risk Signal?

Dalam situasi inferensi praktis, label kebenaran medis (*ground-truth*) tidak tersedia. Namun, sistem dapat meminta model mengevaluasi citra radiologi yang sama menggunakan dua bahasa: bahasa Inggris ($X_{EN}$) dan bahasa Indonesia ($X_{ID}$).
- Sinyal ketidaksepakatan lintas-bahasa didefinisikan sebagai:
  $$D_{lang} = \mathbb{I}(\hat{y}_{EN} \ne \hat{y}_{ID})$$
- **Rasionalitas Teoretis:** Pertanyaan radiologi biner dalam bahasa Inggris dan terjemahan bahasa Indonesianya memiliki makna semantik klinis yang identik. Jika sebuah model menghasilkan diagnosis yang bertolak belakang semata-mata karena perbedaan bahasa prompt, ketidakstabilan semantik ini merefleksikan **ketidakpastian epistemik (*epistemic uncertainty*)** internal model.
- **Bukti Empiris:** Regresi logistik *cluster-robust* membuktikan bahwa kejadian $D_{lang} = 1$ melipatgandakan *odds* terjadinya kesalahan diagnostik lebih dari $2.7$ kali lipat:
  - **Qwen2.5-VL-7B:** $OR(D_{lang}) = \mathbf{3.064}$ ($95\%$ CI $[2.291, 4.098]$, $p < 0.0001$).
  - **HuatuoGPT-Vision-7B:** $OR(D_{lang}) = \mathbf{2.757}$ ($95\%$ CI $[1.866, 4.074]$, $p < 0.0001$).

---

## 6. Apa yang Benar-Benar Baru dari Penelitian Ini? (*Novelty Demarcation*)

Penelitian ini tidak mengklaim sebagai penemu konsep selective prediction (yang dirumuskan oleh Geifman & El-Yaniv, 2017) maupun penemu manipulasi citra hitam (yang digunakan oleh Zafar et al., 2026 dan Agrawal et al., 2018).

**Kebaruan Spesifik yang Defensibel (*Defensible Novelty*):**
1. **Arsitektur Seleksi Multi-Modalitas Waktu-Inferensi (*Zero-Retraining Runtime Framework*):** Integrasi pertama yang memadukan sinyal keyakinan token ($M_1$), probe sensitivitas visual kontrafaktual ($\Delta P_{dir}$), dan indikator ketidaksepakatan terjemahan bilingual ($D_{lang}$) ke dalam fungsi pemeringkatan risiko selektif kontinu untuk VLM medis bilingual tanpa memerlukan pelatihan ulang (*fine-tuning*) bobot model.
2. **Evaluasi Diskrit AURC Berbasis Klaster Citra Fisik:** Studi pertama yang mengevaluasi efisiensi kurva Risk–Coverage pada tugas VQA radiologi bahasa Indonesia menggunakan inferensi non-parametrik yang terkontrol terhadap korelasi intra-klaster citra fisik radiografi.

---

## 7. Taksonomi Kontribusi Ilmiah: Terbukti vs Hipotesis yang Tidak Terbukti

Untuk menjaga integritas ilmiah dan menghindari penolakan reviewer akibat *overclaiming*, kontribusi penelitian diklasifikasikan secara tegas:

### A. Kontribusi yang Terbukti Secara Empiris (*Supported Findings*)
1. **Dampak Pergeseran Bahasa Bersifat Asimetris:** Model spesialis medis HuatuoGPT-Vision mengalami degradasi konsistensi diagnostik yang jauh lebih parah (*flip rate* $50.81\%$) dibanding model fondasi umum Qwen2.5-VL ($28.79\%$).
2. **$D_{lang}$ Merupakan Prediktor Risiko Universal:** Penambahan sinyal ketidaksepakatan lintas-bahasa memberikan nilai tambah inkremental yang terbukti signifikan secara statistik pada kedua model (menurunkan AURC sebesar $-0.0381$ pada Qwen, $p=0.0130$; dan $-0.0335$ pada Huatuo, $p=0.0010$).
3. **Model Multivariat Penuh ($LR_{full}$) Mengungguli Baseline:** Sistem selektif multivariat terpelajar mereduksi AURC diskrit secara signifikan pada kedua model (Qwen $\Delta = -0.0369, p=0.0310$; Huatuo $\Delta = -0.0705, p=0.0010$) serta memangkas selective risk pada target coverage $80\%$ hingga $5.16$ persentase poin.
4. **Patologi Negative Visual Reliance pada Huatuo:** Audit mengungkap bahwa $80.0\%$ prediksi Huatuo pada kondisi kanvas hitam menghasilkan probabilitas yang lebih tinggi dibanding citra asli, membuktikan adanya bias overconfidence pada input OOD yang berhasil dimanfaatkan fitur $\Delta P_{dir}$ sebagai penalti halusinasi.

### B. Temuan Negatif dan Hipotesis yang Tidak Terbukti (*Negative & Disproven Findings*)
1. **Kegagalan Heuristik $M_{norm}$:** Hipotesis bahwa kombinasi aritmatika normalisasi min-max sederhana ($M_{norm}$) dapat menurunkan AURC **DITOLAK** ($p = 0.2298$ pada Qwen, $p = 0.3846$ pada Huatuo).
2. **Ketiadaan Nilai Inkremental Visual Sensitivity pada Qwen:** Penambahan $\Delta P_{dir}$ saja ke dalam skor keyakinan (`LR_visual` vs `LR_conf`) **TIDAK memberikan perbaikan statistik pada Qwen** ($\Delta\text{AURC} = +0.0012, p = 0.6693$).
3. **Bukan Bukti Pemahaman Klinis Kausal:** Intervensi kanvas hitam adalah pengujian sensitivitas piksel pada kondisi OOD; **tidak membuktikan bahwa model memahami anatomi atau lokalisasi patologi secara kausal**.
