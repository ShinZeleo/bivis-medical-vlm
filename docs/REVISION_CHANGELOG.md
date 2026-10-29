# REVISION CHANGELOG — VERSION 10.0
## Tracking Methodological, Theoretical, and Experimental Protocol Evolution (v9.0 $\to$ v10.0)

**Dokumen:** `REVISION_CHANGELOG_10.md`  
**Dari Dokumen Asal:** `RESEARCH_MASTER.md` (Versi 9.0, rev05-master)  
**Ke Dokumen Tujuan:** `RESEARCH_MASTER(10).md` (Versi 10.0)  
**Tanggal Pengesahan:** 21 Oktober 2026  
**Auditor Metodologi:** Senior ML Researcher & Quantitative Methodologist  
**Status Audit:** *Preserved v9.0 intact; Full Traceability Guaranteed*

---

## 1. Ringkasan Eksekutif Perubahan Utama

Revisi Versi 10.0 **mempertahankan 100% fondasi metodologis inti versi 9.0** (dataset IndoRad-VQA, model Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B, intervensi citra hitam deterministik, klaster citra tanpa kebocoran, serta estimator primer $M_{\text{norm}}$ vs baseline $M_1$).

Evolusi substantif pada Versi 10.0 berfokus pada:
1. **Penetapan Sinyal Risiko Baru (Cross-Lingual Disagreement $D_{\text{lang}}$):** Mengoperasionalkan ketidaksepakatan biner antara prediksi prompt bahasa Inggris dan bahasa Indonesia ($\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}}$) sebagai sinyal risiko inkremental.
2. **Desain Analisis Sekunder (Ablation Study via Regresi Logistik):** Menambahkan 4 estimator terstruktur (`M1`, `LR-conf`, `LR-visual`, `LR-full`) untuk mengisolasi secara adil nilai tambah murni sensitivitas visual dan nilai tambah murni ketidaksepakatan bahasa di atas skor keyakinan.
3. **Pencegahan Kebocoran Data Supervised:** Menegakkan aturan bahwa seluruh fitting regresi logistik hanya boleh menggunakan Development Split ($N=323$, 88 klaster citra) dengan validasi silang GroupKFold berbasis klaster citra kanonis, menjaga Test Split ($N=799$, 204 klaster) tetap murni beku.
4. **Pembaruan Hierarki Research Questions:** Menata ulang menjadi 5 RQ terstruktur (RQ1: Language Shift; RQ2: Visual Sensitivity; RQ3: Primary Selective Prediction $M_{\text{norm}}$ vs $M_1$; RQ4: Incremental Cross-Lingual Signal $D_{\text{lang}}$; RQ5: Model Comparison).
5. **Harmonisasi Literatur Internasional Mutakhir:** Memetakan posisi novelty terhadap karya-karya primer CVPR 2023/2024/2026 dan MICCAI 2026 (Khan & Fu, Dancette et al., Sadanandan & Behzadan, Zafar et al., Khanmohammadi et al., Byun et al.).
6. **Pertimbangan Judul Alternatif:** Memperkenalkan judul kerja alternatif yang mencerminkan kedua sinyal multimodal tanpa overclaim.

---

## 2. Tabel Perbandingan Detail per Bagian Dokumen

| Bagian Dokumen | Status di Versi 9.0 | Pembaruan Substantif di Versi 10.0 | Dasar Rasional & Bukti Ilmiah | Dampak terhadap Desain & Eksperimen |
|:---|:---|:---|:---|:---|
| **Judul Penelitian** | *Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift* | **Dipertahankan sebagai Judul Kerja Utama.** Ditambahkan alternatif formal: *Cross-Lingual Disagreement and Visual-Deletion Sensitivity for Selective Risk Estimation in English–Indonesian Medical VQA*. | Mengakomodasi potensi perluasan fokus jika analisis sekunder membuktikan $D_{\text{lang}}$ memberikan nilai tambah inkremental yang signifikan. | Menyiapkan opsi judul publikasi artikel jurnal yang lebih komprehensif tanpa mengubah judul skripsi sebelum bukti final diperoleh. |
| **Sinyal Masukan Reliabilitas** | Hanya menggunakan 2 variabel: Keyakinan verbal ($p_{\text{orig}}$) dan Sensitivitas visual berarah ($\Delta P_{\text{dir}}$). | Menambahkan variabel ketiga: **Cross-Lingual Disagreement ($D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$)**. | Terinspirasi oleh Khan & Fu (CVPR 2024) dan Dancette et al. (CVPR 2023) yang membuktikan bahwa inkonsistensi respons adalah proksi ketidakpastian yang kuat. | Tidak membutuhkan inferensi GPU tambahan karena $\hat{y}_{\text{EN}}$ dan $\hat{y}_{\text{ID}}$ sudah dihitung di checkpoint rev05. |
| **Karakterisasi Sinyal $D_{\text{lang}}$** | Belum didefinisikan secara formal. | Didefinisikan secara eksplisit sebagai indikator ketidaksepakatan, **BUKAN bukti kesalahan klinis mutlak**. Ditegaskan bahwa jawaban yang sama antar-bahasa bisa saja sama-sama salah. | Menghindari ekuivokasi antara konsistensi linguistik dan kebenaran faktual klinis (Sadanandan & Behzadan, CVPR 2026). | Memastikan interpretasi hasil RQ4 objektif dan tidak menyamakan kesepakatan dengan akurasi. |
| **Estimator Selektif yang Diuji** | Hanya 5 formula matematis langsung: $M_1, M_2, M_{\text{dir}}, M_{\text{norm}}, M_3^{\text{old}}$. | **Dua Jalur Evaluasi:**<br>1. *Primary Test:* $M_{\text{norm}}$ vs $M_1$ (Aturan langsung tanpa fitting).<br>2. *Secondary Ablation:* $M_1$, `LR-conf`, `LR-visual`, `LR-full`. | Membedakan manfaat fitur ($\Delta P_{\text{dir}}$ dan $D_{\text{lang}}$) dari manfaat pemodelan risiko terpelajar (*learned risk model*). | Memungkinkan dekomposisi nilai tambah inkremental setiap sinyal secara kuantitatif. |
| **Penyelarasan Arah Skor Reliabilitas** | Skor langsung bernilai tinggi = lebih dipercaya ($[0.5, 1.0]$). | Skor regresi logistik risiko $\hat{P}(\text{error} = 1)$ diselaraskan menjadi skor retensi: $S_{\text{LR}} = 1 - \hat{P}(\text{error} = 1)$. | Konsistensi arah skor dengan Geifman & El-Yaniv (2017) agar kurva Risk-Coverage dan AURC diskrit dapat dibandingkan langsung antar seluruh estimator. | Menjamin perhitungan metrik AURC diskritRiemann identik dan valid tanpa distorsi inversi peringkat. |
| **Pencegahan Data Leakage** | Protokol split level citra fisik kanonis sudah bebas kebocoran (0% overlap). | Diperketat: **Semua fitting regresi logistik hanya pada Dev Set ($N=323$)** menggunakan GroupKFold ($K=5$) berbasis citra. Test Set ($N=799$) dibekukan mutlak. | Mencegah overfitting model risiko terpelajar pada test set retrospektif; standar wajib machine learning kuantitatif. | Menghasilkan estimasi generalisasi yang tidak bias dan dapat dipertanggungjawabkan pada peer-review jurnal internasional. |
| **Research Questions (RQ)** | 4 Research Questions (RQ1: Language Shift; RQ2: Visual Sensitivity; RQ3: M_norm Selective Prediction; RQ4: Model Comparison). | **5 Research Questions:**<br>- RQ1: Language Shift & Reliability Gap<br>- RQ2: Visual-Deletion Sensitivity & Error Association<br>- RQ3: Primary Selective Prediction ($M_{\text{norm}}$ vs $M_1$)<br>- **RQ4: Incremental Cross-Lingual Signal ($D_{\text{lang}}$ via `LR-full` vs `LR-visual`)**<br>- RQ5: General vs Medical Model Comparison. | Memisahkan pengujian hipotesis primer aturan langsung ($M_{\text{norm}}$) dari pengujian hipotesis sekunder sinyal bahasa inkremental ($D_{\text{lang}}$). | Menjaga hierarki pengujian hipotesis tetap bersih dan transparan; RQ4 diposisikan eksplisit sebagai analisis sekunder. |
| **Tinjauan Literatur & Peta Kebaruan** | Rujukan terbatas pada Yudhistira et al., Sadanandan & Behzadan, Khanmohammadi et al., dan teori klasik. | Ditambahkan rujukan terkini: Khan & Fu (CVPR 2024), Dancette et al. (CVPR 2023), Byun et al. (2026), Song et al. (MICCAI 2026), Zafar et al. (2026). | Memperkuat pemetaan empat rumpun literatur dan menutup celah ketidaktahuan terkait selective VQA dan counterfactual medical grounding. | Mempertegas posisi novelty skripsi di antara literatur global 2023–2026. |
| **Klaim Kontribusi Ilmiah** | Dituliskan sebagai 3 kontribusi umum. | Direformulasi menjadi **4 kontribusi potensial yang terukur dan dapat dipertahankan**, dengan batasan tegas bahwa pembuktian empiris bergantung pada data konfirmatori. | Mencegah klaim universal berlebih (*overclaim*) sebelum eksekusi selesai. | Memenuhi kriteria penerimaan reviewer jurnal bereputasi. |
| **Status Dokumen Asal v9.0** | File aktif tunggal di root (`RESEARCH_MASTER.md`). | **Dipelihara utuh tanpa penimpaan (*zero overwrite*).** Versi revisi baru disimpan terpisah sebagai `RESEARCH_MASTER(10).md`. | Mematuhi aturan integritas pelestarian jejak audit akademik (*audit trail preservation*). | Menjamin riwayat keputusan historis tetap dapat diperiksa sewaktu-waktu. |

---

## 3. Matriks Kepatuhan terhadap Batasan Metodologis (Compliance Audit)

- [x] **Tidak menambah model baru:** Tetap Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B.
- [x] **Tidak kembali ke fine-tuning LoRA/QLoRA:** Pendekatan tetap murni waktu-inferensi (*inference-time selective prediction*).
- [x] **Tidak menambah intervensi visual baru:** Tetap $P_0$ (asli) dan $P_1$ (hitam polos deterministik).
- [x] **Intervensi citra hitam tetap diperlakukan sebagai OOD:** Ditafsirkan sebagai sensitivitas penghapusan informasi visual, bukan pemahaman klinis atau lokalisasi kausal.
- [x] **Estimator primer tetap $M_{\text{norm}}$:** `LR-full` tidak menggantikan posisi $M_{\text{norm}}$ sebagai hipotesis utama.
- [x] **Zero test leakage:** Regresi logistik dilatih secara eksklusif pada Dev Set menggunakan GroupKFold tingkat klaster citra.
- [x] **Tidak ada angka empiris fiktif:** Seluruh status eksperimen GPU skala penuh tetap berstatus `PENDING_CONFIRMATORY_GPU`.
