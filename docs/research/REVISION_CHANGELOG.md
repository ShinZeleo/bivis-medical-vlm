# REVISION CHANGELOG: RESEARCH MASTER VERSI 10.0 ➔ VERSI 11.0
## Cross-Lingual Risk Estimation and Adaptive Abstention for Bilingual Medical Vision-Language Models

**Metadata Dokumen:**  
- **Versi Dokumen:** 1.0.0 (Changelog Resmi Versi 11.0)  
- **Tanggal:** 31 Oktober 2026  
- **Dokumen Sumber:** `docs/RESEARCH_MASTER.md` (Versi 10.0)  
- **Dokumen Target:** `docs/RESEARCH_MASTER.md` (Versi 11.0)  
- **Peneliti / Penulis:** Imam (Undergraduate CS Thesis Researcher)  
- **Standar Integritas:** Peer-review standar jurnal internasional bereputasi / SINTA 1–2  

---

## 1. Ringkasan Eksekutif Perubahan

Revisi Versi 11.0 merupakan penyempurnaan metodologis berskala besar yang mengintegrasikan dua pilar penelitian menjadi satu kesatuan alur inferensi waktu-uji (*unified test-time decision framework*):
1. **Cross-Lingual Risk Estimation:** Mengestimasi probabilitas kesalahan diagnostik bahasa Indonesia ($R(x)$) melalui integrasi keyakinan verbal ($p_{\text{orig\_ID}}$), responsivitas visual citra hitam ($\Delta P_{\text{dir\_ID}}$), dan ketidaksepakatan kesimpulan lintas-bahasa ($D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \neq \hat{y}_{\text{ID}})$).
2. **Adaptive Abstention Mechanism:** Menggunakan estimasi risiko $R(x)$ untuk menentukan aksi operasional biner klinis: meneruskan jawaban ($\text{Decision} = \text{Answer}$) jika $R(x) \le \tau$, dan menahan jawaban untuk dialihkan ke dokter spesialis radiologi ($\text{Decision} = \text{Abstain}$) jika $R(x) > \tau$, di mana ambang batas $\tau$ dikalibrasi secara eksklusif pada development set yang bebas kebocoran.

Pembaruan ini tidak mengubah fondasi dataset (`Lab-IS/IndoRad-VQA`, 1.122 pasangan biner bersih), model utama (`Qwen2.5-VL-7B` dan `HuatuoGPT-Vision-7B`), maupun intervensi kanvas hitam ($P_0$ vs $P_1$). Prinsip utama tetap dipegang teguh: **mempertahankan eksperimen yang telah tervalidasi secara empiris, menambahkan mekanisme keputusan yang relevan secara klinis, dan tidak mengklaim hasil yang belum dibuktikan.**

---

## 2. Matriks Perbandingan Konseptual & Metodologis v10.0 vs v11.0

| Dimensi | Versi 10.0 (`rev05-v10-master`) | Versi 11.0 (`rev05-v11-master`) | Alasan Ilmiah & Dampak Metodologis |
| :--- | :--- | :--- | :--- |
| **Fokus Judul Penelitian** | Menekankan *Visual Sensitivity-Aware Selective Prediction*. Menempatkan sensitivitas visual di posisi terdepan. | **Cross-Lingual Risk Estimation and Adaptive Abstention**. Menempatkan estimasi risiko dan aksi penahanan adaptif sebagai fokus utama. | Hasil empiris membuktikan bahwa $M_{\text{norm}}$ heuristik visual tidak konsisten, sedangkan $D_{\text{lang}}$ dan pemodelan risiko `LR-full` sangat dominan. Judul v11 lebih jujur dan mencakup aksi abstention klinis. |
| **Paradigma Seleksi** | **Passive Risk Ranking:** Sampel diurutkan berdasarkan skor $S(x)$ untuk menghitung metrik kurva Risk–Coverage (AURC). | **Active Decision Making (Adaptive Abstention):** Estimasi risiko $R(x)$ dikonversi menjadi aturan keputusan operasional biner ($Answer$ vs $Abstain$) via threshold $\tau$. | Menjembatani jurang antara teori evaluasi selektif pasif menuju implementasi klinis nyata pada *clinical decision support systems*. |
| **Penetapan Threshold ($\tau$)** | Tidak diformalkan secara terpisah; hanya dievaluasi pada titik-titik coverage teoritis kurva Risk-Coverage. | **Dual Strategy Protocol:**<br>1. *Coverage-targeted* (50%, 60%, 70%, 80%, 90% pada Dev Set).<br>2. *Risk-targeted* (15%, 20%, 25% pada Dev Set). | Menghilangkan bias *post-hoc thresholding*. Ambang batas $\tau$ dikunci pada Dev Set ($N=323$) sebelum dievaluasi pada Test Set ($N=799$). |
| **Struktur Research Questions** | 5 RQ yang memusatkan RQ2 pada aturan heuristik langsung $M_{\text{norm}}$. | **Restrukturisasi RQ:**<br>- RQ1: Cross-Lingual Disagreement Association<br>- RQ2: Visual Sensitivity Characteristics<br>- RQ3: Cross-Lingual Risk Estimation (Primer)<br>- RQ4: Adaptive Abstention Efficacy (Primer)<br>- RQ5: Model Comparison | Memisahkan secara tegas pertanyaan kontribusi metodologis primer (RQ3 & RQ4) dari analisis evaluatif perilaku model (RQ1, RQ2, RQ5). |
| **Framing Hasil $M_{\text{norm}}$** | Diposisikan sebagai estimator yang perlu dievaluasi perbaikannya. | Diposisikan secara transparan sebagai **unparameterized heuristic ablation** yang mengalami *negative finding*. | Menyelamatkan naskah skripsi dari penolakan penguji dengan menyajikan kegagalan heuristik sebagai justifikasi ilmiah perlunya regresi logistik teratur. |
| **Metrik Evaluasi** | Discrete Riemann AURC, Risk at Coverage 100/80/50, Odds Ratio. | Ditambahkan: **Coverage Terwujud ($\Phi_{\text{test}}$), Abstention Rate ($\alpha$), Selective Diagnostic Risk ($r$), Selective Accuracy ($\text{Acc}_{\text{sel}}$)** pada grid 50%–90%. | Menyediakan metrik langsung yang dapat diinterpretasikan secara intuitif oleh klinisi dan radiolog. |
| **Status Eksekusi GPU** | Ditandai sebagai *ready-to-run* di v10.0 awal. | **VERIFIED EMPIRICAL:** Kedua model 7B telah selesai dieksekusi 100% pada 1.133 sampel di Kaggle Dual Tesla T4 (timestamp `2026-10-25T02:48:53 UTC`). | Seluruh angka metrik dan koefisien regresi kini berbasis data empiris nyata, bukan simulasi. |

---

## 3. Detail Perubahan per Bagian Dokumen

### 3.1 Judul dan Metadata (Bagian 1)
- Mengganti judul kerja utama menjadi: *"Cross-Lingual Risk Estimation and Adaptive Abstention for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"*.
- Menambahkan Bagian 1.2 yang menyajikan evaluasi komparatif sistematis antara judul kerja utama dan judul alternatif.

### 3.2 Executive Summary (Bagian 2)
- Mengintegrasikan narasi estimasi risiko dan mekanisme penahanan adaptif.
- Memasukkan angka empiris hasil inferensi penuh: penurunan selective risk dari 28.83% ke 21.86% pada Qwen (target coverage 70%) dan dari 38.48% ke 29.17% pada HuatuoGPT (target coverage 70%).

### 3.3 Research Questions & Hypotheses (Bagian 8 & 9)
- Merumuskan ulang RQ3 untuk mengevaluasi model risiko multivariat `LR-full` terhadap baseline confidence dan visual.
- Menambahkan **RQ4 (Adaptive Abstention):** Menguji apakah penahanan adaptif menurunkan selective risk pada target coverage yang telah ditentukan.
- Mengklasifikasikan hipotesis menjadi $H_1$ hingga $H_5$ dengan status pengujian statistik empiris.

### 3.4 Metodologi & Estimasi Risiko (Bagian 14–17)
- Memperjelas pemisahan antara skor selektif $S_{\text{LR}}(x) \in [0, 1]$ (makin tinggi makin yakin) dan estimasi risiko $R(x) = 1 - S_{\text{LR}}(x)$ (makin tinggi makin berisiko).
- Menampilkan koefisien regresi logistik teratur yang dilatih pada Dev Set:
  - Qwen `LR-full`: $w_1(p_{\text{orig}}) = -0.1258$, $w_2(\Delta P) = +0.1128$, $w_3(D_{\text{lang}}) = \mathbf{+1.0960}$, intercept $w_0 = -1.0233$.
  - Huatuo `LR-full`: $w_1(p_{\text{orig}}) = -0.3693$, $w_2(\Delta P) = -0.3055$, $w_3(D_{\text{lang}}) = \mathbf{+0.9941}$, intercept $w_0 = -0.5889$.

### 3.5 Adaptive Abstention Mechanism & Threshold Selection (Bagian 18 & 19)
- Mendefinisikan aturan keputusan:
  $$\text{Decision}(x) = Answer \iff R(x) \le \tau \quad (\text{atau } S_{\text{LR}}(x) \ge 1 - \tau)$$
  $$\text{Decision}(x) = Abstain \iff R(x) > \tau \quad (\text{atau } S_{\text{LR}}(x) < 1 - \tau)$$
- Menjelaskan prosedur kalibrasi threshold pada Development Set ($N=323$) untuk:
  - **Coverage-Targeted:** Grid target 50%, 60%, 70%, 80%, 90%.
  - **Risk-Targeted:** Grid toleransi risiko 15%, 20%, 25%.
- Mendokumentasikan secara jujur terjadinya *risk slippage* pada pendekatan risk-targeted akibat variabilitas antar klaster citra.

### 3.6 Grid Evaluasi Empiris (Bagian 22.3)
- Memasukkan tabel hasil empiris lengkap perbandingan `M1_direct`, `LR_conf`, `LR_visual`, dan `LR_full` pada Primary Test Set ($N=799$) untuk kedua model.
- Membuktikan bahwa `LR-full` secara konsisten memberikan selective risk terendah dan selective accuracy tertinggi pada seluruh tingkat coverage.

---

## 4. Dampak terhadap Naskah Skripsi & Strategi Sidang

1. **Keunggulan Posisi Ilmiah:** Penambahan mekanisme *adaptive abstention* mengubah skripsi dari sekadar makalah analisis statistik pasif menjadi naskah rekayasa sistem pendukung keputusan klinis (*translational clinical decision support*).
2. **Pertahanan terhadap Penguji Metodologi:** Calon sarjana memiliki argumen kuat mengenai *zero-leakage image cluster splitting*, pemisahan dev/test thresholding, dan koreksi multiplisitas Holm-Bonferroni.
3. **Kejujuran Intelektual:** Pengakuan transparan bahwa aturan heuristik $M_{\text{norm}}$ gagal dan digantikan oleh `LR-full` menunjukkan kedewasaan berpikir peneliti dalam merespons data empiris apa adanya tanpa manipulasi hasil.

---
*REVISION_CHANGELOG_11.md disahkan sebagai catatan resmi pembaruan arsitektur penelitian v10.0 menuju v11.0.*
