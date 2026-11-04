# REGISTER OF REMAINING ISSUES & RESOLUTIONS
## Medical Vision-Language Models Under English-Indonesian Language Shift
**Target Milestone:** SINTA 2 Journal Submission & Thesis Defense  
**Audit Source:** Second-Stage Empirical & Statistical Audit (`AUDIT-STAT-STAGE2-REV05`)  
**Date Logged:** 2026-11-01 03:36:02 UTC  
**Status:** `RESOLVED VIA FORMAL DOCUMENTATION & RECONCILIATION`

---

## Executive Overview
Audit tahap kedua mengidentifikasi empat isu teknis, dokumentasi, dan pembingkaian narasi (*framing*). Seluruh isu telah dianalisis secara mendalam dan diberikan panduan mitigasi definitif untuk penulisan naskah artikel ilmiah. **Tidak ada satupun isu yang membatalkan kesimpulan utama penelitian atau memerlukan eksekusi komputasi baru.**

---

## Structured Issue Registry

### Issue 1: Historical Baseline $M_1$ Misattribution in Legacy Drafts
- **Kategori:** *Data Reporting & Metric Consistency*
- **Tingkat Keparahan:** `MEDIUM` (Dapat menimbulkan kebingungan pembaca jika tidak direkonsiliasi).
- **Deskripsi Masalah:**
  Pada dokumen draf awal (`RESEARCH_MASTER(10).md` dan laporan pendahuluan), nilai baseline confidence AURC dilaporkan sebesar **0.2434** untuk Qwen dan **0.2513** untuk Huatuo. Audit numerik membuktikan bahwa angka tersebut adalah test AURC dari model dua-fitur `LR_visual` ($M_1 + \Delta P_{dir}$). Baseline kepercayaan tunggal murni ($M_1$) sebenarnya adalah **0.2422** untuk Qwen dan **0.2884** untuk Huatuo.
- **Dampak Ilmiah:**
  - Untuk Huatuo: Perbaikan ini justru menguntungkan naskah. Penurunan AURC aktual dari $LR_{full}$ ($0.2179$) terhadap baseline murni $M_1$ ($0.2884$) adalah **$\Delta = -0.0705$ (penurunan $24.4\%$)**, jauh lebih impresif daripada klaim awal ($-0.0334$).
  - Untuk Qwen: Pergeseran dari $0.2434$ ke $0.2422$ adalah marjinal ($\Delta = 0.0012$).
- **Solusi & Panduan Manuskrip:**
  1. Manuskrip harus mempublikasikan tabel ablasi lengkap dengan empat tingkat:
     - Baseline Kepercayaan Murni ($M_1$ / `LR_conf`): Qwen = $0.2422$, Huatuo = $0.2884$.
     - Model Sensitivitas Visual (`LR_visual`): Qwen = $0.2434$, Huatuo = $0.2513$.
     - Model Terpelajar Penuh (`LR_full`): Qwen = $0.2053$, Huatuo = $0.2179$.
     - Model Heuristik Normalisasi ($M_{norm}$): Qwen = $0.2753$, Huatuo = $0.2552$.
  2. Dokumentasikan bahwa `0.2434` dan `0.2513` pada catatan historis merupakan `LR_visual`.
- **Status:** **`RESOLVED`** (Tercatat resmi di `CLAIM_EVIDENCE_MATRIX.csv`).

---

### Issue 2: Rigorous Mathematical Phrasing of Permutation Tests
- **Kategori:** *Statistical Methodology Rigor*
- **Tingkat Keparahan:** `LOW-MEDIUM` (Ketepatan terminologi metodologis untuk *peer-review*).
- **Deskripsi Masalah:**
  Draf historis terkadang menyebut pengujian signifikansi sebagai "Exact Cluster Permutation Test". Dengan $K=204$ klaster fisik radiografi, ruang permutasi tanda penuh adalah $2^{204} \approx 2.57 \times 10^{61}$, sehingga enumerasi eksak secara komputasi mustahil dilakukan.
- **Dampak Ilmiah:**
  Reviewer statistik yang teliti dapat mempertanyakan klaim "exact".
- **Solusi & Panduan Manuskrip:**
  1. Dalam bab *Methods*, sebutkan pengujian secara presisi sebagai:  
     *"Monte Carlo Paired Cluster Sign-Flip Permutation Test with Rademacher cluster-level sign perturbation ($B=1,000$ iterations, random seed=42)"*.
  2. Laporkan batas kesalahan standar Monte Carlo secara eksplisit: $SE(\hat{p}) = \sqrt{\hat{p}(1-\hat{p})/B}$.
  3. Konfirmasikan bahwa p-value menggunakan koreksi sampel terhingga Davison & Hinkley (1997): $p = (1 + M) / (B + 1)$.
- **Status:** **`RESOLVED`** (Diverifikasi dan dirinci dalam `PERMUTATION_IMPLEMENTATION_AUDIT.md`).

---

### Issue 3: Epistemic Boundary of Visual Deletion Intervention
- **Kategori:** *Scientific Modesty & Clinical Grounding*
- **Tingkat Keparahan:** `MEDIUM` (Mencegah *overclaiming* saat sidang atau *peer-review* medis).
- **Deskripsi Masalah:**
  Intervensi penghapusan citra visual menggunakan kanvas hitam pekat ($P1$). Kanvas hitam merupakan stimulus yang sepenuhnya *out-of-distribution* (OOD) bagi model visi medis. Pada HuatuoGPT-Vision, $80.0\%$ sampel menunjukkan probabilitas yang lebih tinggi pada kanvas hitam daripada citra asli.
- **Dampak Ilmiah:**
  Klaim bahwa $\Delta P_{dir}$ merefleksikan "pemahaman klinis terhadap patologi radiologi" atau "grounding anatomis kausal" adalah klaim yang keliru (*epistemic overclaiming*).
- **Solusi & Panduan Manuskrip:**
  1. Jangan gunakan istilah "causal clinical reasoning" atau "anatomical comprehension" untuk intervensi kanvas hitam.
  2. Definisikan intervensi secara jujur sebagai:  
     *"Pixel-level visual sensitivity probe under an uninformative baseline intervention"*.
  3. Jelaskan bahwa nilai $\Delta P_{dir}$ yang negatif pada Huatuo mencerminkan fenomena *hallucinatory overconfidence on uninformative inputs*, dan fitur ini bekerja dalam regresi logistik sebagai penalti halusinasi (*hallucination penalty*).
- **Status:** **`RESOLVED`** (Ditegaskan dalam `RQ_DECISION_MATRIX.csv` dan `FINAL_RESEARCH_READINESS_REPORT.md`).

---

### Issue 4: Narrative Framing of Negative Result on $M_{norm}$
- **Kategori:** *Academic Framing & Theoretical Contribution*
- **Tingkat Keparahan:** `LOW` (Penyusunan narasi ilmiah).
- **Deskripsi Masalah:**
  Metode kombinasi aritmatika normalisasi min-max sederhana ($M_{norm} = \tilde{M}_1 + \Delta \tilde{P}_{dir}$) gagal meningkatkan performa selektif secara signifikan ($p = 0.2298$ pada Qwen, $p = 0.3846$ pada Huatuo). Pada Qwen, $M_{norm}$ bahkan memperburuk AURC dari $0.2422$ menjadi $0.2753$.
- **Dampak Ilmiah:**
  Peneliti pemula sering menganggap hasil non-signifikan sebagai kegagalan eksperimen dan mencoba menyembunyikannya.
- **Solusi & Panduan Manuskrip:**
  1. Jadikan kegagalan $M_{norm}$ sebagai **kontribusi ilmiah positif (negative finding)** dalam naskah.
  2. Jelaskan rasionalitas teoretisnya: Menggabungkan sinyal keyakinan token dan sensitivitas visual dengan bobot seragam ($1:1$) gagal karena skala ketidakpastian dan karakteristik distribusi kedua modalitas bersifat asimetris.
  3. Kontraskan kegagalan $M_{norm}$ dengan keberhasilan $LR_{full}$ untuk membuktikan bahwa pembobotan multivariat terpelajar (*learned multivariate combination*) adalah prasyarat mutlak dalam *multimodal selective prediction*.
- **Status:** **`RESOLVED`** (Diformulasikan dalam `FINAL_RESEARCH_READINESS_REPORT.md`).

---

## Ringkasan Kesiapan Aksi
| Isu | Area | Status Mitigasi | Siap Ditulis ke Naskah? |
|---|---|:---:|:---:|
| 1. Baseline Misattribution | Results / Table | Selesai (Angka diperbarui) | **YA** |
| 2. Permutation Phrasing | Methods | Selesai (Terminologi eksak Monte Carlo) | **YA** |
| 3. Visual Epistemic Boundary | Discussion | Selesai (Batasan klaim didefinisikan) | **YA** |
| 4. $M_{norm}$ Negative Finding | Discussion | Selesai (Dibingkai sebagai argumen teoretis) | **YA** |
