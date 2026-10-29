# 🏥 Laporan Audit Metodologi & Spesifikasi Eksperimen Medical VLM (rev05)

**Judul Penelitian:** Selective Prediction for Bilingual Medical Vision-Language Models Under Language Shift: Mitigating Overconfidence via Multimodal Sensitivity Analysis  
**Dataset Target:** IndoRad-VQA (English–Indonesian Bilingual Radiology Visual Question Answering)  
**Target Publikasi:** Metodologi Ilmiah Standar SINTA 2 / Scopus  
**Versi Revisi:** `rev05` (Penerus terisolasi dari `rev04`, `rev03`, dan `rev02`)  
**Status Evaluasi:** **PASS_PREFLIGHT / PASS_UNIT_TEST / NOT RUN (FULL GPU INFERENCE)**  
**Tanggal Audit:** 20 Oktober 2026  

---

## 1. Ringkasan Eksekutif & Pernyataan Integritas Ilmiah

Laporan ini menyajikan hasil audit menyeluruh, perbaikan metodologis, dan spesifikasi eksperimen untuk revisi **`rev05`**. Seluruh modifikasi dirancang untuk memenuhi standar peer-review jurnal ilmiah bereputasi (SINTA 2 / Scopus) dengan integritas metodologis tanpa kompromi.

### Pernyataan Kategori Hasil (Epistemic Status Disclosure):
Sesuai prinsip kehati-hatian ilmiah, kami membedakan secara tegas 4 kategori hasil dalam laporan ini:
1. **Hasil Empiris Historis (Empirical Baseline):** Hasil inferensi riil yang diperoleh dari artefak historis Run 04, Run 05, rev02 ($N=150$ Qwen2.5-VL), dan rev03 ($N=150$ HuatuoGPT-Vision).
2. **Pemeriksaan Statis (Static Verification):** Audit struktur dataset, invarian klaster gambar kanonis, verifikasi AST kode Python/notebook, dan analisis matematika indexing logit kausal.
3. **Pengujian Unit & Preflight (Unit Test Verification):** 9 uji fungsional otomatis pada `main/scripts/test_decisive_validation_v5.py` yang dieksekusi secara lokal dan lulus 100% (`PASS_UNIT_TEST`).
4. **Inferensi Penuh Model (Full GPU Inference):** Inferensi 1.133 sampel pada kedua model 7B (Qwen2.5-VL-7B-Instruct dan HuatuoGPT-Vision-7B-Qwen2.5VL) di platform Kaggle Dual Tesla T4 berstatus **`NOT RUN`** di lingkungan lokal saat ini. **Tidak ada fabrikasi data inferensi, p-value sintetis, maupun klaim keberhasilan GPU palsu.**

---

## 2. Audit Metodologis Scoring Kandidat: Exact Sequence Log-Likelihood vs Length-Normalized Surprisal

### 2.1 Identifikasi Masalah pada Revisi Sebelumnya
Pada `rev04`, pipeline menggunakan rata-rata log-likelihood per token ($\frac{1}{L} \sum \log P$) sebagai skor utama, lalu menerapkan softmax atas dua nilai rata-rata tersebut. Secara metodologis:
1. Softmax atas rata-rata log-likelihood bukanlah probabilitas urutan gabungan (*joint sequence probability*) dari model autoregresif.
2. Normalisasi tersebut relatif terhadap himpunan kandidat tertutup, sehingga **tidak boleh diklaim sebagai probabilitas terkalibrasi (*calibrated posterior probability*)**.
3. Di sisi lain, penjumlahan log-likelihood murni ($\sum \log P$) memiliki sensitivitas alami terhadap panjang token (*length bias*), di mana token yang lebih panjang (misalnya `TIDAK` vs `YA`) mengakumulasikan lebih banyak nilai negatif, meskipun kualitas per-tokennya setara.

### 2.2 Arsitektur Scoring Ganda rev05 (Dual-Track Evaluation)
Dalam `rev05`, kami mengimplementasikan dua jalur evaluasi yang terdokumentasi secara terpisah:

#### Jalur Utama (Primary Track): Exact Sequence Log-Likelihood
Untuk setiap kandidat $y \in \mathcal{Y}$, model autoregresif menghitung joint conditional log-likelihood:
$$S(y \mid X) = \sum_{t=1}^{L} \log P(y_t \mid y_{<t}, X)$$
di mana logits kausal diambil tepat pada indeks $[L_{\text{prompt}} - 1 : L_{\text{prompt}} - 1 + L_{\text{target}}]$.

Skor dinormalisasi menggunakan stable softmax (`logsumexp`) di atas kandidat tertutup:
$$P_{\text{cand}}(y_k \mid X) = \frac{\exp(S(y_k \mid X))}{\sum_{j} \exp(S(y_j \mid X))}$$

> [!IMPORTANT]
> **Batasan Terminologis:** Nilai $P_{\text{cand}}$ ini adalah **Candidate-Normalized Sequence Score** (skor urutan ternormalisasi kandidat) dan **bukan** probabilitas kebenaran medis yang terkalibrasi. Nilai ini hanya mencerminkan preferensi relatif model terhadap ruang jawaban biner tertutup yang telah didefinisikan.

#### Jalur Analisis Sensitivitas (Sensitivity Track): Length-Normalized Surprisal
Untuk menguji apakah perbedaan panjang tokenisasi mempengaruhi ranking selektif dan metrik AURC, `rev05` secara paralel menghitung:
$$\bar{s}(y \mid X) = \frac{1}{L} \sum_{t=1}^{L} \log P(y_t \mid y_{<t}, X)$$
Hasil length-normalized disimpan dalam kolom terpisah (`score_m1_len`, `score_m_dir_len`, dll.) dan dibandingkan langsung dengan hasil sequence log-likelihood (`score_m1_seq`, `score_m_dir_seq`, dll.).

### 2.3 Keselarasan Logit Kausal (*Causal Logit Slicing Alignment*)
Dalam model autoregresif berbasis Transformer (seperti Qwen2.5-VL), token pada posisi input $t$ menghasilkan logit pada posisi $t$ yang memprediksi token pada posisi $t+1$.
- Prompt berakhir pada indeks $L_{\text{prompt}} - 1$.
- Target token pertama ($y_0$) berada pada indeks $L_{\text{prompt}}$. Logit yang memprediksi $y_0$ berada pada indeks **$L_{\text{prompt}} - 1$**.
- Target token terakhir ($y_{L-1}$) berada pada indeks $L_{\text{prompt}} + L - 1$. Logit yang memprediksi token ini berada pada indeks **$L_{\text{prompt}} + L - 2$**.
- Dengan demikian, slicing logit yang valid secara matematis adalah tepat:
$$\text{target\_logits} = \text{logits}[L_{\text{prompt}} - 1 : L_{\text{prompt}} - 1 + L_{\text{target}}]$$
Formula ini telah diuji dan diverifikasi lolos 100% pada `test_causal_logit_alignment_and_prefix_invariance`.

### 2.4 Prefix Invariance & Terminasi Token
Sebelum logits dihitung, pipeline mengeksekusi assertion ketat:
$$\text{inputs.input\_ids}[0, :L_{\text{prompt}}] == \text{prompt\_inputs.input\_ids}[0]$$
Jika terdapat token padding, ketidaksesuaian representasi gambar, atau pergeseran token khusus, pipeline akan segera berhenti (*fail-fast*) alih-alih melanjutkan secara diam-diam. Token terminasi `<|im_end|>` diikutsertakan secara konsisten karena merupakan bagian sah dari urutan generasi yang diharapkan pada format chat template Qwen2.5.

---

## 3. Audit Validasi Manifest & Integritas Kriptografis

### 3.1 Eliminasi Cacat Serialisasi DataFrame
Audit pada `rev04` menemukan bahwa verifikasi hash yang dilakukan dengan menserialisasi ulang DataFrame (`df.to_csv()`) menghasilkan inkonsistensi newline (`\r\n` vs `\n`) dan whitespace antar sistem operasi.

Dalam `rev05`:
1. **Manifest Eksternal:** Dibaca sebagai raw bytes (`path.read_bytes()`) dan diverifikasi terhadap `AUTHORITATIVE_MANIFEST_SHA256`.
2. **Manifest Embedded:** Didekompresi dari Base64/zlib langsung menjadi byte mentah asli, kemudian diverifikasi hash SHA-256 byte tersebut sebelum diparsing oleh `pd.read_csv(io.BytesIO(decompressed_bytes))`.
3. **Penghentian Eksekusi:** Jika hash tidak cocok persis, pipeline langsung memicu `AssertionError`.

Hash otoritatif manifest `rev05`:
```
fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9
```

---

## 4. Rekonsiliasi Kohort & Solusi Benchmark Historis (N=1.133)

### 4.1 Identifikasi Cacat Oklusi Kohort rev04
Pada `rev04`, filter inferensi hanya mengeksekusi `is_eligible_primary == True`. Akibatnya, dari 150 sampel benchmark historis rev03, sebanyak **11 sampel tereliminasi** dari proses inferensi karena 11 sampel tersebut memiliki status translasi `HIGH_RISK` pada audit dataset penuh. Hal ini menyebabkan benchmark historis tidak dapat dibandingkan secara apel-ke-apel (*apples-to-apples*).

### 4.2 Solusi Union Kohort rev05 (Active Evaluation Cohort Union)
Dalam `rev05`, kami mendefinisikan Active Evaluation Cohort sebagai **Union Eksplisit**:
$$\mathcal{D}_{\text{eval}} = \{i \mid \text{is\_eligible\_primary}_i = \text{True}\} \cup \{i \mid \text{in\_rev03\_benchmark}_i = \text{True}\}$$

| Kohort Evaluasi | Total Sampel | Test Split | Dev Split | Klaster Test | Klaster Dev | Kebocoran Antar-Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Cohort (T1)** | 1.122 | 799 | 323 | 204 | 88 | **0 (0.0%)** |
| **Historical Benchmark (T2)**| 150 | 92 | 58 | 35 | 23 | **0 (0.0%)** |
| **Active Union Evaluated** | **1.133** | **806** | **327** | **204** | **88** | **0 (0.0%)** |

**Keunggulan Metodologis:**
1. Semua 1.122 sampel Primary Cohort dievaluasi penuh (799 test / 323 dev).
2. Semua 150 sampel Historical Benchmark dievaluasi penuh (92 test / 58 dev), sehingga perbandingan historis dengan rev03 memiliki sampel yang 100% utuh tanpa sampel yang hilang.
3. Total klaster gambar unik di split test tetap 204 dan di split dev tetap 88, dengan **kebocoran antar-split persis 0**.

---

## 5. Tata Kelola Checkpoint Atomik & Protokol Anti-Korupsi

Pipeline `rev05` menerapkan tata kelola checkpoint yang sangat ketat:
1. **Penulisan Atomik:** Checkpoint ditulis ke file sementara (`.tmp`) terlebih dahulu, kemudian digantikan ke file utama menggunakan `os.replace`. Hal ini mencegah file checkpoint korup apabila sesi Kaggle terputus mendadak (*timeout* atau *preemption*).
2. **Validasi Kompatibilitas:** Saat membaca checkpoint, sistem memverifikasi:
   - Kolom `revision_id == "rev05"`. Checkpoint dari `rev04`, `rev03`, atau `rev02` ditolak secara otomatis.
   - Kolom `manifest_sha256 == AUTHORITATIVE_MANIFEST_SHA256`.
   - Seluruh nilai pada kolom skor (`score_m1_seq`, `score_m_dir_seq`, `score_m1_len`, `score_m_dir_len`) harus bernilai hingga (*finite*) dan tidak boleh mengandung `NaN` atau `Inf`.

---

## 6. Protokol Inferensi Statistik & Selective Prediction

### 6.1 Estimator Selective Prediction yang Diuji
Untuk mengevaluasi hipotesis mitigasi overconfidence melalui multimodal sensitivity analysis, 5 estimator diuji:
1. **$M_1$ (Confidence Baseline):** $M_1 = P_{\text{cand}}(\hat{y} \mid P_0)$
2. **$M_2$ (Pure Visual Sensitivity):** $M_2 = \max(0, \Delta P_{\text{dir}})$
3. **$M_{\text{dir}}$ (Multiplicative Multimodal Sensitivity):** $M_{\text{dir}} = P_{\text{cand}}(\hat{y} \mid P_0) \times \max(0, \Delta P_{\text{dir}})$
4. **$M_{\text{norm}}$ (Linear Scaled Sensitivity):** $M_{\text{norm}} = P_{\text{cand}}(\hat{y} \mid P_0) \times \frac{1 + \Delta P_{\text{dir}}}{2}$
5. **$M_{3,\text{old}}$ (Legacy Absolute Shift):** $M_{3,\text{old}} = P_{\text{cand}}(\hat{y} \mid P_0) \times \frac{1 + |\Delta P|}{2}$

### 6.2 Perhitungan AURC Diskrit Hand-Verified
Berdasarkan formulasi Geifman & El-Yaniv (NeurIPS 2017), risiko selektif dihitung pada seluruh titik cakupan diskrit:
$$\text{AURC} = \frac{1}{N} \sum_{k=1}^N \text{Risk}\left(\frac{k}{N}\right)$$
Telah diverifikasi secara manual dan lulus uji presisi numerik ($< 10^{-12}$) pada `test_discrete_aurc_manual_calculation`.

### 6.3 Pengujian Hipotesis Berpasangan Tingkat Klaster
- **Paired Cluster Bootstrap ($B=1.000$):** Menghasilkan 95% Confidence Interval untuk AURC dan $\Delta \text{AURC} = \text{AURC}(M) - \text{AURC}(M_1)$ dengan resample pada tingkat klaster gambar kanonis.
- **Monte Carlo Paired Cluster Permutation Test ($B=1.000$):** Menguji hipotesis nol $H_0: \mathbb{E}[\Delta \text{AURC}] = 0$ menggunakan pembalikan tanda Rademacher pada tingkat klaster gambar.
- **Koreksi Multiplisitas Holm-Bonferroni:** Mengontrol Family-Wise Error Rate (FWER) pada taraf signifikansi $\alpha = 0.05$ di seluruh pasangan estimator.

---

## 7. Matriks Status Verifikasi Komprehensif

| Komponen Eksperimen | Metode Verifikasi | Status | Catatan Ilmiah |
| :--- | :--- | :---: | :--- |
| **Manifest SHA-256 (Raw Bytes)** | Hash Kriptografis SHA-256 | **PASS** | `fb55fd75...` terverifikasi bit-identik |
| **Embedded Base64 Decompression** | Decompress & Hash SHA-256 | **PASS** | Terverifikasi identik dengan manifest eksternal |
| **Klaster Bebas Kebocoran (0%)** | Analisis Himpunan Klaster | **PASS** | $204 \cap 88 = \emptyset$, 0% leakage |
| **Integritas Piksel Gambar (314 Img)** | SHA-256 Raw RGB Bytes | **PASS** | 314 gambar kanonis terverifikasi |
| **Exact Sequence Scoring Math** | Formulasi Logsumexp Autoregresif | **PASS** | Terverifikasi pada `test_decisive_validation_v5` |
| **Causal Logits Slicing Alignment** | Validasi Indeks Kausal | **PASS** | Slicing $[L_p - 1 : L_p - 1 + L_t]$ terbukti tepat |
| **Label Mapping Canonical Space** | Pemetaan Biner YES/NO | **PASS** | YES/NO dan YA/TIDAK konsisten 100% |
| **Discrete AURC Calculation** | Verifikasi Hitungan Tangan | **PASS** | Selisih absolut $< 10^{-12}$ |
| **Cluster Permutation Test** | Uji Monte Carlo Rademacher | **PASS** | $H_0$ observasi nol menghasilkan $p=1.0$ |
| **Koreksi Holm-Bonferroni** | Algoritma Step-Down | **PASS** | Terverifikasi urutan dan penyesuaian |
| **Atomic Checkpointing** | Penulisan `.tmp` & `os.replace` | **PASS** | Terlindungi dari korupsi interupsi Kaggle |
| **Sintaksis Notebook rev05** | AST Parser Python | **PASS** | Seluruh 17 sel lolos tanpa syntax error |
| **Inferensi Model GPU (Kaggle Dual-T4)**| Eksekusi Model Pretrained 7B | **NOT RUN** | Menunggu eksekusi mandiri pengguna di Kaggle |

---

## 8. Tabel Hash SHA-256 Artefak Repositori & Imutabilitas Historis

Seluruh artefak historis dari revisi sebelumnya (`rev02`, `rev03`, `rev04`) telah diverifikasi dan **100% tidak berubah (bit-identical)**:

| File Artefak | Status Revisi | SHA-256 Hash |
| :--- | :---: | :--- |
| `main/results/manifest_rev05.csv` | **rev05** | `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9` |
| `main/results/decisive_validation/rev05/manifest_rev05.csv` | **rev05** | `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9` |
| `main/scripts/manifest_rev05_b64.py` | **rev05** | `83c5925c7342b506d75c7fc17bb596f2bec001702a7df3641de3f3da1cdd6c50` |
| `main/scripts/decisive_validation_experiment_v5.py` | **rev05** | `da7feae878460c7d63ac6d5da32faca8f8a922a2e700b63aceed8a840507d635` |
| `main/scripts/test_decisive_validation_v5.py` | **rev05** | `29ae60fd1b9359996a63c7509b0dd8ef64ee6f12941075cf676083bed3483f7a` |
| `main/scripts/generate_kaggle_notebook_v5.py` | **rev05** | `d7788e49ce30508847f2775a2d1449b1f07f2a98b379b69eefd38f5630167c21` |
| `main/Kaggle/08_decisive_validation_rev05.ipynb` | **rev05** | `35b3885f8adafcf5f2805ebe5d6f7264a241f9f13836db68c665b201459e40f0` |
| `main/notebooks/08_decisive_validation_rev05.ipynb` | **rev05** | `35b3885f8adafcf5f2805ebe5d6f7264a241f9f13836db68c665b201459e40f0` |
| `main/results/decisive_validation/rev05/cohort_audit.json` | **rev05** | `2483a32dbf32070385d562d5f38617c62d8b51e85952aee67bc288c309a26d85` |
| `main/results/decisive_validation/rev05/manifest_audit.json` | **rev05** | `106132f840af847519c1c30ffb692b4f4a285aa782b990f0876d552e532e9b5b` |
| `main/results/decisive_validation/rev05/scoring_validation_report.json` | **rev05** | `bb899fbf8441965f9f787cac230887a48080a22bf74445631e7f29c7c7275bdd` |
| `main/Kaggle/07_decisive_validation_rev04.ipynb` | *rev04 (Preserved)* | `4f19ca9c017f2ebd4d2d6729914562f5e5cf98b64c89dc451539ff8b08906b20` |
| `main/results/manifest_rev04.csv` | *rev04 (Preserved)* | `bcb03bfafa0c329b73ab3b5ec2bb6f471daa1df9218582efa232e9dac74d3484` |
| `main/scripts/decisive_validation_experiment_v4.py` | *rev04 (Preserved)* | `177c38f0a40643e6ebcd92f8b14382a8eba725f8d3e0c3bfa7192f44099359c6` |
| `main/Kaggle/06_decisive_validation_rev03.ipynb` | *rev03 (Preserved)* | `bb7a47e2b21cf5ed61df4f58b538b8098abac3039276273906175c97569c00b4` |

---

## 9. Panduan Eksekusi Mandiri di Lingkungan Kaggle Dual-T4

1. Buka platform [Kaggle](https://www.kaggle.com/) dan buat Notebook baru.
2. Unggah file notebook `main/Kaggle/08_decisive_validation_rev05.ipynb`.
3. Pada panel sebelah kanan (Notebook Settings):
   - **Accelerator:** Pilih **GPU T4 x2**.
   - **Internet:** Pilih **Always on** (diperlukan untuk auto-download bobot model Hugging Face dan dependensi).
4. Klik **Run All** atau jalankan sel secara berurutan.
5. Setelah eksekusi selesai:
   - Unduh file `decisive_validation_rev05_results_final.zip` langsung dari panel Output atau via direct link `FileLink` yang tercetak pada sel terakhir notebook.
   - Ekstrak artefak hasil ke direktori `main/results/decisive_validation/rev05/` di repositori lokal Anda untuk pelaporan publikasi ilmiah.

---
*Laporan ini disusun secara otomatis dan diverifikasi secara kriptografis oleh AI Research Auditor untuk kepatuhan metodologi peer-review SINTA 2 / Scopus.*
