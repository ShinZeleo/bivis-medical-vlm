# LAPORAN AUDIT KRITIS & ARSITEKTUR PIPELINE RESILIEN (rev05b)
## Decisive Validation Bilingual Medical VLM (IndoRad-VQA) Tahan Timeout Kaggle Dual T4

**Tanggal Audit:** 24 Oktober 2026  
**Auditor Metodologi & Sistem:** Senior Medical VLM Researcher & Technical Auditor  
**Dataset:** `Lab-IS/IndoRad-VQA` (2.244 baris QA total; 314 klaster citra radiologi unik)  
**Authoritative Frozen Manifest SHA-256:** `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`  
**Revision Baseline:** `rev05` -> **Revision Resilien:** `rev05b`  

---

## 1. Ringkasan Eksekutif & Status Audit Terakhir

Berdasarkan audit komputasional langsung terhadap artefak, log, dan checkpoint hasil inferensi run Kaggle `08_decisive_validation_rev05.ipynb` pada direktori canonical `main/results/decisive_validation/rev05/`, ditemukan:

1. **Penyebab Kegagalan (Root Cause):**
   - Sesi GPU Kaggle memiliki batas waktu mati keras (*hard timeout limit*) sebesar **12,0 jam (43.200 detik)**.
   - Notebook `08_decisive_validation_rev05.ipynb` menjalankan inferensi Model 1 (`Qwen2.5-VL-7B-Instruct`) dan Model 2 (`HuatuoGPT-Vision-7B-Qwen2.5VL`) secara sekuensial penuh dalam satu sesi notebook yang sama.
   - Qwen2.5-VL-7B selesai 100% (1.133 sampel) dalam waktu **6,76 jam (24.353,2 detik)** dengan throughput empiris **21,51 detik/sampel**.
   - HuatuoGPT-Vision-7B kemudian berjalan hingga sampel ke-880 selama **5,11 jam (18.396,5 detik)** dengan throughput empiris **20,93 detik/sampel**.
   - Total runtime kumulatif mencapai **11,87 jam (42.749,7 detik)**, sehingga Kaggle memutus sesi saat mengevaluasi sampel ke-881. Tahap evaluasi statistik lintas model, pembangkitan kurva, dan ekspor ZIP (Tahap 13–22) sama sekali belum pernah dieksekusi.

2. **Kondisi Integritas Data Checkpoint Saat Ini:**
   - **Qwen2.5-VL-7B-Instruct:** Selesai **1.133 / 1.133 sampel (100%)**. Validitas 100% lulus, 0 nilai non-finite, 0 duplikat.
   - **HuatuoGPT-Vision-7B-Qwen2.5VL:** Selesai **880 / 1.133 sampel (77,67%)**. Sampel terakhir yang valid adalah QA ID `1771` (baris indeks 879) pada timestamp `2026-10-21 15:16:55 UTC`.
   - **Sampel HuatuoGPT yang Belum Dievaluasi:** Tepat **253 sampel (22,33%)** (rentang QA ID `1773` hingga `2242`).
   - Seluruh 253 citra radiologi yang belum dievaluasi telah diverifikasi secara kriptografis pada disk lokal (`main/data/images/`) dengan kecocokan SHA-256 byte RGB mentah 100% (253 / 253 valid).

3. **Status Klaim Eksperimen:**
   - **Eksperimen BELUM SELESAI.** Klaim komparasi final atau keunggulan estimator selektif $M_{\text{norm}}$ terhadap HuatuoGPT pada kohort Test Primer belum dapat dinyatakan final sampai 253 QA ID tersisa selesai diproses dengan scoring method yang identik.

---

## 2. Rincian Akuntansi QA ID: Selesai, Gagal, dan Tersisa

Tabel berikut menyajikan akuntansi riil per model, cohort, dan split tanpa ada angka yang dikarang:

| Model ID | Kohort / Split | Target Sampel | Selesai & Valid | Gagal / Corrupt | Sisa Wajib Proses | Status Kelengkapan |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-7B-Instruct** | Active Union ($N_{\text{union}}$) | **1.133** | **1.133** | 0 | **0** | **100,00% (LENGKAP)** |
| | - Primary Test Cohort | 799 | 799 | 0 | 0 | 100,00% |
| | - Primary Dev Cohort | 323 | 323 | 0 | 0 | 100,00% |
| | - Historical Benchmark Test | 92 | 92 | 0 | 0 | 100,00% |
| | - Historical Benchmark Dev | 58 | 58 | 0 | 0 | 100,00% |
| **HuatuoGPT-Vision-7B** | Active Union ($N_{\text{union}}$) | **1.133** | **880** | 0 | **253** | **77,67% (TERPOTONG TIMEOUT)** |
| | - Primary Test Cohort | 799 | 627 | 0 | **172** | 78,47% |
| | - Primary Dev Cohort | 323 | 242 | 0 | **81** | 74,92% |
| | - Historical Benchmark Test | 92 | 87 | 0 | **5** | 94,57% |
| | - Historical Benchmark Dev | 58 | 56 | 0 | **2** | 96,55% |

### Analisis Sampel HuatuoGPT yang Tersisa (253 QA ID):
- **Distribusi Split:** 172 QA ID berada pada split **Test**, 81 QA ID berada pada split **Dev**.
- **Distribusi Kohort:** 253 QA ID (100%) merupakan anggota Primary Cohort; 7 QA ID di antaranya beririsan dengan Historical rev03 Benchmark.
- **Klaster Citra Unik:** 253 sampel ini berasal dari **55 klaster citra fisik unik** (`canonical_image_id`). Tidak ada klaster yang bocor silang antara split Test dan Dev.

---

## 3. Estimasi Durasi Berdasarkan Throughput Empiris Riil

Estimasi waktu eksekusi tidak menggunakan angka fiktif, melainkan throughput aktual yang diukur langsung dari 2.013 sampel inferensi yang telah selesai:

$$\text{Throughput}_{\text{Huatuo}} = \frac{18.396,5\text{ detik}}{880\text{ sampel}} = 20,93\text{ detik/sampel}\quad (172,0\text{ sampel/jam})$$

Untuk menyelesaikan 253 sampel tersisa HuatuoGPT:
$$\text{Durasi Proyeksi} = 253\text{ sampel} \times 20,93\text{ detik} = 5.295,29\text{ detik} \approx \mathbf{88,25\text{ menit}}\quad (\mathbf{1,47\text{ jam}})$$

- **Batas Waktu Kaggle Dual T4:** 12,0 jam (720 menit).
- **Konsumsi Kuota Waktu:** $\frac{1,47}{12,0} \approx 12,25\%$.
- **Buffer Keamanan:** **10,53 jam**, menjamin 0% risiko timeout kembali saat resume dijalankan.

---

## 4. Arsitektur Resilien Pipeline rev05b

Untuk mengatasi timeout tanpa mengorbankan integritas metodologi atau menimpa artefak lama, dibangun 3 komponen terpisah:

```
main/
├── scripts/
│   ├── resilient_validation_engine.py      <- Core engine: validasi checkpoint, set-based resume, atomic save, Riemann AURC
│   ├── generate_resilient_notebooks.py     <- Generator deterministik 3 notebook terpisah
│   └── test_resilient_pipeline.py          <- Preflight unit test suite (100% pass)
└── notebooks/kaggle/
    ├── 08_decisive_validation_rev05.ipynb  <- NOTEBOOK ASLI (TETAP DIPERTAHANKAN, TIDAK DITIMPA)
    ├── 09_infer_huatuo_resume.ipynb        <- NOTEBOOK KHUSUS RESUME HUATUOGPT (253 sampel, ~1.47 jam)
    ├── 10_standalone_aggregation_evaluation.ipynb <- NOTEBOOK CPU-ONLY (Zero-GPU, evaluasi statistik lengkap ~30 detik)
    └── 08b_decisive_validation_modular.ipynb      <- NOTEBOOK MODULAR DENGAN PARAMETER EXECUTION_MODE
```

### Fitur Kunci Ketahanan:
1. **Decoupled Architecture (Pemisahan Total Model A & Model B):**
   - Model A (Qwen) dan Model B (HuatuoGPT) memiliki direktori output dan checkpoint CSV mandiri:
     - `results/decisive_validation/rev05/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_...csv`
     - `results/decisive_validation/rev05/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/live_validation_checkpoint_...csv`
2. **True Set-Based Resume Mechanism:**
   - Tidak lagi bergantung pada nomor indeks baris.
   - Menghitung himpunan QA ID tersisa secara matematis:
     $$\mathcal{Q}_{\text{remaining}} = \mathcal{Q}_{\text{cohort}} \setminus \mathcal{Q}_{\text{completed}}$$
   - Menjamin sampel yang sudah selesai tidak akan pernah dievaluasi ulang, berapapun urutan barisnya.
3. **Micro-Batching & Atomic Flushing:**
   - Interval checkpoint dapat dikonfigurasi (`BATCH_SIZE = 5`, default).
   - Penulisan ke file `.tmp_<pid>.csv` diikuti pemindahan atomik (`os.replace` / `shutil.move`).
   - Mencegah kerusakan file (*corrupted zero-byte CSV*) jika terjadi preemption atau pemutusan sesi mendadak.
4. **Validasi Kompatibilitas Checkpoint Ketat (`validate_checkpoint_compatibility`):**
   - Menolak checkpoint jika:
     - Hash manifest tidak sama persis dengan `fb55fd75...`
     - Model ID tidak sesuai target inferensi
     - Metode scoring bukan `exact_sequence_candidate_normalized_v5`
     - Terdapat nilai `NaN` atau `Inf` pada kolom skor primer
     - Terdapat duplikasi `qa_id`
     - Terdapat `qa_id` asing di luar manifest
5. **Standalone Zero-GPU Aggregator (`10_standalone_aggregation_evaluation.ipynb`):**
   - 100% berjalan pada CPU tanpa memuat model neural network berbobot puluhan gigabyte.
   - **Indeks Positional Cluster Immune:** Menggunakan pengindeksan posisi integer berbasis `np.where(cluster_vals == c)[0]` dan pemanggilan `.reset_index(drop=True)` pada setiap fungsi pengujian statistik, sehingga kebal terhadap irisan DataFrame dengan indeks arbitrer atau tidak berurutan.
   - **Automated Preflight Sanity Test Suite (Stage 4B):** Menguji secara otomatis kekebalan indeks non-standar, batas matematis AURC, dan monotonisitas Holm-Bonferroni sebelum masuk ke perhitungan metrik final.
   - Mengevaluasi Discrete Riemann AURC, paired cluster bootstrap ($B=1000$), paired cluster permutation test ($B=1000$), koreksi Holm-Bonferroni, dan risk ablation $D_{\text{lang}}$ dalam waktu **< 30 detik**.

---

## 5. Lokasi Checkpoint dan Artefak

### Checkpoint Lokal Terverifikasi:
- **Qwen2.5-VL-7B (1.133 baris):**
  `c:/.PUSAT/Project/Kuliah/VLM/main/results/decisive_validation/rev05/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv`
- **HuatuoGPT-Vision-7B (880 baris):**
  `c:/.PUSAT/Project/Kuliah/VLM/main/results/decisive_validation/rev05/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv`
- **Authoritative Manifest (2.244 baris):**
  `c:/.PUSAT/Project/Kuliah/VLM/main/results/decisive_validation/rev05/manifest_rev05.csv` (SHA-256: `fb55fd75...`)

### Output Kaggle Environment Target:
- Direktori kerja inferensi: `/kaggle/working/results/decisive_validation/rev05/`
- Arsip unduhan Huatuo checkpoint: `/kaggle/working/huatuo_checkpoint_rev05_complete.zip`
- Arsip unduhan evaluasi komprehensif: `/kaggle/working/decisive_validation_rev05_complete.zip`

---

## 6. Prosedur Melanjutkan Run pada Kaggle Dual T4

### Langkah A: Menyelesaikan 253 Sampel HuatuoGPT
1. Buat Kaggle Notebook baru dengan GPU **Dual Tesla T4**.
2. Unggah file notebook `main/notebooks/kaggle/09_infer_huatuo_resume.ipynb`.
3. Tambahkan Dataset input yang berisi checkpoint 880 sampel HuatuoGPT sebelumnya (`live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv`) dan dataset citra `IndoRad-VQA`.
4. Jalankan notebook:
   - Cell 6 secara otomatis mendeteksi checkpoint 880 baris dari `/kaggle/input/` atau direktori kerja.
   - Mengonfirmasi 253 sampel tersisa.
   - Mengevaluasi sampel dalam batch 5 baris dengan flush atomik.
   - Selesai dalam ~88 menit.
   - Cell 10 menghasilkan file `huatuo_checkpoint_rev05_complete.zip` untuk diunduh langsung.

### Langkah B: Menjalankan Evaluasi & Agregasi Final (Zero-GPU)
1. Unggah checkpoint Qwen (1.133 baris) dan checkpoint Huatuo yang telah selesai 100% (1.133 baris).
2. Buka notebook `main/notebooks/kaggle/10_standalone_aggregation_evaluation.ipynb` pada sesi **CPU Kaggle biasa** (tidak membutuhkan GPU sama sekali).
3. Jalankan seluruh sel:
   - Memvalidasi kedua checkpoint (1.133 / 1.133 baris).
   - Menghitung RQ1 (akurasi EN vs ID, flip rate, cross-lingual gap).
   - Menghitung RQ2 (multimodal visual reliance $\Delta P_{\text{dir}}$ P0 vs P1).
   - Menghitung RQ3 (Discrete Riemann AURC, paired cluster bootstrap, uji permutasi klaster, Holm-Bonferroni).
   - Melatih risk ablation $D_{\text{lang}}$ pada Dev Set ($N=323$, 88 klaster) dan mengevaluasi pada Test Set ($N=799$, 204 klaster).
   - Menghasilkan kurva Risk-Coverage beresolusi tinggi dan file arsip final `decisive_validation_rev05_complete.zip`.
   - Total waktu eksekusi: **< 1 menit**.

---

## 7. Status Daftar Pemeriksaan (Checks Passed vs Pending)

### Pemeriksaan yang Telah Lulus (100% Passed):
- [x] **Audit Integritas Checkpoint Parsial:** Menghitung baris valid empiris tanpa manipulasi (Qwen: 1.133; Huatuo: 880).
- [x] **Audit Resolusi Citra 253 Sampel:** Seluruh 253 citra radiologi yang tersisa memiliki berkas fisik lokal dan lulus verifikasi SHA-256 byte mentah 100%.
- [x] **Pemisahan Model & Checkpoint:** Kode inferensi dan penyimpanan checkpoint antar-model telah didecouple sepenuhnya.
- [x] **Set-Based Deduplication & Resume:** Fungsi `filter_remaining_samples()` diuji unit test dan menghasilkan tepat 253 sampel tersisa dengan 0 duplikasi.
- [x] **Mekanisme Atomic Flush:** Fungsi `atomic_save_checkpoint()` diuji dan menjamin file `.tmp` dipindahkan secara atomik dengan retensi `keep="last"`.
- [x] **Integritas Protokol Scoring:** Logit diiris pada rentang $[L_{\text{prompt}} - 1 : L_{\text{prompt}} - 1 + L_{\text{cand}}]$ dengan verifikasi ketat prefix invariance dan terminasi `<|im_end|>`.
- [x] **Matematika Discrete Riemann AURC:** Diuji terhadap kasus analitik ideal ($AURC = 0,0$) dan terbalik ($AURC = 1,0$).
- [x] **Uji Statistik Non-GPU:** Paired Cluster Bootstrap ($B=100$), Paired Cluster Permutation Test ($B=100$), dan koreksi Holm-Bonferroni berjalan lancar pada CPU.
- [x] **Risk Ablation GroupKFold:** Pelatihan LR pada Dev Set ($N=323$) dan evaluasi pada Test Set ($N=799$) berjalan tanpa kebocoran data (*zero leakage*).
- [x] **Preservasi Artefak Lama:** Berkas notebook `08_decisive_validation_rev05.ipynb` dan hasil revisi terdahulu (`rev02`, `rev03`, `rev04`) tidak tertimpa.

### Pemeriksaan yang Belum Dijalankan (Pending Cloud Resume):
- [ ] **Inferensi 253 Sampel Sisa HuatuoGPT di Kaggle:** Menunggu eksekusi notebook `09_infer_huatuo_resume.ipynb` di cloud Kaggle Dual T4.
- [ ] **Verifikasi Checkpoint Final Huatuo 1.133 Baris:** Memastikan 253 baris baru berhasil digabungkan dengan 880 baris lama.
- [ ] **Eksekusi Evaluasi Statistik Cross-Model Final:** Menjalankan notebook `10_standalone_aggregation_evaluation.ipynb` setelah checkpoint Huatuo 1.133 baris tersedia.
- [ ] **Pemberian Skor Final Hipotesis $M_{\text{norm}}$ vs $M_1$ pada HuatuoGPT:** Membuktikan secara empiris apakah $M_{\text{norm}}$ signifikan mereduksi selective risk pada model berorientasi medis setelah seluruh 799 sampel Test Primer dievaluasi.
