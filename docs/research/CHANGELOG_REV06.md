# CHANGELOG: AUDIT, REFACTORING & PROTOKOL rev06

Dokumen ini mencatat seluruh perubahan kode, refactoring arsitektur modul, dan spesifikasi protokol revisi baru (`rev06`) sesuai standar integritas ilmiah dan reproduktibilitas penelitian.

Perubahan dikelompokkan secara ketat menjadi 4 kategori:
1. **Perbaikan Bug (Bug Fixes)**
2. **Tambahan Non-Pengubah-Hasil (Non-Breaking Enhancements)**
3. **Perubahan Protokol rev06 (rev06 Protocol Specification)**
4. **Pembaruan Pasca-Review & Penguatan Audit (Post-Review Hardening)**

---

## 1. Perbaikan Bug (Bug Fixes)

* **Perbaikan Batas Resolusi $p$-value Permutasi (Task A5 & E2):**
  * *Isu:* Pada `RESEARCH_MASTER.md` v11 tertulis nilai $p = 0.0005$ pada uji permutasi klaster dengan $B=1000$. Secara matematis, dengan formula standar Laplace $p = (\text{extreme} + 1) / (B + 1)$, batas resolusi minimum pada $B=1000$ adalah $1/1001 \approx 0.000999$, sehingga $p = 0.0005$ adalah anomali komputasi.
  * *Perbaikan:* Jumlah permutasi dinaikkan menjadi $B = 10.000$ (resolusi minimum $1/10001 \approx 0.0001$). Uji ulang menghasilkan $p_{\text{raw}} = 0.00020$ dan $0.00030$ pada HuatuoGPT, mengonfirmasi signifikansi secara sah dan terkalibrasi.
* **Perbaikan Monotonisitas Penyesuaian Holm-Bonferroni (Task D7):**
  * *Isu:* Pada implementasi awal, pengurutan ulang dictionary tanpa pelacakan running maximum berpotensi menghasilkan pelanggaran monotonisitas jika nilai $p \times (m - r + 1)$ tidak monoton.
  * *Perbaikan:* Menambahkan penjaminan running maximum `running_max = max(running_max, adj_p)` pada `src/stats_engine.py:apply_holm_bonferroni` sehingga $p_{(1)} \le p_{(2)} \le \dots \le 1.0$.
* **Perbaikan Batas Numerik Wilson Score Interval (Task A1 & D7):**
  * *Isu:* Kasus ekstrem $k=0$ menghasilkan nilai residual floating point $3.47 \times 10^{-18}$ bukannya tepat $0.0$.
  * *Perbaikan:* Menambahkan penanganan batas eksak pada $k=0$ (lower bound = $0.0$) dan $k=n$ (upper bound = $1.0$) pada `src/stats_engine.py:wilson_ci`.
* **Perbaikan Path Direktori Pasca Restrukturisasi (Task D):**
  * *Isu:* File script `scripts/run_notebook_10.py` dan pencarian manifest di notebook 10 masih mengarah ke subdirektori lama `main/notebooks/...` dan `main/results/...`.
  * *Perbaikan:* Menyelaraskan seluruh path pencarian agar mendeteksi struktur root repository modern (`notebooks/kaggle/...`, `artifacts/experiments/...`, `data/manifests/...`).
* **Pencegahan Deprecation Warning `LogisticRegression` (Task D6):**
  * *Isu:* Inisialisasi `LogisticRegression(penalty="l2", C=1.0)` memicu peringatan di scikit-learn $\ge 1.8$.
  * *Perbaikan:* Mengganti menjadi `LogisticRegression(C=1.0, random_state=42)` yang mempertahankan penalti $L_2$ secara default dan terverifikasi menghasilkan koefisien yang 100% identik hingga 4 desimal dengan bobot pada RESEARCH_MASTER §17.
* **Perbaikan Dependensi Visualisasi RQ3 (`import matplotlib.pyplot as plt`):**
  * *Isu:* Sel RQ3 di notebook standalone mengasumsikan `plt` sudah terimpor dari sel sebelumnya; jika sel dijalankan terpisah, memicu `NameError`.
  * *Perbaikan:* Menambahkan `import matplotlib.pyplot as plt` eksplisit di awal sel RQ3 pada seluruh notebook (`rev05` dan `rev06`). Kategori: *bug fix, hasil tidak berubah*.
* **Pembersihan Variabel Dangling di Notebook Single-Model:**
  * *Isu:* Sel RQ3 memuat sisa baris `df_qwen_test = df_qwen[...]` dan `df_huatuo_test = df_huatuo[...]` yang berpotensi error karena dataframe lokal bernama `df_final`.
  * *Perbaikan:* Menghapus dua baris tidak terpakai tersebut. Kategori: *bug fix, hasil tidak berubah*.
* **Koreksi Label Progress Bar Qwen:**
  * *Isu:* Sel inferensi notebook `08_qwen` memuat `desc="HuatuoGPT Inference"`.
  * *Perbaikan:* Memperbaiki label menjadi `desc="Qwen2.5-VL Inference"`. Kategori: *bug fix, hasil tidak berubah*.

---

## 2. Tambahan Non-Pengubah-Hasil (Non-Breaking Enhancements)

* **Sentralisasi Modul Statistik (`src/stats_engine.py`) (Task D7):**
  * Mengekstrak seluruh fungsi statistik yang sebelumnya diduplikasi di notebook 08, 09, dan 10 ke dalam modul tunggal `src/stats_engine.py`.
  * Modul mencakup: `calculate_discrete_aurc`, `paired_cluster_permutation_test`, `paired_cluster_bootstrap`, `apply_holm_bonferroni`, `wilson_ci`, dan `run_mcnemar`.
* **Modul Metrik Keandalan Lanjutan (`src/metrics.py`) (Task C):**
  * Implementasi Expected Calibration Error (ECE, 15 bin) dan Brier Score.
  * Implementasi Confidently-Wrong Rate pada ambang batas $\ge 0.90$.
  * Implementasi Visual Flip Rate ($P_0$ vs $P_1$ kanvas hitam).
  * Implementasi Odds Ratio $D_{\text{lang}}$ vs `error_id` dengan uji eksak Fisher dan regresi logistik cluster-robust SE.
  * Implementasi pipeline adaptif Out-of-Fold (OOF) via GroupKFold ($K=5$) dengan unit test zero-leakage antar klaster citra.
  * Implementasi interpolasi selective risk pada cakupan nyata yang disamakan (50%, 60%, 70%, 80%, 90%).
* **Suite Pengujian Unit Komprehensif (`tests/`):**
  * Seluruh 46 tes di repository lulus 100% (`pytest tests/`), mencakup logika Gate (a) kelayakan pilot GPU dan metrik deskriptif (b).
* **Optimasi Caching Verifikasi Hash Gambar (Task D3):**
  * Menambahkan caching verifikasi pixel SHA-256 per `canonical_image_id` pada Stage 4 notebook inferensi sehingga 314 citra fisik unik hanya dihitung hash-nya satu kali, bukan diulang pada seluruh 1.133 baris QA.
* **Pembersihan Keamanan Remote Code (Task D4):**
  * Menguji dan menghapus flag `trust_remote_code=True` pada loading tokenizer dan model Qwen2.5-VL dan HuatuoGPT-Vision, membuktikan eksekusi aman natively didukung pada Transformers $\ge 4.49.0$.
* **Penyelarasan Dependensi Lingkungan (Task D5):**
  * Memperbarui `requirements.txt` dengan penambahan `bitsandbytes`, menaikkan batas `transformers>=4.49.0`, menghapus paket yang tidak diimpor.
  * Membuat file kunci dependensi deterministik `requirements-lock.txt` dari `pip freeze`.
* **Penyelarasan RESEARCH_MASTER.md ke Versi 12.0 (Task E):**
  * Menyelaraskan seluruh angka $p$-value dan resolusi permutasi $B=10.000$.
  * Menambahkan Bagian 21.3 ("Baseline Pembanding Eksternal") dan mendiskusikan secara jujur batas kemampuan `LR_full` terhadap baseline bahasa Inggris pada HuatuoGPT.
  * Menambahkan 4 poin keterbatasan baru pada Bagian 26 (Limitations).

---

## 3. Perubahan Protokol rev06 (rev06 Protocol Changes)

Perubahan ini membentuk **jalur evaluasi baru (`rev06`)** yang terisolasi dari `rev05` tanpa menimpa checkpoint atau kolom lama:

* **Multi-Token Candidate Sets via LogSumExp (Task B1):**
  * *Latar Belakang:* Diagnostik A1–A2 membuktikan bahwa string huruf besar kaku `TIDAK<|im_end|>` terfragmentasi menjadi 4 sub-token `['T', 'ID', 'AK', '<|im_end|>']` pada tokenizer Qwen/Huatuo, sementara variasi huruf kecil `" tidak"` hanya bernilai 1 token.
  * *Spesifikasi rev06:* Mengevaluasi kelas biner menggunakan himpunan token natural yang di-logsumexp per kelas:
    * EN YES: `["YES", " YES", "Yes", " Yes"]`
    * EN NO: `["NO", " NO", "No", " No"]`
    * ID YA: `["YA", " YA", "Ya", " Ya", "ya", " ya"]`
    * ID TIDAK: `["TIDAK", " TIDAK", "Tidak", " Tidak", "tidak", " tidak"]`
  * *Penyimpanan Kolom:* Disimpan pada kolom baru berakhiran `_v6` (`pred_en_v6`, `conf_en_v6`, `pred_id_v6`, `conf_id_v6`, `score_m1_v6`, dll.). Kolom `_seq` dan `_len` dari rev05 tetap beku dan tidak diubah.
* **Prefix Key-Value Caching (Task B2):**
  * *Spesifikasi:* Untuk setiap kondisi evaluasi (EN-$P_0$, ID-$P_0$, ID-$P_1$), forward pass multimodal hanya dilakukan **satu kali pada prefix prompt+gambar** dengan `use_cache=True`.
  * *Efisiensi Komputasi:* Evaluasi kandidat token dilakukan langsung menggunakan prefix logits yang tersimpan di cache.
* **Tie-Aware Random Expectation AURC (Task D1):**
  * *Latar Belakang:* Diagnostik A4 membuktikan bahwa clipping $\max(0, \Delta P_{\text{dir}})$ menyebabkan 40%–80% sampel memiliki nilai identik $0.0$, memicu variasi AURC mergesort hingga 0.2907 murni akibat urutan baris data.
  * *Spesifikasi rev06:* Mengadopsi random expectation tie breaking (`tie_mode="random"`) sebagai metrik utama untuk mengevaluasi estimator berbasis $\Delta P_{\text{dir}}$.
* **Pemisahan Checkpoint & Manifest Konfigurasi (Task B3 & F):**
  * Menetapkan `revision_id="rev06"` dengan target penyimpanan checkpoint terpisah di `results/decisive_validation/rev06/`.
  * Memperbarui `configs/evaluation_config.yaml` dengan blok konfigurasi baru `decisive_validation_rev06` tanpa mengubah blok `decisive_validation_rev05`.

---

## 4. Pembaruan Pasca-Review & Penguatan Audit (Post-Review Hardening)

* **Pemisahan Fisik Notebook & Git Tagging (`v11.0-rev05-frozen`):**
  * Git tag `v11.0-rev05-frozen` dibuat pada commit beku `881ce06`.
  * Notebook murni rev05 diekspor ke subdirektori tersendiri `notebooks/kaggle/rev05/` (`08_qwen_decisive_validation_rev05.ipynb`, `09_huatuo_decisive_validation_rev05.ipynb`, `10_standalone_aggregation_evaluation_rev05.ipynb`) bebas dari kode rev06 untuk menjamin replikasi independen.
  * Kedua notebook inferensi rev05 (`08_qwen` dan `09_huatuo`) memiliki struktur simetris 16 sel kode lengkap yang mencakup alur inferensi hingga evaluasi RQ1–RQ3 dan risk ablation.
  * **Status Replikasi rev05:** Klaim bahwa inferensi ulang notebook rev05 mereproduksi hasil numerik identik berstatus **belum diverifikasi di GPU nyata** (lingkungan lokal telah memverifikasi validitas sintaks AST 100%, konsistensi manifest, dependensi, dan logika non-pengubah-hasil, namun verifikasi numerik floating-point riil mensyaratkan eksekusi di lingkungan GPU target Kaggle Dual-T4).
* **Penegakan Paritas Dtype (`torch.bfloat16` vs `torch.float16`):**
  * Menggunakan `torch.bfloat16` secara default untuk paritas arsitektural. Jika Gate (a) mendeteksi ketidakcocokan numerik akibat emulasi bfloat16 pada GPU T4, notebook menyediakan instruksi eksplisit untuk menguji `torch.float16` dan melaporkan perbandingannya.
* **Protokol Verifikasi Preflight Pilot GPU ($N=40$):**
  * Notebook inferensi 08 dan 09 dilengkapi parameter `PILOT_MODE = True` yang melakukan sampling terstratifikasi (12 Dev + 28 Test, Seed 42).
  * Menulis output terpisah ke `pilot_checkpoint_{MODEL_SLUG}.csv`, terisolasi total dari file resume full run.
  * **Gate (a) Ekuivalensi Numerik:**
    * Satuan ambang $\ge 98.0\%$ didefinisikan secara tegas **per-prediksi** ($N_{\text{pred}} = 120$ untuk 3 kondisi: EN, ID-$P_0$, ID-$P_1$), dengan pelaporan paralel tingkat kecocokan per-sampel ($N_{\text{sample}} = 40$).
    * Mean $|\Delta p_{\text{orig}}| < 0.02$.
    * Nol kemunculan nilai NaN atau Inf.
    * Melaporkan tabel rincian seluruh ketidakcocokan label beserta margin keputusan $|p - 0.5|$.
    * **Jalur Keputusan Berhenti:** Jika `bfloat16` dan `float16` sama-sama gagal memenuhi Gate (a), notebook mencetak instruksi tegas:
      `"🚨 STOP: jangan jalankan full run; periksa KV-cache/MRoPE/versi library"`.
  * **Pengukuran Latensi Riil:** `elapsed_time_sec` dibungkus dengan sinkronisasi CUDA aman `if torch.cuda.is_available(): torch.cuda.synchronize()`.
* **Demarkasi Integritas Ilmiah Data rev06:**
  * Menegaskan bahwa evaluasi rev06 pada Notebook 10 saat ini berstatus **data tiruan (mock preflight)** untuk memvalidasi alur agregasi sebelum GPU dijalankan.
  * Dipastikan bahwa `RESEARCH_MASTER.md` v12.0 **tidak memuat angka sintetis rev06** sama sekali; seluruh isi dokumen v12 tetap 100% berakar pada data empiris beku rev05.
