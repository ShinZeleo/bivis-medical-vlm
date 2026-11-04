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


---

## 5. Tata Kelola Revisi, Atribusi Checkpoint & Penguatan Gate (a) rev06

* **Audit Asal-Usul Checkpoint rev05 (Atribusi Checkpoint & Komparasi Skema):**
  * *Pemeriksaan Kolom:* Sebanyak 62 kolom pada `live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv` dan `live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv` dibandingkan secara kanonikal terhadap kamus `new_records.append` pada seluruh kandidat notebook:
    1. `08_decisive_validation_rev05.ipynb` (monolitik dual-model di tag `v11.0-rev05-frozen`, memiliki "STAGE 21 & 22"): **Cocok sempurna 62/62 kolom dalam urutan yang identik**.
    2. `09_infer_huatuo_resume.ipynb` (dedicated Huatuo resume di tag `v11.0-rev05-frozen`): **Cocok sempurna 62/62 kolom dalam urutan yang identik**.
    3. `08_qwen_decisive_validation.ipynb` (modular rev05b): Menggunakan skema yang sama (62 kolom) sebelum penambahan track `_v6`.
  * *Nilai Metadata Checkpoint:* Nilai tersimpan pada baris CSV adalah `revision_id = "rev05"` dan `scoring_method = "exact_sequence_candidate_normalized_v5"`.
  * *Kesimpulan Atribusi:* Checkpoint rev05 dibangkitkan oleh pipeline `08_decisive_validation_rev05.ipynb` monolitik (dual-model) untuk inferensi awal Qwen, dan diselesaikan/diresume menggunakan `09_infer_huatuo_resume.ipynb` untuk Huatuo akibat keterbatasan batas waktu (timeout) Kaggle Dual-T4.

* **Restorasi Berkas Asli rev05 (Byte-Identik) di `notebooks/kaggle/rev05/`:**
  * File `09_huatuo_decisive_validation_rev05.ipynb` (10 sel kode) yang sebelumnya keliru diekspor dari resume notebook telah **DIHAPUS**.
  * Direktori `notebooks/kaggle/rev05/` kini memuat 3 file asli murni yang disediakan pengguna tanpa modifikasi apa pun (byte-identik):
    * `notebooks/kaggle/rev05/08_qwen_decisive_validation.ipynb` (16 sel kode):
      `SHA-256: 415b90549eb33e7be514e67cf9053b7a25b2fd54b4199a379a25213156624066`
    * `notebooks/kaggle/rev05/09_huatuo_decisive_validation.ipynb` (16 sel kode):
      `SHA-256: f58634568080746a622adda2b8814bd47c0c79979ea0559dd2f89b5df1913316`
    * `notebooks/kaggle/rev05/10_standalone_aggregation_evaluation.ipynb` (11 sel kode):
      `SHA-256: 1f6c0d62ea4dfbb18149ac12200df7910e7c22fb90777ca503a36e4ac1b4de24`

* **Penyimpanan Salinan Patch Bug Non-Pengubah-Hasil di `notebooks/kaggle/rev05_patched/`:**
  * File asli pada `rev05/` dipertahankan byte-identik. Seluruh perbaikan bug sintaks/library non-pengubah-hasil diisolasi ke dalam direktori `notebooks/kaggle/rev05_patched/`:
    * `08_qwen_decisive_validation.ipynb`:
      `SHA-256: 92f0d1ef0ca8820a7f84ab39f61a1f4c334d5b90e8d94db5094d46a833111806`
      *Diff:* Koreksi label progress bar `desc="HuatuoGPT Inference"` -> `desc="Qwen2.5-VL Inference"` (Sel 8); penambahan `import matplotlib.pyplot as plt` (Sel 14); penghapusan baris variabel tak terpakai `df_qwen_test = df_qwen[...]` (Sel 14); pembersihan `penalty="l2"` pada `LogisticRegression` (Sel 15).
    * `09_huatuo_decisive_validation.ipynb`:
      `SHA-256: d1bb31ecea0c085e7540003bf91c8afdc3b6e49c29edc59a5a81d38047516cce`
      *Diff:* Penambahan `import matplotlib.pyplot as plt` (Sel 14); penghapusan variabel tak terpakai `df_huatuo_test = df_huatuo[...]` (Sel 14); pembersihan `penalty="l2"` pada `LogisticRegression` (Sel 15).
    * `10_standalone_aggregation_evaluation.ipynb`:
      `SHA-256: 65ea047ed5a533b6c57a148aebc3a47747fd169c7426604357b7ad81a067c945`
      *Diff:* Penambahan `import matplotlib.pyplot as plt` di awal sel RQ3 (Sel 7); pembersihan `penalty="l2"` pada `LogisticRegression` (Sel 8).

* **Pembersihan Root Kaggle & Penempatan Notebook `rev06/`:**
  * Seluruh notebook inferensi dan evaluasi berprotokol `rev06` dipindahkan dari root `notebooks/kaggle/` ke `notebooks/kaggle/rev06/`:
    * `notebooks/kaggle/rev06/08_qwen_decisive_validation.ipynb` (16 sel kode)
    * `notebooks/kaggle/rev06/09_huatuo_decisive_validation.ipynb` (16 sel kode)
    * `notebooks/kaggle/rev06/10_standalone_aggregation_evaluation.ipynb` (11 sel kode)
  * Berkas notebook di root `notebooks/kaggle/` dihapus sehingga tidak ada ambiguitas berkas.
  * Total notebook pada repositori: 9 notebook (3 di `rev05/`, 3 di `rev05_patched/`, 3 di `rev06/`), memuat 129 sel kode yang 100% lolos verifikasi `verify_ast.py`.

* **Pengetatan Kriteria Gate (a) di Sel 8B rev06:**
  * Lolos Gate (a) mensyaratkan konjungsi 4 kriteria ketat:
    1. `pred_match_rate_3cond >= 98.0%` dihitung per-prediksi ($N_{	ext{pred}} = 120$ untuk 3 kondisi: EN-$P_0$, ID-$P_0$, ID-$P_1$).
    2. **Marginalitas Mismatch:** Setiap ketidakcocokan label WAJIB memiliki margin $|p - 0.5| < 0.02$ pada **KEDUA** nilai (tersimpan dan pilot). Jika ada satu sampel saja dengan margin $|p - 0.5| \ge 0.02$ yang labelnya berbeda, Gate (a) otomatis **GAGAL**.
    3. `mean_dp_orig < 0.02` pada probabilitas ID-$P_0$, serta 0 kemunculan NaN atau Inf.
    4. **Integritas Sumber Data:** Jika berkas checkpoint rev05 tidak ditemukan di `/kaggle/input`, eksekusi segera dihentikan dengan melempar `RuntimeError("Stored rev05 checkpoint not found in /kaggle/input! Cannot verify Gate (a) equivalence.")` (bukan sekadar peringatan cetak).
  * *Klaim Status:* Seluruh pengujian pipeline berstatus **lolos ast.parse dan pytest pada data tiruan; belum diverifikasi di GPU nyata**.

* **Pembersihan Checkpoint Tiruan rev06 & Penegakan Guardrail `is_mock`:**
  * Direktori output checkpoint tiruan `artifacts/experiments/decisive_validation/rev06/` telah dihapus secara permanen.
  * `scripts/generate_mock_rev06_checkpoints.py` diperbarui untuk secara eksplisit menyertakan kolom `is_mock = True`.
  * Loader checkpoint di `notebooks/kaggle/rev06/10_standalone_aggregation_evaluation.ipynb` (Cell 4) diperketat: secara aktif menolak berkas/jalur dengan nama mock atau dataframe dengan kolom `is_mock == True` dengan melempar `RuntimeError("Mock checkpoint detected! Standalone aggregation refuses mock data.")`.
  * Verifikasi manifest SHA-256 (Cell 3) diperketat dengan melempar `RuntimeError` jika manifest tidak ditemukan dan fallback embedded tidak tersedia atau corrupt.
  * Seluruh output tersimpan (`outputs: []`, `execution_count: None`) pada ketiga notebook `notebooks/kaggle/rev06/*.ipynb` telah dibersihkan hingga bersih (0 cell dengan output).
