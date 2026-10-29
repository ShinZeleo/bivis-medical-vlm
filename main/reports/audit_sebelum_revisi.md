# Laporan Audit Metodologis Sebelum Revisi: Protokol Intervensi Visual & Penilaian Reliabilitas

**Dokumen:** `reports/audit_sebelum_revisi.md`  
**Konteks:** Sinkronisasi Implementasi Pasca-Persetujuan Dosen Pembimbing  
**Auditor:** Research Engineer & Auditor Metodologi Medical VLM  
**Status:** Audit Pra-Revisi Selesai  

---

## 1. Ringkasan Eksekutif & Keputusan Dosen

Dosen pembimbing telah menyetujui arah penelitian skripsi dengan satu keputusan metodologis penting:
> **"Blur tidak perlu digunakan karena kualitas gambar sudah baik dan blur diperkirakan tidak memberikan pengaruh berarti. Intervensi gambar blank hitam diperbolehkan."**

Implikasi langsung terhadap protokol penelitian:
1. **Peniadaan Blur:** Seluruh opsi blur (Gaussian blur) dihapus dari protokol eksperimen utama dan tidak boleh ditambahkan kembali tanpa persetujuan eksplisit.
2. **Dua Kondisi Utama yang Ditetapkan:**
   * `P0_ORIGINAL`: Citra radiologi asli tanpa manipulasi.
   * `P1_BLACK_BLANK`: Citra hitam deterministik ($(0, 0, 0)$) berdimensi identik, menghapus seluruh informasi visual radiologis.
3. **Penyelarasan Proposal PPTX:** Slide 17 dari `Medical_VLM_Proposal.pptx` menyajikan tiga kondisi (P0 asli, P1 blank, P2 pixel-shuffled). Sesuai arahan, P0 dan P1 dijadikan dua kondisi utama, sementara pixel-shuffled dipindahkan menjadi eksperimen opsional sekunder/dihapus dari protokol utama.
4. **Epistemic Humility:** Tidak boleh mengklaim bahwa hilangnya respons pada citra hitam membuktikan "pemahaman klinis/radiologis yang mendalam" (*clinical grounding*), melainkan sebagai uji stres terhadap dominasi prior teks (*textual bias / out-of-distribution sensitivity*).

---

## 2. Audit Komponen Teknis Eksisting (Kode, Scoring, & Evaluasi)

### 2.1 Lokasi Pembuatan Gambar Blank dan Jenis Perturbasi dalam Kode
* **`main/Kaggle/03-kaggle-master.ipynb` (Cell 16):**
  ```python
  orig_img = Image.open(orig_img_path)
  blank_img = Image.new("RGB", orig_img.size, color=(0, 0, 0))
  blank_img.save(blank_img_path)
  ```
  Citra disimpan ke `data/audits/scratch_perturbations/blank_{orig_img_path.name}`.
* **`main/scripts/visual_perturbation.py` (Baris 18–23):**
  Fungsi `create_blank_image(image_path)` menggunakan `Image.new('RGB', orig.size, (0, 0, 0))`.
* **`main/scripts/decisive_validation_experiment.py` (Baris 281–284):**
  Membuat kanvas hitam deterministik `Image.new("RGB", orig_img.size, color=(0, 0, 0))` yang disimpan di `data/audits/scratch_perturbations/blank_{cid}.png`.
* **Status Arsip Fisik:** Sebanyak 150 citra blank hitam beresolusi identik telah tersedia di dalam arsip `main/Kaggle/indorad-vqa-research.zip` pada direktori `data/audits/scratch_perturbations/`.

### 2.2 Cara Menghitung Confidence untuk Jawaban Kandidat
* **Metode Lama (Cell 11 Kaggle & `scripts/confidence.py`):**
  Mengambil probabilitas softmax dari logit token pertama (`outputs.scores[0][0]`).
  * *Cacat Kritis:* Token yang didaftarkan di Cell 11 hanya variasi bahasa Inggris (`['YES', ' YES', 'Yes', ' Yes']` dan `['NO', ' NO', 'No', ' No']`), tanpa menyertakan kata bahasa Indonesia (`"Ya"`, `"Tidak"`). Akibatnya, pada prompt bahasa Indonesia, skrip membagi residu logit bahasa Inggris yang sangat kecil, memicu overconfidence artifisial ($c \approx 0.98 - 0.99$).
  * *Kelemahan Kalimat:* Jika model menjawab berupa kalimat (misal: *"Berdasarkan gambaran radiologi, tidak tampak..."*), token pertama adalah kata pengantar, bukan afirmasi/negasi.
* **Metode Baru yang Distandarkan (`Target Candidate Sequence Scoring`):**
  Mengevaluasi log-likelihood kondisional sekuens kandidat secara langsung pada konteks multimodal $C = (I, Q)$:
  * Bahasa Inggris: $A_{\text{aff}} = \text{"Yes"}$, $A_{\text{neg}} = \text{"No"}$
  * Bahasa Indonesia: $A_{\text{aff}} = \text{"Ya"}$, $A_{\text{neg}} = \text{"Tidak"}$
  $$\log P(A \mid C) = \sum_{t=1}^L \log P(w_t \mid C, w_{<t})$$
  $$\tilde{P}(A_{\text{aff}}) = \frac{\exp(\log P(A_{\text{aff}}))}{\exp(\log P(A_{\text{aff}})) + \exp(\log P(A_{\text{neg}}))}, \quad c = \max(\tilde{P}(A_{\text{aff}}), 1 - \tilde{P}(A_{\text{aff}}))$$
  * *Catatan Penting:* Nilai ini adalah probabilitas preferensi relatif antara dua kandidat, **bukan** probabilitas absolut terkalibrasi bahwa diagnosis benar secara klinis.

### 2.3 Cara Menghitung Perubahan Probabilitas Visual
* **Metode Lama (Cell 16 Kaggle):**
  $\Delta P_{\text{abs}} = |c_{\text{orig}} - c_{\text{blank}}|$.
  * *Cacat:* Menggunakan nilai mutlak tak berarah (*unsigned*). Jika penghapusan citra justru menaikkan keyakinan pada jawaban salah (model kolaps ke bias teks), formula lama justru menaikkan skor reliabilitas!
  * *Cacat Bahasa:* Cell 16 mengevaluasi $\Delta P$ pada prompt bahasa Inggris, lalu Cell 20 mengalikannya dengan confidence bahasa Indonesia.
* **Metode Baru yang Distandarkan (Sensitivitas Visual Berarah / $\Delta_{\text{dir}}$):**
  Biarkan $\hat{y}$ adalah jawaban prediksi pada citra asli ($P_0$).
  $$p_{\text{orig}} = P(\hat{y} \mid I_{\text{orig}}, Q), \quad p_{\text{blank}} = P(\hat{y} \mid I_{\text{blank}}, Q)$$
  $$\Delta_{\text{dir}} = p_{\text{orig}} - p_{\text{blank}} \in [-1.0, 1.0]$$
  Dievaluasi secara monolingual pada pertanyaan bahasa Indonesia yang persis sama.

### 2.4 Cara Menentukan Prediksi Benar atau Salah
* Jawaban prediksi: $\hat{y} = \text{"YES"}$ jika $\tilde{P}(A_{\text{aff}}) \ge 0.5$, dan $\text{"NO"}$ jika sebaliknya.
* Label acuan: $y^* \in \{\text{"YES"}, \text{"NO"}\}$ dari kolom `reference_answer` pada manifest beku.
* Kebenaran: $\text{Correct} = \mathbb{I}(\hat{y} = y^*)$.

### 2.5 Cara Menghitung AURC dan Risk–Coverage
* **Metode Lama (Cell 20 Kaggle):** 50 threshold linear (`np.linspace`). Menyebabkan pengulangan titik coverage dan integrasi kasar.
* **Metode Baru yang Distandarkan (Diskrit Sample-Level):**
  Urutkan sampel secara menurun berdasarkan skor ($S_{(1)} \ge S_{(2)} \dots \ge S_{(N)}$).
  Pada setiap sampel $k = 1, \dots, N$:
  $$\phi_k = \frac{k}{N}, \quad r_k = 1 - \frac{1}{k}\sum_{j=1}^k \text{Correct}_{(j)}, \quad \text{AURC} = \frac{1}{N}\sum_{k=1}^N r_k$$
  Integritas statistik diuji menggunakan **paired image-cluster bootstrap** (1.000 iterasi pada 58 kluster citra independen).

---

## 3. Audit Artefak Historis: Yang Dapat Dipakai vs. Wajib Dihitung Ulang

| Artefak / Berkas | Status Audit | Tindakan & Alasan |
| :--- | :--- | :--- |
| [`main/results/bilingual_manifest_sensitivity.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/results/bilingual_manifest_sensitivity.csv) | **VALID** | **DIPERTAHANKAN:** 150 pasangan bilingual beku, 58 kluster citra, 0% kebocoran hash. Menjadi manifest acuan evaluasi. |
| [`main/data/manifests/image_identity_manifest.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/data/manifests/image_identity_manifest.csv) | **VALID** | **DIPERTAHANKAN:** Berisi 314 hash piksel SHA-256 unik dan pemetaan kluster `canonical_image_id`. |
| [`main/Kaggle/actual_experimental_intervention_results.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/Kaggle/actual_experimental_intervention_results.csv) | **VALID (Bukti Historis)** | **DIPERTAHANKAN:** Bukti empiris kegagalan LoRA B2 (flip rate naik ke 24%, akurasi ID anjlok ke 58%). |
| `main/Kaggle/strong_model_validation_predictions.csv` | **TERDISTORSI TOKENIZER** | **HARUS DIHITUNG ULANG:** Confidence bahasa Indonesia terdampak bug token `TOKENS_YES/NO` di Cell 11. |
| Angka AURC Kaggle ($0.2786 \to 0.2173$) | **EKSPLORATIF / AWAL** | **HARUS DIHITUNG ULANG:** Menggunakan 50 grid, transfer bahasa EN $\to$ ID, dan formula absolut unsigned. |
| Proposal `Medical_VLM_Proposal.pptx` (Slide 17) | **PERLU SINKRONISASI** | **BUAT SALINAN REVISI:** Hapus blur, jadikan P0 dan P1 sebagai kondisi utama, pindahkan P2 menjadi opsional. |

---

## 4. Daftar File yang Akan Diubah dan Ditambahkan pada Tahap Berikutnya

1. **Kode Intervensi & Eksekusi:** `main/scripts/decisive_validation_experiment.py` (pastikan hanya memuat P0 Original vs P1 Black Blank, tanpa opsi blur).
2. **Unit Test:** `main/scripts/test_visual_intervention.py` (menguji determinisme blank image, kandidat sequence scoring, dan discrete AURC sintetis).
3. **Dokumentasi Proposal:** Buat `Medical_VLM_Proposal_Revisi_Dosen.pptx` beserta berkas diff teksnya.
4. **Dokumen Induk:** Perbarui `RESEARCH_MASTER.md` dan `main/reports/3_DECISIVE_VALIDATION_PROTOCOL.md` agar konsisten dengan keputusan dosen (menghapus blur, mendokumentasikan alasan blank hitam, OOD, dsb.).
