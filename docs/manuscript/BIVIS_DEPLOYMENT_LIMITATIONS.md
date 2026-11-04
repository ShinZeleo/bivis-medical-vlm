# Bi-ViS Deployment Limitations & Clinical Translation Boundaries
## Engineering, Clinical Workflow, and Epistemic Risk Assessment

**Target Document:** `docs/manuscript/BIVIS_DEPLOYMENT_LIMITATIONS.md`  
**Target Publication Scope:** SINTA 2 / International Journal Discussion & Limitations Section  
**Status:** `VERIFIED_DEPLOYMENT_ASSESSMENT`  
**Date:** 2026-11-02  

---

## 1. Executive Summary & Epistemic Humility

Metode **Bilingual Visual-Sensitivity-Aware Selective Prediction (Bi-ViS)** terbukti secara statistik mampu mereduksi risiko diagnostik Vision-Language Models (VLM) medis pada saat mengalami pergeseran bahasa Inggris–Indonesia tanpa membutuhkan pelatihan ulang model (*zero-retraining*). 

Namun demikian, penerapan metode ini dalam ekosistem klinis rumah sakit yang sebenarnya (*real-world clinical deployment*) memiliki sejumlah batasan teknis, operasional, dan metodologis yang wajib dideklarasikan secara transparan demi keselamatan pasien (*patient safety*) dan integritas akademik (*epistemic humility*).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TAKSONOMI BATASAN DEPLOYMENT METODE BIVIS                                 │
├──────────────────────────────┬───────────────────────────────┬─────────────────────────────────────────┤
│ Dimensi Batasan              │ Status Kesiapan Klinis        │ Implikasi Praktis & Mitigasi            │
├──────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┤
│ 1. Latensi & Biaya Komputasi │ Terbatas pada Kasus Asinkron  │ Overhead 3× forward pass (~21.7s);      │
│                              │ (Non-Emergency Radiology)     │ tidak layak untuk IGD/trauma akut       │
├──────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┤
│ 2. Lingkup Tugas VQA         │ Terbatas pada Format Biner    │ Hanya mengevaluasi Yes/No; belum mampu  │
│                              │ (Closed-Set Binary VQA)       │ menyaring laporan naratif bebas         │
├──────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┤
│ 3. Cakupan Bahasa Daerah     │ Terbatas pada Bahasa Baku     │ Belum teruji pada dialek lokal Jawa,    │
│                              │ (Bahasa Indonesia Standar)    │ Sunda, atau bahasa campur (code-mixed)  │
├──────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┤
│ 4. Sifat Uji Kanvas Hitam    │ Uji Diagnostik OOD            │ Mengukur dependensi informasi visual,   │
│                              │                               │ bukan penalaran kausal anatomis         │
├──────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┤
│ 5. Validasi Multi-Senter     │ Terbatas pada IndoRad-VQA     │ Wajib validasi eksternal pada PACS      │
│                              │ (314 Citra Fisik)             │ rumah sakit lokal sebelum produksi      │
└──────────────────────────────┴───────────────────────────────┴─────────────────────────────────────────┘
```

---

## 2. Batasan Komputasi dan Latensi Operasional

### 2.1. Overhead Latensi Tiga Forward Pass (Multiplicative Compute Cost)
Bi-ViS membutuhkan tiga kali evaluasi forward pass model dasar VLM untuk setiap kueri diagnostik:
1. Inferensi citra asli dengan prompt Bahasa Indonesia: $(I_{\text{orig}}, Q_{\text{ID}})$.
2. Inferensi kanvas hitam dengan prompt Bahasa Indonesia: $(I_{\text{blank}}, Q_{\text{ID}})$.
3. Inferensi citra asli dengan prompt Bahasa Inggris: $(I_{\text{orig}}, Q_{\text{EN}})$.

**Konsekuensi Operasional:**
- **Latensi per Kasus:** Berdasarkan log eksekusi aktual pada GPU NVIDIA modern, median latensi satu sampel adalah **$21.7$ detik** (dibandingkan $\approx 7.2$ detik pada baseline $M_1$).
- **Throughput:** Satu kartu GPU kelas data-center (48 GB VRAM) hanya mampu memproses $\approx 165$ kueri per jam.
- **Kesesuaian Alur Kerja:** Latensi ini dapat diterima untuk alur kerja radiologi elektif (*asynchronous second-reader triage* atau *tele-radiology audit*), namun **tidak dapat diterapkan** pada situasi kegawatdaruratan medis (*critical trauma triage* atau *stroke emergency*) yang menuntut latensi sub-detik.

### 2.2. Kebutuhan Alokasi Memori GPU (VRAM Footprint)
- Model dasar 7B parameter (Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B) dalam presisi `bfloat16` membutuhkan alokasi memori minimal **$16.5 \text{ GB}$ VRAM**.
- Fungsionalitas ekstraksi fitur Bi-ViS tidak menambah memori model karena dijalankan secara sekuensial. Namun, deployment pada fasilitas kesehatan tingkat primer (Puskesmas/Klinik Daerah) yang tidak memiliki server komputasi GPU khusus memerlukan arsitektur berbasis cloud atau kuantisasi model lebih lanjut (misal: INT4/AWQ).

---

## 3. Batasan Alur Kerja Klinis (*Clinical Workflow Boundaries*)

### 3.1. Restriksi pada Format Pertanyaan Biner (*Closed-Set Binary VQA*)
- Evaluasi Bi-ViS pada studi ini dirancang khusus untuk pertanyaan biner (*Yes/No questions*), di mana model memilih antara kandidat kelas ekuivalen:
  $$\{\text{"Ya"}, \text{"Tidak"}\} \iff \{\text{"Yes"}, \text{"No"}\}$$
- **Keterbatasan:** Bi-ViS belum diformulasikan untuk tugas *open-ended medical reporting*, seperti:
  - Generasi naratif deskripsi radiologi bebas (*free-text radiological findings*).
  - Klasifikasi diferensial diagnosis multi-kelas terbuka (*multi-class disease taxonomy*).
  - Lokalisasi lesi visual (*bounding box grounding* atau segmentasi piksel).
- Ekstensi ke ranah naratif terbuka membutuhkan perumusan ulang sinyal ketidaksepakatan ($D_{\text{lang}}$) menggunakan metrik semantik lanjutan (misal: *BERTScore* atau *Medical Concept NLI*).

### 3.2. Penanganan Kasus Abstensi (*Abstention Handling Protocol*)
- Ketika model mengabstensi prediksi ($S_{\text{Bi-ViS}} < \tau$), sistem harus memiliki alur eskalasi yang jelas:
  $$\text{Keputusan} = \text{"ABSTAIN"} \implies \text{Rujuk Kasus Langsung ke Radiolog Manusia}$$
- **Risiko Klinis:** Kegagalan menyediakan mekanisme *human-in-the-loop* yang responsif dapat menyebabkan penundaan diagnosis (*diagnostic backlog*). Pada cakupan $80\%$, sekitar $20\%$ kueri akan ditolak oleh sistem dan membutuhkan verifikasi manual langsung oleh dokter spesialis radiologi.

---

## 4. Batasan Bahasa dan Generalisasi Dialek

### 4.1. Spesifisitas Pasangan Bahasa Inggris–Indonesia Baku
- Seluruh eksperimen dievaluasi pada dataset IndoRad-VQA yang menggunakan terjemahan Bahasa Indonesia formal dan baku.
- **Status Validasi Dialek Lokal:** `NOT VERIFIED`.
  Belum ada bukti empiris bahwa Bi-ViS dapat mempertahankan kinerjanya pada:
  - Ragam bahasa medis informal atau singkatan klinis lokal (*clinical jargon* dokter Indonesia, misal: *"cor/pulmo dlm batas normal"*, *"infiltrat minimal sup/inf"*).
  - Percakapan dwibahasa campur (*code-mixed Indonesian-English*).
  - Bahasa daerah berpenutur besar di Indonesia seperti Bahasa Jawa, Sunda, atau Melayu dialek regional.

---

## 5. Sifat Intervensi Kanvas Hitam (*Counterfactual Epistemics*)

### 5.1. Kanvas Hitam sebagai Stimulus Out-of-Distribution (OOD)
- Kanvas hitam polos ($I_{\text{blank}} = \mathbf{0}$) adalah citra artifisial yang tidak merepresentasikan distribusi radiologi fisik alami.
- **Interpretasi Ilmiah:** Penurunan probabilitas $\Delta P_{\text{dir}}$ mengukur **kehilangan informasi visual secara agregat (*aggregate visual dependency*)**, dan **bukan** bukti pemahaman kausal patologis yang terlokalisasi pada lesi anatomi tertentu.
- Pada model tertentu (seperti HuatuoGPT-Vision), fenomena *negative visual reliance* ($\Delta P_{\text{dir}} < 0$) dapat terjadi akibat ketidakmampuan model menyelaraskan representasi visual dengan prompt bahasa Indonesia, bukan karena citra benar-benar bebas dari kelainan klinis.

---

## 6. Batasan Dataset dan Kebutuhan Validasi Multi-Senter

### 6.1. Ukuran Sampel Dataset IndoRad-VQA
- Meskipun kecukupan Events Per Variable (EPV) telah terverifikasi secara statistik ($EPV = 32.0$ dan $43.0$ pada Dev set), dataset IndoRad-VQA berasal dari dataset induk VQA-RAD yang memiliki ukuran sampel terbatas ($314$ citra fisik radiologi, $1.122$ kueri biner primer).
- Sebelum model diintegrasikan ke dalam sistem PACS (*Picture Archiving and Communication System*) rumah sakit umum, **diperlukan validasi prospektif multi-senter** menggunakan citra radiologi lokal dari berbagai mesin modalitas (X-Ray, CT-Scan) dan variasi demografi pasien di Indonesia.

---

## 7. Rekomendasi Mitigasi dan Arah Riset Lanjutan

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PANDUAN MITIGASI DAN ROADMAP PENELITIAN LANJUTAN                          │
├──────────────────────────────┬─────────────────────────────────────────────────────────────────────────┤
│ Aspek Keterbatasan           │ Tindakan Mitigasi Terjadwal                                             │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Komputasi & Latensi          │ Terapkan pemrosesan batch asinkron; evaluasi kuantisasi 4-bit (AWQ/GPTQ)│
│                              │ untuk mempercepat forward pass pada infrastruktur klinik terbatas.      │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Domain Tugas Terbuka         │ Kembangkan varian Bi-ViS generasi bebas (Free-Text Bi-ViS) menggunakan  │
│                              │ semantic entailment lintas-bahasa untuk ekstraksi D_lang.               │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Jargon Klinis Lokal          │ Bangun korpus fine-tuning/few-shot khusus singkatan radiologi Indonesia │
│                              │ guna menekan false disagreement pada prompt informal.                   │
├──────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Validasi Multi-Senter        │ Lakukan kerja sama uji klinis multisenter dengan rumah sakit rujukan    │
│                              │ nasional untuk menguji stabilitas ambang batas τ pada populasi baru.    │
└──────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> **Pernyataan Penafian Medis (*Medical Disclaimer*):** Metode Bi-ViS adalah instrumen seleksi risiko komputasional untuk mendukung riset kecerdasan buatan biomedis dan **bukan** perangkat medis diagnostik mandiri (*autonomous medical diagnostic device*). Sistem ini dirancang untuk mendampingi radiolog manusia sebagai penyaring risiko (*decision-support risk filter*), dan setiap keputusan medis akhir wajib berada di bawah wewenang klinisi yang berkualifikasi.
