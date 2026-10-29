# RESEARCH MASTER: SINGLE SOURCE OF TRUTH (AUTHORITATIVE THESIS BLUEPRINT)

# Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift

**Judul Bahasa Indonesia:**  
*Prediksi Selektif Berbasis Sensitivitas Visual pada Model Vision-Language Medis Bilingual dalam Pergeseran Bahasa Inggris–Indonesia*

**Metadata Dokumen:**  
- **Versi:** 8.0 (Audited & Revised Master Edition)  
- **Identifier Revisi Teknis:** `rev05-master`  
- **Status Dokumen:** Single Source of Truth (Authoritative, Methodologically Grounded, Frozen Protocol)  
- **Tanggal Audit & Pengesahan:** 21 Oktober 2026  
- **Tingkat Akademik:** Skripsi Sarjana (S1) Teknik Informatika / Ilmu Komputer  
- **Bidang Keilmuan:** Medical Multimodal Machine Learning, Trustworthy AI, & Selective Prediction  
- **Peneliti / Penulis:** Imam (Undergraduate CS Thesis Researcher)  
- **Standar Integritas:** Peer-review standar jurnal internasional bereputasi / SINTA 1–2; memprioritaskan validitas ilmiah di atas hasil yang tampak bagus.

---

## 1. Metadata, Versi, dan Status Dokumen

Dokumen ini merupakan **satu-satunya acuan resmi (Single Source of Truth / SSoT)** bagi pelaksanaan, evaluasi, penulisan skripsi, dan naskah publikasi ilmiah. Seluruh klaim, formula, notasi, manifest data, skrip inferensi, dan interpretasi statistik wajib merujuk secara konsisten pada spesifikasi dalam dokumen ini.

### 1.1 Klasifikasi Status Klaim dan Hasil (Epistemic Status Disclosure)
Untuk mencegah fabrikasi dan overclaim, dokumen ini memisahkan status data dan temuan ke dalam empat tingkat kepastian:
1. **`VERIFIED_STATIC`:** Telah diverifikasi secara kriptografis dan deterministik pada artefak lokal (struktur manifest, hash piksel gambar, konsistensi label biner, parsing AST kode).
2. **`VERIFIED_LOCAL_TEST`:** Telah diverifikasi melalui eksekusi unit test dan smoke test lokal pada model lokal (misal: Qwen2.5-VL-3B-Instruct / GPU lokal).
3. **`UNVERIFIED_PILOT_ARTIFACT`:** Angka-angka historis dari notebook pilot awal (misal: ECE 0.2350, conditional error 36.63%, penurunan AURC 22% pada Run 01–03) yang terbukti terdistorsi oleh bug tokenizer/asimetri kandidat dan **tidak boleh dikutip sebagai fakta klinis terbukti**.
4. **`PENDING_CONFIRMATORY_GPU`:** Inferensi skala penuh 7B (Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B) pada cloud GPU (Kaggle Dual Tesla T4) dengan protokol `rev05` berstatus siap eksekusi (*ready-to-run*).

---

## 2. Executive Summary

Penelitian ini mengevaluasi apakah sinyal sensitivitas visual terhadap penghilangan informasi citra (*sensitivity to visual deletion*) dapat memberikan nilai tambah di luar skor keyakinan (*confidence*) internal model dalam mengurutkan risiko kesalahan (*selective prediction*) pada Medical Vision-Language Model (Medical VLM) ketika menghadapi pertanyaan radiologi bilingual Inggris–Indonesia.

### Temuan dan Keputusan Metodologis Utama:
1. **Penghentian Pendekatan Fine-Tuning (LoRA Ditinggalkan):** Eksperimen awal membuktikan bahwa fine-tuning QLoRA dengan penambahan loss konsistensi semantik biner ($B_2$) menurunkan akurasi bahasa Indonesia sebesar $-12.00\%$ ($70\% \to 58\%$) dan meningkatkan *flip rate* sebesar $+4.00\%$. Selain itu, model 32B mengalami *CUDA OutOfMemoryError* pada dual-T4 GPU. Seluruh pendekatan pelatihan ulang resmi dihentikan demi pendekatan *inference-time selective prediction*.
2. **Penyebab Utama Cacat Pilot Terdahulu (Audit Tokenizer & Scoring):** Audit independen mengungkap bahwa klaim awal *extreme overconfidence* pada HuatuoGPT (ECE 0.2350 dan *conditional error* 36.63%) sebagian besar dipicu oleh *tokenizer bug* (hanya mendaftarkan token bahasa Inggris `YES/NO` sehingga membagi residu kecil probabilitas pada teks Indonesia) serta penalti panjang token (*length bias*) pada kata `"Tidak"` (multi-token) vs `"Ya"` (single-token).
3. **Desain Intervensi Dua Kondisi:** Berdasarkan persetujuan dosen pembimbing, manipulasi citra dibatasi secara deterministik pada citra radiologi asli (`P0_ORIGINAL`) dan citra kanvas hitam polos berdimensi identik (`P1_BLACK_BLANK`). Pengaburan citra (*Gaussian blur*) resmi ditiadakan.
4. **Sifat Intervensi Citra Hitam:** Citra hitam merupakan masukan *out-of-distribution (OOD)* bagi vision encoder. Penurunan probabilitas terarah ($\Delta P_{\text{dir}}$) mencerminkan sensitivitas perilaku terhadap ketiadaan masukan visual, bukan bukti langsung pemahaman anatomi atau lokalisasi kausal lesi klinis.
5. **Estimator Primer Pra-Registrasi:** Estimator kombinasi linear ternormalisasi ($M_{\text{norm}}$) ditetapkan sebagai hipotesis primer karena memberikan penalti halus proporsional tanpa keruntuhan nilai nol (*zero collapse*), sedangkan $M_1$ adalah baseline acuan, $M_{\text{dir}}$ dan $M_2$ sebagai analisis sekunder, dan $M_{3,\text{old}}$ sebagai kontrol historis.
6. **Integritas Temuan Negatif:** Jika hasil inferensi konfirmatori menunjukkan bahwa estimator visual tidak mengungguli $M_1$ secara signifikan ($p \ge 0.05$ setelah koreksi Holm-Bonferroni), hasil tersebut dilaporkan secara jujur sebagai temuan negatif bahwa penghilangan gambar sederhana tidak menambah informasi selektif di atas skor preferensi kandidat.

---

## 3. Keputusan Penelitian dan Riwayat Revisi

### 3.1 Ringkasan Trajektori Metodologis
- **Fase Awal (Kandidat A - Abandoned):** Pelatihan ulang model (LoRA/QLoRA) dengan semantic consistency loss. Ditinggalkan karena merusak performa bahasa Indonesia dan kendala komputasi VRAM.
- **Fase Pilot (Kandidat B - Exploratory Run 01–03):** Pengujian selective prediction awal melaporkan penurunan AURC dari 0.2786 ke 0.2173 ($M_3$ unsigned). Namun, audit menemukan bug evaluasi lintas-bahasa (Cell 16 mengevaluasi bahasa Inggris, Cell 20 mengevaluasi bahasa Indonesia) dan bug tokenisasi.
- **Fase rev02 (Kaggle Dual-T4 Live Run):** Evaluasi 92 sampel test. Menemukan fenomena *length penalty* di mana raw sequence log-likelihood membiaskan prediksi 100% ke arah `"YA"` (karena `"Tidak"` adalah 2 token: `['T', 'idak']`). Estimator visual menghasilkan AURC lebih buruk daripada $M_1$.
- **Fase rev03 (Length-Normalization & Paired Cluster Test):** Memperbaiki token penutup `<|im_end|>`, normalisasi panjang surprisal, Discrete Riemann AURC, dan exact cluster permutation test. Diverifikasi lolos pada model lokal 3B.
- **Fase rev04 (Full Dataset Audit):** Mengaudit seluruh 2.244 baris IndoRad-VQA. Mengeksklusi 1.053 pertanyaan terbuka dan 69 pasangan translasi cacat (tersisa 1.122 pasangan biner CLEAN).
- **Fase rev05 (Active Cohort Union & Preflight Locked):** Membangun Active Evaluation Cohort Union ($N=1.133$) yang mencakup Primary Cohort ($N=1.122$) dan Historical Benchmark ($N=150$) dengan 0% kebocoran klaster citra. Mengunci protokol dual-track scoring (Exact Sequence vs Length-Normalized).

---

## 4. Latar Belakang dan Definisi Masalah

### 4.1 Latar Belakang Klinis dan Multimodal
Penerapan Medical VLM di negara berkembang seperti Indonesia menghadapi kendala mendasar: sebagian besar model fondasi dilatih menggunakan literatur dan anotasi berbahasa Inggris atau Mandarin. Ketika diinstruksikan dalam bahasa Indonesia, model mengalami degradasi kinerja akibat pergeseran bahasa (*language shift*).

### 4.2 Definisi Masalah
1. **Inkonsistensi Lintas-Bahasa (*Cross-Lingual Flip*):** Pertanyaan klinis yang identik mengenai radiograf yang sama dapat menghasilkan kesimpulan medis yang bertolak belakang semata-mata karena pergantian bahasa instruksi dari bahasa Inggris ke bahasa Indonesia.
2. **Ketergantungan pada Prior Teks (*Visual Ungroundedness*):** Model berisiko menjawab berdasarkan pola frekuensi kata dalam data pelatihan teks daripada memeriksa bukti visual pada radiograf (fenomena "Dangerous Quadrant", Sadanandan & Behzadan, CVPR 2026).
3. **Kegagalan Deteksi Kesalahan oleh Confidence Tunggal ($M_1$):** Model dapat memberikan jawaban salah dengan keyakinan verbal yang sangat tinggi pada bahasa target, sehingga mekanisme penolakan (*abstention*) berbasis confidence internal gagal menyaring kegagalan tersebut.

### 4.3 Demarkasi Tiga Konstruk yang Berbeda
Untuk menjaga ketepatan metodologis, tiga konstruk berikut dibedakan secara tegas:
- **Konsistensi Lintas-Bahasa:** Sejauh mana prediksi model pada prompt bahasa Indonesia identik dengan prediksinya pada prompt bahasa Inggris ($\hat{y}_{\text{ID}} == \hat{y}_{\text{EN}}$).
- **Reliabilitas Prediksi:** Probabilitas atau proporsi prediksi model sesuai dengan *ground-truth* medis ($\hat{y} == y^*$).
- **Ketergantungan Bukti Visual (*Visual Reliance*):** Besarnya perubahan respons atau skor kandidat ketika citra radiologi dihapus ($\Delta P_{\text{dir}} = P_{\text{orig}}(\hat{y}) - P_{\text{blank}}(\hat{y})$).

---

## 5. Research Gap dan Novelty

### 5.1 Rumusan Research Gap
> **Research Gap:** Belum diketahui secara empiris apakah sinyal sensitivitas prediksi terhadap penghilangan informasi visual berarah ($\Delta P_{\text{dir}}$) memberikan nilai tambah (*value-add*) yang signifikan dibandingkan skor preferensi kandidat internal ($M_1$) dalam mengurutkan risiko kesalahan (*selective prediction*) pada Medical VLM di bawah pergeseran bahasa Inggris–Indonesia pada data radiologi biner tertutup.

### 5.2 Demarkasi Kontribusi dari Pekerjaan Terdahulu
- **Terhadap Yudhistira et al. (CVPR 2026 MMFM-BIOMED, arXiv:2606.03693):** Yudhistira et al. mendokumentasikan fenomena penurunan performa IndoRad-VQA secara deskriptif pada evaluasi statis unselective. Penelitian ini mengembangkan dan mengevaluasi mekanisme deteksi kegagalan waktu-inferensi (*inference-time selective prediction*).
- **Terhadap Geifman & El-Yaniv (NeurIPS 2017) & Kamath et al. (ACL 2020):** Teori selective classification diperluas dari visi unimodal klasifikasi tertutup dan NLP teks tunggal ke model multimodal vision-language autoregresif bilingual.
- **Terhadap Sadanandan & Behzadan (CVPR 2026 Med-Reasoner, arXiv:2603.20985):** Sadanandan & Behzadan membuktikan bahwa konsistensi parafrase tanpa uji *image-reliance* adalah jebakan reliabilitas palsu (*false reliability trap*). Penelitian ini mengoperasionalkan penghilangan gambar ke dalam fungsi pengurutan risiko selektif kuantitatif (AURC).
- **Terhadap Khanmohammadi et al. (arXiv:2605.10893, 2026, BICR):** BICR melatih probe eksternal dengan loss kontras citra hitam. Penelitian ini mengevaluasi intervensi inferensi tanpa melatih bobot tambahan apa pun (*zero-parameter runtime rule*).

---

## 6. Tujuan Penelitian dan Kontribusi

### 6.1 Tujuan Penelitian
1. Mengukur dampak pergeseran bahasa (Inggris ke Indonesia) terhadap akurasi, inkonsistensi jawaban, dan distribusi skor keyakinan pada model VLM umum dan medis.
2. Menganalisis tingkat sensitivitas perilaku model terhadap penghilangan citra radiologi pada pertanyaan klinis berbahasa Indonesia.
3. Mengevaluasi secara empiris apakah estimator selective prediction berbasis sensitivitas visual berarah menghasilkan diskriminasi kesalahan (AURC) yang lebih baik daripada baseline confidence internal.
4. Membandingkan karakteristik reliabilitas antara VLM fondasi umum (`Qwen2.5-VL-7B`) dan VLM spesialis medis (`HuatuoGPT-Vision-7B`).

### 6.2 Batasan Kontribusi Penelitian (Maksimal 3 Poin Terukur)
1. **Kontribusi Metodologis:** Perumusan dan standardisasi protokol evaluasi selective prediction multimodal pada model autoregresif bilingual yang menggabungkan candidate-normalized sequence scoring dengan analisis sensitivitas visual berarah, dievaluasi menggunakan Discrete Riemann AURC dan exact paired image-cluster permutation test.
2. **Kontribusi Empiris:** Bukti komparatif berskala penuh pada IndoRad-VQA yang memetakan disparitas akurasi, cross-lingual flip rate, dan efektivitas intervensi citra hitam pada dua kelas arsitektur model 7B.
3. **Kontribusi Praktis / Evaluatif:** Penilaian kritis mengenai kelayakan dan batas kemampuan intervensi visual waktu-inferensi sebagai mekanisme rujukan klinis (*clinical triage/deferral*) tanpa retraining.

---

## 7. Research Questions dan Hipotesis

| Elemen | RQ1: Language Shift & Reliability | RQ2: Visual Sensitivity to Deletion | RQ3: Selective Prediction Value-Add | RQ4: General vs Medical Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **Pertanyaan Penelitian** | Sejauh mana pergeseran bahasa Inggris ke Indonesia menurunkan akurasi, memicu inkonsistensi flip, dan mengubah distribusi skor keyakinan? | Sejauh mana prediksi model pada bahasa Indonesia berubah ketika citra dihitamkan ($\Delta P_{\text{dir}}$), dan apakah sensitivitas berkorelasi dengan kebenaran? | Apakah estimator sensitivitas visual berarah menghasilkan penurunan AURC yang signifikan dibandingkan baseline confidence $M_1$? | Bagaimana perbedaan profil reliabilitas, degradasi bahasa, dan responsivitas selektif antara model umum dan medis? |
| **Motivasi Ilmiah** | Menilai apakah pergeseran bahasa menciptakan kerentanan diagnostik klinis pada model multimodal. | Menguji apakah prediksi didorong oleh bukti visual atau terpaku pada bias prior bahasa (Sadanandan & Behzadan, 2026). | Menguji apakah uji kontrafaktual visual waktu-inferensi dapat menyaring prediksi salah sebelum diserahkan ke pengguna. | Menilai apakah fine-tuning domain medis meningkatkan ketahanan lintas-bahasa atau justru memperparah kalibrasi. |
| **Variabel Bebas (IV)** | Bahasa prompt ($X_{\text{lang}} \in \{\text{EN}, \text{ID}\}$). | Kondisi citra ($I \in \{P_0\text{ (Asli)}, P_1\text{ (Hitam)}\}$). | Estimator reliabilitas ($M_1, M_2, M_{\text{dir}}, M_{\text{norm}}, M_3^{\text{old}}$). | Arsitektur model (`Qwen2.5-VL-7B` vs `HuatuoGPT-Vision-7B`). |
| **Variabel Terikat (DV)** | Akurasi, Flip Rate, Joint Correctness, ECE, Brier Score. | Visual Flip Rate, Mean $\Delta P_{\text{dir}}$, Korelasi Titik-Biserial $\Delta P_{\text{dir}}$ dgn Correctness. | Discrete AURC, Selective Risk pada Cakupan 80% ($r_{0.80}$), Empirical Coverage. | Selisih komparatif metrik RQ1, RQ2, dan $\Delta\text{AURC}$ antar kedua model. |
| **Unit Analisis** | Pasangan pertanyaan bilingual pada klaster citra fisik kanonis. | Pertanyaan bahasa Indonesia pada kondisi citra P0 vs P1. | Pasangan prediksi dan label biner terurut pada set uji beku. | Perbedaan kinerja berpasangan antar model pada split yang identik. |
| **Hipotesis Kerja ($H_1$)** | Pergeseran bahasa memicu penurunan akurasi ($\Delta_{\text{acc}} \ge 5\%$) dan flip rate non-trivial ($\ge 15\%$). | Prediksi yang benar memiliki mean $\Delta P_{\text{dir}}$ positif yang lebih tinggi secara signifikan dibanding prediksi salah. | Estimator primer $M_{\text{norm}}$ menghasilkan AURC lebih rendah secara signifikan dibanding $M_1$ ($p < 0.05$ Holm-Bonferroni). | Model medis HuatuoGPT menunjukkan penurunan akurasi bahasa Indonesia yang tidak lebih kecil dari model umum. |
| **Hipotesis Nol ($H_0$)** | Tidak ada perbedaan akurasi antara bahasa Inggris dan Indonesia ($\mathbb{E}[\text{Acc}_{\text{EN}} - \text{Acc}_{\text{ID}}] = 0$). | Tidak ada perbedaan sensitivitas visual antara prediksi benar dan salah ($\mathbb{E}[\Delta P_{\text{dir}} \mid \text{Benar}] = \mathbb{E}[\Delta P_{\text{dir}} \mid \text{Salah}]$). | Estimator visual tidak memberikan perbaikan ranking risiko dibanding $M_1$ ($\mathbb{E}[\text{AURC}(M) - \text{AURC}(M_1)] \ge 0$). | Tidak ada perbedaan performa selektif $\Delta\text{AURC}$ antara model umum dan medis. |
| **Metrik Utama** | $\text{Acc}_{\text{EN}}, \text{Acc}_{\text{ID}}, \text{Flip Rate}, \text{ECE}_{\text{ID}}$. | $\text{Mean } \Delta P_{\text{dir}}, \text{Visual Flip Rate } P_0 \to P_1$. | $\text{Discrete AURC}, r_{0.80}, \Delta\text{AURC}_{\text{abs}}$. | $\Delta\text{AURC}_{\text{Qwen}} \text{ vs } \Delta\text{AURC}_{\text{Huatuo}}$. |
| **Kriteria Keputusan** | Tolak $H_0$ jika paired $t$-test / permutation $p < 0.05$. | Tolak $H_0$ jika Mann-Whitney $U$ / permutation $p < 0.05$. | Tolak $H_0$ jika paired cluster permutation $p_{\text{adj}} < 0.05$ & $\Delta\text{AURC} < 0$. | Deskriptif komparatif dan uji interaksi model $\times$ estimator. |
| **Batas Interpretasi** | Penurunan akurasi mencerminkan sensitivitas terhadap bahasa, bukan ketidakmampuan membaca citra saja. | $\Delta P \approx 0$ mencerminkan ketidakresponsifan terhadap intervensi citra hitam, dengan batas OOD. | Penurunan AURC adalah perbaikan ranking selektif statistik, bukan jaminan keselamatan klinis. | Perbedaan mencerminkan totalitas sistem model, bukan kausalitas tunggal data medis. |

---

## 8. Definisi Operasional dan Notasi

### 8.1 Notasi Matematis
- $\mathcal{D} = \{(I_i, Q_{i,\text{en}}, Q_{i,\text{id}}, y_i^*)\}_{i=1}^N$: Dataset evaluasi berpasangan.
- $C_i \in \{1, \dots, K\}$: Indeks klaster citra fisik kanonis (`canonical_image_id`).
- $P_0 = I_i$: Citra radiologi asli.
- $P_1 = \mathbf{0}_{(W_i, H_i, 3)}$: Citra kanvas hitam dengan dimensi dan kanal identik.
- $\mathcal{Y} = \{c_{\text{aff}}, c_{\text{neg}}\}$: Ruang kandidat jawaban biner tertutup.
  - Bahasa Inggris: $c_{\text{aff}} = \text{"YES<|im_end|>"}$, $c_{\text{neg}} = \text{"NO<|im_end|>"}$.
  - Bahasa Indonesia: $c_{\text{aff}} = \text{"YA<|im_end|>"}$, $c_{\text{neg}} = \text{"TIDAK<|im_end|>"}$.
- $S(y \mid X) = \sum_{t=1}^L \log P(y_t \mid y_{<t}, X)$: Log-likelihood sekuens target kausal.
- $\bar{s}(y \mid X) = \frac{1}{L} \sum_{t=1}^L \log P(y_t \mid y_{<t}, X)$: Surprisal rata-rata ternormalisasi panjang.
- $P_{\text{cand}}(y \mid X) = \frac{\exp(\bar{s}(y \mid X))}{\sum_{y' \in \mathcal{Y}} \exp(\bar{s}(y' \mid X))}$: Candidate-normalized preference score.
- $\hat{y} = \arg\max_{y \in \mathcal{Y}} P_{\text{cand}}(y \mid P_0, Q_{\text{id}})$: Prediksi diskrit model pada citra asli.
- $p_{\text{orig}} = P_{\text{cand}}(\hat{y} \mid P_0, Q_{\text{id}})$: Keyakinan model pada jawaban terpilih dengan citra asli.
- $p_{\text{blank}} = P_{\text{cand}}(\hat{y} \mid P_1, Q_{\text{id}})$: Skor yang diberikan model pada jawaban yang sama ketika citra dihitamkan.
- $\Delta P_{\text{dir}} = p_{\text{orig}} - p_{\text{blank}} \in [-1.0, 1.0]$: Sensitivitas visual berarah (*directed visual support*).

---

## 9. Dataset, Sampling, Manifest, dan Kriteria Inklusi

### 9.1 Dataset Sumber
Dataset yang digunakan adalah `Lab-IS/IndoRad-VQA` (Yudhistira et al., CVPR 2026 MMFM-BIOMED, 2.244 baris pertanyaan-jawaban klinis dari 314 citra radiologi unik).

### 9.2 Alur Akuntansi Eksklusi dan Kriteria Inklusi
```
Total Dataset IndoRad-VQA: 2.244 Pasangan QA (314 Citra Fisik Unik)
├── Eksklusi Pertanyaan Terbuka: 1.053 QA (Tidak memenuhi asumsi ruang probabilitas biner tertutup)
└── Pertanyaan Biner Yes/No: 1.191 Pasangan QA
    ├── Eksklusi Cacat Translasi Kritis (T2, T4, T5, T6, T7, T8): 69 QA (Pembalikan negasi, salah organ, laterality flip)
    └── Primary Cohort (CLEAN T0/T1): 1.122 Pasangan QA (292 Citra Fisik Unik)
        ├── Primary Test Split: 799 QA Pairs (204 Klaster Citra Unik)
        └── Calibration Dev Split: 323 QA Pairs (88 Klaster Citra Unik)
            └── Kebocoran Lintas-Split (Cross-Split Leakage): PERSIS 0 KLASTER (0.0%)
```

### 9.3 Rekonsiliasi Dual-Track Kohort Evaluasi (Active Union $N=1.133$)
Untuk memungkinkan evaluasi skala penuh sekaligus menjaga auditabilitas terhadap revisi terdahulu, dibentuk Active Evaluation Cohort Union:
$$\mathcal{D}_{\text{eval}} = \mathcal{D}_{\text{primary}} \cup \mathcal{D}_{\text{rev03\_benchmark}}$$

| Kohort Evaluasi | Total Sampel | Test Split ($N_{\text{test}}$) | Dev Split ($N_{\text{dev}}$) | Klaster Test | Klaster Dev | Overlap Antar-Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track 1: Primary Cohort (T1)** | 1.122 | **799** | 323 | 204 | 88 | 0 (0.0%) |
| **Track 2: Historical Benchmark (T2)** | 150 | **92** | 58 | 35 | 23 | 0 (0.0%) |
| **Active Union Evaluated** | **1.133** | **806** | **327** | **204** | **88** | 0 (0.0%) |

- **SHA-256 Manifest Otoritatif (`manifest_rev05.csv`):**
  `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`
- **Integritas Citra:** Seluruh 314 citra terverifikasi bit-identik melalui hash SHA-256 piksel mentah.

---

## 10. Pemilihan Model dan Alasan Teknis

### 10.1 Model yang Dievaluasi
1. **Model Fondasi Umum:** `Qwen/Qwen2.5-VL-7B-Instruct`
2. **Model Spesialis Medis:** `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` (dilatih di atas backbone Qwen2.5-VL-7B menggunakan dataset PubMedVision).

### 10.2 Alasan Teknis dan Kesetaraan Arsitektur
- **Arsitektur Identik:** Kedua model menggunakan kelas arsitektur yang sama (`Qwen2_5_VLForConditionalGeneration`), patch vision encoder yang sama, dan vocabulary tokenizer BPE yang sama (~152.064 token). Hal ini meminimalkan variabel pengacau arsitektur.
- **Keterbatasan Perangkat Keras:** Pengujian model 32B terbukti menghasilkan CUDA OOM pada dual-T4 GPU (15 GiB VRAM per akselerator). Varian 7B dengan kuantisasi 4-bit NF4 berjalan stabil dalam batas kapasitas memori.
- **Batas Inferensi Kausal:** Perbedaan performa antara Qwen-7B dan HuatuoGPT-7B dilaporkan sebagai perbandingan dua sistem model terwujud, bukan klaim kausal terisolasi atas satu-satunya faktor domain medis.

---

## 11. Protokol Inferensi dan Preprocessing

### 11.1 Format Prompt Simetris 1-to-1
Untuk menghindari bias instruksi:
- **Bahasa Inggris (EN):**
  ```text
  Answer only YES or NO.
  Question: {question_text}
  ```
- **Bahasa Indonesia (ID):**
  ```text
  Jawab hanya dengan YA atau TIDAK.
  Pertanyaan: {question_text}
  ```

### 11.2 Chat Template dan Prefix Invariance
Prompt dibungkus menggunakan chat template resmi Qwen2.5:
`<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt_text}<|im_end|>\n<|im_start|>assistant\n`

Setiap pemanggilan inferensi memverifikasi *prefix invariance assertion*:
$$\text{input\_ids}[0, :L_{\text{prompt}}] == \text{prompt\_input\_ids}[0]$$

---

## 12. Validasi Confidence Score

### 12.1 Mekanisme Causal Logit Slicing
Dalam arsitektur autoregresif Transformer, logit pada indeks $t$ memprediksi token pada indeks $t+1$. Slicing logit target yang tepat secara matematis adalah:
$$\text{target\_logits} = \text{logits}[L_{\text{prompt}} - 1 : L_{\text{prompt}} - 1 + L_{\text{target}}]$$

### 12.2 Length-Normalized Surprisal vs Raw Sequence Sum
- **Masalah Raw Log-Likelihood:** Pada tokenizer Qwen, `"Ya"` adalah 1 token (`[62893]`), sedangkan `"Tidak"` adalah 2 token (`[51, 60389]`). Menjumlahkan log-likelihood negatif tanpa normalisasi memberikan penalti panjang yang berat pada `"Tidak"`, menyebabkan model bias 100% memilih `"Ya"` (seperti yang terjadi pada rev02).
- **Normalisasi Panjang:**
  $$\bar{s}(c) = \frac{1}{L_c} \sum_{t=1}^{L_c} \log P(w_t^{(c)} \mid w_{<t}^{(c)}, X)$$
- **Candidate-Normalized Preference Score:**
  $$P_{\text{cand}}(c_1) = \frac{\exp(\bar{s}(c_1))}{\exp(\bar{s}(c_1)) + \exp(\bar{s}(c_2))}$$
- **Status Probabilitas:** Nilai ini adalah skor preferensi komparatif tertutup, bukan probabilitas posterior global terkalibrasi. Istilah yang digunakan adalah *pairwise candidate score*.

---

## 13. Desain Intervensi Visual

### 13.1 Protokol Dua Kondisi
1. **`P0_ORIGINAL`:** Citra radiologi asli RGB sesuai resolusi asli dalam dataset.
2. **`P1_BLACK_BLANK`:** Citra kanvas hitam deterministik ($(0,0,0)$) dengan dimensi lebar ($W$), tinggi ($H$), rasio aspek, kanal warna (RGB), dan normalisasi piksel yang identik dengan P0.

### 13.2 Sifat Out-Of-Distribution (OOD) dan Batas Epistemik
> [!IMPORTANT]
> Citra hitam polos adalah masukan OOD bagi vision encoder. Respon model terhadap P1 mengukur **sensitivitas perilaku terhadap ketiadaan informasi visual**, bukan bukti bahwa model melakukan lokalisasi lesi atau memahami penalaran radiologis kausal. Jika model mempertahankan jawaban pada P1, hal ini menunjukkan bahwa model bersandar pada prior teks bahasa pertanyaan.

---

## 14. Formula dan Interpretasi Estimator

Lima estimator dievaluasi secara sejajar pada sampel yang identik:

| Simbol | Formula Estimator | Rentang | Rasional Metodologis & Perilaku Ekstrem |
| :--- | :--- | :---: | :--- |
| **$M_1$** | $S_{M1} = p_{\text{orig}}$ | $[0.5, 1.0]$ | **Baseline Keyakinan Internal:** Mengabaikan informasi visual; mengurutkan semata-mata berdasarkan preferensi teks internal model. |
| **$M_2$** | $S_{M2} = \max(0, \Delta P_{\text{dir}})$ | $[0.0, 1.0]$ | **Sensitivitas Visual Murni:** Mengabaikan confidence; hanya mempertahankan prediksi jika penghapusan citra menurunkan probabilitas. Risiko: memberikan skor 0 jika $\Delta P \le 0$ meskipun prediksi benar. |
| **$M_{\text{dir}}$** | $S_{\text{dir}} = p_{\text{orig}} \times \max(0, \Delta P_{\text{dir}})$ | $[0.0, 1.0]$ | **Kombinasi Multiplikatif:** Memberikan penalti nol keras (*hard zero collapse*) jika $\Delta P \le 0$. Risiko: terlalu agresif menolak jawaban benar yang memiliki sensitivitas visual marginal. |
| **$M_{\text{norm}}$** | $S_{\text{norm}} = p_{\text{orig}} \times \frac{1 + \Delta P_{\text{dir}}}{2}$ | $[0.0, 1.0]$ | **Estimator Primer (Kombinasi Linear Halus):** Jika $\Delta P = 0$, skor menjadi $0.5 \times p_{\text{orig}}$; jika $\Delta P = -1$, skor $0$; jika $\Delta P = 1$, skor $p_{\text{orig}}$. Mencegah keruntuhan nol mendadak pada prediksi yang benar. |
| **$M_{3,\text{old}}$** | $S_{M3} = p_{\text{orig}} \times \frac{1 + \|\Delta P_{\text{dir}}\|}{2}$ | $[0.5, 1.0]$ | **Kontrol Historis Heuristik Unsigned:** Dipertahankan murni sebagai pembanding historis. Memiliki cacat teoretis: memberikan skor tinggi pada sampel di mana citra dihapus justru menaikkan keyakinan model. |

---

## 15. Protokol Selective Prediction

### 15.1 Pemeringkatan dan Resolusi Nilai Seri (Tie-Breaking)
Sampel diurutkan secara menurun berdasarkan skor reliabilitas $S$:
$$S_{(1)} \ge S_{(2)} \ge \dots \ge S_{(N)}$$
- **Resolusi Seri Deterministik:** Jika $S_{(i)} == S_{(j)}$, urutkan berdasarkan $p_{\text{orig}}$ menurun; jika tetap sama, urutkan berdasarkan `qa_id` menaik.

### 15.2 Definisi Cakupan (Coverage) dan Risiko Selektif (Selective Risk)
Untuk setiap tingkat retensi $k \in \{1, \dots, N\}$:
$$\text{Coverage } \phi_k = \frac{k}{N} \in (0, 1]$$
$$\text{Selective Risk } r_k = 1 - \frac{1}{k} \sum_{j=1}^k \mathbb{I}(\hat{y}_{(j)} = y_{(j)}^*)$$

### 15.3 Perhitungan Area Under the Risk–Coverage Curve (Discrete Riemann AURC)
Sesuai formulasi Geifman & El-Yaniv (NeurIPS 2017), AURC diskrit dihitung sebagai jumlahan Riemann exact kurva tangga:
$$\text{AURC} = \frac{1}{N} \sum_{k=1}^N r_k$$

### 15.4 Ambang Keputusan Abstention dan Evaluasi Titik Operasional
Sistem mengevaluasi risiko selektif pada lima titik cakupan acuan:
- **Coverage 1.00 (100%):** Kinerja tanpa penolakan (ekuivalen dengan $1 - \text{Akurasi}$).
- **Coverage 0.90 (90%):** 10% kasus paling meragukan dirujuk ke radiolog.
- **Coverage 0.80 (80%):** **Titik operasional acuan rujukan klinis** (20% kasus ditolak).
- **Coverage 0.70 (70%):** 30% kasus ditolak.
- **Coverage 0.50 (50%):** Separuh kasus paling dipercaya dipertahankan.

---

## 16. Metrik Evaluasi

1. **Metrik Klasifikasi & Bahasa (RQ1):**
   - $\text{Acc}_{\text{EN}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i,\text{en}} = y_i^*)$
   - $\text{Acc}_{\text{ID}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i,\text{id}} = y_i^*)$
   - $\text{Language Gap} = \text{Acc}_{\text{EN}} - \text{Acc}_{\text{ID}}$
   - $\text{Cross-Lingual Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i,\text{en}} \ne \hat{y}_{i,\text{id}})$
   - $\text{Joint Correctness} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i,\text{en}} = y_i^* \land \hat{y}_{i,\text{id}} = y_i^*)$
2. **Metrik Kalibrasi:**
   - Expected Calibration Error (ECE) dengan 10 bin seragam bobot populasi.
   - Brier Score: $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_{\text{orig},i} - \mathbb{I}(\hat{y}_i = y_i^*))^2$.
   - Confidently Wrong Rate ($c \ge 0.90$): Dilaporkan dengan dua denominator yang jelas (proporsi populasi total vs proporsi kondisional pada subset $c \ge 0.90$).
3. **Metrik Sensitivitas Visual (RQ2):**
   - $\text{Mean } \Delta P_{\text{dir}} = \frac{1}{N} \sum_{i=1}^N (P_{\text{orig}}(\hat{y}_i) - P_{\text{blank}}(\hat{y}_i))$
   - $\text{Visual Flip Rate } (P_0 \to P_1) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{i,P_0} \ne \hat{y}_{i,P_1})$
4. **Metrik Prediksi Selektif (RQ3):**
   - Discrete AURC.
   - $\Delta\text{AURC} = \text{AURC}(M) - \text{AURC}(M_1)$ (nilai negatif menunjukkan perbaikan).
   - Selective Risk pada Coverage 80% ($r_{0.80}$).

---

## 17. Rancangan Analisis Statistik

### 17.1 Ketergantungan Data Klaster Citra Fisik
Beberapa pertanyaan berasal dari citra radiologi fisik yang sama (rata-rata 3.9 baris per citra pada primary cohort). Mengasumsikan observasi independen akan meremehkan *standard error*. Seluruh prosedur inferensial menggunakan **klaster citra fisik kanonis (`canonical_image_id`)** sebagai unit sampling.

### 17.2 Paired Image-Cluster Bootstrap ($B = 1.000$)
- Unit resampling: Klaster citra unik (204 klaster pada Primary Test, 35 klaster pada Benchmark Test).
- Jumlah replikasi: $B = 1.000$, seed acak: 42.
- Mengestimasi 95% Confidence Interval empiris (persentil 2.5% dan 97.5%) untuk AURC dan $\Delta\text{AURC}$.

### 17.3 Exact Paired Image-Cluster Permutation Test
Untuk menguji signifikansi perbaikan terhadap $M_1$ ($H_0: \mathbb{E}[\Delta\text{AURC}] = 0$):
1. Untuk setiap klaster $c \in \{1, \dots, K\}$, bangkitkan variabel Rademacher acak $\epsilon_c \in \{-1, +1\}$.
2. Jika $\epsilon_c = -1$, tukar vektor skor estimator $M$ dan $M_1$ untuk seluruh baris pertanyaan di dalam klaster $c$.
3. Hitung $\Delta\text{AURC}^{(b)}$ pada data permutasi.
4. Nilai $p$ exact dua sisi:
   $$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(|\Delta\text{AURC}^{(b)}| \ge |\Delta\text{AURC}_{\text{obs}}|)}{B + 1}$$

### 17.4 Pengendalian Multiplisitas Uji Hipotesis (Holm-Bonferroni Correction)
Untuk mengontrol Family-Wise Error Rate (FWER) pada taraf $\alpha = 0.05$ di atas 4 pengujian hipotesis sekunder ($M_2, M_{\text{dir}}, M_{\text{norm}}, M_3^{\text{old}}$ vs $M_1$), nilai $p$ disesuaikan menggunakan prosedur *step-down* Holm-Bonferroni.

---

## 18. Eksperimen Utama dan Analisis Sensitivitas

### 18.1 Matriks Eksperimen Utama (Primary Confirmatory Cohort)
- **Dataset:** Primary Test Split ($N = 799$, 204 klaster citra fisik).
- **Model:** `Qwen2.5-VL-7B-Instruct` dan `HuatuoGPT-Vision-7B-Qwen2.5VL`.
- **Intervensi:** Evaluasi $P_0$ (asli) dan $P_1$ (hitam polos).
- **Scoring:** Length-normalized candidate sequence score dengan terminasi `<|im_end|>`.

### 18.2 Analisis Sensitivitas yang Direncanakan
1. **Sensitivitas terhadap Cacat Translasi (Benchmark 150 vs CLEAN 139):** Mengevaluasi apakah penyertaan 11 sampel berisiko translasi mengubah ranking estimator.
2. **Sensitivitas terhadap Skema Scoring (Exact Sequence vs Length-Normalized):** Membandingkan ranking AURC dari $\sum \log P$ vs $\frac{1}{L} \sum \log P$.
3. **Sensitivitas terhadap Klaster Dominan:** Memeriksa kestabilan metrik ketika citra dengan jumlah pertanyaan terbanyak dieliminasi via *jackknife*.

---

## 19. Validasi, Reproducibility, dan Quality Control

### 19.1 Rangkaian Pengujian Unit Otomatis (Preflight Test Suite)
Implementasi kode diverifikasi melalui rangkaian unit test (`test_decisive_validation_v5.py`):
- `test_manifest_hash_and_invariants`: Lulus verifikasi hash bit-identik dan invarian split.
- `test_image_verification_and_tamper_rejection`: Lulus deteksi manipulasi piksel SHA-256.
- `test_causal_logit_alignment_and_prefix_invariance`: Lulus slicing kausal $[L_p - 1 : L_p - 1 + L_t]$.
- `test_discrete_aurc_manual_calculation`: Lulus verifikasi presisi numerik diskrit Riemann ($< 10^{-12}$).
- `test_paired_cluster_permutation_test`: Lulus verifikasi uji permutasi Rademacher tingkat klaster.
- `test_holm_bonferroni`: Lulus penyesuaian nilai $p$ step-down.
- `test_atomic_checkpoint_governance`: Lulus mekanisme penulisan atomik `.tmp` dan proteksi resume.

### 19.2 Protokol Pencegahan Korupsi Checkpoint
Penyimpanan checkpoint pada GPU cloud menggunakan penulisan atomik via `os.replace` dan validasi ketat `revision_id == "rev05"`, `manifest_sha256`, serta ketiadaan nilai `NaN`/`Inf`.

---

## 20. Hasil Pilot dan Status Verifikasi

Untuk menjaga transparansi akademik penuh, tabel berikut mencatat riwayat hasil eksperimen dan status validitas ilmiahnya:

| Iterasi Eksperimen | Sampel Evaluasi | Model | Estimator | Hasil Kunci | Status Validitas | Catatan Auditor |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| **Kaggle Pilot 01–03** | $N=150$ | HuatuoGPT-7B | $M_1 \to M_3^{\text{old}}$ | AURC 0.2786 $\to$ 0.2173 (-22%) | `UNVERIFIED_PILOT_ARTIFACT` | Terdistorsi oleh tokenizer bug Cell 11 dan evaluasi lintas-bahasa Cell 16/20. |
| **Kaggle Pilot 01–03** | $N=150$ | HuatuoGPT-7B | Baseline ECE | ECE 0.2350, Cond. Err 36.63% | `UNVERIFIED_PILOT_ARTIFACT` | Keyakinan terinflasi secara artifisial akibat residu token bahasa Inggris pada teks Indonesia. |
| **Fine-Tuning QLoRA** | $N=150$ | Qwen-7B / 32B | LoRA ($B_2$) | Acc ID turun -12%, Flip naik +4% | `ABANDONED_EMPIRICAL` | Fine-tuning terbukti merusak akurasi bahasa Indonesia; 32B memicu CUDA OOM. Ditinggalkan resmi. |
| **rev02 Live Kaggle** | $N=92$ (Test) | Qwen-7B | $M_1 \to M_{\text{dir}}$ | AURC 0.3146 $\to$ 0.4119 (Worse) | `EMPIRICAL_FLAWED` | Terdistorsi length penalty (90/92 prediksi YA). Menunjukkan bahwa $M_{\text{dir}}$ jatuh akibat zero collapse. |
| **rev02 Live Kaggle** | $N=92$ (Test) | HuatuoGPT-7B | $M_1 \to M_{\text{dir}}$ | AURC 0.3584 $\to$ 0.5041 (Worse) | `EMPIRICAL_FLAWED` | Terdistorsi length penalty (92/92 prediksi YA). |
| **Dry-Run Simulation** | $N=150$ | Sintetis (Seed 42) | $M_1 \to M_{\text{norm}}$ | AURC 0.1142 $\to$ 0.0714 | `VERIFIED_CODE_ONLY` | Simulasi offline untuk verifikasi fungsi matematika, bukan bukti inferensi neural riil. |
| **rev05 Confirmatory** | $N=799$ (Primary) | Qwen-7B & Huatuo-7B | $M_1, M_2, M_{\text{dir}}, M_{\text{norm}}$ | Menunggu eksekusi GPU Dual T4 | `PENDING_CONFIRMATORY_GPU` | Pipeline terkunci 100%, siap dieksekusi di Kaggle `08_decisive_validation_rev05.ipynb`. |

---

## 21. Ancaman terhadap Validitas (Threats to Validity)

### 21.1 Validitas Internal
- **Efek OOD Citra Hitam:** Respons model pada citra hitam dapat dipengaruhi oleh anomali representasi fitur vision encoder, bukan murni hilangnya informasi diagnostik.
- **Normalisasi Ruang Kandidat Tertutup:** Evaluasi biner tertutup (`Yes/No`) tidak mencakup kemampuan generasi bebas teks medis.

### 21.2 Validitas Eksternal
- **Generalisasi Modalitas:** Dataset IndoRad-VQA terbatas pada citra radiologi (X-ray, CT, MRI) dari domain VQA-RAD; temuan tidak dapat digeneralisasi langsung ke histopatologi atau USG tanpa evaluasi tambahan.
- **Skala Model:** Keterbatasan komputasi membatasi evaluasi pada model kelas 7B.

### 21.3 Validitas Konstruk
- Perubahan nilai AURC mencerminkan efisiensi pemeringkatan kesalahan retrospektif, bukan ukuran keselamatan klinis prospektif.

---

## 22. Batasan dan Pertimbangan Keselamatan Medis

> [!CAUTION]
> **PERINGATAN KESELAMATAN MEDIS & BATASAN EPISTEMIK:**
> 1. Sistem selective prediction ini **BUKAN alat diagnosis medis klinis** dan tidak boleh digunakan untuk pengambilan keputusan medis pada pasien tanpa pengawasan dokter spesialis radiologi berlisensi.
> 2. Penurunan nilai AURC semata-mata membuktikan kemampuan statistik pengurutan risiko kesalahan pada dataset uji retrospektif.
> 3. Metrik ini tidak menjamin eliminasi kesalahan diagnosis fatal (*silent diagnostic failure*).
> 4. Model kecerdasan buatan dalam penelitian ini berstatus purwarupa penelitian akademis (*academic research prototype*).

---

## 23. Tinjauan Literatur dan Pemetaan Referensi Terverifikasi

### 23.1 Literature Mapping Table

| Referensi & Sitasi Formal | Fokus Penelitian | Dataset & Modalitas | Metode yang Digunakan | Temuan Kunci yang Relevan | Keterbatasan Utama | Hubungan dengan Penelitian Ini |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Yudhistira, Malik, & Yudistira (2026)**<br>CVPR 2026 MMFM-BIOMED<br>`arXiv:2606.03693` | Evaluasi language shift pada Medical VLM di Indonesia | IndoRad-VQA (2.244 QA, 314 citra radiologi) | Zero-shot & few-shot prompting model VLM umum dan regional | Penurunan akurasi 8%–25% saat prompt diterjemahkan ke bahasa Indonesia; flip rate tinggi. | Bersifat deskriptif; tidak mengembangkan mekanisme mitigasi atau selective prediction. | Menyediakan dataset sumber dan membuktikan eksistensi masalah *language shift* pada populasi target. |
| **Sadanandan & Behzadan (2026)**<br>CVPR 2026 Med-Reasoner<br>`arXiv:2603.20985` | Per-sample safety classification & evaluasi visual reliance | MIMIC-CXR, PadChest (Chest X-ray) | Taksonomi 4-kuadran berbasis konsistensi parafrase vs uji teks-saja (image ablation) | Fine-tuning LoRA menurunkan flip rate semu tetapi mendorong model ke "Dangerous Quadrant" (terpaku prior teks). | Tidak merumuskan fungsi reject option atau metrik selective risk-coverage kuantitatif. | Memberikan landasan teoretis bahwa penghilangan citra esensial untuk mendeteksi *false reliability* akibat prior teks. |
| **Khanmohammadi et al. (2026)**<br>`arXiv:2605.10893` | Estimasi keyakinan VLM via Blind-Image Contrastive Ranking (BICR) | VLCB Benchmark (VQA, Medical, DocVQA) | Training lightweight probe menggunakan contrastive ranking loss (citra asli vs citra hitam) | Menghukum keyakinan tinggi pada citra hitam meningkatkan kalibrasi (ECE) dan diskriminasi (AUROC). | Memerlukan training probe eksternal pada representasi tersembunyi (*hidden states*). | Mendukung hipotesis bahwa kontras citra hitam memuat sinyal reliabilitas, tetapi riset kita mengujinya pada waktu inferensi tanpa probe. |
| **Geifman & El-Yaniv (2017)**<br>NeurIPS 2017 | Teori Selective Classification untuk Deep Neural Networks | ImageNet, CIFAR-10 (Visi Unimodal) | Pemeringkatan probabilitas softmax & kurva Risk–Coverage | Merumuskan metrik formal Area Under the Risk–Coverage Curve (AURC) untuk trade-off coverage-risk. | Terbatas pada visi unimodal klasifikasi tertutup; mengasumsikan distribusi i.i.d. tanpa domain shift. | Menjadi dasar perumusan matematis Discrete Riemann AURC dan protokol selective prediction dalam skripsi ini. |
| **Kamath, Jia, & Liang (2020)**<br>ACL 2020 | Selective Question Answering di bawah pergeseran domain | SQuAD, TriviaQA (Teks Unimodal) | Kalibrator eksternal dan ambang penolakan (*reject option*) | Penolakan berbasis ketidakpastian efektif menekan risiko kesalahan saat terjadi pergeseran distribusi teks. | Unimodal teks murni; tidak memiliki komponen intervensi modalitas silang visual. | Memberikan preseden metodologis evaluasi selective prediction pada tugas Question Answering di bawah *domain shift*. |
| **Agrawal et al. (2018)**<br>CVPR 2018 | Evaluasi ketergantungan prior bahasa pada VQA (VQA-CP) | VQA-CP v1 & v2 (Natural Images) | Reorganisasi split pertanyaan dan branch pertanyaan adversarial | VLM standar sangat rentan bersandar pada korelasi statistik kata tanya daripada bukti visual. | Digunakan sebagai dataset pengujian diagnostik, bukan aturan keputusan selective prediction runtime. | Memperkuat urgensi pengujian intervensi visual untuk mendeteksi ketergantungan model pada prior bahasa. |
| **Chen et al. (2024)**<br>`arXiv:2406.19280` | Pengembangan model fondasi medis multimodal HuatuoGPT-Vision | PubMedVision (1.3M QA medis multimodal) | Visual instruction tuning pada arsitektur Qwen / Yi | Mencapai performa SOTA open-source pada benchmark medis (MMMU Health); memperkuat penalaran klinis. | Tidak mengevaluasi kalibrasi atau ketahanan pada bahasa di luar bahasa Inggris dan Mandarin. | Model `HuatuoGPT-Vision-7B` menjadi representasi utama model spesialis medis dalam komparasi skripsi ini. |

### 23.2 Sintesis Literatur Ilmiah
Literatur terkini menunjukkan konsensus bahwa model vision-language autoregresif sangat rentan terhadap *shortcut learning* dan *visual ungroundedness*, di mana model menghasilkan jawaban meyakinkan semata-mata berdasarkan prioritas teks (Agrawal et al., 2018; Sadanandan & Behzadan, 2026). Masalah ini semakin parah ketika model menghadapi bahasa non-latih (*language shift*) seperti bahasa Indonesia pada domain radiologi (Yudhistira et al., 2026). Meskipun framework selective prediction telah matang pada visi dan NLP unimodal (Geifman & El-Yaniv, 2017; Kamath et al., 2020), penggabungan sinyal intervensi visual kontrafaktual pada waktu inferensi untuk mitigasi overconfidence lintas-bahasa merupakan area yang belum terjawab secara definitif. Penelitian ini mengisi celah tersebut dengan mengevaluasi secara kritis batas efektivitas sinyal sensitivitas visual waktu-inferensi.

---

## 24. Implementation Checklist

Pekerjaan implementasi dibagi ke dalam urutan prioritas ketat:

### Prioritas 1: Kesiapan Pra-Inferensi (Tuntas Lolos)
- [x] Audit integritas dataset IndoRad-VQA dan pembersihan translasi biner (`manifest_rev05.csv` SHA-256: `fb55fd75...`).
- [x] Implementasi causal logit slicing dan prefix invariance assertion.
- [x] Implementasi length-normalized candidate sequence scoring dengan token `<|im_end|>`.
- [x] Implementasi Discrete Riemann AURC hand-verified dan Monte Carlo Paired Cluster Permutation Test.
- [x] Pembangkitan notebook eksekusi Kaggle Dual-T4 (`08_decisive_validation_rev05.ipynb`) dengan AST parse 100% OK.
- [x] Eksekusi unit test suite `test_decisive_validation_v5.py` (9/9 Unit Test PASS).

### Prioritas 2: Eksekusi GPU dan Ekstraksi Artefak (Menunggu Eksekusi Cloud)
- [ ] Upload notebook `08_decisive_validation_rev05.ipynb` ke Kaggle dengan akselerator Dual Tesla T4.
- [ ] Jalankan "Run All" untuk mengevaluasi Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B pada Active Union ($N=1.133$).
- [ ] Unduh arsip hasil resmi `decisive_validation_rev05_results_final.zip`.
- [ ] Ekstrak checkpoint, tabel komparasi CSV, dan kurva risk-coverage ke `main/results/decisive_validation/rev05/`.

### Prioritas 3: Analisis Statistik Inferensial dan Pelaporan
- [ ] Hitung selisih berpasangan $\Delta\text{AURC}$ dan 95% Cluster Bootstrap CI pada Primary Test Split ($N=799$).
- [ ] Terapkan penyesuaian nilai $p$ Holm-Bonferroni pada keluarga 4 komparasi estimator terhadap $M_1$.
- [ ] Analisis korelasi sensitivitas visual ($\Delta P_{\text{dir}}$) dengan kebenaran prediksi (RQ2).
- [ ] Bandingkan pola degradasi bahasa dan performa selektif antara Qwen-7B dan HuatuoGPT-7B (RQ4).
- [ ] Jalankan analisis sensitivitas pada subset Historical Benchmark ($N=92$) dan analisis dampak translasi FLAGGED.

### Prioritas 4: Finalisasi Naskah Skripsi
- [ ] Sinkronkan seluruh angka bab IV naskah skripsi dengan artefak CSV rev05 resmi.
- [ ] Dokumentasikan temuan secara objektif (termasuk jika hipotesis keunggulan $M_{\text{norm}}$ tidak terbukti secara signifikan).
- [ ] Perbarui slide presentasi skripsi dengan kurva Risk-Coverage 300 DPI rev05.

---

## 25. Experiment Decision Gates

| Decision Gate | Kriteria Keberhasilan | Bukti Verifikasi yang Dibutuhkan | Tindakan Jika Lolos | Tindakan Jika Gagal |
| :--- | :--- | :--- | :--- | :--- |
| **Gate A: Validitas Implementasi** | Tokenizer konsisten, logit slicing tepat kausal, prompt simetris, manifest hash bit-identik. | 9/9 Unit test pada `test_decisive_validation_v5.py` berstatus PASS. | Lanjut ke eksekusi inferensi model GPU. | **STOP.** Perbaiki indexing logit atau tokenisasi sebelum inferensi. Dilarang lanjut. |
| **Gate B: Ketiadaan Anomali Scoring** | Distribusi prediksi bahasa Indonesia tidak terdistorsi 100% ke satu kelas; candidate score berhingga tanpa NaN. | Evaluasi Dev Set ($N=327$) menunjukkan rasio prediksi `YA` dan `TIDAK` berada dalam rentang wajar ($20\% - 80\%$). | Lanjut ke evaluasi Test Set ($N=806$). | Periksa normalisasi panjang surprisal atau prefix prompt. |
| **Gate C: Kepatuhan Protokol Citra** | Dimensi, rasio, kanal, dan preprocessing P0 dan P1 identik secara deterministik. | Verifikasi fungsi `create_black_blank_image` pada unit test. | Lanjut ke komputasi $\Delta P_{\text{dir}}$. | Perbaiki fungsi pembuatan citra hitam. |
| **Gate D: Kestabilan Checkpoint** | 100% dari 1.133 sampel tereksekusi tanpa crash; penulisan atomik utuh. | Checkpoint CSV memuat seluruh kolom skor numerik berhingga untuk kedua model. | Lanjut ke perhitungan metrik selective prediction. | Resume inferensi menggunakan fungsi atomic checkpoint. |
| **Gate E: Bukti Nilai Tambah Selektif** | Estimator primer $M_{\text{norm}}$ menghasilkan $\Delta\text{AURC} < 0$ dengan $p_{\text{adj}} < 0.05$ pada Primary Test Split. | Tabel komparasi estimator dan hasil uji permutasi klaster berpasangan. | Klaim bahwa sensitivitas visual memberikan nilai tambah terdukung secara empiris. | **Tetap laporkan hasil sebagai temuan negatif yang informatif.** Dilarang memanipulasi formula atau data. |
| **Gate F: Tata Kelola Klaim Ilmiah** | Naskah tidak mengklaim keselamatan klinis mutlak; judul dan kontribusi proporsional dengan data. | Audit redaksional naskah skripsi dan abstract. | Siap untuk sidang skripsi dan submission jurnal. | Revisi redaksi naskah; turunkan klaim yang melampaui bukti. |

---

## 26. Rencana Kerja dan Prioritas

1. **Tahap 1 (Hari 1):** Eksekusi mandiri notebook `08_decisive_validation_rev05.ipynb` pada Kaggle Dual Tesla T4 (estimasi runtime 40–50 menit).
2. **Tahap 2 (Hari 1–2):** Verifikasi integritas file hasil unduhan, pengecekan ketiadaan NaN, dan ekstraksi tabel metrik utama.
3. **Tahap 3 (Hari 2–3):** Penyusunan tabel Bab IV Skripsi berdasarkan output empiris rev05, pembuatan visualisasi kurva Risk-Coverage final 300 DPI, dan rekapitulasi nilai $p$ statistik.
4. **Tahap 4 (Hari 4):** Finalisasi Bab V (Pembahasan dan Batasan) yang membedakan secara tegas antara kontribusi terbukti dan batasan klinis.

---

## 27. Daftar Keputusan yang Masih Terbuka (Open Issues)

1. **Isu Terbuka 1: Superioritas Empiris Estimator Primer $M_{\text{norm}}$ vs Baseline $M_1$**
   - *Status:* `OPEN / AWAITING_GPU_RUN`
   - *Bukti yang Dibutuhkan:* Nilai $\Delta\text{AURC}$ dan nilai $p$ permutasi klaster dari inferensi penuh rev05 pada $N=799$ test set.
   - *Dampak:* Jika $M_{\text{norm}}$ tidak mengungguli $M_1$ ($p \ge 0.05$), kesimpulan skripsi bergeser menjadi evaluasi kritis bahwa perturbasi citra hitam pada waktu inferensi tidak cukup untuk mengoreksi miskalibrasi VLM autoregresif.
2. **Isu Terbuka 2: Perilaku Disparitas Model Medis HuatuoGPT pada Scoring rev05**
   - *Status:* `OPEN / AWAITING_GPU_RUN`
   - *Bukti yang Dibutuhkan:* Perbandingan ECE dan flip rate HuatuoGPT vs Qwen pada skema scoring length-normalized.
   - *Dampak:* Mengonfirmasi apakah overconfidence ekstrem HuatuoGPT benar-benar karakteristik inheren model medis ataukah murni artifak implementasi tokenizer lama.

---

## 28. Changelog

### Versi 8.0 (`rev05-master`, 21 Oktober 2026)
- **Restrukturisasi Total Dokumen SSoT:** Menyusun ulang seluruh 28 bagian sesuai standar peer-review metodologi internasional.
- **Revisi Judul dan Klaim Kontribusi:** Mengganti judul menjadi versi netral non-overclaiming: *"Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"*.
- **Pembersihan Klaim Pilot:** Mengklasifikasikan angka historis Run 01–03 (AURC 0.2173, ECE 0.2350, conditional error 36.63%) sebagai `UNVERIFIED_PILOT_ARTIFACT` akibat distorsi tokenizer dan panjang token.
- **Penyelarasan Referensi Primer:** Memperbaiki sitasi Yudhistira et al. menjadi `arXiv:2606.03693` (CVPR 2026 MMFM-BIOMED) dan mengintegrasikan referensi mutakhir Sadanandan & Behzadan (CVPR 2026, `arXiv:2603.20985`) serta Khanmohammadi et al. (2026, `arXiv:2605.10893`).
- **Penetapan Estimator Primer:** Menetapkan $M_{\text{norm}}$ sebagai estimator hipotesis primer terdaftar untuk mencegah *selection bias*, dengan $M_1$ sebagai baseline acuan.
- **Formalisasi Desain Statistik Klaster:** Menetapkan unit analisis bootstrap dan permutasi pada tingkat klaster citra fisik kanonis (`canonical_image_id`) dengan koreksi multiplisitas Holm-Bonferroni.
