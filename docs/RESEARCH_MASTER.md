# RESEARCH MASTER: SINGLE SOURCE OF TRUTH (AUTHORITATIVE THESIS BLUEPRINT) — VERSI 10.0

# Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift

**Judul Bahasa Indonesia:**  
*Prediksi Selektif Berbasis Sensitivitas Visual pada Model Vision-Language Medis Bilingual dalam Pergeseran Bahasa Inggris–Indonesia*

**Judul Alternatif yang Dipertimbangkan (Conditional Working Title):**  
*Cross-Lingual Disagreement and Visual-Deletion Sensitivity for Selective Risk Estimation in English–Indonesian Medical VQA*  
*(Hanya diadopsi secara resmi jika analisis sekunder membuktikan ketidaksepakatan bilingual memberikan nilai tambah inkremental yang signifikan di atas sensitivitas visual).*

**Metadata Dokumen:**  
- **Versi:** 10.0 (Comprehensive Revision Edition: Cross-Lingual Disagreement & Risk Ablation Integration)  
- **Identifier Revisi Teknis:** `rev05-v10-master`  
- **Status Dokumen:** Single Source of Truth (Authoritative, Methodologically Grounded, Pre-Registered Frozen Protocol)  
- **Tanggal Audit & Pengesahan:** 21 Oktober 2026  
- **Tingkat Akademik:** Skripsi Sarjana (S1) Teknik Informatika / Ilmu Komputer  
- **Bidang Keilmuan:** Medical Multimodal Machine Learning, Trustworthy AI, & Selective Prediction  
- **Peneliti / Penulis:** Imam (Undergraduate CS Thesis Researcher)  
- **Standar Integritas:** Peer-review standar jurnal internasional bereputasi / SINTA 1–2; memprioritaskan validitas ilmiah di atas hasil yang tampak bagus.

---

## 1. Metadata, Versi, dan Status Dokumen

Dokumen ini merupakan **satu-satunya acuan resmi (Single Source of Truth / SSoT) Versi 10.0** bagi pelaksanaan, evaluasi statistik, penulisan skripsi, dan naskah publikasi ilmiah. Dokumen ini mempertahankan fondasi metodologis versi 9.0 (`RESEARCH_MASTER.md`) tanpa menimpanya, sekaligus memperluas kerangka evaluasi dengan memasukkan sinyal ketidaksepakatan lintas-bahasa (*cross-lingual disagreement*) dan studi ablasi estimasi risiko terpelajar (*learned risk modeling*).

### 1.1 Klasifikasi Status Klaim dan Hasil (Epistemic Status Disclosure)
Untuk mencegah fabrikasi data dan klaim berlebih (*overclaim*), seluruh data dan temuan diklasifikasikan ke dalam empat tingkat kepastian:
1. **`VERIFIED_STATIC`:** Telah diverifikasi secara kriptografis dan deterministik pada artefak lokal (struktur manifest, hash SHA-256 piksel gambar, konsistensi label biner, parsing AST kode).
2. **`VERIFIED_LOCAL_TEST`:** Telah diverifikasi melalui eksekusi unit test otomatis dan smoke test lokal pada model lokal (misal: Qwen2.5-VL-3B-Instruct / GPU lokal).
3. **`UNVERIFIED_PILOT_ARTIFACT`:** Angka-angka historis dari notebook pilot awal (misal: ECE 0.2350, conditional error 36.63%, penurunan AURC 22% pada Run 01–03) yang terbukti terdistorsi oleh bug tokenizer/asimetri kandidat dan **tidak boleh dikutip sebagai fakta klinis terbukti**.
4. **`PENDING_CONFIRMATORY_GPU`:** Inferensi skala penuh 7B (Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B) pada cloud GPU (Kaggle Dual Tesla T4) dengan protokol `rev05` berstatus siap eksekusi (*ready-to-run*).

---

## 2. Executive Summary

Penelitian ini mengevaluasi apakah sinyal perilaku berupa **sensitivitas terhadap penghilangan informasi citra (*sensitivity to visual deletion*)** dan **ketidaksepakatan prediksi lintas-bahasa (*cross-lingual disagreement*)** dapat memberikan nilai tambah di luar skor preferensi kandidat (*confidence*) internal model dalam mengurutkan risiko kesalahan (*selective prediction*) pada Medical Vision-Language Model (Medical VLM) ketika menjawab pertanyaan radiologi bilingual Inggris–Indonesia.

### Ringkasan Keputusan Metodologis Utama:
1. **Penghentian Pendekatan Fine-Tuning (LoRA Ditinggalkan):** Eksperimen historis membuktikan bahwa fine-tuning QLoRA dengan loss konsistensi semantik biner ($B_2$) menurunkan akurasi bahasa Indonesia sebesar $-12.00\%$ ($70\% \to 58\%$) dan meningkatkan *flip rate* sebesar $+4.00\%$, serta model 32B memicu *CUDA OutOfMemoryError*. Pendekatan pelatihan ulang resmi dihentikan demi pendekatan *inference-time selective prediction*.
2. **Pembersihan Cacat Pilot Terdahulu:** Audit independen mengungkap bahwa klaim awal *extreme overconfidence* pada HuatuoGPT (ECE 0.2350 dan *conditional error* 36.63%) sebagian besar dipicu oleh *tokenizer bug* (hanya mendaftarkan token bahasa Inggris `YES/NO` sehingga membagi residu kecil probabilitas pada teks Indonesia) serta penalti panjang token (*length bias*) pada kata `"Tidak"` (multi-token) vs `"Ya"` (single-token).
3. **Intervensi Citra Dua Kondisi:** Manipulasi citra dibatasi secara deterministik pada citra radiologi asli (`P0_ORIGINAL`) dan citra kanvas hitam polos berdimensi identik (`P1_BLACK_BLANK`). Pengaburan citra (*Gaussian blur*) ditiadakan total sesuai persetujuan dosen pembimbing.
4. **Sifat Out-Of-Distribution (OOD) Citra Hitam:** Citra hitam merupakan masukan OOD bagi vision encoder. Penurunan probabilitas terarah ($\Delta P_{\text{dir}}$) mencerminkan sensitivitas perilaku terhadap ketiadaan masukan visual, bukan bukti langsung pemahaman anatomi atau lokalisasi kausal lesi klinis.
5. **Estimator Primer Pra-Registrasi:** Estimator kombinasi linear halus ($M_{\text{norm}}$) dipertahankan sebagai hipotesis primer karena memberikan penalti halus proporsional tanpa keruntuhan nilai nol (*zero collapse*), dengan $M_1$ sebagai baseline acuan.
6. **Integrasi Sinyal Ketidaksepakatan Lintas-Bahasa ($D_{\text{lang}}$):** Sinyal biner $D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$ dioperasionalkan sebagai indikator ketidakpastian inkremental tanpa memerlukan forward pass GPU tambahan.
7. **Ablation Study Model Risiko Terpelajar:** Diperkenalkan studi ablasi sekunder berbasis regresi logistik teratur pada Development Set ($N=323$) untuk menguji nilai tambah inkremental setiap sinyal: `LR-conf` (confidence saja), `LR-visual` (confidence + visual), dan `LR-full` (confidence + visual + $D_{\text{lang}}$).
8. **Integritas Temuan Negatif:** Jika hasil inferensi konfirmatori menunjukkan bahwa estimator visual atau $D_{\text{lang}}$ tidak mengungguli $M_1$ secara signifikan ($p_{\text{adj}} \ge 0.05$ setelah koreksi Holm-Bonferroni), hasil tersebut dilaporkan secara jujur sebagai temuan negatif yang informatif.

---

## 3. Latar Belakang dan Definisi Masalah

### 3.1 Latar Belakang Klinis dan Multimodal
Dalam penerapan klinis di negara berkembang seperti Indonesia, Medical VLM sering kali dituntut untuk memproses instruksi atau pertanyaan radiologi dalam bahasa lokal (Bahasa Indonesia). Namun, mayoritas model fondasi multimodal dilatih menggunakan data biomedis berbahasa Inggris (misal: MIMIC-CXR, PubMedVision) atau Mandarin. Fenomena pergeseran bahasa (*language shift*) ini menimbulkan degradasi keandalan yang berisiko fatal pada sistem pendukung keputusan medis.

### 3.2 Definisi Masalah Utama
1. **Inkonsistensi Diagnostik Lintas-Bahasa (*Cross-Lingual Flip*):** Pasien yang sama dapat menerima kesimpulan radiologis bertolak belakang semata-mata karena pertanyaan diajukan dalam bahasa Indonesia alih-alih bahasa Inggris ($\hat{y}_{\text{ID}} \ne \hat{y}_{\text{EN}}$).
2. **Ketergantungan pada Prior Teks (*Visual Ungroundedness*):** Model berisiko memberikan jawaban yang benar atau salah secara kebetulan berdasarkan pola frekuensi kata dalam data pelatihan teks daripada memeriksa bukti radiografis (fenomena "Dangerous Quadrant", Sadanandan & Behzadan, CVPR 2026; Zafar et al., 2026).
3. **Kegagalan Deteksi Kesalahan oleh Confidence Tunggal ($M_1$):** Model dapat memberikan jawaban salah dengan keyakinan verbal yang sangat tinggi pada bahasa target, sehingga mekanisme penolakan (*abstention*) berbasis confidence internal gagal menyaring kegagalan tersebut (*silent failure*).

### 3.3 Demarkasi Tiga Konstruk yang Berbeda
Untuk menjaga ketepatan metodologis, tiga konstruk berikut dibedakan secara tegas:
- **Konsistensi Lintas-Bahasa:** Sejauh mana prediksi model pada prompt bahasa Indonesia identik dengan prediksinya pada prompt bahasa Inggris ($\hat{y}_{\text{ID}} == \hat{y}_{\text{EN}}$). *Catatan: Dua prediksi yang konsisten bisa saja sama-sama salah.*
- **Reliabilitas Prediksi:** Probabilitas atau proporsi prediksi model sesuai dengan *ground-truth* medis ($\hat{y} == y^*$).
- **Ketergantungan Bukti Visual (*Visual Reliance*):** Besarnya perubahan respons atau skor kandidat ketika citra radiologi dihapus ($\Delta P_{\text{dir}} = P_{\text{orig}}(\hat{y}) - P_{\text{blank}}(\hat{y})$).

---

## 4. Tinjauan Pustaka (Related Work)

Penelitian ini memetakan posisi ilmiahnya terhadap empat rumpun literatur:

### 4.1 Selective VQA & Learning to Abstain
Selective prediction pada deep learning dirumuskan secara teoritis oleh Geifman & El-Yaniv (NeurIPS 2017) melalui konsep kurva Risk–Coverage dan Area Under the Risk–Coverage Curve (AURC). Kamath et al. (ACL 2020) memperluasnya ke domain NLP Question Answering di bawah pergeseran distribusi teks menggunakan kalibrator eksternal. Dalam ranah VQA multimodal, Dancette et al. (CVPR 2023) memperkenalkan *Learning from Your Peers (LYP)* yang melatih fungsi seleksi multimodal berbasis ketidaksepakatan ensemble. Baru-baru ini, Khan & Fu (CVPR 2024) membuktikan bahwa pada VLM black-box, konsistensi jawaban pada variasi parafrase pertanyaan efektif mengidentifikasi respons yang tidak reliabel tanpa memerlukan akses bobot internal.

### 4.2 Cross-Lingual Consistency & Language Shift
Yudhistira et al. (CVPR 2026 MMFM-BIOMED, `arXiv:2606.03693`) mengevaluasi fenomena *language shift* pada dataset IndoRad-VQA dan menemukan penurunan akurasi sebesar 8%–25% serta *flip rate* yang tinggi saat model dialihkan dari bahasa Inggris ke bahasa Indonesia. Namun, studi mereka bersifat evaluasi statis unselective tanpa mekanisme deteksi kegagalan waktu inferensi.

### 4.3 Confidence Calibration pada Medical VLM
Byun et al. (2026, Microsoft Health & Life Sciences) menunjukkan bahwa overconfidence bersifat endemik di seluruh skala model VLM medis (2B hingga 38B) dan tidak terselesaikan oleh *chain-of-thought* atau scaling. Lebih kritis lagi, mereka membuktikan bahwa metode kalibrasi probabilitas standar (seperti Platt scaling) bersifat monotonik ketat sehingga **tidak mengubah urutan pemeringkatan diskriminatif (AUROC/AURC)**. Mereka mengusulkan *Hallucination-Aware Calibration (HAC)* yang memanfaatkan sinyal grounding visual untuk mengubah ranking.

### 4.4 Visual Grounding & Counterfactual Visual Evaluation
Sadanandan & Behzadan (CVPR 2026 Med-Reasoner, `arXiv:2603.20985`) membuktikan bahwa evaluasi konsistensi parafrase teks saja merupakan "jebakan reliabilitas palsu" (*false reliability trap*); model dapat terlihat sangat konsisten semata-mata karena bersandar pada prior teks, sehingga uji ketidakhadiran citra (*image removal baseline*) wajib disertakan. Zafar et al. (2026, `arXiv:2607.03647`) menemukan bahwa mengganti citra medis dengan citra kosong hanya menurunkan akurasi sebesar $\le 6.5$ pp, membuktikan lemahnya grounding visual. Khanmohammadi et al. (2026, `arXiv:2605.10893`, BICR) melatih probe berbasis loss kontras citra hitam untuk menekan ECE, dan Song et al. (MICCAI 2026, `arXiv:2606.28520`, CounterVHD) mengukur ketidakpastian visual via perturbasi kontrafaktual.

---

## 5. Research Gap dan Demarkasi Kebaruan (Novelty)

### 5.1 Rumusan Research Gap yang Defensibel

1. **Research Gap Primer (Sensitivitas Visual terhadap Penghilangan Citra):**  
   *Belum diketahui secara empiris apakah sensitivitas skor kandidat terhadap penghilangan informasi visual berarah ($\Delta P_{\text{dir}}$) mampu meningkatkan efisiensi pemeringkatan risiko selektif (menurunkan AURC diskrit) secara signifikan dibandingkan skor preferensi kandidat internal ($M_1$) pada Medical VLM yang menjawab pertanyaan radiologi dalam bahasa Indonesia.*
2. **Research Gap Sekunder (Nilai Tambah Inkremental Ketidaksepakatan Lintas-Bahasa):**  
   *Belum diketahui apakah ketidaksepakatan prediksi pada pasangan pertanyaan radiologi bilingual Inggris–Indonesia ($D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$) memberikan informasi tambahan yang signifikan untuk mengurutkan risiko kesalahan pada jawaban bahasa Indonesia, setelah skor preferensi kandidat ($p_{\text{orig}}$) dan sensitivitas visual ($\Delta P_{\text{dir}}$) diperhitungkan.*

### 5.2 Demarkasi dari Pekerjaan Terdahulu
- **Bukan klaim VQA selektif pertama secara umum:** Selective classification telah dirumuskan oleh Geifman & El-Yaniv (2017) dan Dancette et al. (2023).
- **Bukan klaim pertama penggunaan citra hitam:** Penghilangan citra telah digunakan oleh Agrawal et al. (2018), Sadanandan & Behzadan (2026), Zafar et al. (2026), dan Khanmohammadi et al. (2026).
- **Kebaruan Spesifik:** Integrasi sinyal sensitivitas visual berarah waktu-inferensi tanpa pelatihan ulang model (*zero-retraining runtime rule*) dan evaluasi nilai tambah inkremental ketidaksepakatan bilingual ($D_{\text{lang}}$) untuk mitigasi risiko pada VLM radiologi bilingual di bawah pergeseran bahasa.

---

## 6. Tujuan Penelitian dan Kontribusi yang Dapat Dipertahankan

### 6.1 Tujuan Penelitian
1. Mengukur dampak pergeseran bahasa (Inggris ke Indonesia) terhadap akurasi dan konsistensi prediksi model VLM umum dan medis.
2. Menganalisis sejauh mana sensitivitas terhadap penghilangan citra ($\Delta P_{\text{dir}}$) berasosiasi dengan kesalahan prediksi bahasa Indonesia.
3. Menguji secara formal apakah estimator visual primer $M_{\text{norm}}$ mengungguli baseline confidence $M_1$ pada test set beku.
4. Menguji secara empiris melalui studi ablasi apakah penambahan fitur $D_{\text{lang}}$ memberikan peningkatan inkremental dalam pemodelan risiko kesalahan.
5. Membandingkan karakteristik reliabilitas dan respons selektif antara model VLM umum (`Qwen2.5-VL-7B`) dan model spesialis medis (`HuatuoGPT-Vision-7B`).

### 6.2 Kontribusi Potensial yang Terukur (Maksimal 4 Poin)
1. **Evaluasi Terkontrol Sinyal Sensitivitas Visual:** Bukti empiris mengenai apakah penurunan skor akibat penghilangan visual berkorelasi dengan kebenaran diagnostik dan bermanfaat untuk seleksi risiko.
2. **Analisis Nilai Tambah Inkremental Sinyal Lintas-Bahasa:** Penilaian kuantitatif apakah ketidaksepakatan bilingual ($D_{\text{lang}}$) memuat sinyal ortogonal yang memperbaiki estimasi risiko kesalahan bahasa Indonesia.
3. **Studi Ablasi Komparatif Pemodelan Risiko:** Perbandingan terstruktur antara aturan seleksi langsung ($M_1, M_{\text{norm}}$) dan model risiko terpelajar (`LR-conf`, `LR-visual`, `LR-full`) menggunakan pemisahan data berbasis klaster citra fisik.
4. **Pemetaan Karakteristik Keandalan Dua Sistem VLM:** Dokumentasi komparatif profil reliabilitas, degradasi bahasa, dan perilaku OOD citra hitam pada model fondasi umum dan spesialis medis.

---

## 7. Research Questions dan Hipotesis

| Elemen | RQ1: Language Shift & Reliability | RQ2: Visual Sensitivity to Deletion | RQ3: Primary Selective Prediction ($M_{\text{norm}}$) | RQ4: Incremental Cross-Lingual Signal ($D_{\text{lang}}$) | RQ5: General vs Medical Model Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pertanyaan Penelitian** | Bagaimana perubahan bahasa Inggris ke Indonesia memengaruhi akurasi dan konsistensi prediksi kedua model VLM? | Apakah sensitivitas skor kandidat terhadap penghilangan informasi visual berasosiasi dengan kesalahan prediksi pada pertanyaan bahasa Indonesia? | Apakah estimator primer $M_{\text{norm}}$ meningkatkan kinerja selective prediction (AURC lebih rendah) dibanding $M_1$ pada test set? | Apakah $D_{\text{lang}}$ memberikan peningkatan inkremental dalam pemodelan risiko dibanding model yang sudah menggunakan confidence dan sensitivitas visual? | Bagaimana perbedaan profil reliabilitas, degradasi bahasa, dan efektivitas selektif antara model umum dan medis? |
| **Motivasi Ilmiah** | Menilai kerentanan diagnostik akibat transfer bahasa non-dominan (Yudhistira et al., 2026). | Menguji apakah model benar-benar memanfaatkan citra atau terpaku pada prior teks (Sadanandan & Behzadan, 2026). | Menguji efektivitas aturan keputusan langsung berbasis bukti visual waktu-inferensi tanpa pelatihan tambahan. | Menguji apakah inkonsistensi bilingual memuat informasi ketidakpastian tambahan (Khan & Fu, 2024; Dancette et al., 2023). | Membandingkan apakah fine-tuning medis memperkuat atau justru mendistorsi ketahanan lintas-bahasa. |
| **Variabel Bebas (IV)** | Bahasa prompt ($X_{\text{lang}} \in \{\text{EN}, \text{ID}\}$). | Kondisi citra ($I \in \{P_0, P_1\}$). | Estimator selektif ($M_1$ vs $M_{\text{norm}}$). | Himpunan fitur risiko: $[p_{\text{orig}}]$ vs $[p_{\text{orig}}, \Delta P]$ vs $[p_{\text{orig}}, \Delta P, D_{\text{lang}}]$. | Arsitektur model (`Qwen2.5-VL-7B` vs `HuatuoGPT-Vision-7B`). |
| **Variabel Terikat (DV)** | Akurasi, Flip Rate, Joint Correctness, ECE, Brier Score. | Mean $\Delta P_{\text{dir}}$, Korelasi Titik-Biserial $\Delta P_{\text{dir}}$ dgn Error. | Discrete Riemann AURC, Selective Risk pada Coverage 80% ($r_{0.80}$). | $\Delta\text{AURC}(\text{LR-full} - \text{LR-visual})$, Koefisien $\beta_3$, Likelihood Ratio Test. | Perbedaan komparatif metrik RQ1–RQ4 antar kedua model. |
| **Unit Analisis** | Pasangan pertanyaan bilingual pada klaster citra fisik kanonis. | Pertanyaan bahasa Indonesia pada P0 vs P1. | Pasangan prediksi dan label biner terurut pada Test Set beku ($N=799$). | Prediksi pada Test Set beku dari model yang difit pada Dev Set ($N=323$). | Perbandingan berpasangan antar model pada split yang identik. |
| **Hipotesis Alternatif ($H_1$)** | Pergeseran bahasa memicu penurunan akurasi ($\Delta_{\text{acc}} \ge 5\%$) dan flip rate non-trivial ($\ge 15\%$). | Prediksi yang salah memiliki mean $\Delta P_{\text{dir}}$ yang lebih rendah secara signifikan dibanding prediksi benar. | Estimator $M_{\text{norm}}$ menghasilkan AURC lebih rendah secara signifikan dibanding $M_1$ ($p_{\text{adj}} < 0.05$). | `LR-full` menghasilkan AURC lebih rendah secara signifikan dibanding `LR-visual` ($p_{\text{adj}} < 0.05$). | Model medis menunjukkan degradasi bahasa yang tidak lebih kecil dari model umum. |
| **Hipotesis Nol ($H_0$)** | Tidak ada perbedaan akurasi antara bahasa Inggris dan Indonesia ($\mathbb{E}[\Delta_{\text{acc}}] = 0$). | Sensitivitas visual tidak berasosiasi dengan kesalahan prediksi ($\mathbb{E}[\Delta P_{\text{dir}} \mid \text{Benar}] = \mathbb{E}[\Delta P_{\text{dir}} \mid \text{Salah}]$). | Estimator $M_{\text{norm}}$ tidak menurunkan AURC dibanding $M_1$ ($\mathbb{E}[\Delta\text{AURC}] \ge 0$). | $D_{\text{lang}}$ tidak memberikan perbaikan inkremental ($\mathbb{E}[\text{AURC}(\text{LR-full}) - \text{AURC}(\text{LR-visual})] \ge 0$). | Tidak ada perbedaan performa selektif antar kedua model. |
| **Status Pengujian** | Hipotesis Konfirmatori Primer | Hipotesis Asosiasi Diagnostik | **PENGUJIAN HIPOTESIS PRIMER** | **ANALISIS SEKUNDER ABLASI** | Analisis Komparatif Eksploratif |
| **Kriteria Keputusan** | Tolak $H_0$ jika paired permutation $p < 0.05$. | Tolak $H_0$ jika Mann-Whitney $U$ / permutation $p < 0.05$. | Tolak $H_0$ jika permutation $p_{\text{adj}} < 0.05$ & $\Delta\text{AURC} < 0$. | Tolak $H_0$ jika permutation $p_{\text{adj}} < 0.05$ & $\Delta\text{AURC} < 0$. | Deskriptif komparatif & interaksi model $\times$ estimator. |

---

## 8. Definisi Operasional, Notasi, dan Penyelarasan Arah Skor

### 8.1 Notasi Matematis
- $\mathcal{D} = \{(I_i, Q_{i,\text{en}}, Q_{i,\text{id}}, y_i^*)\}_{i=1}^N$: Dataset evaluasi bilingual.
- $C_i \in \{1, \dots, K\}$: Indeks klaster citra fisik kanonis (`canonical_image_id`).
- $P_0 = I_i$: Citra radiologi asli.
- $P_1 = \mathbf{0}_{(W_i, H_i, 3)}$: Citra kanvas hitam deterministik berdimensi dan kanal identik.
- $\mathcal{Y} = \{c_{\text{aff}}, c_{\text{neg}}\}$: Ruang kandidat biner tertutup (`YES/NO` atau `YA/TIDAK`).
- $S(y \mid X) = \sum_{t=1}^L \log P(y_t \mid y_{<t}, X)$: Log-likelihood sekuens target kausal.
- $\bar{s}(y \mid X) = \frac{1}{L} \sum_{t=1}^L \log P(y_t \mid y_{<t}, X)$: Surprisal rata-rata ternormalisasi panjang.
- $P_{\text{cand}}(y \mid X) = \frac{\exp(\bar{s}(y \mid X))}{\sum_{y' \in \mathcal{Y}} \exp(\bar{s}(y' \mid X))}$: Candidate-normalized preference score.
- $\hat{y}_{\text{ID}} = \arg\max_{y \in \mathcal{Y}} P_{\text{cand}}(y \mid P_0, Q_{\text{id}})$: Prediksi diskrit bahasa Indonesia pada citra asli.
- $\hat{y}_{\text{EN}} = \arg\max_{y \in \mathcal{Y}} P_{\text{cand}}(y \mid P_0, Q_{\text{en}})$: Prediksi diskrit bahasa Inggris pada citra asli.
- $p_{\text{orig\_ID}} = P_{\text{cand}}(\hat{y}_{\text{ID}} \mid P_0, Q_{\text{id}})$: Keyakinan pada jawaban terpilih dengan citra asli.
- $p_{\text{blank\_ID}} = P_{\text{cand}}(\hat{y}_{\text{ID}} \mid P_1, Q_{\text{id}})$: Skor yang diberikan pada jawaban yang sama saat citra dihitamkan.
- $\Delta P_{\text{dir}} = p_{\text{orig\_ID}} - p_{\text{blank\_ID}} \in [-1.0, 1.0]$: Sensitivitas visual berarah.
- $D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}}) \in \{0, 1\}$: Indikator ketidaksepakatan prediksi lintas-bahasa.
- $\text{error}_{\text{ID}} = \mathbb{I}(\hat{y}_{\text{ID}} \ne y_{\text{ID}}^*) \in \{0, 1\}$: Indikator kesalahan prediksi bahasa Indonesia terhadap ground truth.

### 8.2 Penyelarasan Arah Skor Reliabilitas (Score Orientation Alignment)
Dalam kerangka selective prediction (Geifman & El-Yaniv, 2017), sampel diurutkan secara menurun berdasarkan **skor reliabilitas / keyakinan $S$**:
$$S_{(1)} \ge S_{(2)} \ge \dots \ge S_{(N)}$$
Sampel dengan skor tertinggi dijawab terlebih dahulu (prioritas retensi tinggi).
Untuk model risiko terpelajar yang memprediksi probabilitas kesalahan $\hat{P}(\text{error}_{\text{ID}} = 1 \mid \mathbf{x})$, skor reliabilitas didefinisikan secara konsisten sebagai:
$$S_{\text{LR}} = 1 - \hat{P}(\text{error}_{\text{ID}} = 1 \mid \mathbf{x}) \in [0, 1]$$
Konvensi ini memastikan bahwa nilai yang lebih tinggi selalu berarti keyakinan yang lebih tinggi / risiko kesalahan yang lebih rendah, sehingga seluruh perbandingan kurva Risk-Coverage dan AURC diskrit berjalan secara harmonis dan adil antar-estimator.

---

## 9. Dataset, Kohort, dan Tata Kelola Bebas Kebocoran

### 9.1 Dataset Sumber
Dataset adalah `Lab-IS/IndoRad-VQA` (Yudhistira et al., CVPR 2026 MMFM-BIOMED, 2.244 baris pertanyaan-jawaban klinis dari 314 citra radiologi unik).

### 9.2 Alur Akuntansi Eksklusi Data
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

### 9.3 Rekonsiliasi Kohort Evaluasi Dual-Track ($N=1.133$)

| Kohort Evaluasi | Total Sampel | Test Split ($N_{\text{test}}$) | Dev Split ($N_{\text{dev}}$) | Klaster Test | Klaster Dev | Overlap Antar-Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track 1: Primary Cohort (T1)** | 1.122 | **799** | 323 | 204 | 88 | 0 (0.0%) |
| **Track 2: Historical Benchmark (T2)** | 150 | **92** | 58 | 35 | 23 | 0 (0.0%) |
| **Active Union Evaluated** | **1.133** | **806** | **327** | **204** | **88** | 0 (0.0%) |

- **SHA-256 Manifest Otoritatif (`manifest_rev05.csv`):**
  `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`
- **Integritas Citra:** Seluruh 314 citra terverifikasi bit-identik melalui hash SHA-256 piksel mentah RGB.

---

## 10. Pemilihan Model dan Alasan Teknis

### 10.1 Model yang Dievaluasi
1. **Model Fondasi Umum:** `Qwen/Qwen2.5-VL-7B-Instruct`
2. **Model Spesialis Medis:** `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` (dilatih di atas backbone Qwen2.5-VL-7B menggunakan dataset PubMedVision).

### 10.2 Alasan Teknis dan Batasan Inferensi Kausal
- Kedua model berbagi arsitektur dasar yang sama (`Qwen2_5_VLForConditionalGeneration`), patch vision encoder yang sama, dan tokenizer BPE (~152.064 token), meminimalkan variabel pengacau struktural.
- Perbedaan hasil antara Qwen-7B dan HuatuoGPT-7B diperlakukan sebagai perbandingan dua sistem model yang dipilih, bukan isolasi kausal satu-satunya efek spesialisasi medis (karena resep pelatihan, data alignment, dan regularisasi HuatuoGPT berbeda secara menyeluruh).
- Keterbatasan perangkat keras: Model 32B memicu CUDA OOM pada dual-T4 GPU (15 GiB limit). Kuantisasi 4-bit NF4 pada model 7B berjalan stabil.

---

## 11. Protokol Inferensi dan Validasi Scoring

### 11.1 Format Prompt Simetris 1-to-1
- **Bahasa Inggris (EN):** `"Answer only YES or NO.\nQuestion: {question_text}"`
- **Bahasa Indonesia (ID):** `"Jawab hanya dengan YA atau TIDAK.\nPertanyaan: {question_text}"`
- Chat template resmi Qwen2.5 digunakan dengan token penutup generasi `<|im_end|>`. Prefix invariance assertion dieksekusi sebelum inferensi.

### 11.2 Slicing Logit Kausal dan Length-Normalized Surprisal
Slicing logit target kausal dieksekusi tepat pada:
$$\text{target\_logits} = \text{logits}[L_{\text{prompt}} - 1 : L_{\text{prompt}} - 1 + L_{\text{target}}]$$
Normalisasi panjang mengevaluasi rata-rata surprisal per-token:
$$\bar{s}(c) = \frac{1}{L_c} \sum_{t=1}^{L_c} \log P(w_t^{(c)} \mid w_{<t}^{(c)}, X)$$
Skor preferensi kandidat dihitung via stable softmax (`logsumexp`):
$$P_{\text{cand}}(c_1) = \frac{\exp(\bar{s}(c_1))}{\exp(\bar{s}(c_1)) + \exp(\bar{s}(c_2))}$$
Skor ini adalah skor preferensi biner komparatif tertutup, bukan probabilitas posterior global terkalibrasi.

---

## 12. Desain Intervensi Visual dan Karakterisasi OOD

- `P0_ORIGINAL`: Citra radiologi asli RGB.
- `P1_BLACK_BLANK`: Citra kanvas hitam deterministik ($(0,0,0)$) berdimensi piksel $(W, H)$, rasio aspek, dan kanal identik.
- **Batas Interpretasi:** Citra hitam polos adalah masukan OOD. Perubahan probabilitas $\Delta P_{\text{dir}}$ mengukur sensitivitas perilaku terhadap ketiadaan masukan visual, bukan bukti langsung pemahaman anatomi atau lokalisasi kausal bukti klinis.

---

## 13. Taksonomi Estimator: Aturan Langsung vs Model Risiko Terpelajar

### 13.1 Himpunan Estimator Langsung (Direct Decision Rules — Primary Track)
Diuji tanpa pelatihan model tambahan pada data uji:
1. **$M_1$ (Baseline Confidence):** $S_{M1} = p_{\text{orig\_ID}}$
2. **$M_2$ (Sensitivitas Murni):** $S_{M2} = \max(0, \Delta P_{\text{dir\_ID}})$
3. **$M_{\text{dir}}$ (Multiplikatif):** $S_{\text{dir}} = p_{\text{orig\_ID}} \times \max(0, \Delta P_{\text{dir\_ID}})$
4. **$M_{\text{norm}}$ (Estimator Primer):** $S_{\text{norm}} = p_{\text{orig\_ID}} \times \frac{1 + \Delta P_{\text{dir\_ID}}}{2}$
5. **$M_{3,\text{old}}$ (Kontrol Historis Heuristik Unsigned):** $S_{M3} = p_{\text{orig\_ID}} \times \frac{1 + |\Delta P_{\text{dir\_ID}}|}{2}$

### 13.2 Himpunan Model Risiko Terpelajar (Learned Risk Models — Secondary Ablation Track)
Model regresi logistik sederhana dengan regularisasi $L_2$ ($C=1.0$) yang difit secara eksklusif pada Development Split ($N=323$):
1. **`LR-conf` (Model Risiko Confidence-Only):**
   $$\text{Logit}(\hat{P}(\text{error}_{\text{ID}} = 1)) = \beta_0 + \beta_1 \cdot z(p_{\text{orig\_ID}})$$
   Skor selektif: $S_{\text{LR-conf}} = 1 - \hat{P}(\text{error}_{\text{ID}} = 1)$.
2. **`LR-visual` (Model Risiko Visual):**
   $$\text{Logit}(\hat{P}(\text{error}_{\text{ID}} = 1)) = \beta_0 + \beta_1 \cdot z(p_{\text{orig\_ID}}) + \beta_2 \cdot z(\Delta P_{\text{dir\_ID}})$$
   Skor selektif: $S_{\text{LR-visual}} = 1 - \hat{P}(\text{error}_{\text{ID}} = 1)$.
3. **`LR-full` (Model Risiko Gabungan Multimodal & Bilingual):**
   $$\text{Logit}(\hat{P}(\text{error}_{\text{ID}} = 1)) = \beta_0 + \beta_1 \cdot z(p_{\text{orig\_ID}}) + \beta_2 \cdot z(\Delta P_{\text{dir\_ID}}) + \beta_3 \cdot D_{\text{lang}}$$
   Skor selektif: $S_{\text{LR-full}} = 1 - \hat{P}(\text{error}_{\text{ID}} = 1)$.

*Catatan:* $z(\cdot)$ adalah standardisasi z-score yang dihitung hanya dari Dev Set. $D_{\text{lang}} \in \{0, 1\}$ dimasukkan sebagai fitur biner mentah.

---

## 14. Protokol Selective Prediction dan Metrik Evaluasi

### 14.1 Pemeringkatan dan Perhitungan Diskrit Riemann AURC
Sampel diurutkan secara menurun berdasarkan skor reliabilitas $S$:
$$S_{(1)} \ge S_{(2)} \ge \dots \ge S_{(N)}$$
- Resolusi seri deterministik: urutkan sekunder berdasarkan $p_{\text{orig\_ID}}$, lalu berdasarkan `qa_id` menaik.
- Cakupan empiris pada langkah $k$: $\phi_k = \frac{k}{N} \in (0, 1]$.
- Risiko selektif empiris pada langkah $k$: $r_k = 1 - \frac{1}{k} \sum_{j=1}^k \mathbb{I}(\hat{y}_{(j)} == y_{(j)}^*)$.
- **Discrete Riemann AURC (Geifman & El-Yaniv, 2017):**
  $$\text{AURC} = \frac{1}{N} \sum_{k=1}^N r_k$$
- Titik operasional acuan rujukan klinis: **Selective Risk pada Coverage 80% ($r_{0.80}$)**.

---

## 15. Protokol Studi Ablasi & Pencegahan Data Leakage

```
                       [ARSITEKTUR ANTI-LEAKAGE ABLASI]
                                  
     Dataset Primer Biner CLEAN (N = 1.122 QA / 292 Klaster Citra Kanonis)
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
  DEVELOPMENT SPLIT (N = 323)                           TEST SPLIT (N = 799)
   [88 Klaster Citra Kanonis]                        [204 Klaster Citra Kanonis]
           │                                                     │
           ├─ 1. Hitung StandardScaler (mu, sigma)               │
           ├─ 2. GroupKFold (K=5) berbasis klaster citra         │ [DIBEKUKAN MUTLAK]
           ├─ 3. Fit LR-conf, LR-visual, LR-full                 │ (Zero tuning, zero
           ├─ 4. Evaluasi stabilitas koefisien beta              │  threshold selection)
           │                                                     │
           └─ 5. Latih final LR pada 100% Dev Set                │
                                 │                               │
                                 └───────────────┬───────────────┘
                                                 ▼
                                     EVALUASI AKHIR SATU KALI
                                    - M1 vs M_norm (Uji Primer)
                                    - LR-conf vs M1 (Efek Model)
                                    - LR-visual vs LR-conf (Efek Visual)
                                    - LR-full vs LR-visual (Efek D_lang)
```

### Pencegahan Overfitting pada Dev Set Kecil
Dev Set memiliki $N=323$ baris pada 88 klaster citra. Prevalensi kesalahan empiris pada Dev Set dilaporkan secara transparan. Regularisasi $L_2$ ($C=1.0$) diterapkan untuk mencegah over-weighting koefisien $\beta$. Penggunaan model linear sederhana menjamin kapasitas VC-dimension yang sangat rendah, meminimalkan risiko overfitting.

---

## 16. Rancangan Analisis Statistik Inferensial

1. **Unit Sampling Klaster Citra Kanonis:** Karena pertanyaan berbagi citra radiologi yang sama, seluruh inferensi resampling menggunakan **klaster citra fisik kanonis (`canonical_image_id`)** sebagai unit independen.
2. **Paired Image-Cluster Bootstrap ($B=1.000$):** Mengestimasi 95% Confidence Interval empiris untuk AURC dan $\Delta\text{AURC}$.
3. **Exact Paired Image-Cluster Permutation Test ($B=1.000$):** Menguji signifikansi perbaikan selisih berpasangan terhadap $H_0: \mathbb{E}[\Delta\text{AURC}] = 0$ menggunakan pembalikan tanda Rademacher $\epsilon_c \in \{-1, +1\}$ pada tingkat klaster citra.
4. **Koreksi Multiplisitas Holm-Bonferroni:** Mengontrol Family-Wise Error Rate (FWER) pada taraf signifikansi $\alpha = 0.05$ di atas keluarga 4 pengujian hipotesis terencana:
   - Komparasi 1 (Primer): $M_{\text{norm}}$ vs $M_1$
   - Komparasi 2 (Ablasi Model): `LR-conf` vs $M_1$
   - Komparasi 3 (Ablasi Visual): `LR-visual` vs `LR-conf`
   - Komparasi 4 (Ablasi Bahasa): `LR-full` vs `LR-visual`

---

## 17. Validasi, Reproducibility, dan Quality Control

### 17.1 Rangkaian Pengujian Unit Preflight
Kode diverifikasi melalui automated test suite (`main/scripts/test_decisive_validation_v5.py` dan `test_risk_ablation_v10.py`):
- `test_manifest_hash_and_invariants`: Lulus verifikasi hash bit-identik dan invarian split (0% leakage).
- `test_image_verification_and_tamper_rejection`: Lulus deteksi manipulasi piksel SHA-256.
- `test_causal_logit_alignment_and_prefix_invariance`: Lulus slicing kausal $[L_p - 1 : L_p - 1 + L_t]$.
- `test_discrete_aurc_manual_calculation`: Lulus verifikasi presisi numerik Riemann hand-verified ($< 10^{-12}$).
- `test_groupkfold_cluster_isolation`: Lulus verifikasi ketiadaan tumpang tindih klaster gambar antar fold.
- `test_score_orientation_consistency`: Lulus verifikasi konsistensi korelasi arah skor retensi $S_{\text{LR}}$.

### 17.2 Protokol Pencegahan Korupsi Checkpoint
Penyimpanan checkpoint pada GPU cloud menggunakan penulisan atomik via `.tmp` dan `os.replace` dengan validasi ketat integritas numerik (ketiadaan `NaN`/`Inf`).

---

## 18. Hasil Pilot Historis dan Status Verifikasi

| Iterasi Eksperimen | Sampel Evaluasi | Model | Estimator | Hasil Kunci | Status Validitas | Catatan Auditor |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| **Kaggle Pilot 01–03** | $N=150$ | HuatuoGPT-7B | $M_1 \to M_3^{\text{old}}$ | AURC 0.2786 $\to$ 0.2173 (-22%) | `UNVERIFIED_PILOT_ARTIFACT` | Terdistorsi oleh tokenizer bug Cell 11 dan evaluasi lintas-bahasa Cell 16/20. |
| **Kaggle Pilot 01–03** | $N=150$ | HuatuoGPT-7B | Baseline ECE | ECE 0.2350, Cond. Err 36.63% | `UNVERIFIED_PILOT_ARTIFACT` | Keyakinan terinflasi secara artifisial akibat residu token bahasa Inggris pada teks Indonesia. |
| **Fine-Tuning QLoRA** | $N=150$ | Qwen-7B / 32B | LoRA ($B_2$) | Acc ID turun -12%, Flip naik +4% | `ABANDONED_EMPIRICAL` | Fine-tuning terbukti merusak akurasi bahasa Indonesia; 32B memicu CUDA OOM. Ditinggalkan resmi. |
| **rev02 Live Kaggle** | $N=92$ (Test) | Qwen-7B | $M_1 \to M_{\text{dir}}$ | AURC 0.3146 $\to$ 0.4119 (Worse) | `EMPIRICAL_FLAWED` | Terdistorsi length penalty (90/92 prediksi YA). Menunjukkan bahwa $M_{\text{dir}}$ jatuh akibat zero collapse. |
| **rev02 Live Kaggle** | $N=92$ (Test) | HuatuoGPT-7B | $M_1 \to M_{\text{dir}}$ | AURC 0.3584 $\to$ 0.5041 (Worse) | `EMPIRICAL_FLAWED` | Terdistorsi length penalty (92/92 prediksi YA). |
| **Dry-Run Simulation** | $N=150$ | Sintetis (Seed 42) | $M_1 \to M_{\text{norm}}$ | AURC 0.1142 $\to$ 0.0714 | `VERIFIED_CODE_ONLY` | Simulasi offline untuk verifikasi fungsi matematika, bukan bukti inferensi neural riil. |
| **rev05 Confirmatory** | $N=799$ (Primary) | Qwen-7B & Huatuo-7B | $M_1, M_{\text{norm}}, \text{LR-models}$ | Menunggu eksekusi GPU Dual T4 | `PENDING_CONFIRMATORY_GPU` | Pipeline terkunci 100%, siap dieksekusi di Kaggle `08_decisive_validation_rev05.ipynb`. |

---

## 19. Ancaman terhadap Validitas (Threats to Validity)

### 19.1 Validitas Internal
- **Efek OOD Citra Hitam:** Respons model pada citra hitam dapat dipengaruhi oleh anomali representasi fitur vision encoder, bukan murni hilangnya informasi diagnostik.
- **Normalisasi Ruang Kandidat Tertutup:** Evaluasi biner tertutup (`Yes/No`) tidak mencakup kemampuan generasi bebas teks medis.
- **Kapasitas Sampel Dev Set:** Ukuran Dev Set ($N=323$, 88 klaster) membatasi kompleksitas model regresi logistik; risiko estimasi varians parameter ditangani via GroupKFold dan regularisasi $L_2$.

### 19.2 Validitas Eksternal
- **Generalisasi Modalitas:** Dataset IndoRad-VQA terbatas pada citra radiologi (X-ray, CT, MRI); temuan tidak dapat digeneralisasi langsung ke histopatologi atau USG.
- **Skala Model:** Keterbatasan komputasi membatasi evaluasi pada model kelas 7B.

### 19.3 Validitas Konstruk
- Perubahan nilai AURC mencerminkan efisiensi pemeringkatan kesalahan retrospektif, bukan ukuran keselamatan klinis prospektif.
- $D_{\text{lang}} = 1$ mencerminkan ketidaksepakatan perilaku lintas-bahasa, bukan bukti definitif bahwa salah satu bahasa salah secara patologis.

---

## 20. Batasan dan Pertimbangan Keselamatan Medis

> [!CAUTION]
> **PERINGATAN KESELAMATAN MEDIS & BATASAN EPISTEMIK:**
> 1. Sistem selective prediction ini **BUKAN alat diagnosis medis klinis** dan tidak boleh digunakan untuk pengambilan keputusan medis pada pasien tanpa pengawasan dokter spesialis radiologi berlisensi.
> 2. Penurunan nilai AURC semata-mata membuktikan kemampuan statistik pengurutan risiko kesalahan pada dataset uji retrospektif.
> 3. Metrik ini tidak menjamin eliminasi kesalahan diagnosis fatal (*silent diagnostic failure*).
> 4. Model kecerdasan buatan dalam penelitian ini berstatus purwarupa penelitian akademis (*academic research prototype*).

---

## 21. Implementation Checklist

### Prioritas 1: Kesiapan Pra-Inferensi (Tuntas Lolos)
- [x] Audit integritas dataset IndoRad-VQA dan pembersihan translasi biner (`manifest_rev05.csv` SHA-256: `fb55fd75...`).
- [x] Implementasi causal logit slicing dan prefix invariance assertion.
- [x] Implementasi length-normalized candidate sequence scoring dengan token `<|im_end|>`.
- [x] Formulasi sinyal $D_{\text{lang}}$ dan penegasan orientasi skor retensi $S_{\text{LR}} = 1 - \hat{P}(\text{error})$.
- [x] Pembangkitan notebook eksekusi Kaggle Dual-T4 (`08_decisive_validation_rev05.ipynb`) dengan AST parse 100% OK.
- [x] Eksekusi unit test suite `test_decisive_validation_v5.py` (9/9 Unit Test PASS).

### Prioritas 2: Eksekusi GPU dan Ekstraksi Artefak (Menunggu Eksekusi Cloud)
- [ ] Upload notebook `08_decisive_validation_rev05.ipynb` ke Kaggle dengan akselerator Dual Tesla T4.
- [ ] Jalankan "Run All" untuk mengevaluasi Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B pada Active Union ($N=1.133$).
- [ ] Unduh arsip hasil resmi `decisive_validation_rev05_results_final.zip`.
- [ ] Ekstrak checkpoint, tabel komparasi CSV, dan kurva risk-coverage ke `main/results/decisive_validation/rev05/`.

### Prioritas 3: Eksekusi Ablation Study Regresi Logistik (CPU Post-Processing)
- [ ] Eksekusi GroupKFold CV pada Dev Set ($N=323$) untuk fitting `LR-conf`, `LR-visual`, dan `LR-full`.
- [ ] Evaluasi performa out-of-fold dan catat nilai bobot regresi $\beta_1, \beta_2, \beta_3$.
- [ ] Evaluasi model pada Test Set beku ($N=799$) dan hitung Discrete Riemann AURC untuk seluruh 5 estimator.
- [ ] Hitung exact cluster permutation test $p$-value dan aplikasikan penyesuaian Holm-Bonferroni pada 4 komparasi terencana.

### Prioritas 4: Finalisasi Naskah Skripsi
- [ ] Sinkronkan seluruh angka bab IV naskah skripsi dengan artefak CSV rev05/v10 resmi.
- [ ] Dokumentasikan temuan secara objektif (laporkan secara jujur baik temuan positif maupun temuan negatif).
- [ ] Perbarui visualisasi kurva Risk-Coverage 300 DPI dan tabel evaluasi.

---

## 22. Experiment Decision Gates

| Decision Gate | Kriteria Keberhasilan | Bukti Verifikasi yang Dibutuhkan | Tindakan Jika Lolos | Tindakan Jika Gagal |
| :--- | :--- | :--- | :--- | :--- |
| **Gate A: Validitas Implementasi** | Tokenizer konsisten, logit slicing tepat kausal, prompt simetris, manifest hash bit-identik. | 9/9 Unit test pada `test_decisive_validation_v5.py` berstatus PASS. | Lanjut ke eksekusi inferensi model GPU. | **STOP.** Perbaiki indexing logit atau tokenisasi sebelum inferensi. Dilarang lanjut. |
| **Gate B: Ketiadaan Anomali Scoring** | Distribusi prediksi bahasa Indonesia tidak terdistorsi 100% ke satu kelas; candidate score berhingga tanpa NaN. | Evaluasi Dev Set ($N=327$) menunjukkan rasio prediksi `YA` dan `TIDAK` berada dalam rentang wajar ($20\% - 80\%$). | Lanjut ke evaluasi Test Set ($N=806$). | Periksa normalisasi panjang surprisal atau prefix prompt. |
| **Gate C: Kepatuhan Protokol Citra** | Dimensi, rasio, kanal, dan preprocessing P0 dan P1 identik secara deterministik. | Verifikasi fungsi `create_black_blank_image` pada unit test. | Lanjut ke komputasi $\Delta P_{\text{dir}}$. | Perbaiki fungsi pembuatan citra hitam. |
| **Gate D: Kestabilan Checkpoint** | 100% dari 1.133 sampel tereksekusi tanpa crash; penulisan atomik utuh. | Checkpoint CSV memuat seluruh kolom skor numerik berhingga untuk kedua model. | Lanjut ke perhitungan metrik selective prediction. | Resume inferensi menggunakan fungsi atomic checkpoint. |
| **Gate E: Bukti Nilai Tambah Selektif** | Estimator primer $M_{\text{norm}}$ menghasilkan $\Delta\text{AURC} < 0$ dengan $p_{\text{adj}} < 0.05$ pada Primary Test Split. | Tabel komparasi estimator dan hasil uji permutasi klaster berpasangan. | Klaim bahwa sensitivitas visual memberikan nilai tambah terdukung secara empiris. | **Tetap laporkan hasil sebagai temuan negatif yang informatif.** Dilarang memanipulasi formula atau data. |
| **Gate F: Nilai Tambah Inkremental $D_{\text{lang}}$** | `LR-full` menghasilkan penurunan AURC yang signifikan dibanding `LR-visual` ($p_{\text{adj}} < 0.05$). | Perbandingan AURC dan Likelihood Ratio Test pada Test Split beku. | Adopsi judul alternatif dan klaim peran ketidaksepakatan bilingual didukung bukti. | Pertahankan judul kerja utama; laporkan bahwa $D_{\text{lang}}$ tidak memberikan sinyal inkremental di atas visual sensitivity. |

---

## 23. Rencana Kerja dan Prioritas

1. **Tahap 1 (Hari 1):** Eksekusi mandiri notebook `08_decisive_validation_rev05.ipynb` pada Kaggle Dual Tesla T4 (estimasi runtime 40–50 menit).
2. **Tahap 2 (Hari 1–2):** Verifikasi integritas checkpoint unduhan, ekstraksi kolom $D_{\text{lang}}$ dan $\text{error}_{\text{ID}}$.
3. **Tahap 3 (Hari 2):** Eksekusi skrip post-processing regresi logistik ablasi pada Dev Set dan evaluasi pada Test Set.
4. **Tahap 4 (Hari 3):** Rekapitulasi nilai $p$ statistik berpasangan (permutasi klaster dan Holm-Bonferroni) untuk Uji Primer ($M_{\text{norm}}$ vs $M_1$) dan Uji Sekunder (`LR-full` vs `LR-visual`).
5. **Tahap 5 (Hari 4):** Finalisasi penulisan Bab IV dan Bab V Skripsi sesuai standar kejujuran saintifik.

---

## 24. Daftar Keputusan yang Masih Terbuka (Open Issues)

1. **Isu Terbuka 1: Superioritas Empiris Estimator Primer $M_{\text{norm}}$ vs Baseline $M_1$**
   - *Status:* `OPEN / AWAITING_GPU_RUN`
   - *Bukti yang Dibutuhkan:* Nilai $\Delta\text{AURC}$ dan nilai $p$ permutasi klaster dari inferensi penuh rev05 pada $N=799$ test set.
   - *Dampak:* Menentukan apakah intervensi citra hitam langsung memberikan sinyal seleksi risiko yang superior.
2. **Isu Terbuka 2: Nilai Tambah Inkremental Sinyal Ketidaksepakatan Bilingual ($D_{\text{lang}}$)**
   - *Status:* `OPEN / AWAITING_GPU_RUN`
   - *Bukti yang Dibutuhkan:* Nilai $\Delta\text{AURC}(\text{LR-full} - \text{LR-visual})$ pada Test Set beku.
   - *Dampak:* Menentukan apakah judul alternatif diadopsi dan apakah inkonsistensi bahasa terbukti menjadi prediktor kesalahan independen.
3. **Isu Terbuka 3: Perilaku Disparitas Model Medis HuatuoGPT pada Scoring rev05**
   - *Status:* `OPEN / AWAITING_GPU_RUN`
   - *Bukti yang Dibutuhkan:* Perbandingan ECE dan flip rate HuatuoGPT vs Qwen pada skema scoring length-normalized.
   - *Dampak:* Mengonfirmasi apakah overconfidence ekstrem HuatuoGPT merupakan sifat inheren model medis atau artifak tokenizer lama.

---

## 25. Target Publikasi dan Standar Akademik

- **Target Diseminasi Skripsi:** Ujian Sidang Sarjana (S1) Ilmu Komputer / Teknik Informatika dengan predikat keunggulan metodologis (*rigorous methodology*).
- **Target Publikasi Ilmiah:** Naskah disiapkan untuk disubmit ke jurnal nasional bereputasi terakreditasi **SINTA 1 atau SINTA 2**, atau konferensi internasional bereputasi bidang medical AI / computer vision (IEEE/ACM/Springer).
- **Pernyataan Kejujuran:** Dokumen ini menegaskan bahwa peringkat akreditasi jurnal (SINTA 1/2) merupakan target kualitas penulisan dan ketelitian metodologis, bukan jaminan penerimaan sebelum proses peer-review independen dilalui.
