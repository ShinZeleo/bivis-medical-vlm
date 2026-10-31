# 🎓 Supervisor Defense Question Bank (Bank Pertanyaan Kritis Bimbingan & Sidang)
**Thesis Project:** Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift  
**Target:** Persiapan Ujian Seminar Proposal, Bimbingan Dosen Pembimbing, dan Sidang Skripsi Sarjana (S1)  
**Standard:** 35 Pertanyaan Kritis Tingkat Senior Auditor Lengkap dengan Bukti Empiris, Lokasi Berkas, dan Landasan Teoretis

---

## 📌 DAFTAR KATEGORI
- [Kategori 1: Motivasi Penelitian & Konteks Klinis (Q1–Q3)](#kategori-1-motivasi-penelitian--konteks-klinis)
- [Kategori 2: Research Gap & Batas Kebaruan / Novelty (Q4–Q7)](#kategori-2-research-gap--batas-kebaruan--novelty)
- [Kategori 3: Landasan Literatur & Preseden Ilmiah (Q8–Q10)](#kategori-3-landasan-literatur--preseden-ilmiah)
- [Kategori 4: Tata Kelola Dataset & Pencegahan Kebocoran (Q11–Q14)](#kategori-4-tata-kelola-dataset--pencegahan-kebocoran)
- [Kategori 5: Pemilihan Model & Batasan Komputasi (Q15–Q17)](#kategori-5-pemilihan-model--batasan-komputasi)
- [Kategori 6: Formulasi Matematis & 5 Estimator Selektif (Q18–Q22)](#kategori-6-formulasi-matematis--5-estimator-selektif)
- [Kategori 7: Intervensi Visual & Karakteristik OOD Kanvas Hitam (Q23–Q26)](#kategori-7-intervensi-visual--karakteristik-ood-kanvas-hitam)
- [Kategori 8: Sinyal Ketidaksepakatan Lintas-Bahasa $D_{\text{lang}}$ (Q27–Q29)](#kategori-8-sinyal-ketidaksepakatan-lintas-bahasa-d_lang)
- [Kategori 9: Model Risiko Terpelajar & Desain Ablasi (Q30–Q32)](#kategori-9-model-risiko-terpelajar--desain-ablasi)
- [Kategori 10: Analisis Statistik Inferensial & Multiplisitas (Q33–Q34)](#kategori-10-analisis-statistik-inferensial--multiplisitas)
- [Kategori 11: Batasan Epistemik & Keselamatan Klinis Pasien (Q35)](#kategori-11-batasan-epistemik--keselamatan-klinis-pasien)

---

### Kategori 1: Motivasi Penelitian & Konteks Klinis

#### Q1: "Mengapa kita perlu meneliti Medical VLM dalam Bahasa Indonesia jika dokter radiologi di Indonesia juga terbiasa membaca literatur bahasa Inggris?"
- **Jawaban Berbasis Bukti:** Meskipun dokter spesialis radiologi memahami istilah medis bahasa Inggris, sistem pendukung keputusan klinis terintegrasi di rumah sakit daerah (RSUD/Puskesmas), resume medis elektronik (EHR), dan interaksi konsultasi pasien di Indonesia beroperasi dalam Bahasa Indonesia. Yudhistira et al. (2026, `arXiv:2606.03693`) membuktikan bahwa model VLM mengalami penurunan akurasi 8%–25% dan *flip rate* hingga 50.81% (HuatuoGPT) saat instruksi dialihkan ke Bahasa Indonesia. Fenomena ini membahayakan keselamatan pasien jika sistem mengeluarkan diagnosis kontradiktif semata-mata karena pergeseran bahasa.
- **Lokasi Bukti:** `paper/2606.03693v1.pdf`, `docs/RESEARCH_MASTER.md` Bagian 3.1, `main/results/decisive_validation/rev05/rq1_bilingual_metrics.csv`.

#### Q2: "Mengapa Anda memilih pendekatan Selective Prediction (Learning to Abstain) alih-alih melatih ulang / fine-tuning model agar lebih pintar?"
- **Jawaban Berbasis Bukti:** Eksperimen historis kami membuktikan bahwa melatih ulang model medis (QLoRA dengan semantic consistency loss) justru menurunkan akurasi bahasa Indonesia sebesar $-12.00\%$ ($70\% \to 58\%$) dan model 32B memicu *CUDA OutOfMemoryError*. Sadanandan & Behzadan (CHIL 2026, `paper/sadanandan26a.pdf`) juga membuktikan bahwa pelatihan konsistensi murni tanpa supervisi label yang sangat presisi akan memicu *mode collapse* (model memprediksi "YES" untuk semua kasus). Oleh karena itu, pendekatan *selective prediction* waktu-inferensi (*zero-retraining*) jauh lebih aman, realistis secara komputasi, dan klinis: model otomatis merujuk (*refer*) kasus berisiko tinggi kepada dokter manusia.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 2 (Poin 1), `paper/sadanandan26a.pdf` Bagian 5.2.

#### Q3: "Apa bahaya klinis terbesar jika sistem Medical VLM beroperasi secara unselective?"
- **Jawaban Berbasis Bukti:** Bahaya terbesar adalah *silent diagnostic failure* di kuadran berbahaya (*Dangerous Quadrant*, Sadanandan & Behzadan, CVPR 2026). Model memberikan diagnosis yang salah dengan tingkat keyakinan (*confidence*) verbal yang sangat tinggi karena bersandar pada pola frekuensi teks pelatihan, sementara ia sebenarnya mengabaikan citra rontgen pasien. Sistem unselective akan membiarkan kesalahan fatal ini lolos ke rekam medis pasien tanpa peringatan.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 3.2, `docs/LITERATURE_AUDIT.md` Rumpun 4.

---

### Kategori 2: Research Gap & Batas Kebaruan / Novelty

#### Q4: "Apakah penelitian Anda ini merupakan penelitian Selective VQA pertama di dunia?"
- **Jawaban Berbasis Bukti:** Bukan. Selective classification telah dirumuskan secara teoritis oleh Geifman & El-Yaniv (NeurIPS 2017), diperluas ke NLP domain shift oleh Kamath et al. (ACL 2020), dan ke VQA umum oleh Dancette et al. (CVPR 2023) serta Khan & Fu (CVPR 2024). Kebaruan spesifik penelitian kami terletak pada **integrasi sinyal sensitivitas visual kontrafaktual ($\Delta P_{\text{dir}}$) dan ketidaksepakatan bilingual ($D_{\text{lang}}$) tanpa pelatihan ulang model untuk selective risk estimation pada Medical VLM di bawah pergeseran bahasa Inggris–Indonesia**.
- **Lokasi Bukti:** `docs/LITERATURE_AUDIT.md` Bagian 3, `docs/RESEARCH_MASTER.md` Bagian 5.2.

#### Q5: "Apakah penggunaan kanvas hitam (black image) adalah kebaruan orisinal Anda?"
- **Jawaban Berbasis Bukti:** Bukan. Penghilangan citra atau *blind image test* telah digunakan sebagai uji diagnostik oleh Agrawal et al. (2018), Sadanandan & Behzadan (2026), Zafar et al. (2026), dan Khanmohammadi et al. (2026). Kebaruan penelitian kami adalah **mengoperasionalkan penurunan probabilitas terarah pada kanvas hitam ($\Delta P_{\text{dir}}$) sebagai fitur waktu-inferensi (*runtime decision rule*) untuk meminimalkan metrik AURC pada kurva Risk–Coverage**.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 5.2, `docs/audit/LITERATURE_VERIFICATION_REPORT.md` Bagian 4.

#### Q6: "Bagaimana rumusan Research Gap Primer Anda yang dapat dipertahankan secara metodologis?"
- **Jawaban Berbasis Bukti:** "Belum diketahui secara empiris apakah sensitivitas skor kandidat terhadap penghilangan informasi visual berarah ($\Delta P_{\text{dir}}$) mampu meningkatkan efisiensi pemeringkatan risiko selektif (menurunkan Discrete Riemann AURC) secara signifikan dibandingkan skor preferensi kandidat internal ($M_1$) pada Medical VLM yang menjawab pertanyaan radiologi dalam Bahasa Indonesia."
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 5.1.

#### Q7: "Bagaimana rumusan Research Gap Sekunder Anda mengenai ketidaksepakatan lintas-bahasa?"
- **Jawaban Berbasis Bukti:** "Belum diketahui apakah ketidaksepakatan prediksi pada pasangan pertanyaan radiologi bilingual Inggris–Indonesia ($D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$) memberikan informasi tambahan yang signifikan untuk mengurutkan risiko kesalahan pada jawaban bahasa Indonesia, setelah skor preferensi kandidat ($p_{\text{orig}}$) dan sensitivitas visual ($\Delta P_{\text{dir}}$) diperhitungkan."
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 5.1.

---

### Kategori 3: Landasan Literatur & Preseden Ilmiah

#### Q8: "Mengapa metode kalibrasi standar seperti Platt Scaling atau Temperature Scaling tidak Anda gunakan sebagai solusi utama?"
- **Jawaban Berbasis Bukti:** Byun et al. (2026, Microsoft Health, OpenReview) membuktikan secara matematis bahwa metode kalibrasi probabilitas pasca-proses standar (seperti Platt scaling atau Temperature scaling) merupakan transformasi monotonik ketat. Transformasi monotonik **sama sekali tidak mengubah urutan pemeringkatan sampel (*ranking order*)**, sehingga nilai AUROC dan metrik Selective Prediction AURC tidak akan berubah. Untuk memperbaiki pemeringkatan risiko selektif, diperlukan sinyal tambahan non-monotonik dari bukti multimodal (seperti $\Delta P_{\text{dir}}$ dan $D_{\text{lang}}$).
- **Lokasi Bukti:** `docs/LITERATURE_AUDIT.md` Tabel 2 Baris 4, `docs/RESEARCH_MASTER.md` Bagian 4.3.

#### Q9: "Paper Sadanandan & Behzadan (2026) di repositori Anda (`paper/sadanandan26a.pdf`) meneliti apa dan apa relevansinya?"
- **Jawaban Berbasis Bukti:** Paper tersebut dipublikasikan di PMLR (CHIL 2026) berjudul *"Mechanistically Guided LoRA Improves Paraphrase Consistency in Medical Vision-Language Models"*. Mereka meneliti mengapa model MedGemma tidak konsisten menjawab parafrase klinis bahasa Inggris dan membuktikan bahwa pelatihan konsistensi murni tanpa supervisi akurasi akan memicu *mode collapse*. Naskah ini menjadi landasan mengapa penelitian kami memfokuskan diri pada seleksi waktu-inferensi daripada memaksakan pelatihan ulang model.
- **Lokasi Bukti:** `paper/sadanandan26a.pdf`, `docs/audit/LITERATURE_VERIFICATION_REPORT.md` Bagian 3.

#### Q10: "Siapa yang mendasari penggunaan regresi logistik sederhana pada development set untuk selective QA?"
- **Jawaban Berbasis Bukti:** Kamath, Jia, & Liang (ACL 2020) dalam paper *"Selective Question Answering under Domain Shift"*. Mereka membuktikan bahwa di bawah pergeseran domain, model regresi logistik sederhana yang dilatih pada development set efektif memadukan beberapa sinyal ketidakpastian untuk memprediksi probabilitas kebenaran dan menekan selective risk.
- **Lokasi Bukti:** `docs/LITERATURE_AUDIT.md` Tabel 2 Baris 10, `docs/RESEARCH_MASTER.md` Bagian 4.1.

---

### Kategori 4: Tata Kelola Dataset & Pencegahan Kebocoran

#### Q11: "Berapa jumlah sampel data asli IndoRad-VQA dan berapa yang Anda gunakan dalam evaluasi primer?"
- **Jawaban Berbasis Bukti:** Dataset mentah IndoRad-VQA memiliki 2.244 baris pertanyaan-jawaban klinis dari 314 citra radiologi unik. Dari 2.244 baris tersebut, 1.053 baris pertanyaan terbuka dieksklusi (karena tidak memenuhi asumsi probabilitas biner tertutup). Dari 1.191 pertanyaan yes/no, 69 baris dieksklusi karena cacat translasi kritis (pembalikan negasi, salah organ). Hasilnya adalah **Primary Clean Cohort sebesar tepat 1.122 pasangan QA (799 Test Set / 323 Dev Set)**.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 9.2, `main/results/decisive_validation/rev05/cohort_audit.json`.

#### Q12: "Bagaimana Anda membuktikan bahwa tidak ada data leakage antara Development Set dan Test Set?"
- **Jawaban Berbasis Bukti:** Pemisahan dataset kami **TIDAK dilakukan secara acak per-baris QA**, melainkan dikelompokkan secara ketat pada tingkat **klaster citra fisik kanonis (`canonical_image_id`)** melalui hash SHA-256 piksel mentah RGB. Primary Test Set memuat 204 klaster citra unik, dan Dev Set memuat 88 klaster citra unik. Irisan klaster antara Test dan Dev adalah **PERSIS 0 KLASTER (0.00% leakage)**. Kode pengujian `test_cohort_invariants_and_zero_cluster_leakage` memverifikasi hal ini dengan penegasan biner yang lulus 100%.
- **Lokasi Bukti:** `main/scripts/test_decisive_validation_v5.py` line 90–143, `main/results/decisive_validation/rev05/cohort_audit.json`.

#### Q13: "Apakah pasangan pertanyaan bahasa Inggris dan bahasa Indonesia dijamin menggunakan citra yang sama?"
- **Jawaban Berbasis Bukti:** Ya, 100% terjamin. Setiap baris manifest merepresentasikan satu unit pasangan bilingual yang memiliki `qid` tunggal, mengikat `question_en`, `question_id`, `reference_answer`, dan `canonical_image_id` ke berkas citra fisik yang sama.
- **Lokasi Bukti:** `main/results/manifest_rev05.csv`, `docs/audit/DATASET_INTEGRITY_REPORT.md` Bagian 4.

#### Q14: "Apakah distribusi kelas YES dan NO pada Test Set seimbang?"
- **Jawaban Berbasis Bukti:** Ya, sangat seimbang. Pada Primary Test Set ($N=799$), terdapat 407 sampel YES (50.94%) dan 392 sampel NO (49.06%), rasio $1.038 : 1$. Pada Dev Set ($N=323$), terdapat 147 YES (45.51%) dan 176 NO (54.49%). Keseimbangan alami ini menjamin bahwa metrik akurasi dan selective risk tidak terdistorsi oleh prevalensi kelas mayoritas.
- **Lokasi Bukti:** `main/results/decisive_validation/rev05/cohort_audit.json`.

---

### Kategori 5: Pemilihan Model & Batasan Komputasi

#### Q15: "Mengapa Anda memilih Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B?"
- **Jawaban Berbasis Bukti:** Kedua model berbagi arsitektur dasar yang sama (`Qwen2_5_VLForConditionalGeneration`), patch visual encoder yang sama, dan tokenizer BPE yang sama (~152.064 token). Qwen2.5-VL mewakili model fondasi multimodal umum tercanggih (SOTA open-source), sedangkan HuatuoGPT-Vision-7B mewakili model spesialis medis yang dilatih di atas backbone tersebut menggunakan 1.3 juta data biomedis PubMedVision. Perbandingan ini menguji apakah spesialisasi medis memperkuat atau justru melemahkan ketahanan model terhadap pergeseran bahasa.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 10.1–10.2.

#### Q16: "Apakah perbedaan hasil antara Qwen dan HuatuoGPT membuktikan hubungan kausal murni dari fine-tuning medis?"
- **Jawaban Berbasis Bukti:** Tidak boleh diklaim sebagai hubungan kausal murni. Dalam dokumen resmi kami (Research Master v10.0 Bagian 10.2), kami secara tegas mencantumkan batasan epistemik: perbedaan hasil antara kedua model diperlakukan sebagai perbandingan dua sistem model yang dipilih, bukan isolasi kausal satu faktor, karena dataset alignment, resep instruksi, dan optimasi HuatuoGPT berbeda secara komprehensif.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 10.2.

#### Q17: "Mengapa tidak menguji model 32B atau 72B?"
- **Jawaban Berbasis Bukti:** Keterbatasan perangkat keras. Model multimodal 32B memicu *CUDA OutOfMemoryError* pada lingkungan GPU dual-T4 (15 GB VRAM limit) saat memproses resolusi visual radiologi dinamis. Model 7B dengan kuantisasi 4-bit NF4 mengonsumsi ~8.5 GB VRAM per GPU dan terbukti stabil mengeksekusi 1.133 sampel tanpa kegagalan memori.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 10.2, `main/reports/DECISIVE_VALIDATION_REV05_AUDIT.md`.

---

### Kategori 6: Formulasi Matematis & 5 Estimator Selektif

#### Q18: "Apa perbedaan matematis antara Exact Sequence Log-Likelihood dan Length-Normalized Surprisal?"
- **Jawaban Berbasis Bukti:** Exact sequence log-likelihood menghitung probabilitas gabungan sekuens $S(y \mid X) = \sum_{t=1}^L \log P(y_t \mid y_{<t}, X)$, yang merupakan probabilitas sejati dari model autoregresif. Namun, penjumlahan murni memiliki *length bias* alami (token lebih panjang mendapat akumulasi nilai negatif lebih besar). Length-normalized surprisal menghitung rata-rata surprisal per-token $\bar{s} = \frac{1}{L} \sum \log P$. Pada `rev05`, kami mengimplementasikan arsitektur dual-track: sequence log-likelihood sebagai evaluasi primer, dan length-normalized sebagai analisis sensitivitas.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 11.2, `main/scripts/test_decisive_validation_v5.py` line 165–198.

#### Q19: "Apakah skor $P_{\text{cand}}$ yang Anda peroleh adalah probabilitas terkalibrasi (*calibrated posterior probability*)?"
- **Jawaban Berbasis Bukti:** **BUKAN.** Nilai tersebut adalah **Candidate-Normalized Sequence Score** pada ruang biner tertutup `{"YES", "NO"}` yang dinormalisasi via softmax. Nilai ini hanya mencerminkan preferensi relatif model terhadap dua pilihan tersebut, dan tidak boleh diklaim sebagai probabilitas kebenaran medis global.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 8.1 & 11.2, `main/results/decisive_validation/rev05/scoring_validation_report.json`.

#### Q20: "Mengapa Anda mengevaluasi probabilitas pada label yang sama saat menghitung $\Delta P_{\text{dir}}$?"
- **Jawaban Berbasis Bukti:** $\Delta P_{\text{dir}} = p_{\text{orig}} - p_{\text{blank}}$ di mana $p_{\text{blank}} = P_{\text{cand}}(\hat{y}_{\text{ID}} \mid P_1)$. Jika kita mengevaluasi $\max_y P(y \mid P_1)$ (arg-max baru pada citra blank), kita akan membandingkan probabilitas dari dua label yang berbeda ketika prediksi model berbalik. Dengan mengevaluasi probabilitas pada label terpilih yang sama $\hat{y}_{\text{ID}}$, $\Delta P_{\text{dir}}$ secara murni mengukur seberapa besar dukungan terhadap hipotesis klinis tersebut runtuh saat informasi visual dihilangkan.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 8.1, `main/scripts/decisive_validation_experiment_v5.py` line 687–690.

#### Q21: "Apa kelemahan matematis utama dari estimator $M_{\text{dir}} = p_{\text{orig}} \times \max(0, \Delta P_{\text{dir}})$?"
- **Jawaban Berbasis Bukti:** Kelemahan utamanya adalah **Zero Collapse**. Ketika model mengalami ketergantungan visual negatif ($\Delta P_{\text{dir}} \le 0$), $\max(0, \Delta)$ bernilai persis $0.0$. Akibatnya, seluruh sampel ungrounded mendapat skor identik $0.0$ tanpa memedulikan nilai $p_{\text{orig}}$ (misal: $p_{\text{orig}}=0.99$ dan $p_{\text{orig}}=0.51$ sama-sama mendapat skor $0$). Hal ini memicu ikatan nilai masif (*massive ties*) dan merusak kurva selective prediction (AURC melonjak buruk menjadi $0.3158$ pada Qwen dan $0.3394$ pada Huatuo).
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 13.1 & 18 (rev02 Live Kaggle), `main/results/decisive_validation/rev05/rq3_selective_prediction_primary_test.csv`.

#### Q22: "Mengapa estimator primer $M_{\text{norm}} = p_{\text{orig}} \times \frac{1 + \Delta P_{\text{dir}}}{2}$ dirancang demikian?"
- **Jawaban Berbasis Bukti:** $M_{\text{norm}}$ memetakan nilai $\Delta P_{\text{dir}} \in [-1, 1]$ secara linier dan halus ke interval $[0, 1]$ melalui fungsi $\frac{1 + \Delta}{2}$. Ketika model memiliki sensitivitas visual sempurna ($\Delta = 1$), skornya adalah $p_{\text{orig}} \times 1.0 = p_{\text{orig}}$. Ketika model netral ($\Delta = 0$), skornya dihukum proporsional menjadi $p_{\text{orig}} \times 0.5$. Ketika model memiliki prior bias negatif ekstrem ($\Delta = -1$), skornya terdegradasi ke $0$. Desain ini mencegah zero collapse dan memberikan penalti halus.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 13.1, `docs/audit/FORMULA_VERIFICATION_REPORT.md` Bagian 4.

---

### Kategori 7: Intervensi Visual & Karakteristik OOD Kanvas Hitam

#### Q23: "Mengapa Anda menggunakan kanvas hitam polos ($P_1$) dan bukan Gaussian blur?"
- **Jawaban Berbasis Bukti:** Pengaburan citra (*Gaussian blur*) ditiadakan secara permanen atas arahan dan persetujuan dosen pembimbing. Gaussian blur memperkenalkan hyperparameter arbitrer (radius kernel, sigma $\sigma$) yang sulit distandarisasi di seluruh modalitas radiologi berbeda (CT vs X-ray). Kanvas hitam murni `RGB(0, 0, 0)` berdimensi dan kanal identik adalah intervensi deterministik biner tanpa hyperparameter (*zero-parameter deletion*) yang mengeliminasi seluruh bukti foton radiologis.
- **Lokasi Bukti:** `docs/REVISION_NOTES.md` Bagian 1, `docs/RESEARCH_MASTER.md` Bagian 2 (Poin 3) & 12.

#### Q24: "Bukankah citra hitam polos adalah kondisi Out-of-Distribution (OOD) bagi vision encoder model medis?"
- **Jawaban Berbasis Bukti:** **BENAR.** Kami secara eksplisit mengakui dalam naskah skripsi bahwa kanvas hitam adalah masukan OOD. Oleh karena itu, penurunan probabilitas $\Delta P_{\text{dir}}$ dilaporkan secara jujur sebagai **sensitivitas perilaku terhadap ketiadaan masukan visual**, dan **BUKAN** bukti langsung pemahaman anatomi klinis kausal atau lokalisasi patologi lesi.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 2 (Poin 4) & Bagian 12.

#### Q25: "Bagaimana hasil empiris $\Delta P_{\text{dir}}$ pada kedua model dan apa artinya?"
- **Jawaban Berbasis Bukti:** Pada Primary Test Set ($N=799$), Qwen2.5-VL-7B memiliki Mean $\Delta P_{\text{dir}} = +0.1405$ dengan 60.08% sampel mengalami penurunan keyakinan (sensitif visual). Sebaliknya, HuatuoGPT-Vision-7B memiliki Mean $\Delta P_{\text{dir}} = -0.0411$ di mana pada **79.97% sampel**, keyakinannya tidak turun atau justru meningkat saat citra dihitamkan. Ini membuktikan bahwa model spesialis medis HuatuoGPT menderita bias prior teks biomedis yang sangat parah (*severe visual ungroundedness*).
- **Lokasi Bukti:** `main/results/decisive_validation/rev05/rq2_visual_intervention_metrics.csv`.

#### Q26: "Apakah estimator primer $M_{\text{norm}}$ mengungguli baseline $M_1$ secara signifikan?"
- **Jawaban Berbasis Bukti:** Berdasarkan hasil inferensi konfirmatori resmi pada Test Set ($N=799$), kami melaporkan **temuan negatif yang jujur**:
  - Pada Qwen2.5-VL-7B, $M_{\text{norm}}$ menghasilkan AURC $0.2753$ vs $M_1$ $0.2422$ (lebih buruk).
  - Pada HuatuoGPT-Vision-7B, $M_{\text{norm}}$ menurunkan AURC dari $0.2884 \to 0.2552$ ($\Delta = -0.0332$), namun setelah uji permutasi klaster berpasangan, nilai p-value penyesuaian Holm-Bonferroni adalah $p_{\text{adj}} = 0.7812 > 0.05$ (tidak signifikan).
  - Ini membuktikan bahwa aturan heuristik langsung tidak cukup konsisten, membenarkan kebutuhan pemodelan risiko terpelajar (`LR-full`).
- **Lokasi Bukti:** `main/results/decisive_validation/rev05/rq3_selective_prediction_primary_test.csv`.

---

### Kategori 8: Sinyal Ketidaksepakatan Lintas-Bahasa $D_{\text{lang}}$

#### Q27: "Bagaimana sinyal $D_{\text{lang}}$ dioperasionalkan secara matematis dan komputasi?"
- **Jawaban Berbasis Bukti:** $D_{\text{lang}} = \mathbb{I}(\hat{y}_{\text{EN}} \ne \hat{y}_{\text{ID}})$. Bernilai $1$ jika kesimpulan model pada prompt bahasa Inggris berbeda dengan bahasa Indonesia, dan bernilai $0$ jika sepakat. Sinyal ini diperoleh langsung dari perbandingan output diskrit pada pasangan pertanyaan bilingual yang sudah dievaluasi, sehingga **memerlukan zero additional forward pass pada GPU**.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 8.1, `main/scripts/resilient_validation_engine.py` line 640.

#### Q28: "Apakah kesepakatan dua bahasa ($D_{\text{lang}} = 0$) menjamin bahwa jawaban model benar?"
- **Jawaban Berbasis Bukti:** Tidak. Model bisa saja konsisten menjawab salah pada kedua bahasa (*jointly incorrect*). Pada Qwen, terdapat 45.06% kasus yang tidak benar bersamaan. $D_{\text{lang}}$ bukan sertifikat kebenaran mutlak, melainkan **indikator ketidakpastian representasi multimodal**: jika model tidak konsisten pada bahasa yang berbeda, risiko kesalahannya melonjak tajam.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 3.3 & 19.3.

#### Q29: "Apakah penambahan $D_{\text{lang}}$ terbukti secara statistik memberikan nilai tambah di atas visual sensitivity?"
- **Jawaban Berbasis Bukti:** **YA, SANGAT SIGNIFIKAN.** Uji permutasi klaster berpasangan ($B=2.000$) antara `LR-full` ($[p_{\text{orig}}, \Delta P_{\text{dir}}, D_{\text{lang}}]$) vs `LR-visual` ($[p_{\text{orig}}, \Delta P_{\text{dir}}]$) membuktikan:
  - Pada Qwen2.5-VL-7B: $\Delta\text{AURC} = -0.0381$, $p = \mathbf{0.0120} < 0.05$.
  - Pada HuatuoGPT-Vision-7B: $\Delta\text{AURC} = -0.0335$, $p = \mathbf{0.0005} < 0.001$.
  - Koefisien regresi $D_{\text{lang}}$ bernilai positif besar ($\beta_3 = +1.096$ pada Qwen dan $+0.994$ pada Huatuo), membuktikan $D_{\text{lang}}$ secara signifikan memprediksi error bahasa Indonesia.
- **Lokasi Bukti:** `main/results/decisive_validation/rev05/risk_ablation_dev_test.csv`, `docs/audit/EXPERIMENT_OUTPUT_AUDIT.md` Bagian 3.4.

---

### Kategori 9: Model Risiko Terpelajar & Desain Ablasi

#### Q30: "Bagaimana arsitektur pelatihan model regresi logistik ablasi?"
- **Jawaban Berbasis Bukti:** Model dilatih secara ketat hanya pada Calibration Dev Set ($N=323$, 88 klaster citra). Fitur distandarisasi menggunakan `StandardScaler` yang parameternya $(\mu, \sigma)$ dihitung eksklusif dari Dev Set. Menggunakan regularisasi $L_2$ ($C=1.0$) untuk mencegah overfitting pada sampel kecil. Validasi silang dilakukan menggunakan `GroupKFold(n_splits=5)` berbasis klaster citra. Model final yang telah difit kemudian dievaluasi satu kali pada Test Set beku ($N=799$, 204 klaster citra) yang sama sekali tidak tersentuh selama pelatihan.
- **Lokasi Bukti:** `main/scripts/resilient_validation_engine.py` line 620–700, `docs/RESEARCH_MASTER.md` Bagian 15.

#### Q31: "Bagaimana Anda menyelaraskan arah skor risiko regresi logistik agar adil dibandingkan dengan confidence $M_1$?"
- **Jawaban Berbasis Bukti:** Model regresi logistik memprediksi probabilitas error $\hat{P}(\text{error}_{\text{ID}} = 1)$. Agar selaras dengan prinsip selective classification (di mana skor tinggi berarti sampel lebih andal), skor selektif didefinisikan secara invers:
  $$S_{\text{LR}} = 1.0 - \hat{P}(\text{error}_{\text{ID}} = 1 \mid \mathbf{x}) = \hat{P}(\text{correct}_{\text{ID}} = 1 \mid \mathbf{x})$$
  Dengan demikian, sampel dengan risiko error terendah berada di urutan teratas, persis sama dengan arah ranking pada $M_1$ dan $M_{\text{norm}}$.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 8.2, `main/scripts/resilient_validation_engine.py` line 675 & 687.

#### Q32: "Mengapa perbandingan utama pembuktian $D_{\text{lang}}$ harus `LR-full vs LR-visual` dan bukan `LR-full vs M1`?"
- **Jawaban Berbasis Bukti:** Karena `LR-full` memuat 3 fitur ($p_{\text{orig}}$, $\Delta P_{\text{dir}}$, dan $D_{\text{lang}}$). Jika kita hanya membandingkan `LR-full` dengan $M_1$, kita tidak dapat membedakan apakah penurunan AURC disebabkan oleh fitur visual $\Delta P_{\text{dir}}$ atau sinyal bahasa $D_{\text{lang}}$. Perbandingan `LR-full vs LR-visual` mengisolasi murni kontribusi inkremental dari penambahan $D_{\text{lang}}$ ketika fitur visual sudah dikontrol dalam model.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 16, `docs/audit/EXPERIMENT_OUTPUT_AUDIT.md` Bagian 3.4.

---

### Kategori 10: Analisis Statistik Inferensial & Multiplisitas

#### Q33: "Mengapa Anda melakukan resampling bootstrap dan permutasi pada tingkat klaster citra dan bukan baris QA?"
- **Jawaban Berbasis Bukti:** Karena dalam dataset IndoRad-VQA, satu citra radiologi fisik diajukan rata-rata 7 pertanyaan berbeda. Pertanyaan yang merujuk pada citra yang sama memiliki keterikatan kovarians (*intra-cluster correlation*) dan melanggar asumsi *independent and identically distributed (i.i.d.)*. Resampling pada tingkat baris akan meremehkan varians (*underestimating standard error*) dan menghasilkan p-value palsu yang terlalu optimistik. Resampling klaster citra menjamin bahwa unit sampling yang ditarik adalah unit fisik yang benar-benar independen.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 16, `main/scripts/decisive_validation_experiment_v5.py` line 250–343.

#### Q34: "Bagaimana Anda menangani masalah pengujian hipotesis berganda (*multiplicity*)?"
- **Jawaban Berbasis Bukti:** Kami menerapkan koreksi step-down **Holm-Bonferroni** untuk mengontrol *Family-Wise Error Rate (FWER)* pada taraf signifikansi $\alpha = 0.05$ di atas keluarga 4 pengujian hipotesis terencana:
  1. $M_{\text{norm}}$ vs $M_1$ (Uji Primer Heuristik)
  2. `LR-conf` vs $M_1$ (Efek Pemodelan Risiko)
  3. `LR-visual` vs `LR-conf` (Efek Sinyal Visual Inkremental)
  4. `LR-full` vs `LR-visual` (Efek Sinyal Bahasa Inkremental)
  Implementasi fungsi `apply_holm_bonferroni` memverifikasi penyesuaian p-value secara ketat.
- **Lokasi Bukti:** `main/scripts/decisive_validation_experiment_v5.py` line 346–357, `docs/RESEARCH_MASTER.md` Bagian 16.

---

### Kategori 11: Batasan Epistemik & Keselamatan Klinis Pasien

#### Q35: "Jika sistem selective prediction Anda menghasilkan AURC rendah (misal 0.2053), apakah sistem ini sudah siap dideploy di rumah sakit?"
- **Jawaban Berbasis Bukti:** **TIDAK, BELUM SIAP.** Penurunan nilai AURC semata-mata membuktikan kemampuan statistik pengurutan risiko kesalahan secara retrospektif pada dataset uji tertutup. Sistem ini berstatus **purwarupa penelitian akademis (*academic research prototype*)** dan tidak boleh digunakan untuk pengambilan keputusan medis pada pasien tanpa pengawasan dokter spesialis radiologi berlisensi. Validasi prospektif dalam alur kerja rumah sakit nyata serta uji klinis independen masih mutlak diperlukan sebelum implementasi dunia nyata.
- **Lokasi Bukti:** `docs/RESEARCH_MASTER.md` Bagian 20.
