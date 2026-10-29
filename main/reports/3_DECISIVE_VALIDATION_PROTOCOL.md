# Consolidated Report 3: Pre-Registered Decisive Validation Protocol (Original vs. Black-Blank)

**Dokumen:** `reports/3_DECISIVE_VALIDATION_PROTOCOL.md`  
**Konsolidasi:** `decisive_validation_protocol.md`, `statistical_analysis_plan.md`, `final_research_protocol_gate.md`, `primary_prompt_template.md`  
**Status:** Protokol Pra-Registrasi Final Pasca-Persetujuan Dosen Pembimbing  
**Target Eksekusi:** `scripts/decisive_validation_experiment.py`  

---

## 1. Konteks Persetujuan Dosen & Batasan Metodologis

Dosen pembimbing telah menyetujui arah penelitian skripsi dan memberikan arahan operasional:
> **"Blur tidak perlu digunakan karena kualitas gambar sudah baik dan blur diperkirakan tidak memberikan pengaruh berarti. Intervensi gambar blank hitam diperbolehkan."**

Sesuai arahan tersebut, protokol ini mengunci batasan metodologis berikut:
1. **Peniadaan Blur Total:** Blur (Gaussian blur maupun pengaburan lainnya) dihapus dari protokol eksperimen utama dan tidak boleh digunakan kembali tanpa izin eksplisit dosen pembimbing.
2. **Dua Kondisi Utama yang Ditetapkan:**
   * `P0_ORIGINAL`: Citra radiologi asli tanpa modifikasi.
   * `P1_BLACK_BLANK`: Citra hitam deterministik ($(0,0,0)$) berdimensi piksel, rasio, kanal warna, dan preprocessing yang identik dengan citra asli.
3. **Invariansi Mutlak di Luar Citra:** Pertanyaan evaluasi, prompt instruksi, pasangan kandidat token, arsitektur model, checkpoint bobot, precision bfloat16/4-bit, dan prosedur scoring dijamin **100% identik** antara kondisi P0 dan P1. Satu-satunya perbedaan yang disengaja adalah modalitas citra.
4. **Epistemic Humility & Sifat Out-Of-Distribution (OOD):** Citra hitam $(0,0,0)$ adalah masukan di luar distribusi (*out-of-distribution*) bagi vision encoder (SigLIP/ViT). Responsivitas model terhadap penghapusan citra adalah ukuran sensitivitas perilaku (*behavioral sensitivity to visual deletion*), **bukan bukti pemahaman klinis atau lokalisasi lesi patologis secara kausal sempurna**.
5. **Klaim Klinis vs. Perankingan Statistik:** Nilai AURC yang lebih rendah semata-mata mengukur kemampuan pemeringkatan kesalahan (*selective risk-coverage error ranking*) pada sampel uji retrospektif. AURC yang lebih rendah **tidak sama dengan jaminan keselamatan klinis**, tidak meniadakan risiko salah diagnosis fatal (*false negatives*), dan tidak dapat menggantikan validasi prospektif dokter spesialis radiologi.

---

## 2. Langkah 1: Pengukuran Confidence Bilingual Sekuens Target

### 2.1 Mengapa Bukan Logit Token Pertama?
Model autoregresif instruksi medis sering menjawab dalam bentuk kalimat pembuka (misalnya: *"Ya, terdapat pembesaran jantung..."*, *"Berdasarkan gambaran radiologi, tidak tampak..."*). Pada kasus ini, token pertama adalah `"Berdasarkan"`, sehingga pembacaan logit token pertama gagal mendeteksi afirmasi/negasi. Selain itu, tokenisasi Byte-Pair Encoding (BPE) pada bahasa Indonesia dapat memecah kata dasar (misal: `"Tidak"` = `["Ti", "dak"]`).

### 2.2 Formulasi *Target Candidate Sequence Scoring*
Diberikan konteks multimodal $C = (I, Q)$:
* **Kandidat Bahasa Inggris:** $A_{\text{aff}} = \text{"Yes"}$, $A_{\text{neg}} = \text{"No"}$
* **Kandidat Bahasa Indonesia:** $A_{\text{aff}} = \text{"Ya"}$, $A_{\text{neg}} = \text{"Tidak"}$

Untuk setiap kandidat token $A = (w_1, \dots, w_L)$, evaluasi log-likelihood sekuensial secara kondisional:
$$\log P(A \mid C) = \sum_{t=1}^L \log P(w_t \mid C, w_{<t})$$

Probabilitas preferensi afirmatif ternormalisasi dihitung melalui softmax pasangan biner:
$$\tilde{P}(A_{\text{aff}} \mid C) = \frac{\exp(\log P(A_{\text{aff}} \mid C))}{\exp(\log P(A_{\text{aff}} \mid C)) + \exp(\log P(A_{\text{neg}} \mid C))}, \qquad \tilde{P}(A_{\text{neg}} \mid C) = 1.0 - \tilde{P}(A_{\text{aff}} \mid C)$$

Jawaban prediksi diskrit pada citra asli ($\hat{y}$) dan nilai confidence operasional ($p_{\text{orig}}$):
$$\hat{y} = \begin{cases} \text{"YES"} & \text{jika } \tilde{P}(A_{\text{aff}} \mid C_0) \ge 0.5 \\ \text{"NO"} & \text{lainnya} \end{cases}, \qquad p_{\text{orig}} = \max\left(\tilde{P}(A_{\text{aff}} \mid C_0), \tilde{P}(A_{\text{neg}} \mid C_0)\right) \in [0.5, 1.0]$$

> *Catatan Kalibrasi:* Probabilitas relatif dari dua kandidat ini adalah ukuran preferensi kondisional biner model, bukan probabilitas terkalibrasi absolut bahwa diagnosis tersebut benar secara fakta medis.

---

## 3. Langkah 2: Intervensi Monolingual Bahasa Indonesia (P0 vs P1)

* **Pertanyaan Target:** Menggunakan `question_indonesian` (`lang="id"`).
* **Dua Kondisi Utama:**
  * **$P_0$ (Citra Asli):** $C_0 = (I_{\text{orig}}, Q_{\text{id}})$. Menghasilkan probabilitas prediksi asli:
    $$p_{\text{orig}} = P(\hat{y} \mid I_{\text{orig}}, Q_{\text{id}})$$
  * **$P_1$ (Black Blank):** $C_1 = (I_{\text{blank}}, Q_{\text{id}})$. Menghasilkan probabilitas yang diberikan model pada **jawaban yang sama $\hat{y}$**:
    $$p_{\text{blank}} = P(\hat{y} \mid I_{\text{blank}}, Q_{\text{id}})$$
* **Pembuatan Citra Hitam Deterministik:**
  Kanvas hitam berdimensi piksel $(W, H)$ identik dengan citra asli, kanal RGB $(0, 0, 0)$. Diproses melalui pipeline `AutoProcessor` yang sama persis tanpa normalisasi tak terduga.
* **Template Prompt Tetap:**
  `"Jawab pertanyaan radiologi berdasarkan gambar berikut. Jawab hanya dengan YA atau TIDAK.\nPertanyaan: {question_id}"`

---

## 4. Langkah 3: Perumusan Sensitivitas Visual Berarah & Estimator Reliabilitas

### 4.1 Sensitivitas Visual Berarah ($\Delta_{\text{dir}}$)
$$\Delta_{\text{dir}} = p_{\text{orig}} - p_{\text{blank}} \in [-1.0, 1.0]$$
* **$\Delta_{\text{dir}} > 0$:** Penghapusan citra radiologi *menurunkan* keyakinan model terhadap jawaban $\hat{y}$. Modalitas visual memberikan dukungan positif bagi prediksi tersebut.
* **$\Delta_{\text{dir}} \le 0$:** Penghapusan citra *tidak mengubah* atau *malah menaikkan* keyakinan model terhadap $\hat{y}$. Prediksi tersebut murni didorong oleh bias bahasa teks / halusinasi.

### 4.2 Lima Estimator yang Diuji Sejajar (Fair Comparison):
1. **$M_1$ (Confidence Saja — Baseline):**
   $$S_{M1} = p_{\text{orig}}$$
2. **$M_2$ (Sensitivitas Visual Murni):**
   $$S_{M2} = \max(0, \Delta_{\text{dir}})$$
3. **$M_{\text{dir}}$ (Kombinasi Multiplikatif Berarah):**
   $$S_{\text{dir}} = p_{\text{orig}} \times \max(0, \Delta_{\text{dir}})$$
4. **$M_{\text{norm}}$ (Kombinasi Linear Ternormalisasi):**
   $$S_{\text{norm}} = p_{\text{orig}} \times \frac{1 + \Delta_{\text{dir}}}{2}$$
5. **$M_3^{\text{old}}$ (Baseline Historis Unsigned):**
   $$S_{M3} = p_{\text{orig}} \times \frac{1 + |\Delta_{\text{dir}}|}{2}$$
   *(Dipertahankan murni sebagai pembanding historis terhadap klaim eksperimen terdahulu).*

---

## 5. Langkah 4: Evaluasi Diskrit AURC & Paired Image-Cluster Bootstrap

### 5.1 Perhitungan Diskrit Sample-Level
Sampel diurutkan menurun berdasarkan skor: $S_{(1)} \ge S_{(2)} \dots \ge S_{(N)}$.  
Untuk setiap peringkat sampel $k \in \{1, \dots, N\}$:
$$\text{Coverage } \phi_k = \frac{k}{N}, \qquad \text{Risiko Selektif } r_k = 1 - \frac{1}{k} \sum_{j=1}^k \mathbb{I}(\hat{y}_{(j)} = y_{(j)}^*)$$
$$\text{AURC}_{\text{discrete}} = \frac{1}{N} \sum_{k=1}^N r_k$$

### 5.2 Evaluasi Risiko pada Coverage Tetap
Evaluasi risiko kesalahan pada tingkat cakupan operasional yang telah ditentukan sebelumnya:
* $r(\phi = 1.00)$ (Seluruh data dijawab, $k = 150$)
* $r(\phi = 0.90)$ ($k = 135$)
* $r(\phi = 0.80)$ ($k = 120$ — titik acuan operasional rujukan)
* $r(\phi = 0.70)$ ($k = 105$)
* $r(\phi = 0.50)$ ($k = 75$)

### 5.3 Paired Image-Cluster Bootstrap
* Unit independen adalah **58 kluster citra radiologi unik** (`canonical_image_id`).
* Lakukan $B = 1.000$ resample dengan pengembalian pada tingkat 58 kluster citra (seed = 42).
* Hitung selisih berpasangan terhadap baseline $M_1$:
  $$\Delta \text{AURC} = \text{AURC}(M_{\text{kandidat}}) - \text{AURC}(M_1)$$
* Laporkan interval kepercayaan empiris 95% $[2.5\%, 97.5\%]$ dan nilai $p$ empiris dua arah ($H_0: \Delta \text{AURC} = 0$).

---

## 6. Langkah 5: Interpretasi Ilmiah & Matriks Keputusan

1. **Jika Estimator Visual Berarah Mengungguli $M_1$ Secara Signifikan ($p < 0.05$):**  
   Membuktikan secara empiris bahwa sensitivitas visual berarah pada citra hitam efektif mengidentifikasi dan memotong halusinasi overconfidence bahasa Indonesia.
2. **Jika Estimator Visual Tidak Mengungguli $M_1$ ($p \ge 0.05$):**  
   **Temuan negatif yang informatif.** Membuktikan bahwa intervensi citra kosong pada waktu inferensi tidak memberikan sinyal pemisah tambahan melampaui kalibrasi confidence sekuens target. Pembahasan skripsi mendokumentasikan batas kegunaan perturbasi input pada LLM multimodal.
