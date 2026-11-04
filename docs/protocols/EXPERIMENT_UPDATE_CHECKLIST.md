# EXPERIMENT UPDATE CHECKLIST — VERSION 10.0
## Pipeline Specification, Feature Engineering, Risk Modeling, and Zero-GPU Verification

**Dokumen:** `EXPERIMENT_UPDATE_CHECKLIST_10.md`  
**Target Revisi:** Integrasi Sinyal $D_{\text{lang}}$ dan Ablation Study Regresi Logistik Risiko pada Pipeline `rev05` $\to$ `rev05-v10`  
**Peneliti / Auditor:** Senior ML Research Engineer & Quantitative Methodologist  
**Status Gate:** `AWAITING_CODE_INTEGRATION / ZERO_ADDITIONAL_GPU_PASS_REQUIRED`

---

## 1. Rasional Efisiensi Komputasi (Zero Extra Forward-Passes)

> [!IMPORTANT]
> **Efisiensi Bebas Biaya Komputasi Tambahan:**  
> Arsitektur inferensi `rev05` (`main/scripts/decisive_validation_experiment_v5.py` dan `main/Kaggle/08_decisive_validation_rev05.ipynb`) **telah membangkitkan dan menyimpan seluruh variabel mentah yang dibutuhkan**:
> 1. Prediksi biner bahasa Inggris ($\hat{y}_{\text{EN}}$) dari evaluasi $P_0$ (asli).
> 2. Prediksi biner bahasa Indonesia ($\hat{y}_{\text{ID}}$) dari evaluasi $P_0$ (asli).
> 3. Skor preferensi kandidat asli ($p_{\text{orig\_ID}}$) dan citra hitam ($p_{\text{blank\_ID}}$).
> 4. Dukungan visual berarah ($\Delta P_{\text{dir\_ID}} = p_{\text{orig\_ID}} - p_{\text{blank\_ID}}$).
> 5. Indikator kebenaran terhadap ground truth ($\text{correct}_{\text{ID}}$).
>
> Dengan demikian, implementasi Versi 10.0 **TIDAK MEMERLUKAN FORWARD-PASS GPU TAMBAHAN**. Sinyal $D_{\text{lang}}$ dan pemodelan risiko regresi logistik murni merupakan pemrosesan pasca-inferensi (*post-inference tabular risk modeling*) yang dieksekusi dalam hitungan detik pada CPU.

---

## 2. Rincian Perubahan Modul Kode (Pipeline Updates)

### 2.1 Perumusan Variabel Tabular Baru
Pada checkpoint hasil inferensi, tambahkan komputasi kolom deterministik berikut:

```python
# 1. Penyelarasan ruang label biner (Mapping ID -> Canonical YES/NO)
LABEL_MAP_ID_TO_CANONICAL = {"YA": "YES", "TIDAK": "NO"}
df["pred_id_canonical"] = df["pred_orig_id"].map(LABEL_MAP_ID_TO_CANONICAL)

# 2. Cross-Lingual Disagreement Indicator (Fitur Risiko Baru)
# D_lang = 1 jika prediksi EN != ID; 0 jika prediksi identik
df["d_lang"] = (df["pred_en"] != df["pred_id_canonical"]).astype(int)

# 3. Ground-Truth Binary Error Indicator (Target Regresi)
# error_id = 1 jika prediksi ID salah; 0 jika benar
df["error_id"] = (~df["correct_orig_id"].astype(bool)).astype(int)
```

### 2.2 Penanganan Kasus Ambigu, Output Tidak Valid, & Missing Values
- **Kasus Non-Biner / Gagal Ekstraksi:** Karena kandidat dievaluasi menggunakan *closed candidate scoring* (`YES/NO` dan `YA/TIDAK`), probabilitas selalu terdistribusi 100% pada ruang biner tertutup. Tidak ada teks terbuka yang lolos dari parsing.
- **Kasus Seri ($P(\text{YA}) == P(\text{TIDAK}) == 0.5$):** Ditangani deterministik dengan resolusi default ke `"YA"`, dan dicatat flag `is_tie = True`.
- **Nilai Hilang / NaN:** Checkpoint memvalidasi `assert df[features].isna().sum().sum() == 0` sebelum fitting.

---

## 3. Protokol Anti-Leakage & Spesifikasi Model Regresi Logistik

### 3.1 Pemisahan Data & Pencegahan Kebocoran (Strict Cohort Separation)
- **Development Split ($N=323$, 88 Klaster Citra Kanonis):**
  - Digunakan secara eksklusif untuk fitting scaler, pemilihan parameter regularisasi ($C$), dan validasi silang (*cross-validation*).
  - Validasi silang menggunakan **`GroupKFold(n_splits=5)`** dengan `groups = df_dev["canonical_image_id"]`.
  - Semua baris QA yang berbagi citra yang sama dipastikan berada dalam fold yang sama (0% klaster tumpang tindih antar fold).
- **Test Split ($N=799$, 204 Klaster Citra Kanonis):**
  - **DIBEKUKAN MUTLAK.** Tidak boleh digunakan untuk tuning $C$, pemilihan threshold, atau pemilihan fitur.
  - Model yang telah dilatih pada 100% data Dev Set ($N=323$) langsung dievaluasi satu kali pada Test Set ($N=799$).

### 3.2 Spesifikasi Empat Estimator dalam Ablation Study
Semua model menggunakan `LogisticRegression` sederhana dari `scikit-learn` dengan regularisasi $L_2$:

1. **Estimator A — Baseline Direct ($M_1$):**
   $$S_{M1} = p_{\text{orig\_ID}}$$
   *(Tanpa training; aturan keputusan langsung)*
2. **Estimator B — Confidence-Only Risk Model (`LR-conf`):**
   $$\text{Logit}(P(\text{error}_{\text{ID}} = 1)) = \beta_0 + \beta_1 \cdot z(p_{\text{orig\_ID}})$$
3. **Estimator C — Visual-Sensitivity Risk Model (`LR-visual`):**
   $$\text{Logit}(P(\text{error}_{\text{ID}} = 1)) = \beta_0 + \beta_1 \cdot z(p_{\text{orig\_ID}}) + \beta_2 \cdot z(\Delta P_{\text{dir\_ID}})$$
4. **Estimator D — Full Multimodal Risk Model (`LR-full`):**
   $$\text{Logit}(P(\text{error}_{\text{ID}} = 1)) = \beta_0 + \beta_1 \cdot z(p_{\text{orig\_ID}}) + \beta_2 \cdot z(\Delta P_{\text{dir\_ID}}) + \beta_3 \cdot D_{\text{lang}}$$

*Catatan:* $z(\cdot)$ adalah standardisasi z-score (`StandardScaler`) yang parameternya $(\mu, \sigma)$ dihitung **hanya dari Dev Set**. $D_{\text{lang}} \in \{0, 1\}$ dimasukkan sebagai fitur biner tanpa standardisasi.

### 3.3 Penyelarasan Arah Skor Reliabilitas (Score Orientation Alignment)
Dalam kerangka kerja Selective Prediction (Geifman & El-Yaniv, 2017), sampel diurutkan secara menurun berdasarkan **skor reliabilitas / keyakinan** (sampel paling diyakini benar dipertahankan pertama kali).
Karena regresi logistik memprediksi probabilitas kesalahan $\hat{P}(\text{error}_{\text{ID}} = 1)$, skor selektif didefinisikan secara konsisten sebagai:
$$S_{\text{LR}} = 1 - \hat{P}(\text{error}_{\text{ID}} = 1 \mid \mathbf{x}) \in [0, 1]$$

Dengan konvensi ini:
- Skor $S_{\text{LR}}$ yang tinggi $\implies$ probabilitas error rendah $\implies$ diprioritaskan untuk dijawab.
- Skor $S_{\text{LR}}$ yang rendah $\implies$ probabilitas error tinggi $\implies$ ditolak / diabstain pertama kali.
- Kurva Risk–Coverage dan perhitungan Discrete Riemann AURC berjalan secara 100% harmonis dengan baseline $M_1$ dan $M_{\text{norm}}$.

---

## 4. Matriks Komparasi Hipotesis & Uji Statistik

Untuk menguji nilai tambah setiap komponen secara adil:

| Pasangan Komparasi | Estimator yang Diuji | Pertanyaan Metodologis yang Dijawab | Kriteria Keberhasilan Ilmiah |
|:---|:---|:---|:---|
| **Uji Primer** | $M_{\text{norm}}$ vs $M_1$ | Apakah formula intervensi visual langsung mengungguli confidence verbal? | $\Delta\text{AURC} < 0$ dengan exact permutation $p_{\text{adj}} < 0.05$ |
| **Uji Sekunder 1** | `LR-conf` vs $M_1$ | Apakah pemodelan risiko terpelajar memberikan keuntungan di atas skor preferensi langsung? | Mengukur baseline gain akibat kalibrasi supervised di Dev Set |
| **Uji Sekunder 2** | `LR-visual` vs `LR-conf` | Apakah sensitivitas visual ($\Delta P_{\text{dir}}$) menambah daya pisah risiko kesalahan? | $\Delta\text{AURC} < 0$ (mengisolasi kontribusi murni sinyal visual) |
| **Uji Sekunder 3** | `LR-full` vs `LR-visual` | **Apakah ketidaksepakatan bilingual ($D_{\text{lang}}$) memberikan nilai tambah inkremental?** | $\Delta\text{AURC} < 0$ (mengisolasi kontribusi murni inkonsistensi bahasa) |

- **Uji Signifikansi:** Exact Paired Image-Cluster Permutation Test ($B=1.000$ Rademacher sign-flips) pada tingkat klaster citra fisik kanonis.
- **Koreksi Multiplisitas:** Penyesuaian nilai $p$ step-down Holm-Bonferroni pada keluarga 4 komparasi.

---

## 5. Implementation Checklist: Urutan Tindakan Teknis

```
[ ] TAHAP 1: PEMBARUAN SCRIPT EVALUASI POST-INFERENCE
    ├── File: main/scripts/postprocess_risk_ablation.py (Skrip Python Mandiri)
    ├── Task 1.1: Fungsi komputasi D_lang dan error_id dari checkpoint rev05.
    ├── Task 1.2: Pipeline GroupKFold CV untuk fitting LR-conf, LR-visual, LR-full pada Dev Set.
    ├── Task 1.3: Prediksi S_LR = 1 - P(error) pada Test Set.
    ├── Task 1.4: Fungsi Discrete Riemann AURC terpadu untuk M1, M_norm, LR-conf, LR-visual, LR-full.
    └── Acceptance: Output tabel perbandingan 5 estimator pada Dev dan Test.

[ ] TAHAP 2: UNIT TEST SUITE VALIDASI FITUR & GROUPKFOLD
    ├── File: main/scripts/test_risk_ablation_v10.py
    ├── Test 2.1: Verifikasi tidak ada tumpang-tindih klaster gambar antar fold GroupKFold.
    ├── Test 2.2: Verifikasi penanganan kasus seri D_lang dan pemetaan YES/NO vs YA/TIDAK.
    ├── Test 2.3: Verifikasi konvensi arah skor: korelasi Spearman S_LR dengan p_orig positif.
    ├── Test 2.4: Verifikasi hand-calculated Discrete AURC pada probabilitas LR.
    └── Acceptance: 5/5 Unit Test Lolos 100%.

[ ] TAHAP 3: PEMBARUAN NOTEBOOK KAGGLE REV05 -> REV05_V10
    ├── File: main/Kaggle/08_decisive_validation_rev05.ipynb (atau generator script v5)
    ├── Task 3.1: Tambahkan Cell 18: Eksekusi evaluasi regresi logistik ablation pasca inferensi neural.
    ├── Task 3.2: Simpan artefak tambahan: risk_ablation_comparison_test.csv dan model_weights_dev.json.
    └── Acceptance: Notebook tervalidasi AST parse OK tanpa syntax error.

[ ] TAHAP 4: SINKRONISASI ARTEFAK DOKUMENTASI
    ├── File: main/reports/risk_ablation_results_report.md
    ├── Task 4.1: Catat prevalensi error_id pada Dev Set (N=323) dan Test Set (N=799).
    ├── Task 4.2: Rekapitulasi bobot koefisien regresi beta_1, beta_2, beta_3 pada Dev Set.
    └── Acceptance: Dokumentasi lengkap siap pakai untuk Bab IV Naskah Skripsi.
```
