# 🔬 Dataset Integrity & Anti-Leakage Audit Report (v10.0)
**Thesis Project:** Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift  
**Auditor:** Senior Research Auditor, Machine Learning Engineer & Quantitative Methodologist  
**Audit Dataset:** `Lab-IS/IndoRad-VQA` (Hugging Face Hub)  
**Authoritative Manifest:** `manifest_rev05.csv` (SHA-256: `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`)  
**Status:** `VERIFIED_CRYPTOGRAPHIC` / `ZERO_LEAKAGE_CONFIRMED`

---

## 1. Executive Summary

Laporan ini menyajikan hasil audit forensik menyeluruh terhadap proses kurasi, pembersihan data, pengelompokan citra fisik kanonis (*canonical image clustering*), dan integritas pemisahan split (*zero data leakage*) pada dataset IndoRad-VQA.

### Ringkasan Temuan Audit Dataset:
1. **Integritas Baris & Citra:**  
   - Total baris manifest mentah: **2.244 pasangan QA bilingual**.
   - Total citra fisik unik: **314 citra radiologi** (diverifikasi bit-identik melalui hash SHA-256 piksel mentah RGB).
   - Seluruh baris memiliki `qid` unik (0 baris duplikat).
   - Seluruh baris memiliki data lengkap pada kolom kritis (0 missing values).
2. **Tata Kelola Eksklusi Data:**
   - 1.053 pertanyaan terbuka dieksklusi secara teratur karena tidak memenuhi asumsi ruang probabilitas biner tertutup.
   - 1.191 pertanyaan yes/no disaring lebih lanjut melalui audit translasi: 69 pasangan dieksklusi akibat cacat terjemahan kritis (pembalikan negasi, organ salah, distorsi lateralitas).
   - Menghasilkan **Primary Clean Cohort tepat 1.122 pasangan QA**.
3. **Pemisahan Berbasis Klaster Citra (Zero Data Leakage):**
   - Pemisahan data (`train/dev` vs `test`) dilakukan secara ketat pada unit **klaster citra fisik kanonis (`canonical_image_id`)**, bukan baris QA.
   - Primary Test Split: **799 baris (204 klaster citra)**.
   - Calibration Dev Split: **323 baris (88 klaster citra)**.
   - **Tingkat Kebocoran Lintas Split (*Cross-Split Leakage*): PERSIS 0 KLASTER (0.00%)**. Tidak ada satu pun citra radiologi yang muncul di Dev Set dan Test Set secara bersamaan.

---

## 2. Alur Akuntansi Eksklusi Data (Data Accounting Flow)

```
                       TOTAL INDORAD-VQA DATASET
                   2.244 Baris / 314 Citra Fisik Unik
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
PERTANYAAN TERBUKA                                 PERTANYAAN BINER YES/NO
  1.053 Baris                                            1.191 Baris
(Dieksklusi demi ruang                             (585 YES / 606 NO)
 probabilitas tertutup)                                      │
                                                             ▼
                                                AUDIT KUALITAS TRANSLASI
                                               69 Pasangan Cacat Dieksklusi
                                               - T2 (Pembalikan Negasi)
                                               - T4 (Salah Organ)
                                               - T5 (Distorsi Lateralitas)
                                               - T6-T8 (Ambiguitas Sintaksis)
                                                             │
                                                             ▼
                                                   PRIMARY CLEAN COHORT
                                              1.122 Baris (292 Citra Fisik)
                                                             │
                                   ┌─────────────────────────┴─────────────────────────┐
                                   ▼                                                   ▼
                          PRIMARY TEST SPLIT                                  CALIBRATION DEV SPLIT
                        799 Baris (407 YES / 392 NO)                        323 Baris (147 YES / 176 NO)
                          204 Klaster Citra Unik                              88 Klaster Citra Unik
                                   │                                                   │
                                   └─────────────────┬─────────────────────────────────┘
                                                     ▼
                                       CROSS-SPLIT CLUSTER LEAKAGE:
                                         PERSIS 0 KLASTER (0.0%)
```

---

## 3. Rekonsiliasi Struktur Kohort Dual-Track

Untuk memastikan penelitian ini dapat memvalidasi hipotesis baru sekaligus membandingkannya secara adil (*apples-to-apples*) dengan eksperimen historis terdahulu, sistem mengoperasikan arsitektur dual-track:

| Atribut Evaluasi | Primary Clean Cohort (Track 1) | Historical Benchmark (Track 2) | Active Union Evaluated (Union) |
| :--- | :---: | :---: | :---: |
| **Kriteria Seleksi** | `is_eligible_primary == True` | `in_rev03_benchmark == True` | `in_eval_cohort == True` |
| **Tujuan Evaluasi** | Uji Hipotesis Primer & Generalisasi | Replikasi Historis rev02/rev03 | Evaluasi Komprehensif Satu Kali Run |
| **Total Sampel ($N$)** | **1.122** | **150** | **1.133** |
| **Test Set ($N_{\text{test}}$)** | **799** | **92** | **806** |
| **Dev Set ($N_{\text{dev}}$)** | **323** | **58** | **327** |
| **Klaster Citra Test** | **204** | **35** | **204** |
| **Klaster Citra Dev** | **88** | **23** | **88** |
| **Kebocoran Antar-Split** | **0 Klaster (0.0%)** | **0 Klaster (0.0%)** | **0 Klaster (0.0%)** |
| **Distribusi Label Test** | 407 YES / 392 NO | 47 YES / 45 NO | 412 YES / 394 NO |
| **Distribusi Label Dev** | 147 YES / 176 NO | 24 YES / 34 NO | 149 YES / 178 NO |

> **Penjelasan Selisih Union (1.133 vs 1.122):**  
> Terdapat 11 sampel dari benchmark historis rev03 ($N=150$) yang pada audit translasi rev04/rev05 diklasifikasikan sebagai memiliki cacat translasi ringan. Agar artefak historis $N=150$ dapat direproduksi 100% tanpa menimpa protokol baru, 11 sampel ini diikutsertakan dalam inferensi model (`in_eval_cohort == True`, total 1.133), namun dieksklusi secara ketat saat mengevaluasi metrik Primary Clean Test Set ($N=799$).

---

## 4. Laporan Kualitas Data Forensik

### 4.1 Missing Values
- `qid`: 0 missing (0.0%)
- `canonical_image_id`: 0 missing (0.0%)
- `pixel_sha256`: 0 missing (0.0%)
- `question_en`: 0 missing (0.0%)
- `question_id`: 0 missing (0.0%)
- `reference_answer`: 0 missing (0.0%)
- `split`: 0 missing (0.0%)
- **Kesimpulan:** Seluruh 2.244 baris bebas dari missing values.

### 4.2 Duplikasi & Integritas Relasional
- Unique `qid`: 2.244 dari 2.244 baris (100% unik).
- Unique `qa_id`: 2.244 dari 2.244 baris (100% unik).
- Hubungan QA per Citra: Rata-rata 7.15 pertanyaan per citra fisik (Rentang: 1 hingga 29 pertanyaan per citra).
- **Kesimpulan:** Tidak ada duplikasi baris atau pertanyaan. Pengelompokan baris ke klaster citra fisik terbukti esensial karena pertanyaan-pertanyaan yang berbagi citra yang sama memiliki keterikatan kovarians (*intra-cluster correlation*).

### 4.3 Keseimbangan Kelas Ground Truth (Class Balance)
- **Primary Test Set ($N=799$):**
  - YES: 407 sampel (50.94%)
  - NO: 392 sampel (49.06%)
  - *Rasio Ketidakseimbangan:* $1.038 : 1$ (Sangat seimbang, mendekati ideal 50:50).
- **Primary Dev Set ($N=323$):**
  - NO: 176 sampel (54.49%)
  - YES: 147 sampel (45.51%)
  - *Rasio Ketidakseimbangan:* $1.197 : 1$ (Seimbang).
- **Kesimpulan:** Tidak diperlukan teknik oversampling sintetis (seperti SMOTE) karena distribusi label biner alami pada kedua split sangat seimbang.

### 4.4 Verifikasi Kriptografis Citra Fisik
- Seluruh 314 citra radiologi fisik mentah diverifikasi melalui algoritma hashing SHA-256 pada byte array mentah RGB:
  ```python
  hashlib.sha256(pil_img.convert("RGB").tobytes()).hexdigest()
  ```
- Hasil verifikasi: 100% cocok dengan kolom `pixel_sha256` pada manifest resmi. Tidak ditemukan manipulasi citra, kompresi artefak sekunder, atau distorsi dimensi.

---

## 5. Verifikasi Zero Data Leakage

Salah satu ancaman validitas paling kritis dalam machine learning medis adalah kebocoran citra (*image-level data leakage*), di mana citra pasien yang sama muncul pada data latih/dev dan data uji dengan pertanyaan berbeda.

### Bukti Verifikasi Independen:
```python
test_clusters = set(prim[prim["split"] == "test"]["canonical_image_id"])
dev_clusters = set(prim[prim["split"] == "dev"]["canonical_image_id"])
overlap = test_clusters.intersection(dev_clusters)
assert len(overlap) == 0
```
- Jumlah klaster Primary Test: **204 klaster unik**.
- Jumlah klaster Primary Dev: **88 klaster unik**.
- Jumlah irisan klaster ($test \cap dev$): **0 klaster (0.00%)**.
- Jumlah irisan klaster global ($N=2.244$): **0 klaster (0.00%)**.

**Pernyataan Auditor:**  
Pemisahan dataset pada protokol `rev05` sepenuhnya bebas dari kebocoran citra (*zero cluster leakage*). Model regresi logistik yang dilatih pada Dev Set ($N=323$) tidak pernah melihat informasi citra atau representasi fitur dari Test Set ($N=799$).
