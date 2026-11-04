# LAPORAN DIAGNOSTIK VALIDITAS ILMIAH PIPELINE EVALUASI (TUGAS A)
**Proyek Penelitian:** Risk-Aware Selective Answering untuk Medical VLM Bilingual (EN–ID) pada IndoRad-VQA  
**Revisi Protokol:** `rev05` (Frozen Manifest SHA-256: `fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9`)  
**Kohort Evaluasi:** Primary Test Set ($N=799$, 204 klaster citra `canonical_image_id`) & Primary Dev Set ($N=323$, 88 klaster)  
**Status Audit:** Hasil komputasi nyata end-to-end tanpa fabrikasi numerik.

---

## 1. Ringkasan Eksekutif Hasil Diagnostik

Audit diagnostik validitas ilmiah terhadap checkpoint inferensi asli rev05 (`Qwen2.5-VL-7B-Instruct` dan `HuatuoGPT-Vision-7B-Qwen2.5VL`) mengungkap beberapa temuan kritis yang harus segera diakomodasi sebelum melangkah ke revisi naskah skripsi:

1. **Bias Asimetri & Keruntuhan Prediksi Bahasa Indonesia (A1 & A2):**  
   Penurunan performa drastis pada HuatuoGPT (gap 17,27 pp, flip rate 50,81%) **bukanlah artefak penalti panjang token sequence logprob**, melainkan **keruntuhan representasi asimetris searah**. Model HuatuoGPT memprediksi `NO` (`TIDAK`) sebanyak **88,24%** pada track `seq` dan **95,37%** pada track `len`. Sebanyak **404 dari 406 flip (99,5%)** adalah pembalikan searah dari `EN=YES` menjadi `ID=NO`. Sensitivity (TPR) HuatuoGPT runtuh menjadi **19,90%** pada track `seq` dan **8,11%** pada track `len`.
2. **Kelemahan Realistis Baseline Pembanding (A3):**  
   Meskipun `LR_full` terbukti mengungguli baseline selektif internal bahasa Indonesia ($M_1$), pendekatan selektif bahasa Indonesia ini **kalah telak dibandingkan strategi naif "Jawaban EN + Conf EN"**:
   - Pada HuatuoGPT, akurasi bahasa Inggris tanpa penolakan (coverage 100%) adalah **74,84%**, sedangkan `LR_full` pada coverage 70% hanya mencapai **69,95% (atau 70,83%)**. Artinya, menolak 30% sampel pada bahasa Indonesia justru menghasilkan akurasi yang lebih rendah 4–5 pp dibanding sekadar menjawab dalam bahasa Inggris tanpa penolakan sama sekali.
   - Pada Qwen, `LR_full` pada coverage 70% (76,92% / 77,49%) hanya unggul marjinal (+1,08 hingga +1,65 pp) dibanding jawaban bahasa Inggris unabstained (75,84%), dengan mengorbankan 30% throughput klinis.
3. **Instabilitas AURC akibat Tie Heuristik Visual ($M_2$ & $M_{\text{dir}}$) (A4):**  
   Operasi clipping $\max(0, \Delta P_{\text{dir}})$ menyebabkan **80,73% sampel Huatuo** dan **41,68% sampel Qwen** bernilai sama persis ($0,0$). Hal ini menyebabkan nilai AURC konvensional (mergesort) sangat dipengaruhi oleh urutan baris data, dengan rentang variasi ekstrim (*spread best-worst*) mencapai **0,2907 pada Huatuo**.
4. **Resolusi Uji Permutasi & Koreksi Holm Keluarga 4 (A5):**  
   Klaim $p = 0,0005$ pada $B=1000$ di RESEARCH_MASTER v11 secara matematis mustahil karena batas resolusi minimum $1/(B+1) = 0,000999$. Dengan pengujian ulang $B=10.000$, keunggulan `LR_full` terhadap `LR_conf` dan `LR_visual` tetap signifikan secara statistik pada kedua model setelah koreksi Holm-Bonferroni keluarga 4 ($p_{\text{Holm}} = 0,0304$ untuk Qwen, $p_{\text{Holm}} \le 0,0009$ untuk Huatuo).

---

## 2. Diagnostik A1: Kinerja Diagnostik, Wilson CI 95%, dan Uji McNemar

Pada kohort Primary Test ($N=799$), distribusi ground-truth seimbang: `YES` = 407 (50,94%), `NO` = 392 (49,06%).

### Tabel 1: Metrik Diagnostik per Bahasa (EN vs ID) dan per Track Scoring (seq vs len)
| Model | Track | Lang | Pred-YES Rate (%) [95% Wilson CI] | Overall Acc (%) [95% Wilson CI] | TPR / Sens (%) [95% Wilson CI] | TNR / Spec (%) [95% Wilson CI] | Balanced Acc (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B** | `seq` | EN | 51,56% [48,11 – 55,01] | 75,84% [72,74 – 78,69] | 76,90% [72,56 – 80,75] | 74,74% [70,19 – 78,82] | 75,82% |
| **Qwen2.5-VL-7B** | `seq` | ID | 32,54% [29,37 – 35,88] | 62,83% [59,43 – 66,10] | 45,45% [40,67 – 50,34] | 80,87% [76,67 – 84,47] | 63,16% |
| **Qwen2.5-VL-7B** | `len` | EN | 51,56% [48,11 – 55,01] | 75,84% [72,74 – 78,69] | 76,90% [72,56 – 80,75] | 74,74% [70,19 – 78,82] | 75,82% |
| **Qwen2.5-VL-7B** | `len` | ID | 16,65% [14,21 – 19,40] | 58,45% [55,02 – 61,80] | 25,55% [21,52 – 30,07] | 92,60% [89,54 – 94,83] | 59,08% |
| **HuatuoGPT-Vision-7B** | `seq` | EN | 62,08% [58,65 – 65,39] | 74,84% [71,70 – 77,73] | 86,24% [82,52 – 89,28] | 63,01% [58,07 – 67,69] | 74,63% |
| **HuatuoGPT-Vision-7B** | `seq` | ID | **11,76%** [9,71 – 14,19] | 57,57% [54,14 – 60,94] | **19,90%** [16,27 – 24,11] | **96,68%** [94,39 – 98,06] | **58,29%** |
| **HuatuoGPT-Vision-7B** | `len` | EN | 62,08% [58,65 – 65,39] | 74,84% [71,70 – 77,73] | 86,24% [82,52 – 89,28] | 63,01% [58,07 – 67,69] | 74,63% |
| **HuatuoGPT-Vision-7B** | `len` | ID | **4,63%** [3,37 – 6,33] | 52,69% [49,23 – 56,12] | **8,11%** [5,82 – 11,19] | **98,98%** [97,40 – 99,60] | **53,54%** |

### Tabel 2: Uji McNemar Berpasangan (Pergeseran Klasifikasi EN vs ID)
| Model | Track | $b$ (EN=1, ID=0) | $c$ (EN=0, ID=1) | $a$ (Keduanya 1) | $d$ (Keduanya 0) | $\chi^2_{\text{continuity}}$ | $p$-value Eksak |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B** | `seq` | 167 | 63 | 439 | 130 | 46,13 | $4,79 \times 10^{-12}$ |
| **Qwen2.5-VL-7B** | `len` | 218 | 79 | 388 | 114 | 64,12 | $3,30 \times 10^{-16}$ |
| **HuatuoGPT-Vision-7B** | `seq` | 272 | 134 | 326 | 67 | 46,23 | $6,65 \times 10^{-12}$ |
| **HuatuoGPT-Vision-7B** | `len` | 318 | 141 | 280 | 60 | 67,49 | $8,81 \times 10^{-17}$ |

**Interpretasi A1:**  
Penurunan akurasi lintas bahasa sangat signifikan ($p < 10^{-11}$). Namun, sifat penurunannya sangat asimetris: bukan kesalahan acak dua arah, melainkan pergeseran drastis ke arah kelas negatif (`NO`/`TIDAK`), yang menurunkan TPR secara masif terutama pada model HuatuoGPT.

---

## 3. Diagnostik A2: Investigasi Artefak Panjang Token & Sifat Asimetris Flip Rate Huatuo

### 3.1 Tokenisasi Kandidat pada Tokenizer Qwen / Huatuo
Kedua model menggunakan kosakata tokenizer yang identik (keluarga Qwen2 / Qwen2.5):
- `YES<|im_end|>`: 2 token $\rightarrow$ `[14004, 151645]` (`['YES', '<|im_end|>']`)
- `NO<|im_end|>`: 2 token $\rightarrow$ `[8996, 151645]` (`['NO', '<|im_end|>']`)
- `YA<|im_end|>`: 2 token $\rightarrow$ `[86482, 151645]` (`['YA', '<|im_end|>']`)
- `TIDAK<|im_end|>`: **4 token** $\rightarrow$ `[51, 915, 11907, 151645]` (`['T', 'ID', 'AK', '<|im_end|>']`)

### 3.2 Apakah Flip Rate Huatuo (50,81%) Akibat Penalti Jumlah Token?
**Jawabannya: TIDAK.**  
Hipotesis awal menduga bahwa penjumlahkan log-probabilitas (`seq_logprob = sum(token_lps)`) akan menghukum token yang lebih panjang (`TIDAK` 4 token) karena menjumlahkan 4 nilai negatif. Jika penalti panjang token mendominasi, model seharusnya **bias ke arah `YA`** (hanya 2 token).

Namun, data empiris menunjukkan kebalikannya secara dramatis:
1. **Model justru runtuh ke arah `TIDAK`**: Pada track `seq`, HuatuoGPT memilih `TIDAK` sebanyak **705 dari 799 kali (88,24%)**.
2. **Normalisasi panjang (`len`) memperburuk bias**: Ketika log-probabilitas dinormalisasi terhadap panjang token (`mean_surprisal = seq_logprob / len`), probabilitas `TIDAK` rata-rata per token ternyata lebih tinggi dibanding `YA`, sehingga prediksi `TIDAK` meningkat menjadi **762 dari 799 kali (95,37%)**! Prediksi `YES` runtuh ke 4,63% dan akurasi anjlok ke 52,69% (mendekati tebakan acak).
3. **Asimetri Flip 100% Unidirectional**:
   - Dari 406 pembalikan prediksi (flip rate 50,81%) pada Huatuo track `seq`:
     - $\text{EN}=\text{YES} \rightarrow \text{ID}=\text{NO}$: **404 kasus (99,51% dari total flip, atau 50,56% dari seluruh populasi test set)**.
     - $\text{EN}=\text{NO} \rightarrow \text{ID}=\text{YES}$: **Hanya 2 kasus (0,49%)**.
   - Pada track `len`: 459 flip (57,45%), **seluruh 459 kasus (100%) adalah $\text{EN}=\text{YES} \rightarrow \text{ID}=\text{NO}$**.

### 3.3 Lima Contoh Konkret dari Test Set Huatuo
| `qa_id` | Pertanyaan (EN) | Pertanyaan (ID) | GT | Pred EN (`seq`) | Pred ID (`seq`) | $P(\text{YA})_{\text{seq}}$ | $P(\text{TIDAK})_{\text{seq}}$ | Pred ID (`len`) | $P(\text{YA})_{\text{len}}$ | Catatan Klinis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | are regions of the brain infarcted? | Apakah wilayah otak mengalami infark? | YES | YES | **YES** | 0,7060 | 0,2940 | **NO** | 0,4952 | Flip muncul saat dinormalisasi panjang |
| **3** | is the lesion causing significant herniation? | Apakah lesi menyebabkan herniasi signifikan? | NO | NO | **NO** | 0,1002 | 0,8998 | **NO** | 0,2167 | Konsisten NO |
| **9** | is this image in the transverse plane? | Apakah gambar ini dalam bidang transversal? | YES | YES | **YES** | 0,8071 | 0,1929 | **YES** | 0,5491 | Bertahan YES |
| **10** | are there any pulmonary findings? | Apakah ada temuan pada paru-paru? | NO | YES | **NO** | 0,3328 | 0,6672 | **NO** | 0,3416 | EN salah YES, ID balik ke NO |
| **11** | are the lungs affected? | Apakah paru-paru terpengaruh? | NO | YES | **NO** | 0,3322 | 0,6678 | **NO** | 0,3507 | EN salah YES, ID balik ke NO |

**Kesimpulan A2:**  
HuatuoGPT mengalami keruntuhan inferensi semantik total pada bahasa Indonesia: model spesialis medis ini tidak memiliki representasi istilah klinis bahasa Indonesia yang memadai, sehingga "menyerah" dan jatuh ke prior bias bahasa berupa partikel sanggahan (`TIDAK`). Meskipun bukan penalti panjang token yang menyebabkan bias ini, **kandidat rev05 (`TIDAK<|im_end|>`) memecah kata menjadi 4 sub-token kaku**, yang memperparah distorsi probabilitas posterior dibanding variasi natural bahasa (`" tidak"` / `"Tidak"`).

---

## 4. Diagnostik A3: Tabel Baseline Pembanding & Diskusi Kritis Penjawab Selektif

### Tabel 3: Evaluasi Komprehensif Baseline Pembanding pada Primary Test Set ($N=799$)
| Model | Strategi Evaluasi | Coverage (%) | Selective Accuracy (%) | Selective Risk (%) | AURC Penuh |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B** | Baseline ID Unabstained | 100,00% | 62,83% | 37,17% | - |
| | **Baseline EN Unabstained** | **100,00%** | **75,84%** | **24,16%** | - |
| | $M_1$ (Raw Conf ID) @ Cov 70% | 70,00% | 69,05% | 30,95% | 0,2422 |
| | `LR_visual` @ Cov 70% | 70,00% | 69,23% | 30,77% | 0,2434 |
| | `LR_full` @ Cov 70% | 70,00% | **76,92%** | **23,08%** | **0,2053** |
| | $D_{\text{lang}}$-only @ Cov 70% | 70,00% | 76,92% | 23,08% | 0,2607 |
| | **Gating Naif: Abstain iff $D_{\text{lang}}=1$** | **71,21%** | **77,15%** | **22,85%** | - |
| | **Strategi Superior: Jawaban EN + Conf EN** | **70,00%** | **85,15%** | **14,85%** | **0,1073** |
| **HuatuoGPT-Vision-7B** | Baseline ID Unabstained | 100,00% | 57,57% | 42,43% | - |
| | **Baseline EN Unabstained** | **100,00%** | **74,84%** | **25,16%** | - |
| | $M_1$ (Raw Conf ID) @ Cov 70% | 70,00% | 61,72% | 38,28% | 0,2884 |
| | `LR_visual` @ Cov 70% | 70,00% | 66,73% | 33,27% | 0,2513 |
| | `LR_full` @ Cov 70% | 70,00% | **69,95%** | **30,05%** | **0,2179** |
| | $D_{\text{lang}}$-only @ Cov 70% | 70,00% | 67,26% | 32,74% | 0,2626 |
| | **Gating Naif: Abstain iff $D_{\text{lang}}=1$** | **49,19%** | **82,95%** | **17,05%** | - |
| | **Strategi Superior: Jawaban EN + Conf EN** | **70,00%** | **82,47%** | **17,53%** | **0,1216** |

### Evaluasi Kritis & Demarkasi Batas Klaim (Scientific Honesty):
1. **Temuan Menyakitkan untuk HuatuoGPT:**  
   Pada HuatuoGPT, mengaplikasikan selective answering pada jawaban bahasa Indonesia menggunakan `LR_full` pada coverage 70% menghasilkan selective accuracy **69,95%** (atau 70,83% pada interpolasi continuous threshold). Angka ini **4,0–4,9 poin persentase LEBIH BURUK daripada akurasi bahasa Inggris tanpa penolakan sama sekali (74,84% pada coverage 100%)**.  
   *Implikasi Praktis:* Dalam skenario implementasi nyata di RS Indonesia, jika sistem memiliki modul penerjemah internal untuk memproses prompt bahasa Inggris, **jauh lebih baik menyajikan jawaban bahasa Inggris langsung daripada mencoba menjawab dalam bahasa Indonesia lalu menolak 30% kasus menggunakan `LR_full`**.
2. **Temuan Terbatas pada Qwen2.5-VL:**  
   Pada Qwen, `LR_full` pada coverage 70% mencapai 76,92% (77,49%), hanya mengungguli baseline bahasa Inggris tanpa penolakan (75,84%) sebesar **+1,08 pp hingga +1,65 pp**, dengan biaya membuang 30% throughput konsultasi.
3. **Dominasi Sinyal $D_{\text{lang}}$:**  
   Aturan sederhana "Abstain jika $D_{\text{lang}}=1$" mencapai coverage 71,21% dan akurasi 77,15% pada Qwen, menyamai kinerja model regresi logistik multivariat kompleks `LR_full`.

---

## 5. Diagnostik A4: Analisis Tie-Share dan Instabilitas AURC pada $M_2$ dan $M_{\text{dir}}$

Operasi pemotongan selisih probabilitas non-negatif $\max(0, \Delta P_{\text{dir}})$ pada definisi:
$$M_2 = \max(0, \Delta P_{\text{dir}}), \quad M_{\text{dir}} = \text{conf}_{\text{ID}} \times \max(0, \Delta P_{\text{dir}})$$
menyebabkan lonjakan sampel bernilai identik ($0,0$) secara masif ketika bukti visual kanvas hitam tidak menurunkan keyakinan model.

### Tabel 4: Analisis Tie-Share dan Simulasi Tie-Breaking Acak (200 Ulangan)
| Model | Estimator | Proporsi Nilai 0.0 (%) | Total Tie-Share (%) | AURC Stable (Mergesort) | AURC Random (Mean $\pm$ Std) | Rentang AURC [Best – Worst] | Spread Best-Worst |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B** | $M_1$ (`score_m1_seq`) | 0,00% | 2,50% | 0,2422 | $0,2422 \pm 0,0000$ | [0,2422 – 0,2422] | 0,0000 |
| **Qwen2.5-VL-7B** | $M_2$ (`score_m2_seq`) | **39,92%** | **41,68%** | 0,3209 | $0,3159 \pm 0,0016$ | [0,2920 – 0,3417] | **0,0497** |
| **Qwen2.5-VL-7B** | $M_{\text{dir}}$ (`score_m_dir_seq`) | **39,92%** | **41,68%** | 0,3158 | $0,3107 \pm 0,0016$ | [0,2869 – 0,3366] | **0,0497** |
| **Qwen2.5-VL-7B** | $M_{\text{norm}}$ (`score_m_norm_seq`)| 0,00% | 2,25% | 0,2753 | $0,2753 \pm 0,0000$ | [0,2753 – 0,2753] | 0,0000 |
| **HuatuoGPT-Vision**| $M_1$ (`score_m1_seq`) | 0,00% | 2,25% | 0,2884 | $0,2884 \pm 0,0000$ | [0,2884 – 0,2884] | 0,0000 |
| **HuatuoGPT-Vision**| $M_2$ (`score_m2_seq`) | **79,97%** | **80,73%** | 0,3400 | $0,3119 \pm 0,0068$ | [0,1702 – 0,4609] | **0,2907** |
| **HuatuoGPT-Vision**| $M_{\text{dir}}$ (`score_m_dir_seq`) | **79,97%** | **80,73%** | 0,3394 | $0,3113 \pm 0,0068$ | [0,1695 – 0,4602] | **0,2907** |
| **HuatuoGPT-Vision**| $M_{\text{norm}}$ (`score_m_norm_seq`)| 0,00% | 2,25% | 0,2552 | $0,2552 \pm 0,0000$ | [0,2552 – 0,2552] | 0,0000 |

**Interpretasi A4:**  
1. Pada HuatuoGPT, sebanyak **79,97% sampel memiliki $\Delta P_{\text{dir}} \le 0$**, sehingga terpotong menjadi 0,0.
2. Ketika skor memiliki banyak nilai kembar, fungsi `np.argsort(kind="mergesort")` mempertahankan urutan asli baris file. Jika baris yang kebetulan benar berada di awal blok tie, AURC tampak jauh lebih baik (best = 0,1702), sedangkan jika di akhir blok tie, AURC memburuk drastis (worst = 0,4609).
3. Ini membuktikan bahwa angka AURC $M_2$ dan $M_{\text{dir}}$ pada dokumen rev05 memiliki cacat kestabilan matematis (*tie instability*). Seluruh pembandingan pada rev06 wajib menggunakan **tie-aware random expectation AURC**.

---

## 5. Diagnostik A5: Signifikansi Statistik $B=10.000$ dan Koreksi Holm Keluarga 4

Pada dokumen RESEARCH_MASTER v11, tertulis:  
`LR-full menurunkan AURC... p = 0.0120 pada Qwen, serta p = 0.0005 pada HuatuoGPT`.  
Namun, pada uji permutasi berpasangan klaster dengan $B=1000$ dan koreksi kontinuitas Laplace $p = \frac{\text{extreme} + 1}{B + 1}$, nilai terkecil yang mungkin dicapai saat $\text{extreme} = 0$ adalah $\frac{1}{1001} \approx 0,000999$. Oleh karena itu, $p = 0,0005$ adalah anomali resolusi komputasi.

Berikut hasil uji permutasi presisi tinggi dengan **$B = 10.000$ ulangan per klaster citra** (`canonical_image_id`, 204 klaster):

### Tabel 5: Hasil Uji Permutasi Berpasangan Klaster ($B=10.000$, Holm Keluarga 4)
| Model | Perbandingan Estimator | AURC Full | AURC Baseline | $\Delta$ AURC | Ekstrem / 10.000 | $p$-value Mentah | Multiplier Holm | $p$-value Holm | Signifikansi ($\alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HuatuoGPT-7B** | `LR_full` vs `LR_conf` | 0,2179 | 0,2884 | -0,0705 | 1 | **0,000200** | 4 | **0,000800** | **Signifikan Kuat** |
| **HuatuoGPT-7B** | `LR_full` vs `LR_visual` | 0,2179 | 0,2513 | -0,0335 | 2 | **0,000300** | 3 | **0,000900** | **Signifikan Kuat** |
| **Qwen2.5-VL-7B** | `LR_full` vs `LR_visual` | 0,2053 | 0,2434 | -0,0381 | 151 | **0,015198** | 2 | **0,030397** | **Signifikan** |
| **Qwen2.5-VL-7B** | `LR_full` vs `LR_conf` | 0,2053 | 0,2422 | -0,0369 | 256 | **0,025697** | 1 | **0,030397** | **Signifikan** |

**Interpretasi A5:**  
1. Keunggulan `LR_full` terhadap baseline keyakinan mentah (`LR_conf`) dan baseline visual (`LR_visual`) **tetap terbukti signifikan secara statistik** pada kedua model setelah koreksi Holm-Bonferroni keluarga 4.
2. Pada Qwen, $p$-value mentah adalah $0,0152$ dan $0,0257$, yang setelah koreksi Holm keluarga 4 menjadi **$p_{\text{Holm}} = 0,0304$** (bukan 0,0120).
3. Pada Huatuo, resolusi $B=10.000$ membuktikan $p_{\text{raw}} = 0,00020$ dan $0,00030$, menghasilkan $p_{\text{Holm}} = 0,0008$ dan $0,0009$, mengonfirmasi signifikansi sangat kuat ($p < 0,001$) secara sah.

---

## 6. Adjudikasi Klaim Dokumen: Mana yang Bertahan, Mana yang Harus Dilunakkan

| Pernyataan / Klaim dalam Dokumen v11 | Status Verifikasi Kode | Rekomendasi Revisi untuk Dokumen & Skripsi |
| :--- | :--- | :--- |
| **"Pergeseran bahasa memicu penurunan akurasi tajam (13,02–17,27 pp) dan laju pembalikan prediksi 28,79%–50,81%"** | **TERBUKTI PENUH** (Tabel 1 & 2) | Pertahankan. Uji McNemar membuktikan pergeseran sangat signifikan ($p < 10^{-11}$). |
| **"Ketidakstabilan representasi multimodal memicu keruntuhan kepercayaan"** | **HARUS DILUNAKKAN** (Tabel 1 & Sec A2) | Ubah narasi menjadi **"Keruntuhan prediksi asimetris searah (unidirectional prediction collapse)"**. Pada Huatuo, model tidak berhalusinasi acak, melainkan runtuh ke kelas `TIDAK` (99,5% flip adalah YES $\rightarrow$ NO). |
| **"LR-full secara signifikan menurunkan AURC menjadi 0,2053 (Qwen) dan 0,2179 (Huatuo)"** | **TERBUKTI PENUH** (Tabel 5) | Pertahankan nilai AURC. Perbaiki nilai $p$-value menjadi $p_{\text{Holm}} = 0,0304$ (Qwen) dan $p_{\text{Holm}} \le 0,0009$ (Huatuo). |
| **"Estimasi risiko lintas-bahasa adalah solusi terbaik untuk implementasi klinis VLM di Indonesia"** | **HARUS DIBATASI / DILUNAKKAN KERAS** (Tabel 3) | **Wajib diakui secara jujur:** Pada model spesialis Huatuo, selective answering pada bahasa Indonesia (akurasi 69,95% @ cov 70%) **kalah dari sekadar memakai jawaban bahasa Inggris tanpa abstain (74,84% @ cov 100%)**. Peneliti harus menyatakan bahwa selective answering bahasa lokal hanya relevan bila bahasa Indonesia adalah kendala interaksi mutlak bagi pengguna akhir. |
| **"Aturan heuristik visual $M_2$ dan $M_{\text{dir}}$ memiliki nilai AURC stabil"** | **DITOLAK / INVALID** (Tabel 4) | Akui dalam Limitations bahwa $M_2$ dan $M_{\text{dir}}$ menderita *tie-instability* parah (80% sampel tied pada nilai 0), sehingga AURC mergesort rentan terhadap permutasi baris dataset. |
