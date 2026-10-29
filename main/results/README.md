# 📊 Panduan Hasil & Artefak Evaluasi (`main/results/`)

Direktori ini memuat seluruh data keluaran hasil inferensi, tabel metrik terkomputasi, visualisasi kurva, dan catatan verifikasi kriptografis.

---

## 🏆 Hasil Resmi & Otoritatif: `decisive_validation/rev05/`

Seluruh angka yang dilaporkan dalam skripsi mengacu secara eksklusif pada folder **`decisive_validation/rev05/`**:

| File Hasil | Format | Isi & Makna Klinis |
| :--- | :---: | :--- |
| **`rq1_bilingual_metrics.csv`** | CSV | Metrik reliabilitas bilingual (Akurasi EN, Akurasi ID, Kesenjangan Bahasa, *Flip Rate*, dan *Joint Accuracy*) untuk model Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B pada Primary Test Set ($N=799$). |
| **`rq2_visual_intervention_metrics.csv`** | CSV | Metrik responsivitas intervensi visual pada kondisi $P_0$ (citra asli) vs $P_1$ (kanvas hitam polos) serta nilai pergeseran probabilitas terarah ($\Delta P_{\text{dir}}$). |
| **`rq3_selective_prediction_primary_test.csv`** | CSV | Metrik prediksi selektif komparatif: Nilai **AURC**, selisih $\Delta\text{AURC}$, *confidence interval* 95% bootstrap ($B=2000$), nilai p-value permutasi terklaster pasangannya (*paired cluster permutation test*), serta koreksi Holm-Bonferroni. |
| **`risk_ablation_dev_test.csv`** | CSV | Studi ablasi regresi logistik teratur pada Development Set ($N=323$) dan evaluasi pada Test Set ($N=799$): membuktikan signifikansi kontribusi fitur `LR_full` ($M_1$ + $\Delta P_{\text{dir}}$ + $D_{\text{lang}}$). |
| **`risk_coverage_primary_test_*.png`** | PNG | Grafik kurva *Risk-Coverage* resolusi tinggi untuk masing-masing model, membandingkan kurva baseline ($M_1$) terhadap estimator sensitivitas visual. |
| **`manifest_rev05.csv`** | CSV | Manifest evaluasi resmi rev05 memuat 1.133 pasangan QA dengan pemisahan terkelompok berbasis hash piksel citra unik (Primary Test $N=799$, Dev $N=323$). |
| **`artifact_sha256_manifest.csv`** | CSV | Manifest checksum SHA-256 seluruh file luaran untuk menjamin tidak adanya manipulasi atau korupsi data paska-inferensi. |

---

## 📁 Riwayat Revisi Sebelumnya
- `rev02/`, `rev03/`, `rev04/`: Hasil iterasi evaluasi awal sebelum penambahan sinyal ketidaksepakatan lintas-bahasa ($D_{\text{lang}}$) dan penyesuaian target sequences log-likelihood.
