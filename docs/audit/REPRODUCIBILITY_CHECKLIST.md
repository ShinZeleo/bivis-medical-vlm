# 🔄 Deep Learning Reproducibility & Replication Checklist (v10.0)
**Thesis Project:** Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift  
**Auditor:** Senior Research Auditor, Machine Learning Engineer & Quantitative Methodologist  
**Target:** Reproducibility Standard for Peer-Review / Master/Bachelor Thesis Defense  
**Status:** `FULLY_REPRODUCIBLE` / `SELF_CONTAINED`

---

## 1. Specification of Computational Environment

| Komponen Sistem | Spesifikasi Resmi yang Digunakan | Rekomendasi Replikasi Minimum |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 / Linux Ubuntu 22.04 LTS | Linux x86_64 atau Windows 10/11 64-bit |
| **Python Version** | Python 3.10.x | Python 3.10+ (atau Python 3.9+) |
| **Deep Learning Framework** | `torch==2.1.2+cu121`, `torchvision==0.16.2` | PyTorch 2.1.0+ dengan dukungan CUDA |
| **LLM/VLM Core Libraries** | `transformers>=4.45.0`, `accelerate>=0.28.0` | Transformers 4.45.0+, Accelerate 0.28.0+ |
| **Multimodal Vision Utility** | `qwen-vl-utils>=0.0.8` | `qwen-vl-utils>=0.0.8` (Kritis untuk patch grid Qwen) |
| **Quantization Framework** | `bitsandbytes>=0.42.0` (4-bit NF4) | `bitsandbytes` (Linux) atau `bitsandbytes-windows` |
| **Data Processing & ML** | `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.2.0` | Versi standar pada PyPI |
| **Hardware Akselerasi (Cloud)** | **Kaggle Dual Tesla T4** ($2 \times 16\text{ GB}$ VRAM) | Minimal 1x GPU 16 GB VRAM atau Dual-T4 |
| **Hardware Verifikasi Lokal** | Laptop / PC CPU Standar (tanpa GPU diskrit) | Intel Core i5 / AMD Ryzen 5, 8 GB RAM |
| **Random Seed Global** | `RANDOM_SEED = 42` | Ditetapkan konstan di seluruh skrip |

---

## 2. Tingkat Replikasi (Tiers of Reproducibility)

Sesuai panduan efisiensi komputasi, kami membedakan 3 tingkatan replikasi:

### 🟢 Tier 1: Verifikasi Preflight & Unit Test (Waktu: ~7 Detik, Hardware: CPU Standar)
Tidak memerlukan GPU atau download model berukuran gigabyte. Memverifikasi seluruh logika matematika, slicing kausal Transformer, dan integritas kriptografis data:
```bash
python main/scripts/test_decisive_validation_v5.py
```
*Hasil yang Diharapkan:*  
9/9 pengujian lolos 100% dengan status `ALL 9 PREFLIGHT TESTS PASSED (PASS_UNIT_TEST)`.

### 🟡 Tier 2: Rekomputasi Metrik, Model Risiko, & Signifikansi Statistik (Waktu: ~1 Menit, Hardware: CPU Standar)
Tidak memerlukan inferensi ulang forward-pass neural network. Membaca berkas checkpoint mentah resmi (`live_validation_checkpoint_*.csv`), melatih model risiko terpelajar pada Dev Set, dan menghitung ulang Discrete Riemann AURC serta uji permutasi klaster:
```bash
python -c "
import pandas as pd, numpy as np
from main.scripts.decisive_validation_experiment_v5 import compute_discrete_risk_coverage
qwen_df = pd.read_csv('main/results/decisive_validation/rev05/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv')
p_test = qwen_df[(qwen_df['split']=='test') & (qwen_df['is_eligible_primary']==True)]
_, _, aurc_m1 = compute_discrete_risk_coverage(p_test['score_m1_seq'].values, p_test['correct_id_seq'].values)
_, _, aurc_mnorm = compute_discrete_risk_coverage(p_test['score_m_norm_seq'].values, p_test['correct_id_seq'].values)
print(f'Recomputed Qwen M1 AURC: {aurc_m1:.6f}')
print(f'Recomputed Qwen M_norm AURC: {aurc_mnorm:.6f}')
"
```
*Hasil yang Diharapkan:*  
Menghasilkan angka `0.242187` dan `0.275315` persis sama dengan tabel resmi.

### 🔴 Tier 3: Inferensi Skala Penuh dari Nol (Waktu: ~45 Menit, Hardware: Cloud GPU Dual Tesla T4)
Mengevaluasi 1.133 sampel dari awal menggunakan model bobot neural riil:
1. Buka [Kaggle](https://www.kaggle.com/) dan buat Notebook baru.
2. Unggah berkas `main/notebooks/decisive_validation_primary_rev05.ipynb` atau script `main/scripts/decisive_validation_experiment_v5.py`.
3. Pasang Settings: **Accelerator = GPU T4 x 2**, **Internet = ON**.
4. Eksekusi "Run All".
5. Notebook akan memuat model 7B dalam format NF4, mengevaluasi kondisi $P_0$ (EN & ID) dan $P_1$ (ID Blank), menyimpan checkpoint setiap 10 sampel, dan menghasilkan seluruh tabel artefak.

---

## 3. Checklist Protokol Reproduktibilitas

| No | Parameter Kritis | Status | Verifikasi & Bukti |
| :---: | :--- | :---: | :--- |
| 1 | **Ketersediaan Kode Sumber Lengkap** | ✅ LENGKAP | Seluruh skrip tersimpan di `main/scripts/` dan modul di `main/src/`. |
| 2 | **Manifest Data Tertutup & Terkunci** | ✅ TERKUNCI | SHA-256 byte mentah `fb55fd75...` diverifikasi pada setiap inisialisasi. |
| 3 | **Ketersediaan Citra Resolusi Asli** | ✅ TERVERIFIKASI | 314 citra PNG tersimpan di `main/data/images/` dan cocok dengan hash piksel. |
| 4 | **Pencegahan Token Slippage** | ✅ TERVERIFIKASI | Penegasan tensor `torch.equal` memastikan prompt multimodal tidak terdistorsi. |
| 5 | **Slicing Causal Logit Eksak** | ✅ TERVERIFIKASI | Logit kausal diambil tepat pada $[L_p-1 : L_p-1+L_t]$. |
| 6 | **Pencegahan Kebocoran Split** | ✅ 0.0% LEAKAGE | Pemisahan GroupKFold dan Test-Dev terisolasi pada unit klaster citra fisik. |
| 7 | **Penyelarasan Arah Skor Reliabilitas** | ✅ KONSISTEN | $S_{\text{LR}} = 1.0 - \hat{P}(\text{error}=1)$ memastikan nilai tinggi = risiko rendah. |
| 8 | **Pengujian Statistik Berpasangan** | ✅ NON-PARAMETRIK | Resampling klaster citra ($B=1.000$) dan koreksi Holm-Bonferroni terotomasi. |
| 9 | **Daftar Dependensi Python** | ✅ TERKUNCI | Berkas `requirements.txt` tersedia di root dan subdirektori `main/`. |
| 10| **Integritas Checkpoint Atomik** | ✅ ATOMIK | Penulisan berkas temporer `.tmp` mencegah korupsi checkpoint saat interupsi. |
