# 📓 Panduan Notebook Interaktif (`notebooks/`)

Direktori ini memuat seluruh Jupyter Notebook untuk audit integritas data eksploratif (EDA) dan alur kerja evaluasi model pada cloud GPU (Kaggle Dual-T4).

---

## 🗂 Struktur Direktori

```text
notebooks/
├── 01_dataset_audit_eda.ipynb              # Audit dataset lokal, hash SHA-256, & zero-leakage split
│
└── kaggle/                                 # Alur kerja inferensi resmi di Kaggle Dual-T4
    ├── README.md                           # Panduan eksekusi & spesifikasi lingkungan Kaggle
    ├── rev06/                              # Notebook eksekusi aktif protokol rev06
    │   ├── 08_qwen_decisive_validation.ipynb   # Inferensi aktif Qwen2.5-VL-7B (P0 & P1)
    │   ├── 09_huatuo_decisive_validation.ipynb # Inferensi aktif HuatuoGPT-Vision-7B (P0 & P1)
    │   └── 10_standalone_aggregation_evaluation.ipynb # Agregasi metrik, bootstrap, & kurva AURC
    ├── rev05_patched/                      # Salinan rev05 dengan patch bug library/sintaks non-pengubah-hasil
    └── rev05/                              # Salinan arsip beku byte-identik protokol rev05 awal
```

---

## 📑 Rincian Notebook

### 1. Audit Dataset Lokal
- [**`01_dataset_audit_eda.ipynb`**](01_dataset_audit_eda.ipynb):  
  Melakukan hashing piksel SHA-256 pada seluruh 2.244 baris data IndoRad-VQA, menemukan 314 citra radiologi unik, dan memverifikasi pembagian *image-level split* bebas kebocoran data (**0.00% overlap antara 88 klaster dev dan 204 klaster test**).

### 2. Evaluasi Cloud GPU ([`kaggle/`](kaggle/README.md))
- [**`rev06/08_qwen_decisive_validation.ipynb`**](kaggle/rev06/08_qwen_decisive_validation.ipynb):  
  Mengeksekusi inferensi zero-shot bilingual Qwen2.5-VL-7B pada kondisi citra asli ($P_0$) dan kanvas hitam ($P_1$) dengan auto-recovery checkpoint.
- [**`rev06/09_huatuo_decisive_validation.ipynb`**](kaggle/rev06/09_huatuo_decisive_validation.ipynb):  
  Mengeksekusi inferensi zero-shot bilingual HuatuoGPT-Vision-7B pada kondisi $P_0$ dan $P_1$.
- [**`rev06/10_standalone_aggregation_evaluation.ipynb`**](kaggle/rev06/10_standalone_aggregation_evaluation.ipynb):  
  Mengagregasi file checkpoint inferensi, menghitung metrik RQ1 (akurasi & flip rate), RQ2 (sensitivitas kanvas hitam), RQ3 (optimasi AURC & perankingan Bi-ViS), serta menghasilkan grafik kurva Risk-Coverage resolusi tinggi.

---

> [!NOTE]
> Notebook dari fase eksplorasi awal (02 s.d. 07) telah diarsipkan secara rapi di:  
> [`artifacts/experiments/decisive_validation/historical/notebooks_archive/`](../artifacts/experiments/decisive_validation/historical/notebooks_archive/README.md).
