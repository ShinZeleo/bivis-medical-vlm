# 📓 Panduan Notebook Interaktif (`main/notebooks/`)

Direktori ini memuat seluruh notebook Jupyter yang digunakan untuk analisis data eksploratif (EDA), audit integritas data, dan eksekusi evaluasi pada cloud GPU (Kaggle / Google Colab).

---

## 🟢 Notebook Utama (Siap Eksekusi)

| File Notebook | Lingkungan | Deskripsi |
| :--- | :---: | :--- |
| **`decisive_validation_primary_rev05.ipynb`** | **Kaggle Dual T4 / Colab** | **Notebook Eksperimen Utama (`rev05`)**: Mengeksekusi inferensi bilingual zero-shot Qwen2.5-VL-7B dan HuatuoGPT-Vision-7B pada kondisi $P_0$ (asli) dan $P_1$ (kanvas hitam), menghasilkan tabel metrik dan kurva risk-coverage. |
| **`01_indorad_vqa_dataset_audit_eda.ipynb`** | Lokal / Colab | **Audit Dataset & Eliminasi Kebocoran**: Melakukan hashing piksel SHA-256 pada seluruh 2.244 baris IndoRad-VQA, menemukan 314 citra unik, dan membagi dataset secara *image-level* agar nol kebocoran antar split. |
| **`02_model_pilot_experiments.ipynb`** | GPU Lokal | **Eksperimen Pilot Awal**: Uji kelayakan pemuatan model quantized (4-bit NF4) dan evaluasi sampel awal. |

---

## 📁 Subdirektori Notebook
- **`kaggle/`**: Notebook terfragmentasi untuk eksekusi terdistribusi per model (misal: `09_infer_huatuo_resume.ipynb`, `10_standalone_aggregation_evaluation.ipynb`).
- **`archive/`**: Notebook versi historis (`03_colab_master.ipynb`, `04_decisive_validation_live.ipynb`).
