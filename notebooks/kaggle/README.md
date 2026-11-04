# 🚀 Panduan Eksekusi Notebook Kaggle (`notebooks/kaggle/`)

Direktori ini memuat notebook Jupyter resmi yang dirancang khusus untuk dieksekusi pada lingkungan cloud GPU **Kaggle Dual Tesla T4 (2x15 GB VRAM)**.

---

## ⚙️ Petunjuk Setup di Kaggle

1. **Buat Notebook Baru** di [Kaggle](https://www.kaggle.com/).
2. **File $\rightarrow$ Import Notebook**, lalu unggah salah satu berkas:
   - `08_qwen_decisive_validation.ipynb` untuk Qwen2.5-VL-7B.
   - `09_huatuo_decisive_validation.ipynb` untuk HuatuoGPT-Vision-7B.
   - `10_standalone_aggregation_evaluation.ipynb` untuk kalkulasi metrik & kurva AURC.
3. **Pengaturan Sesi Kaggle:**
   - **Accelerator:** GPU T4 x 2
   - **Environment:** Always use latest environment
   - **Internet:** ON (diperlukan untuk mengunduh bobot model dari HuggingFace)
4. **Jalankan Seluruh Sel:**
   Notebook telah dilengkapi mekanisme *in-flight checkpoint recovery* otomatis sehingga jika terjadi timeout atau disconnect, eksekusi dapat dilanjutkan tanpa kehilangan hasil parsial.

---

## 📑 Struktur & Versi Notebook

| Berkas | Versi Protokol | Deskripsi |
| :--- | :---: | :--- |
| [**`08_qwen_decisive_validation.ipynb`**](08_qwen_decisive_validation.ipynb) | Active | Runner utama inferensi Qwen2.5-VL-7B (kombinasi P0 dan P1). |
| [**`09_huatuo_decisive_validation.ipynb`**](09_huatuo_decisive_validation.ipynb) | Active | Runner utama inferensi HuatuoGPT-Vision-7B (kombinasi P0 dan P1). |
| [**`10_standalone_aggregation_evaluation.ipynb`**](10_standalone_aggregation_evaluation.ipynb) | Active | Evaluasi metrik terpadu, Wilson CI, uji McNemar, dan kalibrasi selektif. |
| [**`rev05/`**](rev05/) | rev05 (Frozen) | Salinan arsip beku tepat sesuai eksekusi historis awal rev05. |
