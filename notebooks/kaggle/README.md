# 🚀 Panduan Eksekusi Notebook Kaggle (`notebooks/kaggle/`)

Direktori ini memuat notebook Jupyter resmi yang dirancang khusus untuk dieksekusi pada lingkungan cloud GPU **Kaggle Dual Tesla T4 (2x15 GB VRAM)**.

---

## ⚙️ Petunjuk Setup di Kaggle

1. **Buat Notebook Baru** di [Kaggle](https://www.kaggle.com/).
2. **File $\rightarrow$ Import Notebook**, lalu unggah berkas dari folder protokol yang sesuai:
   - `rev06/08_qwen_decisive_validation.ipynb` untuk Qwen2.5-VL-7B.
   - `rev06/09_huatuo_decisive_validation.ipynb` untuk HuatuoGPT-Vision-7B.
   - `rev06/10_standalone_aggregation_evaluation.ipynb` untuk kalkulasi metrik & kurva AURC.
3. **Pengaturan Sesi Kaggle:**
   - **Accelerator:** GPU T4 x 2
   - **Environment:** Always use latest environment
   - **Internet:** ON (diperlukan untuk mengunduh bobot model dari HuggingFace)
4. **Jalankan Seluruh Sel:**
   Notebook telah dilengkapi mekanisme *in-flight checkpoint recovery* otomatis sehingga jika terjadi timeout atau disconnect, eksekusi dapat dilanjutkan tanpa kehilangan hasil parsial.

---

## 📑 Struktur & Versi Notebook

| Subdirektori / Berkas | Versi Protokol | Deskripsi |
| :--- | :---: | :--- |
| [**`rev06/`**](rev06/) | **rev06 (Active)** | Notebook eksekusi aktif protokol rev06 (gate check, modul modular `src/metrics.py`, sentralisasi statistik). |
| [**`rev05_patched/`**](rev05_patched/) | **rev05 (Patched)** | Salinan rev05 dengan patch bug library/sintaks non-pengubah-hasil (isolasi bugfix). |
| [**`rev05/`**](rev05/) | **rev05 (Frozen)** | Salinan arsip beku byte-identik sesuai eksekusi historis awal rev05. |
