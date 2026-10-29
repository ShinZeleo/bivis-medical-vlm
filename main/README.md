# IndoRad-VQA: Bilingual Semantic Consistency & Counterfactual Visual Sensitivity in Medical VLMs

**Research Domain:** Medical Vision-Language Models (Med-VLM), Radiology VQA, Cross-Lingual Robustness  
**Languages:** English (EN) & Indonesian (ID)  
**Dataset:** `Lab-IS/IndoRad-VQA` (Hugging Face Datasets)  
**Core Strategy:** QLoRA + Cross-Lingual Semantic Consistency Regularization ($\mathcal{L}_{total} = \mathcal{L}_{VQA} + \lambda_s \mathcal{L}_{semantic}$)  
**Behavioral Evaluation:** Counterfactual Visual Sensitivity ($P_0$ Original vs $P_1$ Blank vs $P_2$)  
**Experimental Design:** $2 \times 3$ Factorial Design (Model Capability $\times$ Training Condition B0/B1/B2)

---

## 🎯 Research Gap & Posisi Literatur (Update Oktober 2026)
* **Precedents:** 
  * Yudhistira, Malik, & Yudistira (2026, arXiv:2606.03693) telah mengukur degradasi zero-shot pada IndoRad-VQA.
  * Sadanandan & Behzadan (2026, CHIL) telah menguji konsistensi parafrasa medis satu bahasa via LoRA.
* **Our Core Contribution (Gap C + Gap D):**
  * **Gap C:** Menguji apakah **bilingual semantic consistency training khusus pasangan bahasa Inggris–Indonesia** memberikan manfaat performa dan konsistensi tambahan melampaui standard supervised VQA fine-tuning ($B_2 - B_1$).
  * **Gap D:** Memastikan peningkatan konsistensi lintas bahasa mempertahankan akurasi klinis dan sensitivitas bukti visual tanpa mengalami mode collapse.

---

## 📁 Struktur Repositori
* `notebooks/`:
  * `01_indorad_vqa_dataset_audit_eda.ipynb`: Audit dataset dan eliminasi kebocoran hash gambar.
  * `02_model_pilot_experiments.ipynb`: Evaluasi pilot lokal.
  * `03_colab_master.ipynb`: Master pipeline eksperimen Google Colab (GPU T4/L4/A100 + Google Drive).
  * `03_kaggle_master.ipynb`: Master pipeline eksperimen Kaggle (Dual GPU T4 x 2).
* `src/`: Modul loader, augmentasi citra, kalkulasi metrik, dan validasi translasi.
* `reports/`: Laporan audit data leakage, protokol validasi, dan verifikasi empiris.
* `results/`: CSV prediksi mentah, tabel metrik, dan visualisasi intervensi.

---

## 🚀 Panduan Menjalankan Eksperimen
1. Jalankan `notebooks/03_kaggle_master.ipynb` di Kaggle (dengan Accelerator **GPU T4 x 2** dan **Internet ON**).
2. Notebook akan mengeksekusi B0 baseline, B1 supervised QLoRA, B2 semantic consistency QLoRA, serta mencetak tabel metrik dan kurva intervensi secara otomatis.
