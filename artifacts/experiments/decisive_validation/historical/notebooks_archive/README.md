# 📦 Historical Notebooks Archive (`notebooks_archive/`)

Direktori ini menyimpan seluruh Jupyter Notebook dari fase iterasi dan eksperimen terdahulu (protokol skrining, pilot, rev02, rev03, dan rev04) sebelum repositori dibekukan pada protokol final **rev05**:

---

## 📑 Inventaris Notebook Historis (02–07)

| Berkas | Ukuran | Fase Eksperimen & Deskripsi |
|---|:---:|---|
| [**`02_model_pilot_experiments.ipynb`**](02_model_pilot_experiments.ipynb) | ~88 KB | **Fase Skrining Awal (Pilot):** Skrining 8 model fondasi multimodal (Med-VLM & General VLM) pada IndoRad-VQA. |
| [**`03_decisive_validation.ipynb`**](03_decisive_validation.ipynb) | ~316 KB | **Prototipe Awal Validasi:** Prototipe skrip pengujian inferensi komparatif Qwen vs Huatuo. |
| [**`04_decisive_validation_live.ipynb`**](04_decisive_validation_live.ipynb) | ~528 KB | **Fase Live Eksekusi Kaggle:** Uji coba eksekusi online dengan GPU Dual-T4. |
| [**`05_decisive_validation_revised.ipynb`**](05_decisive_validation_revised.ipynb) | ~490 KB | **Fase Revisi Format Output:** Penyesuaian skema output probabilitas dan logit kausal. |
| [**`06_decisive_validation_rev03.ipynb`**](06_decisive_validation_rev03.ipynb) | ~74 KB | **Protokol rev03:** Evaluasi pendahuluan intervensi citra kontrol kanvas hitam (superseded). |
| [**`07_decisive_validation_rev04.ipynb`**](07_decisive_validation_rev04.ipynb) | ~305 KB | **Protokol rev04:** Pengujian kohort sebelum isolasi zero cluster leakage rev05 (superseded). |

---

## 🚀 Notebook Aktif Resmi (rev05 Final)

Untuk eksperimen aktif dan reproduksi hasil skripsi utama, gunakan notebook yang berada di direktori kerja resmi:
- [`notebooks/01_dataset_audit_eda.ipynb`](file:///c:/.PUSAT/Project/Kuliah/VLM/notebooks/01_dataset_audit_eda.ipynb): Audit integritas 1.122 pasangan data QA dan SHA-256 citra radiologi.
- [`notebooks/kaggle/08_qwen_decisive_validation.ipynb`](file:///c:/.PUSAT/Project/Kuliah/VLM/notebooks/kaggle/08_qwen_decisive_validation.ipynb): Inferensi resmi Qwen2.5-VL-7B (Kaggle Dual-T4).
- [`notebooks/kaggle/09_huatuo_decisive_validation.ipynb`](file:///c:/.PUSAT/Project/Kuliah/VLM/notebooks/kaggle/09_huatuo_decisive_validation.ipynb): Inferensi resmi HuatuoGPT-Vision-7B (Kaggle Dual-T4).
- [`notebooks/kaggle/10_standalone_aggregation_evaluation.ipynb`](file:///c:/.PUSAT/Project/Kuliah/VLM/notebooks/kaggle/10_standalone_aggregation_evaluation.ipynb): Evaluasi metrik gabungan, bootstrapping, dan perankingan selective prediction.
