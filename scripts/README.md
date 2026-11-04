# 📜 Skrip Otomasi, Inferensi, & Validasi Ilmiah (`scripts/`)

Direktori ini memuat seluruh skrip CLI eksekusi inferensi, verifikasi logit kausal, evaluasi statistik, dan otomasi pipeline penelitian:
> **"Visual Sensitivity–Aware Selective Prediction for Bilingual Medical Vision-Language Models under English–Indonesian Language Shift"**

---

## 🗂 Taksonomi Skrip

### 1. Engine Inferensi & Prediksi Selektif Produksi
| Skrip | Fungsi Utama | Cara Penggunaan |
| :--- | :--- | :--- |
| [**`run_experiment.py`**](run_experiment.py) | **CLI Runner Terpadu:** Entry point utama pengujian inferensi dan dry-run lokal pada CPU/GPU. | `python scripts/run_experiment.py --dry-run` |
| [**`run_decisive_validation.py`**](run_decisive_validation.py) | **Mesin Inferensi Utama rev05:** Evaluasi zero-shot bilingual, sequence log-likelihood scoring, intervensi kanvas hitam ($P_1$), dan bootstrapping klaster. | `python scripts/run_decisive_validation.py --help` |
| [**`adaptive_abstention_engine.py`**](adaptive_abstention_engine.py) | **Engine Kalibrasi Selektif:** Estimasi threshold operasional OOF GroupKFold ($K=5$) pada Dev Set untuk optimasi kurva Risk-Coverage. | Digunakan oleh notebook & runner |
| [**`visual_perturbation.py`**](visual_perturbation.py) | **Generator Kanvas Hitam:** Utilitas pembuatan citra kontrol kanvas hitam murni ($P_1$) berdimensi identik dengan citra radiologi fisik ($P_0$). | `python scripts/visual_perturbation.py` |

### 2. Diagnostik Ilmiah & Pengujian Validitas
| Skrip | Fungsi Utama | Cara Penggunaan |
| :--- | :--- | :--- |
| [**`diagnostics_suite.py`**](diagnostics_suite.py) | **Suite Diagnostik Lengkap:** Analisis bias asimetri bahasa, Wilson score CI, uji McNemar, dan kalibrasi uji permutasi ($B=10.000$). | Diimpor oleh runner diagnostik |
| [**`run_task_a_diagnostics.py`**](run_task_a_diagnostics.py) | **Runner Diagnostik:** Eksekutor diagnostik A1–A5 dan penghasil laporan [`docs/audit/DIAGNOSTICS_REPORT.md`](../docs/audit/DIAGNOSTICS_REPORT.md). | `python scripts/run_task_a_diagnostics.py` |
| [**`audit_bivis_implementation.py`**](audit_bivis_implementation.py) | **Audit Implementasi Bi-ViS:** Verifikasi keselarasan formula matematis, urutan fitur, dan replikasi numerik $0.00\text{e}+00$. | `python scripts/audit_bivis_implementation.py` |
| [**`verify_ast.py`**](verify_ast.py) | **Verifikator Sintaks Notebook:** Pengujian validitas parser AST (`ast.parse`) untuk seluruh sel kode di semua notebook. | `python scripts/verify_ast.py` |
| [**`verify_restructuring.py`**](verify_restructuring.py) | **Verifikator Integritas Repositori:** Audit konsistensi path dan impor pasca-restrukturisasi. | `python scripts/verify_restructuring.py` |

### 3. Evaluasi Statistik & Pembuatan Laporan Ilmiah
| Skrip | Fungsi Utama |
| :--- | :--- |
| [**`audit_rev05_deep.py`**](audit_rev05_deep.py) | Audit empiris mendalam rev05: rekalkulasi metrik RQ1–RQ3, Wilson CI, dan plot Risk-Coverage. |
| [**`audit_second_stage_statistical.py`**](audit_second_stage_statistical.py) | Audit inferensi statistik tahap kedua: paired cluster permutation, paired bootstrap, VIF, dan koreksi Holm. |
| [**`build_final_readiness_documents.py`**](build_final_readiness_documents.py) | Generator dokumen kesiapan riset final dan matriks pembuktian klaim. |
| [**`build_phase3_manuscript_foundation.py`**](build_phase3_manuscript_foundation.py) | Generator fondasi naskah Phase 3 (narasi penelitian & blueprint). |
| [**`evaluate_rev05_huatuo_audit.py`**](evaluate_rev05_huatuo_audit.py) | Audit terfokus kelemahan representasi bahasa lokal model HuatuoGPT-Vision-7B. |
| [**`verify_links.py`**](verify_links.py) | Verifikasi otomatis keutuhan seluruh tautan relatif markdown di repositori. |

### 4. Manajemen Manifest & Utilitas Cloud GPU
| Skrip | Fungsi Utama |
| :--- | :--- |
| [**`audit_and_generate_rev05_manifest.py`**](audit_and_generate_rev05_manifest.py) | Generator manifest resmi `data/manifests/manifest_rev05.csv` berbasis klaster hash piksel. |
| [**`manifest_rev05_b64.py`**](manifest_rev05_b64.py) | Representasi base64 manifest rev05 untuk injeksi langsung ke cloud notebook tanpa dependensi eksternal. |
| [**`generate_resilient_notebooks.py`**](generate_resilient_notebooks.py) | Generator notebook Kaggle yang tahan gangguan jaringan (*network-resilient*). |
| [**`export_pristine_rev05_notebooks.py`**](export_pristine_rev05_notebooks.py) | Utilitas ekspor notebook versi beku (*frozen baseline*) dari git tag. |
| [**`run_notebook_10.py`**](run_notebook_10.py) | Eksekutor headless notebook agregasi evaluasi melalui `nbclient`. |
