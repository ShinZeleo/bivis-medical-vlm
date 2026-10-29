# Consolidated Report 1: IndoRad-VQA Dataset, Split Integrity, and Baseline Model Audit

**Document:** `reports/1_DATASET_AND_BASELINE_AUDIT.md`  
**Consolidates:** `dataset_audit.md`, `final_image_split_audit.md`, `final_split_balance_report.md`, `dataset_final_status.md`, `translation_manual_validation_protocol.md`, `baseline_reproduction.md`, `strong_model_validation_report.md`, `model_capability_comparison.md`, `screening_vs_validation_consistency_audit.md`, `evidence_integrity_audit.md`  
**Status:** Unified Authoritative Audit Document  

---

## 1. Executive Summary

This report establishes the verified empirical foundations of the IndoRad-VQA dataset, evaluates cross-split leakage at the pixel level, validates translation fidelity across English and Indonesian pairs, and benchmarks baseline Vision-Language Models (VLMs) under standardized prompting protocols.

---

## 2. Dataset Architecture & Manifest Audit

### 2.1 Dataset Identity & Schema
* **Source:** HuggingFace `Lab-IS/IndoRad-VQA` (derived from clinical radiology archives, primarily chest X-rays and abdominal/brain CT scans).
* **Total QA Pairs:** 2,244 question-answer rows.
* **Format:** Parquet containing embedded image bytes, English questions/answers, Indonesian questions/answers, question types, and metadata.
* **Null Value Audit:** Columns `question_type`, `image_organ`, and `phrase_type` are 100% null in the raw parquet and were reconstructed via downstream heuristic rules.

### 2.2 Physical Image De-duplication & Cluster Analysis
* **Unique Physical Images:** Across the entire 2,244 rows, pixel-level SHA-256 hashing (`pixel_sha256`) reveals exactly **314 unique physical images** ([`image_identity_manifest.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/data/manifests/image_identity_manifest.csv)).
* **QA Clustering:** Multiple questions are nested within each image. Across the dataset, the number of QA pairs per image ranges from 1 to 14 (mean: 7.15 QA pairs per image).
* **Evaluation Subset ($N = 150$):**
  * Evaluates 150 bilingual pairs.
  * Mapped via `canonical_image_id` to **58 unique physical images**.
  * Questions per image in evaluation subset: mean = 2.59 (range: 1 to 8; cluster `6185dbde43bcbc0c` has 8 questions).
  * **Statistical Requirement:** All paired evaluations must use **image-cluster resampling** to avoid violating independence assumptions.

---

## 3. Split Integrity & Leakage Verification

### 3.1 Patient/Image-Level Split Design
* **Method:** Image-level stratification based on `pixel_sha256`. All questions referencing the same physical image are strictly assigned to either `train` (dev) or `test` splits.
* **Pixel Hash Overlap:**
  $$\text{Train Images} \cap \text{Test Images} = \emptyset \quad (\mathbf{0\%} \text{ Leakage})$$
* **Verification:** Recomputed directly from [`image_level_split_manifest.csv`](file:///c:/.PUSAT/Project/Kuliah/VLM/main/data/manifests/image_level_split_manifest.csv). There is zero exact or pixel-level leakage across splits.

---

## 4. Translation Fidelity & Bilingual Equivalence Audit

### 4.1 Automated Rule-Based Screening
All 2,244 bilingual pairs were screened across a 10-class error taxonomy ($T_0$ to $T_9$):
* **CLEAN ($T_0$ Equivalent / $T_1$ Minor linguistic):** 1,984 pairs ($88.41\%$)
* **FLAGGED ($T_2$ Terminology / $T_7$ Intent shift):** 173 pairs ($7.71\%$)
* **HIGH RISK ($T_4$ Negation / $T_5$ Anatomy / $T_6$ Laterality / $T_8$ Answer mismatch):** 87 pairs ($3.88\%$)

### 4.2 Known Error Patterns Detected
* *False Literal Translations:* "axial plane" translated as *"pesawat aksial"* (airplane) instead of *"bidang aksial"*.
* *Laterality/Negation Inversion:* 14 pairs had reversed left/right (*kiri/kanan*) or inverted negation words (*tidak ada* vs *ada*).
* *Audit Strategy:* Primary evaluations are conducted strictly on **CLEAN pairs** ($N=139$), with full sensitivity analysis on the total cohort ($N=150$).

---

## 5. Baseline Model Benchmarking & Performance Reproduction

Two candidate architectures were evaluated under standardized greedy decoding (`do_sample=False`, `max_new_tokens=4`) on the frozen 150-pair bilingual manifest:
1. **General-Domain VLM:** `Qwen/Qwen2.5-VL-7B-Instruct`
2. **Medical-Domain VLM:** `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

### 5.1 Comparative Empirical Results

| Metric | Qwen2.5-VL-7B-Instruct (General) | HuatuoGPT-Vision-7B (Medical) | Analysis & Contrast |
| :--- | :---: | :---: | :--- |
| **English Accuracy** ($\text{Acc}_{\text{EN}}$) | **82.00%** (123/150) | **76.00%** (114/150) | General model higher overall accuracy |
| **Indonesian Accuracy** ($\text{Acc}_{\text{ID}}$) | **73.33%** (110/150) | **70.00%** (105/150) | Both models degrade under Indonesian prompt |
| **Language Shift Gap** ($\Delta_{\text{lang}}$) | **+8.67 pts** (+0.0867) | **+6.00 pts** (+0.0600) | Statistically significant cross-lingual gap |
| **Cross-Lingual Flip Rate** | **20.67%** (31/150) | **21.33%** (32/150) | ~1 in 5 answers flips simply by translating prompt |
| **Joint Correctness** | **67.33%** (101/150) | **62.67%** (94/150) | Both languages correct simultaneously |
| **Mean Confidence (EN / ID)** | **0.8465 / 0.7448** | **0.7742 / 0.9082** | **HuatuoGPT exhibits severe Indonesian overconfidence** |
| **ECE (EN / ID)** | **0.0435 / 0.0612** | **0.0325 / 0.2350** | HuatuoGPT ID ECE is 3.8x worse than Qwen-7B |
| **Brier Score (EN / ID)** | **0.1387 / 0.1741** | **0.1534 / 0.2713** | Higher Brier score = poorer probability calibration |
| **Confidently Wrong Rate ($\ge 0.90$)** | **0.67%** (1/150) | **24.67%** (37/150) | Denominator: All 150 answers |
| **Conditional Error @ $c \ge 0.90$** | **3.03%** (1/33) | **36.63%** (37/101) | Denominator: Only high-confidence answers |

### 5.2 Key Takeaways from Baseline Reproduction
1. **The Language Shift Vulnerability is Real:** Both general and medical models suffer a 6–9 percentage point drop in accuracy and a ~21% flip rate on identical radiographs.
2. **Medical Specialization Induces Language-Specific Overconfidence:** While general Qwen-7B lowers its confidence in Indonesian (0.7448) matching its accuracy (73.33%), the medical model HuatuoGPT inflates its confidence in Indonesian (0.9082) despite declining accuracy (70.00%), leading to an alarming **36.63% error rate when the model asserts $\ge 90\%$ certainty**.
3. **Hardware Constraint:** The 32B model (`Qwen2.5-VL-32B-Instruct`) encounters CUDA OOM on Kaggle dual-T4 GPUs (14.56 GiB limit per device) during full attention forward passes. Research is strictly locked to 7B-class models.
