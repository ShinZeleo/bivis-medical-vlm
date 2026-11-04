# RQ1 Empirical Audit: Bilingual Performance Under Language Shift
**Research Question:** *How does English-to-Indonesian language shift affect the accuracy and prediction consistency of medical vision-language models?*

## 1. Recomputed Primary Test Metrics (N=799 QA pairs, 204 clusters)
| Model | English Acc (%) | Indonesian Acc (%) | Accuracy Gap (pp) | Flip / Disagreement (%) | Joint Acc (%) | Len-Norm Gap (pp) |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | 75.84% | 62.83% | -13.02 pp | 28.79% | 54.94% | -17.40 pp |
| **HuatuoGPT-Vision-7B** | 74.84% | 57.57% | -17.27 pp | 50.81% | 40.80% | -22.15 pp |

## 2. Transition Analysis (EN -> ID Directional Shift)
| Model | Both Correct (1->1) | Degraded (EN=1 -> ID=0) | Gained (EN=0 -> ID=1) | Both Incorrect (0->0) |
|---|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | 439 (54.9%) | 167 (20.9%) | 63 (7.9%) | 130 (16.3%) |
| **HuatuoGPT-Vision-7B** | 326 (40.8%) | 272 (34.0%) | 134 (16.8%) | 67 (8.4%) |

## 3. Confusion Matrix Breakdown
See [`tables/rq1_confusion_matrices.csv`](tables/rq1_confusion_matrices.csv) for full Sensitivity, Specificity, and Balanced Accuracy across English and Indonesian prompts.

## 4. Key Empirical Takeaways
1. **Severe Degradation:** Both models suffer substantial diagnostic degradation when queried in Indonesian (-13.0 pp for Qwen, -17.3 pp for Huatuo).
2. **Elevated Prediction Disagreement:** Huatuo displays a staggering **50.81% flip rate**, indicating that for more than half of identical clinical images, changing the query language changes the binary diagnosis. Qwen displays a 28.79% flip rate.
3. **Medical Specialization Vulnerability:** HuatuoGPT (medical-specialized) exhibits **greater cross-lingual vulnerability** than generalist Qwen, confirming that medical adaptation in English does not transfer reliably to non-English medical queries.
