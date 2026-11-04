# RQ3 Empirical Audit: Selective Prediction Performance
**Research Question:** *Does visual-sensitivity-aware scoring improve selective prediction compared with candidate preference alone?*

## 1. Discrete Selective Prediction Scorecard (Primary Test, N=799, 204 clusters)
| Model | Estimator | AURC Mean | 95% Bootstrap CI | $\Delta$ AURC vs M1 | Permutation p-raw | Holm-Bonferroni p | Risk @ 80% Cov |
|---|---|---:|:---:|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | `score_m1_seq` | 0.2422 | [0.1977, 0.2889] | +0.0000 | 1.0000 | 1.0000 | 33.65% |
| **Qwen2.5-VL-7B** | `score_m2_seq` | 0.3209 | [0.2626, 0.3638] | +0.0787 | 0.0030 | 0.0120 | 36.15% |
| **Qwen2.5-VL-7B** | `score_m_dir_seq` | 0.3158 | [0.2576, 0.3570] | +0.0736 | 0.0090 | 0.0270 | 36.15% |
| **Qwen2.5-VL-7B** | `score_m_norm_seq` | 0.2753 | [0.2287, 0.3206] | +0.0331 | 0.2298 | 0.4176 | 32.71% |
| **Qwen2.5-VL-7B** | `score_m3_old_seq` | 0.2763 | [0.2302, 0.3231] | +0.0341 | 0.2088 | 0.4176 | 33.18% |
| **HuatuoGPT-Vision-7B** | `score_m1_seq` | 0.2884 | [0.2357, 0.3499] | +0.0000 | 1.0000 | 1.0000 | 40.38% |
| **HuatuoGPT-Vision-7B** | `score_m2_seq` | 0.3400 | [0.2554, 0.3688] | +0.0516 | 0.1968 | 0.7872 | 42.25% |
| **HuatuoGPT-Vision-7B** | `score_m_dir_seq` | 0.3394 | [0.2550, 0.3678] | +0.0510 | 0.2098 | 0.7872 | 42.25% |
| **HuatuoGPT-Vision-7B** | `score_m_norm_seq` | 0.2552 | [0.2068, 0.3084] | -0.0332 | 0.3846 | 0.7872 | 37.40% |
| **HuatuoGPT-Vision-7B** | `score_m3_old_seq` | 0.2614 | [0.2127, 0.3177] | -0.0270 | 0.4835 | 0.7872 | 38.03% |

## 2. Rigorous Evaluation of Heuristic Estimator ($M_{norm}$)
- **Qwen2.5-VL-7B:** Heuristic $M_{norm}$ yields AURC = 0.2753 vs M1 = 0.2422 ($\Delta = +0.0331$, worse than baseline; Holm p = 0.4056).
- **HuatuoGPT-Vision-7B:** Heuristic $M_{norm}$ lowers AURC from 0.2884 to 0.2552 ($\Delta = -0.0332$), but fails to achieve statistical significance under paired cluster permutation (Holm p = 0.7812).

## 3. Scientific Verdict for RQ3
> **Scientific Finding (Negative Result for Heuristic Combination):**  
> Simple arithmetic combinations of candidate preference and visual perturbation ($M_{norm}$) **do NOT reliably or significantly improve selective prediction** over baseline confidence alone under rigorous cluster permutation tests.  
> As shown in Phase 6, selective prediction improvement requires a principled multivariate approach incorporating cross-lingual disagreement ($LR_{full}$).
