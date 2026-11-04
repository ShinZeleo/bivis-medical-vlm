# Risk Model Ablation Audit: Logistic Regression & Cross-Lingual Disagreement
**Protocol:** Trained strictly on Dev Set (N=323, 88 clusters) via GroupKFold(5); evaluated on Test Set (N=799, 204 clusters). Zero test leakage.

## 1. Model Progression & Test AURC
| Model | Configuration | Features | Dev CV AURC | Test AURC | Risk @ 80% Cov | $\Delta$ AURC vs Conf |
|---|---|---|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | `LR_conf` | `score_m1_seq` | 0.1988 | 0.2422 | 33.65% | 0.0000 |
| | `LR_visual` | `score_m1_seq, delta_p_dir_seq` | 0.2015 | 0.2434 | 33.49% | +0.0012 |
| | `LR_full` | `score_m1_seq, delta_p_dir_seq, d_lang` | 0.1918 | **0.2053** | **28.48%** | **-0.0369** |
| **HuatuoGPT-Vision-7B** | `LR_conf` | `score_m1_seq` | 0.2402 | 0.2884 | 40.38% | 0.0000 |
| | `LR_visual` | `score_m1_seq, delta_p_dir_seq` | 0.2164 | 0.2513 | 37.25% | -0.0371 |
| | `LR_full` | `score_m1_seq, delta_p_dir_seq, d_lang` | 0.2068 | **0.2179** | **35.68%** | **-0.0705** |

## 2. Learned Feature Coefficients & Odds Ratios (`LR_full`)
| Model | Feature | Standardized Coef | Standardized OR | Unstandardized Coef | Unstandardized OR |
|---|---|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | `score_m1_seq` | -0.1258 | 0.882 | -0.9381 | 0.391 |
| **Qwen2.5-VL-7B** | `delta_p_dir_seq` | +0.1128 | 1.119 | +0.3926 | 1.481 |
| **Qwen2.5-VL-7B** | `d_lang` | +1.0960 | 2.992 | +2.5607 | 12.945 |
| **HuatuoGPT-Vision-7B** | `score_m1_seq` | -0.3693 | 0.691 | -2.7551 | 0.064 |
| **HuatuoGPT-Vision-7B** | `delta_p_dir_seq` | -0.3055 | 0.737 | -1.5513 | 0.212 |
| **HuatuoGPT-Vision-7B** | `d_lang` | +0.9941 | 2.702 | +1.9890 | 7.308 |

## 3. Strict Verification of Odds Ratio Claim: $OR(D_{lang}) > 2.7$
- **Qwen2.5-VL-7B:** Standardized $\beta = 1.0960 \implies OR = \exp(1.0960) = \mathbf{2.992}$
- **HuatuoGPT-Vision-7B:** Standardized $\beta = 0.9941 \implies OR = \exp(0.9941) = \mathbf{2.702}$
- **Correct Academic Interpretation:**  
  Holding model confidence and visual perturbation constant, cross-lingual prediction disagreement between English and Indonesian queries is associated with approximately **2.7 to 3.0 times higher odds** of an Indonesian diagnostic error.  
  *(Note: An odds ratio of 2.7 is NOT a 2.7x increase in absolute probability).*
