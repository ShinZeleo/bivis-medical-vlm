# RQ2 Empirical Audit: Visual Intervention & Deletion Sensitivity
**Research Question:** *Is sensitivity to visual deletion associated with errors in Indonesian predictions?*

## 1. Visual Sensitivity Formulation
$$\Delta P_{dir} = P(\hat{y}_{ID} \mid X_{orig}) - P(\hat{y}_{ID} \mid X_{blank})$$
Where $X_{blank}$ is a deterministic zero-pixel black image canvas (P1 control).

## 2. Quantitative Summary Statistics (Primary Test, N=799)
| Metric | Qwen2.5-VL-7B | HuatuoGPT-Vision-7B |
|---|---:|---:|
| **Mean $\Delta P_{dir}$** | **+0.1405** | **-0.0411** |
| **Standard Deviation** | 0.3034 | 0.2425 |
| **Median $\Delta P_{dir}$** | 0.0518 | -0.0918 |
| **Interquartile Range (IQR)** | 0.3960 | 0.1444 |
| **Positive Reliance ($\Delta P > 0$)** | **60.1%** | **20.0%** |
| **Negative Reliance ($\Delta P < 0$)** | **39.9%** | **80.0%** |

## 3. Association with Indonesian Prediction Correctness
| Metric | Qwen (Correct vs Incorrect) | Huatuo (Correct vs Incorrect) |
|---|---|---|
| **Mean Delta (Correct ID=1)** | 0.1694 (N=502) | 0.0208 (N=460) |
| **Mean Delta (Incorrect ID=0)** | 0.0917 (N=297) | -0.1252 (N=339) |
| **Difference (Correct - Incorrect)** | +0.0776 | +0.1461 |
| **Effect Size (Cohen's d)** | 0.258 | 0.631 |
| **Mann-Whitney U Test** | U = 85908, **p = 3.1408e-04** | U = 108654, **p = 1.7932e-21** |
| **Cluster-Level Correlation** | r = 0.019 (p = 0.7876) | r = 0.207 (p = 0.0030) |

## 4. Crucial Methodological Warning & Epistemic Boundaries
> [!WARNING]
> **No Causal Claim of Anatomical Comprehension:** A black blank image represents an extreme out-of-distribution (OOD) perturbation.
> - A positive $\Delta P_{dir}$ demonstrates that model predictions are sensitive to the presence of visual pixels versus darkness.
> - This does **NOT** prove causal visual grounding, clinical radiologist reasoning, or anatomical understanding.
> - In HuatuoGPT, 80.0% of predictions exhibit negative reliance ($\Delta P < 0$), meaning the model is often **more confident in darkness** than on the actual chest X-ray!
