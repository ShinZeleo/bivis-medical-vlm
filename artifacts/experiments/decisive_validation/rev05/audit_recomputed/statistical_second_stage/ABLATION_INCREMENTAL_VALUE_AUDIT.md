# Ablation & Incremental Feature Contribution Audit Report
**Generated:** 2026-11-01 03:25:47 UTC  
**Auditor:** Machine Learning Research Scientist  
**Focus:** Isolating the incremental value of Visual Sensitivity ($\Delta P_{dir}$) and Cross-Lingual Disagreement ($D_{lang}$)

## 1. Incremental Comparison Matrix (Primary Test, N=799, 204 clusters)
| Model | Stepwise Transition | Question Addressed | Base AURC | New AURC | $\Delta$ AURC | 95% Cluster Boot CI | Permutation $p$ | Risk @ 80% Cov Drop | Incremental Verdict |
|---|---|---|---:|---:|---:|:---:|---:|---:|:---:|
| **Qwen2.5-VL-7B** | `LR_visual vs LR_conf` | Does adding visual sensitivity to confidence ... | 0.2422 | 0.2434 | **+0.0012** | [-0.0038, 0.0064] | 0.6693 | -0.16 pp | `FAILS` |
| **Qwen2.5-VL-7B** | `LR_full vs LR_visual` | Does adding cross-lingual disagreement to vis... | 0.2434 | 0.2053 | **-0.0381** | [-0.0622, -0.0148] | 0.0130 | -5.01 pp | `INCREMENTAL GAIN` |
| **Qwen2.5-VL-7B** | `LR_full vs LR_conf` | Does full model improve over confidence alone... | 0.2422 | 0.2053 | **-0.0369** | [-0.0589, -0.0159] | 0.0220 | -5.16 pp | `INCREMENTAL GAIN` |
| **Qwen2.5-VL-7B** | `LR_full vs M1` | Does full model improve over raw heuristic co... | 0.2422 | 0.2053 | **-0.0369** | [-0.0589, -0.0159] | 0.0310 | -5.16 pp | `INCREMENTAL GAIN` |
| **Qwen2.5-VL-7B** | `M_norm vs M1` | Does heuristic arithmetic combination improve... | 0.2422 | 0.2753 | **+0.0331** | [0.0078, 0.0580] | 0.2298 | -0.94 pp | `FAILS` |
| **HuatuoGPT-Vision-7B** | `LR_visual vs LR_conf` | Does adding visual sensitivity to confidence ... | 0.2884 | 0.2513 | **-0.0371** | [-0.0643, -0.0085] | 0.0180 | -3.13 pp | `INCREMENTAL GAIN` |
| **HuatuoGPT-Vision-7B** | `LR_full vs LR_visual` | Does adding cross-lingual disagreement to vis... | 0.2513 | 0.2179 | **-0.0335** | [-0.0467, -0.0218] | 0.0010 | -1.56 pp | `INCREMENTAL GAIN` |
| **HuatuoGPT-Vision-7B** | `LR_full vs LR_conf` | Does full model improve over confidence alone... | 0.2884 | 0.2179 | **-0.0705** | [-0.0984, -0.0415] | 0.0010 | -4.69 pp | `INCREMENTAL GAIN` |
| **HuatuoGPT-Vision-7B** | `LR_full vs M1` | Does full model improve over raw heuristic co... | 0.2884 | 0.2179 | **-0.0705** | [-0.0984, -0.0415] | 0.0010 | -4.69 pp | `INCREMENTAL GAIN` |
| **HuatuoGPT-Vision-7B** | `M_norm vs M1` | Does heuristic arithmetic combination improve... | 0.2884 | 0.2552 | **-0.0332** | [-0.0585, -0.0063] | 0.3846 | -2.97 pp | `FAILS` |

## 2. Critical Scientific Insights on Feature Contributions

### A. Incremental Value of Visual Sensitivity ($\Delta P_{dir}$)
- **In Qwen2.5-VL-7B (Generalist):**  
  Adding $\Delta P_{dir}$ to confidence (`LR_conf` $\rightarrow$ `LR_visual`) yields $\Delta \text{AURC} = +0.0012$ ($p = 0.6693$).  
  **Conclusion:** Visual sensitivity alone provides **ZERO incremental predictive value** for Qwen.
- **In HuatuoGPT-Vision-7B (Medical Specialist):**  
  Adding $\Delta P_{dir}$ to confidence yields $\Delta \text{AURC} = -0.0371$ ($p = 0.0180$).  
  **Conclusion:** Visual sensitivity provides **strong, statistically significant incremental value** for Huatuo. Because Huatuo suffers from severe negative reliance on blank images (80% negative), this feature acts as an effective hallucination penalty.

### B. Incremental Value of Cross-Lingual Disagreement ($D_{lang}$)
- **In Qwen2.5-VL-7B:**  
  Adding $D_{lang}$ to visual+confidence (`LR_visual` $\rightarrow$ `LR_full`) drops AURC from 0.2434 to 0.2053 ($\Delta = -0.0381$, $p = 0.0130$). Risk at 80% coverage drops from 33.65% to 29.58% (-4.07 pp).  
  **Conclusion:** $D_{lang}$ is the **primary driver of selective prediction improvement** in Qwen.
- **In HuatuoGPT-Vision-7B:**  
  Adding $D_{lang}$ drops AURC from 0.2513 to 0.2179 ($\Delta = -0.0335$, $p = 0.0010$). Risk at 80% coverage drops from 36.46% to 31.30% (-5.16 pp).  
  **Conclusion:** $D_{lang}$ provides **substantial, statistically significant incremental value** beyond visual sensitivity.
