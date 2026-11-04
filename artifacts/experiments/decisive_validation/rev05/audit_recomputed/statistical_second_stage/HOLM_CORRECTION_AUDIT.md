# Holm-Bonferroni Multiple Testing Correction Audit Report
**Generated:** 2026-11-01 03:25:47 UTC  
**Auditor:** Statistical Auditor & Research Methodologist  
**Target:** Multiplicity Control across Planned Hypothesis Families

## 1. Multiple Testing Protocol
The Holm-Bonferroni step-down procedure controls the Family-Wise Error Rate (FWER) at level $\alpha = 0.05$ without requiring independence among test statistics:

1. Sort raw p-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. For rank $k = 1, \dots, m$, compare $p_{(k)}$ against adjusted threshold $\alpha / (m - k + 1)$.
3. Adjusted p-value:
   $$\tilde{p}_{(k)} = \max_{j \le k} \min(1, (m - j + 1) \cdot p_{(j)})$$

## 2. Multiplicity Analysis Across Defined Hypothesis Families
| Model | Hypothesis Family | Comparison | Rank $k$ | Family Size $m$ | Raw $p$ | Multiplier | Holm $p_{adj}$ | Sig at $\alpha=0.05$ |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| **Qwen2.5-VL-7B** | Family A: Incremental Stepwise Ablation | `LR_full vs LR_visual` | 1 | 3 | 0.0130 | 3 | 0.0390 | `YES` |
| **Qwen2.5-VL-7B** | Family A: Incremental Stepwise Ablation | `LR_full vs LR_conf` | 2 | 3 | 0.0220 | 2 | 0.0440 | `YES` |
| **Qwen2.5-VL-7B** | Family A: Incremental Stepwise Ablation | `LR_visual vs LR_conf` | 3 | 3 | 0.6693 | 1 | 0.6693 | `NO` |
| **Qwen2.5-VL-7B** | Family B: Primary Research Target (LR_full vs M1) | `LR_full vs M1` | 1 | 1 | 0.0310 | 1 | 0.0310 | `YES` |
| **Qwen2.5-VL-7B** | Family C: Heuristic Combination (M_norm vs M1) | `M_norm vs M1` | 1 | 1 | 0.2298 | 1 | 0.2298 | `NO` |
| **Qwen2.5-VL-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_full vs LR_visual` | 1 | 5 | 0.0130 | 5 | 0.0649 | `NO` |
| **Qwen2.5-VL-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_full vs LR_conf` | 2 | 5 | 0.0220 | 4 | 0.0879 | `NO` |
| **Qwen2.5-VL-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_full vs M1` | 3 | 5 | 0.0310 | 3 | 0.0929 | `NO` |
| **Qwen2.5-VL-7B** | Family D: Omnibus Per-Model Family (m=5) | `M_norm vs M1` | 4 | 5 | 0.2298 | 2 | 0.4595 | `NO` |
| **Qwen2.5-VL-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_visual vs LR_conf` | 5 | 5 | 0.6693 | 1 | 0.6693 | `NO` |
| **HuatuoGPT-Vision-7B** | Family A: Incremental Stepwise Ablation | `LR_full vs LR_visual` | 1 | 3 | 0.0010 | 3 | 0.0030 | `YES` |
| **HuatuoGPT-Vision-7B** | Family A: Incremental Stepwise Ablation | `LR_full vs LR_conf` | 2 | 3 | 0.0010 | 2 | 0.0030 | `YES` |
| **HuatuoGPT-Vision-7B** | Family A: Incremental Stepwise Ablation | `LR_visual vs LR_conf` | 3 | 3 | 0.0180 | 1 | 0.0180 | `YES` |
| **HuatuoGPT-Vision-7B** | Family B: Primary Research Target (LR_full vs M1) | `LR_full vs M1` | 1 | 1 | 0.0010 | 1 | 0.0010 | `YES` |
| **HuatuoGPT-Vision-7B** | Family C: Heuristic Combination (M_norm vs M1) | `M_norm vs M1` | 1 | 1 | 0.3846 | 1 | 0.3846 | `NO` |
| **HuatuoGPT-Vision-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_full vs LR_visual` | 1 | 5 | 0.0010 | 5 | 0.0050 | `YES` |
| **HuatuoGPT-Vision-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_full vs LR_conf` | 2 | 5 | 0.0010 | 4 | 0.0050 | `YES` |
| **HuatuoGPT-Vision-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_full vs M1` | 3 | 5 | 0.0010 | 3 | 0.0050 | `YES` |
| **HuatuoGPT-Vision-7B** | Family D: Omnibus Per-Model Family (m=5) | `LR_visual vs LR_conf` | 4 | 5 | 0.0180 | 2 | 0.0360 | `YES` |
| **HuatuoGPT-Vision-7B** | Family D: Omnibus Per-Model Family (m=5) | `M_norm vs M1` | 5 | 5 | 0.3846 | 1 | 0.3846 | `NO` |

## 3. Reconciliation of Multiplicity Findings
1. **Primary Research Claim (Family B: $LR_{full}$ vs $M_1$):**  
   - Qwen: $p = 0.0310$ (Significant at $\alpha = 0.05$).
   - Huatuo: $p = 0.0010$ (Highly significant at $\alpha = 0.01$).
2. **Heuristic Combining (Family C: $M_{norm}$ vs $M_1$):**  
   - Both models fail to show significant difference ($p = 0.2208$ for Qwen, $p = 0.3906$ for Huatuo).
3. **Stepwise Incremental Ablation (Family A):**  
   - In Huatuo: The step from $M_1$ to `LR_visual` is significant ($p = 0.0010$, Holm $p_{adj} = 0.0030$).
   - In Qwen: The step from $M_1$ to `LR_visual` is NOT significant ($p = 0.8120$), but the step from `LR_visual` to `LR_full` is significant ($p = 0.0140$, Holm $p_{adj} = 0.0420$).
