# FINAL STATISTICAL & ABLATION AUDIT REPORT (SECOND-STAGE)
## Medical Vision-Language Models Under English-Indonesian Language Shift
**Audited Target:** `artifacts/experiments/decisive_validation/rev05/`  
**Document Identifier:** `AUDIT-STAT-STAGE2-REV05-FINAL`  
**Audit Completion Date:** 2026-11-01 03:25:47 UTC  
**Auditor Roles:** Senior Statistical Auditor & Research Reproducibility Engineer

---

## 1. Executive Summary & Definitive Decision Scorecard

This second-stage audit performed deep statistical verification of the inference machinery, cluster resampling methods, multiple testing adjustments, incremental ablation contributions, and logistic regression stability.

### The 8 Definitive Statistical Decisions
| # | Audit Decision Question | Status Verdict | Grounded Empirical Justification |
|---|---|:---:|---|
| **1** | **Apakah p-value LR_full vs M1 valid?** | **`PASS`** | Evaluated via paired cluster permutation with Rademacher cluster flips ($B=1,000$). Qwen ($p=0.0310$) and Huatuo ($p=0.0010$) both demonstrate statistically significant reduction in selective risk over baseline confidence. |
| **2** | **Apakah confidence interval AURC valid?** | **`PASS`** | Cluster-level bootstrap preserves block-correlated radiograph dependencies. Percentile 95% CIs correctly capture cluster variance, which is 1.2x to 1.8x wider than naive row bootstrap. |
| **3** | **Apakah M_norm benar-benar tidak signifikan?** | **`PASS`** | Verified definitively. Simple arithmetic combining ($M_{norm}$) yields $p_{perm} = 0.2208$ on Qwen and $p_{perm} = 0.3906$ on Huatuo. It fails Holm-Bonferroni correction ($p_{adj} > 0.40$). |
| **4** | **Apakah LR_visual menambah manfaat dibanding LR_conf?** | **`PASS WITH LIMITATIONS`** | **Model-dependent:** For Huatuo, yes ($\Delta = -0.0371, p = 0.0010$). For Qwen, no ($\Delta = +0.0012, p = 0.8120$). Visual perturbation alone only helps models with severe visual vulnerability. |
| **5** | **Apakah LR_full menambah manfaat dibanding LR_visual?** | **`PASS`** | Yes, for both models. Adding $D_{lang}$ drops AURC by $-0.0381$ on Qwen ($p = 0.0140$) and by $-0.0335$ on Huatuo ($p = 0.0070$). |
| **6** | **Apakah D_lang memberikan incremental predictive value?** | **`PASS`** | Formally confirmed. Beyond odds ratios ($OR > 2.7$), $D_{lang}$ delivers statistically significant incremental AURC reduction and lowers selective risk at 80% coverage by 4.0 to 5.2 percentage points. |
| **7** | **Apakah ukuran dev set cukup untuk estimasi yang stabil?** | **`PASS`** | Dev set has 323 samples across 88 clusters, yielding 32.0 EPV (Qwen) and 43.0 EPV (Huatuo). Exceeds the standard 10–20 EPV threshold. Regularization sensitivity audit confirms stability across $C \in [0.1, 100.0]$. |
| **8** | **Apakah hasil statistik layak dikutip dalam artikel ilmiah?** | **`PASS`** | Fully reproducible, cluster-aware, and mathematically sound for publication in a SINTA 2 accredited journal, provided claims are framed with noted clinical boundaries. |

---

## 2. Statistical Architecture & Methodology Synthesis
1. **Resampling Unit:** All tests strictly enforce `canonical_image_id` (N=204 test clusters, N=88 dev clusters) as the independent resampling unit.
2. **Epistemic Labeling:** Permutation tests must be reported as *Monte Carlo Paired Cluster Sign-Flip Permutation Tests (B=1,000)* rather than "exact permutations".
3. **Hypothesis Family Structure:** Under all reasonable family definitions (stepwise ablation, heuristic combinations, or primary target), the superiority of $LR_{full}$ remains statistically significant at $\alpha = 0.05$.

---

## 3. Directory of Deliverable Reports
- [`PERMUTATION_IMPLEMENTATION_AUDIT.md`](PERMUTATION_IMPLEMENTATION_AUDIT.md) — Algorithmic proof and Monte Carlo bounds.
- [`BOOTSTRAP_IMPLEMENTATION_AUDIT.md`](BOOTSTRAP_IMPLEMENTATION_AUDIT.md) — Cluster vs. naive bootstrap variance inflation analysis.
- [`HOLM_CORRECTION_AUDIT.md`](HOLM_CORRECTION_AUDIT.md) — Multiplicity adjustments across hypothesis families.
- [`ABLATION_INCREMENTAL_VALUE_AUDIT.md`](ABLATION_INCREMENTAL_VALUE_AUDIT.md) — Pairwise incremental contribution breakdown.
- [`LOGISTIC_REGRESSION_VALIDITY_AUDIT.md`](LOGISTIC_REGRESSION_VALIDITY_AUDIT.md) — Cluster-robust standard errors, VIF, and C-sensitivity.
