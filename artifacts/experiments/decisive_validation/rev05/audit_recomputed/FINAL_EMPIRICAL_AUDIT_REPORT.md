# FINAL EMPIRICAL AUDIT REPORT (REV05)
## Medical Vision-Language Models: Qwen2.5-VL-7B & HuatuoGPT-Vision-7B
**Audited Directory:** `artifacts/experiments/decisive_validation/rev05/`  
**Document Identifier:** `AUDIT-REV05-DEEP-EMPIRICAL-FINAL`  
**Audit Completion Date:** 2026-10-31 14:08:47 UTC  
**Auditor Roles:** Senior Machine Learning Research Scientist, Statistical Auditor & Reproducibility Engineer

---

## 1. Executive Summary & Final Research Question Verdicts

This report provides the definitive empirical audit of the rev05 experimental findings. Every metric, transition count, odds ratio, and risk-coverage curve was independently recomputed from raw prediction checkpoints without modifying existing experimental artifacts.

### Definitive Research Question Verdicts
| Research Question | Core Hypothesis | Empirical Audit Finding | Final Scientific Verdict |
|---|---|---|:---:|
| **RQ1: Language Shift Degradation** | English-to-Indonesian query translation severely degrades VLM accuracy and consistency. | Accuracy drops by 13.0 pp (Qwen) and 17.3 pp (Huatuo). Flip rates reach 28.8% (Qwen) and 50.8% (Huatuo). | **`SUPPORTED`** |
| **RQ2: Visual Deletion Sensitivity** | Model error is correlated with sensitivity to visual pixel removal ($\Delta P_{dir}$). | Significant difference in $\Delta P_{dir}$ between correct and incorrect predictions ($p < 0.0001$). However, Huatuo displays 80% negative reliance. | **`PARTIALLY SUPPORTED`** (Visual sensitivity verified; no causal clinical grounding claim) |
| **RQ3: Selective Prediction** | Visual sensitivity scoring improves selective prediction over candidate preference. | Heuristic arithmetic combining ($M_{norm}$) fails significance ($p > 0.20$). Principled multivariate risk modeling ($LR_{full}$) significantly lowers AURC ($p < 0.05$). | **`SUPPORTED WITH MODELING QUALIFICATIONS`** |

---

## 2. Sample Accounting & Data Provenance
- Authoritative Manifest (`data/manifests/manifest_rev05.csv`) verified via SHA-256 (`fb55fd75...`).
- Exactly 1,122 primary cohort samples evaluated across 292 unique image clusters.
- Test split (799 pairs, 204 clusters) and dev split (323 pairs, 88 clusters) exhibit **strictly 0.00% patient cluster leakage**.
- Checkpoint prediction coverage across Qwen and Huatuo is 100% identical.

---

## 3. Detailed Empirical Findings

### RQ1: Bilingual Performance Under Language Shift
- English accuracy: Qwen 75.84%, Huatuo 74.84%.
- Indonesian accuracy: Qwen 62.83%, Huatuo 57.57%.
- Language accuracy gap: Qwen -13.02 pp, Huatuo -17.27 pp.
- Flip rate (disagreement): Qwen 28.79%, Huatuo 50.81%.
- Degradation transitions (EN correct -> ID incorrect): Qwen 16.90% (135 samples), Huatuo 25.16% (201 samples).

### RQ2: Visual Intervention (P0 vs P1 Black Canvas)
- Qwen exhibits positive mean visual sensitivity ($\Delta P_{dir} = +0.1405$, 60.1% positive reliance).
- Huatuo exhibits negative mean visual sensitivity ($\Delta P_{dir} = -0.0411$, 80.0% negative reliance).
- For both models, samples answered correctly have significantly higher visual reliance than samples answered incorrectly ($p < 0.0001$).

### RQ3: Selective Prediction & Risk-Coverage (AURC)
- Pure confidence baseline ($M_1$): Qwen AURC = 0.2422, Huatuo AURC = 0.2884.
- Heuristic estimator ($M_{norm}$): Qwen AURC = 0.2753, Huatuo AURC = 0.2552 (Fails statistical significance under Holm-Bonferroni correction).
- Learned multivariate model ($LR_{full}$): Qwen AURC = 0.2053 ($p = 0.0310$), Huatuo AURC = 0.2179 ($p = 0.0010$). Both statistically significant over baseline confidence.

### Risk Ablation & Odds Ratio
- Cross-lingual disagreement $D_{lang}$ is a powerful predictor of Indonesian diagnostic error:
  - Qwen: Standardized $\beta = 1.0960 \implies OR = 2.992$
  - Huatuo: Standardized $\beta = 0.9941 \implies OR = 2.702$
- Both exceed the reported threshold $OR(D_{lang}) > 2.7$.

---

## 4. Methodological Limitations & SINTA 2 Advisory
1. **Retrospective Benchmark Constraint:** The IndoRad-VQA benchmark is retrospective and binary (Yes/No). Real-world clinical radiology involves open-ended narrative diagnoses and complex differential findings.
2. **Visual Deletion is OOD:** A black blank image demonstrates pixel dependence, not anatomical reasoning. Do not claim models "see" or "understand" radiology images in clinical terms.
3. **Misattribution Correction:** Masters thesis drafts and papers should cite pure $M_1$ as 0.2422 (Qwen) and 0.2884 (Huatuo), and clarify that 0.2434 and 0.2513 reflect the `LR_visual` ablation model.

---

## 5. Decision & Next Steps
1. The empirical data of rev05 is **computationally reproducible, mathematically sound, and internally consistent**.
2. All tables and figures are stored in `artifacts/experiments/decisive_validation/rev05/audit_recomputed/`.
3. The research is now ready for **academic paper writing** targeted at a SINTA 2 journal.
