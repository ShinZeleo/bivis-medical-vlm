#!/usr/bin/env python3
"""
scripts/generate_readiness_assessment.py
========================================
Generates the authoritative Final Research Readiness Assessment deliverables:
1. RQ_DECISION_MATRIX.csv
2. CLAIM_EVIDENCE_MATRIX.csv
3. REMAINING_ISSUES.md
4. PUBLICATION_READINESS_CHECKLIST.md
5. FINAL_RESEARCH_READINESS_REPORT.md

Outputs are written to:
  - artifacts/experiments/decisive_validation/rev05/audit_recomputed/statistical_second_stage/
  - docs/audit/
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "decisive_validation" / "rev05" / "audit_recomputed" / "statistical_second_stage"
DOCS_AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"

STAGE2_DIR.mkdir(parents=True, exist_ok=True)
DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ------------------------------------------------------------------------------
# 1. RQ_DECISION_MATRIX.csv
# ------------------------------------------------------------------------------
rq_matrix_data = [
    {
        "rq_id": "RQ1",
        "research_question": "How does English-to-Indonesian language shift affect accuracy and prediction consistency?",
        "primary_hypothesis": "Translation from EN to ID causes significant diagnostic accuracy degradation and prediction inconsistency.",
        "status": "SUPPORTED",
        "key_evidence": "Accuracy drop: Qwen -13.02 pp (75.84% to 62.83%), Huatuo -17.27 pp (74.84% to 57.57%). Flip rate: Qwen 28.79%, Huatuo 50.81%.",
        "confidence_intervals": "Qwen Acc EN 95% CI [72.7, 78.7] vs ID [59.4, 66.1]; Huatuo Acc EN [71.7, 77.8] vs ID [54.1, 60.9].",
        "corrected_p_value": "p < 0.0001 (McNemar paired test on EN vs ID predictions)",
        "interpretation_boundaries": "Evaluated on binary Yes/No IndoRad-VQA benchmark. Does not evaluate open-ended radiology report generation."
    },
    {
        "rq_id": "RQ2",
        "research_question": "Is sensitivity to visual deletion associated with errors in Indonesian predictions?",
        "primary_hypothesis": "Sensitivity to visual removal (Delta P_dir = P_orig - P_blank) correlates with prediction correctness.",
        "status": "PARTIALLY SUPPORTED",
        "key_evidence": "Correct predictions have significantly higher visual sensitivity than incorrect ones (Qwen: +0.169 vs +0.092, Huatuo: -0.022 vs -0.067, p < 0.0001). However, Huatuo exhibits 80% negative reliance on blank images.",
        "confidence_intervals": "Qwen Delta difference: +0.0777 (Cohen's d=0.347); Huatuo Delta difference: +0.0449 (Cohen's d=0.339).",
        "corrected_p_value": "Qwen Mann-Whitney p = 1.17e-7; Huatuo Mann-Whitney p = 4.28e-7",
        "interpretation_boundaries": "Black blank canvas is out-of-distribution (OOD). Proves pixel sensitivity, NOT causal anatomical grounding or clinical comprehension."
    },
    {
        "rq_id": "RQ3",
        "research_question": "Does visual-sensitivity-aware scoring improve selective prediction compared with candidate preference alone?",
        "primary_hypothesis": "Visual-sensitivity-aware scoring lowers discrete empirical AURC compared to baseline confidence M1.",
        "status": "SUPPORTED WITH MODELING QUALIFICATIONS",
        "key_evidence": "Heuristic combination M_norm fails statistical significance (Qwen p=0.2298, Huatuo p=0.3846). Principled learned multivariate model LR_full significantly lowers AURC (Qwen Delta=-0.0369, p=0.0310; Huatuo Delta=-0.0705, p=0.0010).",
        "confidence_intervals": "Qwen LR_full vs M1: Delta AURC -0.0369, 95% Cluster Boot CI [-0.0589, -0.0159]; Huatuo LR_full vs M1: Delta AURC -0.0705, 95% Cluster Boot CI [-0.0984, -0.0415].",
        "corrected_p_value": "Qwen Holm p_adj = 0.0310; Huatuo Holm p_adj = 0.0010 (under Primary Target Family)",
        "interpretation_boundaries": "Simple arithmetic combination (M_norm) fails. Selective prediction gains require multivariate modeling integrating cross-lingual disagreement (D_lang)."
    }
]

df_rq = pd.DataFrame(rq_matrix_data)

# ------------------------------------------------------------------------------
# 2. CLAIM_EVIDENCE_MATRIX.csv
# ------------------------------------------------------------------------------
claim_matrix_data = [
    {
        "claim_id": "CLM-01",
        "claim_topic": "Language Shift Degradation",
        "claim_statement": "English-to-Indonesian language shift reduces diagnostic accuracy by >10 percentage points.",
        "reported_value": "Qwen: -13.0 pp, Huatuo: -17.3 pp",
        "recomputed_value": "Qwen: -13.02 pp, Huatuo: -17.27 pp",
        "status": "VERIFIED",
        "discrepancy": "None (exact match)",
        "source_evidence": "artifacts/experiments/decisive_validation/rev05/audit_recomputed/tables/rq1_recomputed_metrics.csv"
    },
    {
        "claim_id": "CLM-02",
        "claim_topic": "Specialist vs Generalist Vulnerability",
        "claim_statement": "Medical specialist Huatuo exhibits higher language shift vulnerability than generalist Qwen.",
        "reported_value": "Huatuo flip rate 50.8% vs Qwen 28.8%",
        "recomputed_value": "Huatuo: 50.81% (406/799), Qwen: 28.79% (230/799)",
        "status": "VERIFIED",
        "discrepancy": "None (exact match)",
        "source_evidence": "rq1_recomputed_metrics.csv"
    },
    {
        "claim_id": "CLM-03",
        "claim_topic": "Baseline Confidence AURC (Qwen)",
        "claim_statement": "Qwen baseline confidence AURC is 0.2434.",
        "reported_value": "0.2434",
        "recomputed_value": "0.2422 (M1) vs 0.2434 (LR_visual)",
        "status": "PARTIALLY VERIFIED",
        "discrepancy": "Misattributed: 0.2434 is the test AURC of LR_visual (0.243381). Pure M1 baseline is 0.242187.",
        "source_evidence": "rq3_recomputed_metrics.csv & risk_ablation_dev_test.csv"
    },
    {
        "claim_id": "CLM-04",
        "claim_topic": "Baseline Confidence AURC (Huatuo)",
        "claim_statement": "Huatuo baseline confidence AURC is 0.2513.",
        "reported_value": "0.2513",
        "recomputed_value": "0.2884 (M1) vs 0.2513 (LR_visual)",
        "status": "PARTIALLY VERIFIED",
        "discrepancy": "Misattributed: 0.2513 is the test AURC of LR_visual (0.251330). Pure M1 baseline is 0.288393.",
        "source_evidence": "rq3_recomputed_metrics.csv & risk_ablation_dev_test.csv"
    },
    {
        "claim_id": "CLM-05",
        "claim_topic": "Full Model Selective AURC (Qwen)",
        "claim_statement": "Qwen LR_full achieves test AURC of 0.2053.",
        "reported_value": "0.2053",
        "recomputed_value": "0.205279 (rounds to 0.2053)",
        "status": "VERIFIED",
        "discrepancy": "None (< 0.0001 difference)",
        "source_evidence": "risk_ablation_recomputed.csv"
    },
    {
        "claim_id": "CLM-06",
        "claim_topic": "Full Model Selective AURC (Huatuo)",
        "claim_statement": "Huatuo LR_full achieves test AURC of 0.2179.",
        "reported_value": "0.2179",
        "recomputed_value": "0.217864 (rounds to 0.2179)",
        "status": "VERIFIED",
        "discrepancy": "None (< 0.0001 difference)",
        "source_evidence": "risk_ablation_recomputed.csv"
    },
    {
        "claim_id": "CLM-07",
        "claim_topic": "Cross-Lingual Disagreement Odds Ratio",
        "claim_statement": "Cross-lingual prediction disagreement odds ratio OR(D_lang) > 2.7.",
        "reported_value": "> 2.7",
        "recomputed_value": "Qwen: 3.064 [2.291, 4.098], Huatuo: 2.757 [1.866, 4.074]",
        "status": "VERIFIED",
        "discrepancy": "None (both models exceed 2.7 under cluster-robust Logit estimation)",
        "source_evidence": "logistic_regression_cluster_robust_coefs.csv"
    },
    {
        "claim_id": "CLM-08",
        "claim_topic": "Incremental Value of Visual Sensitivity on Qwen",
        "claim_statement": "Adding visual sensitivity alone (LR_visual) significantly improves selective prediction on Qwen.",
        "reported_value": "Implicitly assumed positive",
        "recomputed_value": "Delta AURC = +0.0012, p = 0.6693, 95% CI [-0.0038, +0.0064]",
        "status": "CONTRADICTED",
        "discrepancy": "No incremental value on Qwen. Visual sensitivity alone does not improve Qwen selective prediction.",
        "source_evidence": "pairwise_ablation_incremental_metrics.csv"
    },
    {
        "claim_id": "CLM-09",
        "claim_topic": "Incremental Value of Visual Sensitivity on Huatuo",
        "claim_statement": "Adding visual sensitivity alone (LR_visual) significantly improves selective prediction on Huatuo.",
        "reported_value": "Significant gain",
        "recomputed_value": "Delta AURC = -0.0371, p = 0.0180, 95% CI [-0.0643, -0.0085]",
        "status": "VERIFIED",
        "discrepancy": "None (significant improvement confirmed on Huatuo)",
        "source_evidence": "pairwise_ablation_incremental_metrics.csv"
    },
    {
        "claim_id": "CLM-10",
        "claim_topic": "Incremental Value of D_lang on Both Models",
        "claim_statement": "Adding D_lang to LR_visual significantly lowers selective risk on both models.",
        "reported_value": "Significant gain",
        "recomputed_value": "Qwen: Delta=-0.0381, p=0.0130; Huatuo: Delta=-0.0335, p=0.0010",
        "status": "VERIFIED",
        "discrepancy": "None (significant incremental gain verified on both models)",
        "source_evidence": "pairwise_ablation_incremental_metrics.csv"
    }
]

df_claims = pd.DataFrame(claim_matrix_data)

# Save CSVs to both locations
for d in [STAGE2_DIR, DOCS_AUDIT_DIR]:
    df_rq.to_csv(d / "RQ_DECISION_MATRIX.csv", index=False)
    df_claims.to_csv(d / "CLAIM_EVIDENCE_MATRIX.csv", index=False)

print("✅ Saved RQ_DECISION_MATRIX.csv and CLAIM_EVIDENCE_MATRIX.csv")
