# Experiment Outputs (`artifacts/experiments/`)

This directory catalogs experimental results across all research iterations.

---

## Directory Organization

1. **[`decisive_validation/rev05/`](decisive_validation/rev05/) (Authoritative):**
   - **`metrics/`**:
     - `rq1_bilingual_metrics.csv`: Cross-lingual accuracy and flip rates.
     - `rq2_visual_intervention_metrics.csv`: Black blank deletion sensitivity ($\Delta P_{\text{dir}}$).
     - `rq3_selective_prediction_primary_test.csv`: Comparative AURC ($M_1$ vs $M_{\text{norm}}$).
     - `risk_ablation_dev_test.csv`: Out-of-sample logistic regression risk estimation.
     - `adaptive_abstention_*.csv`: Operational referral thresholds and coverage targets.
   - **`predictions/`**:
     - `huatuo/`: Checkpoint CSVs for HuatuoGPT-Vision-7B.
     - `qwen/`: Checkpoint CSVs for Qwen2.5-VL-7B.
   - **`figures/`**:
     - `risk_coverage_primary_test_HuatuoGPT_Vision_7B.png`
     - `risk_coverage_primary_test_Qwen2.5_VL_7B.png`
   - **`perturbed_images/`**:
     - 314 black blank control radiographs.
2. **`decisive_validation/historical/`:**
   - Archives of iterations `rev02`, `rev03`, and `rev04`.
3. **`legacy_screening/`:**
   - Initial screening leaderboard and pilot outputs.
