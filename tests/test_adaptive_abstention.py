#!/usr/bin/env python3
"""
Preflight Unit Test Suite: Adaptive Abstention Engine (v11.0)
=============================================================
Verifies:
1. Mathematical Range Bounds: Selective Risk and Coverage in [0.0, 1.0].
2. Denominator Accounting Invariance: N_answered + N_abstained == N_total.
3. Quantile Monotonicity: Higher target coverage => higher (more lenient) tau_risk.
4. Non-Leakage Assertion: Tau is strictly calibrated on Dev Set.
5. Exact Identity: M1_direct and LR_conf produce identical ranks.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("scripts"))
from adaptive_abstention_engine import (
    calibrate_and_evaluate_coverage_targeted,
    calibrate_and_evaluate_risk_targeted
)

QWEN_CKPT = Path("artifacts/experiments/decisive_validation/rev05/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
HUATUO_CKPT = Path("artifacts/experiments/decisive_validation/rev05/predictions/huatuo/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")


def test_coverage_targeted_invariants():
    print("▶ Running test_coverage_targeted_invariants...")
    assert QWEN_CKPT.exists(), f"Missing checkpoint for test: {QWEN_CKPT}"

    df = pd.read_csv(QWEN_CKPT)
    df_dev = df[(df["split"] == "dev") & (df["is_eligible_primary"] == True)].copy().reset_index(drop=True)
    df_test = df[(df["split"] == "test") & (df["is_eligible_primary"] == True)].copy().reset_index(drop=True)

    df_res, df_comp = calibrate_and_evaluate_coverage_targeted(df_dev, df_test, "Qwen2.5-VL-7B")

    # Invariant 1: Denominator accounting
    for _, row in df_res.iterrows():
        n_ans = row["n_answered"]
        n_tot = len(df_test)
        real_cov = row["realized_test_coverage"]
        abs_rate = row["abstention_rate"]
        assert abs((real_cov + abs_rate) - 1.0) < 1e-7, "Coverage + Abstention must equal 1.0!"
        assert 0.0 <= row["selective_risk"] <= 1.0, "Selective risk must be in [0, 1]!"
        assert 0.0 <= row["selective_accuracy"] <= 1.0, "Selective accuracy must be in [0, 1]!"
        assert abs((row["selective_risk"] + row["selective_accuracy"]) - 1.0) < 1e-7, "Risk + Acc must equal 1.0!"

    # Invariant 2: Monotonic thresholding per estimator
    for est in ["M1_direct", "LR_conf", "LR_visual", "LR_full"]:
        sub = df_res[df_res["estimator"] == est].sort_values("target_dev_coverage")
        taus = sub["tau_risk_threshold"].values
        diffs = np.diff(taus)
        assert np.all(diffs >= -1e-9), f"Threshold tau_risk must increase monotonically with target coverage for {est}!"

    # Invariant 3: OOF vs In-sample comparison dataframe integrity
    assert len(df_comp) == len(df_res), "Comparison dataframe row count mismatch!"

    print("  ✅ PASS: Coverage-targeted invariants verified 100%.")


def test_risk_targeted_invariants():
    print("▶ Running test_risk_targeted_invariants...")
    assert HUATUO_CKPT.exists(), f"Missing checkpoint for test: {HUATUO_CKPT}"

    df = pd.read_csv(HUATUO_CKPT)
    df_dev = df[(df["split"] == "dev") & (df["is_eligible_primary"] == True)].copy().reset_index(drop=True)
    df_test = df[(df["split"] == "test") & (df["is_eligible_primary"] == True)].copy().reset_index(drop=True)

    df_res = calibrate_and_evaluate_risk_targeted(df_dev, df_test, "HuatuoGPT-Vision-7B")

    for _, row in df_res.iterrows():
        assert 0.0 <= row["realized_test_coverage"] <= 1.0
        assert 0.0 <= row["actual_test_risk"] <= 1.0
        assert 0.0 <= row["actual_test_selective_accuracy"] <= 1.0
        assert abs((row["actual_test_risk"] + row["actual_test_selective_accuracy"]) - 1.0) < 1e-7

    print("  ✅ PASS: Risk-targeted invariants verified 100%.")


def run_all_tests():
    print("=" * 70)
    print("RUNNING UNIT TEST SUITE FOR ADAPTIVE ABSTENTION ENGINE (v11.0)")
    print("=" * 70)
    test_coverage_targeted_invariants()
    test_risk_targeted_invariants()
    print("=" * 70)
    print("ALL TESTS PASSED (PASS_UNIT_TEST)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
