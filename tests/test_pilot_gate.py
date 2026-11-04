import pytest
import numpy as np
import pandas as pd

def compute_pilot_gate_metrics(df_pilot, df_stored):
    merged = df_pilot.merge(
        df_stored[["qa_id", "pred_en_seq", "p_en_yes_seq", "pred_id_seq", "p_orig_ya_seq", "p_blank_ya_seq", "delta_p_dir_seq"]],
        on="qa_id",
        suffixes=("", "_stored")
    )
    
    # Derive blank predictions
    merged["pred_blank_seq_stored"] = np.where(merged["p_blank_ya_seq_stored"] >= 0.5, "YES", "NO")
    merged["pred_blank_seq"] = np.where(merged["p_blank_ya_seq"] >= 0.5, "YES", "NO")
    
    match_en = merged["pred_en_seq"] == merged["pred_en_seq_stored"]
    match_id_p0 = merged["pred_id_seq"] == merged["pred_id_seq_stored"]
    match_id_p1 = merged["pred_blank_seq"] == merged["pred_blank_seq_stored"]
    
    # 1. Per-prediksi (3 conditions: EN, ID-P0, ID-P1)
    all_3cond = match_en.tolist() + match_id_p0.tolist() + match_id_p1.tolist()
    pred_match_rate_3cond = float(np.mean(all_3cond) * 100.0)
    n_mismatches_3cond = int(len(all_3cond) - sum(all_3cond))
    
    # 2. Per-prediksi (2 conditions: EN, ID-P0)
    all_2cond = match_en.tolist() + match_id_p0.tolist()
    pred_match_rate_2cond = float(np.mean(all_2cond) * 100.0)
    n_mismatches_2cond = int(len(all_2cond) - sum(all_2cond))
    
    # 3. Per-sampel (all conditions match)
    sample_all_match = match_en & match_id_p0 & match_id_p1
    sample_match_rate = float(sample_all_match.mean() * 100.0)
    n_sample_mismatches = int((~sample_all_match).sum())
    
    dp_en = np.abs(merged["p_en_yes_seq"] - merged["p_en_yes_seq_stored"])
    dp_id_p0 = np.abs(merged["p_orig_ya_seq"] - merged["p_orig_ya_seq_stored"])
    dp_id_p1 = np.abs(merged["p_blank_ya_seq"] - merged["p_blank_ya_seq_stored"])
    d_deltap = np.abs(merged["delta_p_dir_seq"] - merged["delta_p_dir_seq_stored"])
    
    all_dp = np.concatenate([dp_en.values, dp_id_p0.values, dp_id_p1.values])
    max_dp = float(np.max(all_dp))
    mean_dp_orig = float(np.mean(dp_id_p0))
    mean_d_deltap = float(np.mean(d_deltap))
    nan_inf_count = int(np.isnan(all_dp).sum() + np.isinf(all_dp).sum())
    
    # Mismatch detail list
    mismatches = []
    for _, row in merged.iterrows():
        qid = row["qa_id"]
        if row["pred_en_seq"] != row["pred_en_seq_stored"]:
            mismatches.append({
                "qa_id": qid, "condition": "EN-P0",
                "abs_dp": abs(row["p_en_yes_seq"] - row["p_en_yes_seq_stored"]),
                "stored_|p-0.5|": abs(row["p_en_yes_seq_stored"] - 0.5),
                "pilot_|p-0.5|": abs(row["p_en_yes_seq"] - 0.5)
            })
        if row["pred_id_seq"] != row["pred_id_seq_stored"]:
            mismatches.append({
                "qa_id": qid, "condition": "ID-P0",
                "abs_dp": abs(row["p_orig_ya_seq"] - row["p_orig_ya_seq_stored"]),
                "stored_|p-0.5|": abs(row["p_orig_ya_seq_stored"] - 0.5),
                "pilot_|p-0.5|": abs(row["p_orig_ya_seq"] - 0.5)
            })
        if row["pred_blank_seq"] != row["pred_blank_seq_stored"]:
            mismatches.append({
                "qa_id": qid, "condition": "ID-P1",
                "abs_dp": abs(row["p_blank_ya_seq"] - row["p_blank_ya_seq_stored"]),
                "stored_|p-0.5|": abs(row["p_blank_ya_seq_stored"] - 0.5),
                "pilot_|p-0.5|": abs(row["p_blank_ya_seq"] - 0.5)
            })
            
    # Gate (a) Criteria:
    # (i) >=98% per-prediksi
    # (ii) SETIAP mismatch memiliki |p-0.5| < 0.02 pada KEDUA nilai (stored dan pilot)
    # (iii) mean |dp_orig| < 0.02, dan 0 NaN/Inf
    all_mismatches_marginal = all(m["stored_|p-0.5|"] < 0.02 and m["pilot_|p-0.5|"] < 0.02 for m in mismatches)
    gate_passed = bool((pred_match_rate_3cond >= 98.0) and all_mismatches_marginal and (mean_dp_orig < 0.02) and (nan_inf_count == 0))
    
    return {
        "pred_match_rate_3cond": pred_match_rate_3cond,
        "n_mismatches_3cond": n_mismatches_3cond,
        "pred_match_rate_2cond": pred_match_rate_2cond,
        "n_mismatches_2cond": n_mismatches_2cond,
        "sample_match_rate": sample_match_rate,
        "n_sample_mismatches": n_sample_mismatches,
        "mean_dp_orig": mean_dp_orig,
        "max_dp": max_dp,
        "mean_d_deltap": mean_d_deltap,
        "nan_inf_count": nan_inf_count,
        "mismatches": mismatches,
        "all_mismatches_marginal": all_mismatches_marginal,
        "gate_passed": gate_passed
    }

def compute_pilot_descriptive_metrics(df_pilot):
    yes_rate_en = (df_pilot["pred_en_v6"] == "YES").mean() * 100.0
    yes_rate_id = (df_pilot["pred_id_v6"] == "YES").mean() * 100.0
    
    tp_id = ((df_pilot["pred_id_v6"] == "YES") & (df_pilot["reference_answer"] == "YES")).sum()
    fn_id = ((df_pilot["pred_id_v6"] == "NO") & (df_pilot["reference_answer"] == "YES")).sum()
    tpr_id = (tp_id / (tp_id + fn_id) * 100.0) if (tp_id + fn_id) > 0 else 0.0
    
    tn_id = ((df_pilot["pred_id_v6"] == "NO") & (df_pilot["reference_answer"] == "NO")).sum()
    fp_id = ((df_pilot["pred_id_v6"] == "YES") & (df_pilot["reference_answer"] == "NO")).sum()
    tnr_id = (tn_id / (tn_id + fp_id) * 100.0) if (tn_id + fp_id) > 0 else 0.0
    
    return {
        "yes_rate_en": yes_rate_en,
        "yes_rate_id": yes_rate_id,
        "tpr_id": tpr_id,
        "tnr_id": tnr_id
    }

def test_pilot_gate_passing():
    np.random.seed(42)
    n = 40
    qa_ids = [f"qa_{i}" for i in range(n)]
    
    df_stored = pd.DataFrame({
        "qa_id": qa_ids,
        "pred_en_seq": ["YES"] * 20 + ["NO"] * 20,
        "p_en_yes_seq": np.random.uniform(0.6, 0.9, n),
        "pred_id_seq": ["YES"] * 18 + ["NO"] * 22,
        "p_orig_ya_seq": np.random.uniform(0.55, 0.85, n),
        "p_blank_ya_seq": np.random.uniform(0.1, 0.4, n),
        "delta_p_dir_seq": np.random.uniform(0.1, 0.4, n)
    })
    
    # Simulating passing pilot: identical predictions, perturbation < 0.005
    df_pilot = df_stored.copy()
    df_pilot["p_en_yes_seq"] += np.random.normal(0, 0.002, n)
    df_pilot["p_orig_ya_seq"] += np.random.normal(0, 0.002, n)
    df_pilot["p_blank_ya_seq"] += np.random.normal(0, 0.002, n)
    df_pilot["delta_p_dir_seq"] = df_pilot["p_orig_ya_seq"] - df_pilot["p_blank_ya_seq"]
    
    res = compute_pilot_gate_metrics(df_pilot, df_stored)
    assert res["pred_match_rate_3cond"] == 100.0
    assert res["sample_match_rate"] == 100.0
    assert res["n_mismatches_3cond"] == 0
    assert len(res["mismatches"]) == 0
    assert res["all_mismatches_marginal"] is True
    assert res["mean_dp_orig"] < 0.02
    assert res["nan_inf_count"] == 0
    assert res["gate_passed"] is True

def test_pilot_gate_failing_on_mismatch_rate():
    n = 40
    qa_ids = [f"qa_{i}" for i in range(n)]
    df_stored = pd.DataFrame({
        "qa_id": qa_ids,
        "pred_en_seq": ["YES"] * 40,
        "p_en_yes_seq": [0.8] * 40,
        "pred_id_seq": ["YES"] * 40,
        "p_orig_ya_seq": [0.8] * 40,
        "p_blank_ya_seq": [0.2] * 40,
        "delta_p_dir_seq": [0.6] * 40
    })
    
    df_pilot = df_stored.copy()
    # 5 flips in ID out of 120 total predictions -> match = 115/120 = 95.83% (< 98%)
    df_pilot.loc[:4, "pred_id_seq"] = "NO"
    df_pilot.loc[:4, "p_orig_ya_seq"] = 0.2
    
    res = compute_pilot_gate_metrics(df_pilot, df_stored)
    assert res["pred_match_rate_3cond"] < 98.0
    assert res["n_mismatches_3cond"] == 5
    assert len(res["mismatches"]) == 5
    for m in res["mismatches"]:
        assert "stored_|p-0.5|" in m
        assert "pilot_|p-0.5|" in m
        assert m["stored_|p-0.5|"] == pytest.approx(0.3)
    assert res["all_mismatches_marginal"] is False
    assert res["gate_passed"] is False

def test_pilot_gate_failing_on_non_marginal_mismatch():
    """Even with match_rate >= 98% (e.g. 119/120 = 99.17%), if mismatch is non-marginal (|p-0.5| >= 0.02), gate FAILS."""
    n = 40
    qa_ids = [f"qa_{i}" for i in range(n)]
    df_stored = pd.DataFrame({
        "qa_id": qa_ids,
        "pred_en_seq": ["YES"] * 40,
        "p_en_yes_seq": [0.8] * 40,
        "pred_id_seq": ["YES"] * 40,
        "p_orig_ya_seq": [0.8] * 40,
        "p_blank_ya_seq": [0.2] * 40,
        "delta_p_dir_seq": [0.6] * 40
    })
    
    df_pilot = df_stored.copy()
    # Exactly 1 flip out of 120 (match rate = 99.17% >= 98%), but high confidence (non-marginal)
    df_pilot.loc[0, "pred_id_seq"] = "NO"
    df_pilot.loc[0, "p_orig_ya_seq"] = 0.2
    
    res = compute_pilot_gate_metrics(df_pilot, df_stored)
    assert res["pred_match_rate_3cond"] >= 98.0  # (i) passes
    assert res["n_mismatches_3cond"] == 1
    assert res["all_mismatches_marginal"] is False  # (ii) fails!
    assert res["gate_passed"] is False  # overall gate fails!

def test_pilot_gate_passing_with_marginal_mismatch():
    """With match_rate >= 98% and mismatch is marginal (|p-0.5| < 0.02 on both), gate PASSES."""
    n = 40
    qa_ids = [f"qa_{i}" for i in range(n)]
    df_stored = pd.DataFrame({
        "qa_id": qa_ids,
        "pred_en_seq": ["YES"] * 40,
        "p_en_yes_seq": [0.8] * 40,
        "pred_id_seq": ["YES"] * 40,
        "p_orig_ya_seq": [0.8] * 40,
        "p_blank_ya_seq": [0.2] * 40,
        "delta_p_dir_seq": [0.6] * 40
    })
    # Set sample 0 very close to decision boundary: stored_p = 0.505
    df_stored.loc[0, "p_orig_ya_seq"] = 0.505
    df_stored.loc[0, "pred_id_seq"] = "YES"
    
    df_pilot = df_stored.copy()
    # Jitter flips it to 0.495 (pilot_p = 0.495 -> NO)
    df_pilot.loc[0, "p_orig_ya_seq"] = 0.495
    df_pilot.loc[0, "pred_id_seq"] = "NO"
    
    res = compute_pilot_gate_metrics(df_pilot, df_stored)
    assert res["pred_match_rate_3cond"] >= 98.0  # (i) passes (119/120 = 99.17%)
    assert res["n_mismatches_3cond"] == 1
    assert res["all_mismatches_marginal"] is True  # (ii) passes because 0.005 < 0.02
    assert res["mean_dp_orig"] < 0.02  # (iii) passes
    assert res["gate_passed"] is True  # overall gate passes!

def test_pilot_gate_failing_on_nan():
    n = 40
    qa_ids = [f"qa_{i}" for i in range(n)]
    df_stored = pd.DataFrame({
        "qa_id": qa_ids,
        "pred_en_seq": ["YES"] * 40,
        "p_en_yes_seq": [0.8] * 40,
        "pred_id_seq": ["YES"] * 40,
        "p_orig_ya_seq": [0.8] * 40,
        "p_blank_ya_seq": [0.4] * 40,
        "delta_p_dir_seq": [0.4] * 40
    })
    
    df_pilot = df_stored.copy()
    df_pilot.loc[0, "p_orig_ya_seq"] = np.nan
    
    res = compute_pilot_gate_metrics(df_pilot, df_stored)
    assert res["nan_inf_count"] > 0
    assert res["gate_passed"] is False

def test_pilot_descriptive():
    df_pilot = pd.DataFrame({
        "qa_id": ["qa_1", "qa_2", "qa_3", "qa_4"],
        "pred_en_v6": ["YES", "YES", "NO", "NO"],
        "pred_id_v6": ["YES", "NO", "YES", "NO"],
        "reference_answer": ["YES", "YES", "NO", "NO"]
    })
    desc = compute_pilot_descriptive_metrics(df_pilot)
    assert desc["yes_rate_en"] == 50.0
    assert desc["yes_rate_id"] == 50.0
