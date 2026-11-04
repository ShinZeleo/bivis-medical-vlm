import pytest
import numpy as np
import pandas as pd
from src.stats_engine import (
    calculate_discrete_aurc,
    get_risk_at_coverage,
    paired_cluster_permutation_test,
    paired_cluster_bootstrap,
    apply_holm_bonferroni,
    wilson_ci,
    run_mcnemar
)


def test_aurc_mathematical_bounds():
    # Perfect predictor: all correct first
    y_perfect = np.array([1, 1, 1, 1, 0, 0])
    s_perfect = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    aurc_perf, cov, risks = calculate_discrete_aurc(y_perfect, s_perfect)
    assert 0.0 <= aurc_perf <= 1.0
    assert risks[0] == 0.0 # 100% acc at k=1

    # Inverted predictor: all wrong first
    y_worst = np.array([0, 0, 1, 1])
    s_worst = np.array([0.9, 0.8, 0.2, 0.1])
    aurc_w, _, risks_w = calculate_discrete_aurc(y_worst, s_worst)
    assert aurc_w > aurc_perf

    # Empty array
    aurc_empty, _, _ = calculate_discrete_aurc(np.array([]), np.array([]))
    assert aurc_empty == 0.0


def test_holm_bonferroni_strict_monotonicity():
    # Test unordered input p-values
    p_raw = {"test1": 0.04, "test2": 0.001, "test3": 0.02, "test4": 0.03}
    adj = apply_holm_bonferroni(p_raw)
    
    # Assert all keys present
    assert set(adj.keys()) == set(p_raw.keys())
    
    # Assert values are <= 1.0
    for v in adj.values():
        assert 0.0 <= v <= 1.0
        
    # Sorted by raw should also be sorted by adjusted (monotonicity)
    sorted_by_raw = sorted(p_raw.keys(), key=lambda k: p_raw[k])
    adj_vals = [adj[k] for k in sorted_by_raw]
    for i in range(len(adj_vals) - 1):
        assert adj_vals[i] <= adj_vals[i + 1]


def test_non_standard_index_immunity():
    # DataFrame with non-standard, gapped, string-like indices
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "canonical_image_id": [f"img_{i % 10}" for i in range(n)],
        "correct_id_seq": np.random.choice([0, 1], size=n),
        "score_m1_seq": np.random.uniform(0.1, 0.9, size=n),
        "score_m2_seq": np.random.uniform(0.1, 0.9, size=n)
    }, index=np.random.permutation(np.arange(100, 100 + n)))
    
    aurc_orig, _, _ = calculate_discrete_aurc(df["correct_id_seq"].values, df["score_m1_seq"].values)
    
    # Reset index DataFrame
    df_reset = df.reset_index(drop=True)
    aurc_reset, _, _ = calculate_discrete_aurc(df_reset["correct_id_seq"].values, df_reset["score_m1_seq"].values)
    
    assert abs(aurc_orig - aurc_reset) < 1e-12
    
    # Test permutation test with non-standard index
    p1 = paired_cluster_permutation_test(df, "score_m2_seq", "score_m1_seq", n_perm=100, seed=42)
    p2 = paired_cluster_permutation_test(df_reset, "score_m2_seq", "score_m1_seq", n_perm=100, seed=42)
    assert abs(p1 - p2) < 1e-12


def test_tie_handling_modes():
    # Data with intentional heavy ties
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # all tied
    
    aurc_best, _, _ = calculate_discrete_aurc(y, s, tie_mode="best")
    aurc_worst, _, _ = calculate_discrete_aurc(y, s, tie_mode="worst")
    aurc_rand, _, _ = calculate_discrete_aurc(y, s, tie_mode="random", n_tie_reps=200, rng=np.random.RandomState(42))
    
    # Best should have strictly lower risk than worst
    assert aurc_best < aurc_worst
    # Random expectation must lie strictly between best and worst
    assert aurc_best <= aurc_rand <= aurc_worst


def test_wilson_ci_bounds():
    low, high = wilson_ci(50, 100, conf=0.95)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    
    # Edge case 0 and N
    l0, h0 = wilson_ci(0, 100)
    assert l0 == 0.0 and h0 > 0.0
    lN, hN = wilson_ci(100, 100)
    assert lN < 1.0 and hN == 1.0
