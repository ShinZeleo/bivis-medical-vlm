import pytest
import numpy as np
import pandas as pd
from src.metrics import (
    compute_calibration_metrics,
    compute_visual_flip_rate,
    compute_odds_ratio_dlang,
    run_oof_abstention_pipeline,
    interpolate_risk_coverage
)


def test_calibration_metrics():
    # Perfect calibration
    conf = np.array([0.9, 0.9, 0.1, 0.1])
    corr = np.array([1, 1, 0, 0])
    res = compute_calibration_metrics(conf, corr, n_bins=10)
    assert 0.0 <= res["ece"] <= 0.2
    assert 0.0 <= res["brier"] <= 0.1
    assert res["conf_wrong_rate"] == 0.0 # No confident wrong predictions
    
    # Confidently wrong cases
    conf_w = np.array([0.95, 0.95, 0.2, 0.2])
    corr_w = np.array([0, 0, 1, 1])
    res_w = compute_calibration_metrics(conf_w, corr_w, n_bins=10, conf_wrong_threshold=0.90)
    assert res_w["conf_wrong_rate"] == 1.0
    assert res_w["conf_wrong_share"] == 0.5


def test_visual_flip_rate():
    p0 = np.array(["YES", "YES", "NO", "NO"])
    p1 = np.array(["YES", "NO", "NO", "YES"])
    res = compute_visual_flip_rate(p0, p1)
    assert res["visual_flip_rate"] == 50.0
    assert res["flip_count"] == 2
    assert res["total"] == 4


def test_odds_ratio_dlang():
    dl = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    err = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    clusters = np.array(["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"])
    res = compute_odds_ratio_dlang(dl, err, clusters=clusters)
    assert res["odds_ratio_naive"] > 1.0
    assert "odds_ratio_cluster_robust" in res
    assert res["odds_ratio_cluster_robust"] > 1.0


def test_oof_abstention_zero_leakage():
    # Generate mock Dev and Test data
    np.random.seed(42)
    n_dev = 100
    n_test = 200
    
    dev_clusters = [f"dev_cluster_{i % 20}" for i in range(n_dev)]
    test_clusters = [f"test_cluster_{i % 40}" for i in range(n_test)]
    
    # Assert dev and test clusters are completely disjoint
    assert set(dev_clusters).isdisjoint(set(test_clusters))
    
    df_dev = pd.DataFrame({
        "canonical_image_id": dev_clusters,
        "score_m1_seq": np.random.uniform(0.3, 0.9, size=n_dev),
        "delta_p_dir_seq": np.random.uniform(-0.2, 0.8, size=n_dev),
        "d_lang": np.random.choice([0, 1], size=n_dev),
        "error_id": np.random.choice([0, 1], size=n_dev),
        "correct_id_seq": np.random.choice([0, 1], size=n_dev)
    })
    
    df_test = pd.DataFrame({
        "canonical_image_id": test_clusters,
        "score_m1_seq": np.random.uniform(0.3, 0.9, size=n_test),
        "delta_p_dir_seq": np.random.uniform(-0.2, 0.8, size=n_test),
        "d_lang": np.random.choice([0, 1], size=n_test),
        "error_id": np.random.choice([0, 1], size=n_test),
        "correct_id_seq": np.random.choice([0, 1], size=n_test)
    })
    
    res = run_oof_abstention_pipeline(df_dev, df_test, n_splits=5)
    
    # Verify thresholds exist for all target coverages
    for cov in [0.5, 0.6, 0.7, 0.8, 0.9]:
        assert cov in res["calibrated_thresholds"]
        
    # Verify thresholds increase monotonically with coverage (higher coverage = higher risk tolerance)
    taus = [res["calibrated_thresholds"][c] for c in [0.5, 0.6, 0.7, 0.8, 0.9]]
    for i in range(len(taus) - 1):
        assert taus[i] <= taus[i + 1]


def test_interpolate_risk_coverage():
    covs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    risks = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    interp = interpolate_risk_coverage(covs, risks, target_coverages=[0.5, 0.7])
    assert abs(interp[0.5] - 0.25) < 1e-6
    assert abs(interp[0.7] - 0.35) < 1e-6
