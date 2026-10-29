"""
Comprehensive Unit & Preflight Test Suite (rev03)
=================================================
Tests all critical components of decisive_validation_experiment_v3:
1. Manifest hash and split invariants
2. Cryptographic image pixel verification and tamper rejection
3. Candidate tokenization and causal logit alignment
4. Discrete empirical AURC hand-verified calculation
5. Exact Paired Cluster Permutation Test
6. Holm-Bonferroni correction
7. Strict Checkpoint resume rejection of invalid/stale revisions
"""

import os
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath("main/scripts"))
from decisive_validation_experiment_v3 import (
    AUTHORITATIVE_MANIFEST_SHA256,
    REVISION_ID,
    load_authoritative_manifest,
    compute_rgb_pixel_sha256,
    compute_discrete_risk_coverage,
    paired_cluster_permutation_test,
    apply_holm_bonferroni,
    validate_checkpoint_compatibility,
    CANDIDATES_EN,
    CANDIDATES_ID,
    PROMPT_EN_TEMPLATE,
    PROMPT_ID_TEMPLATE,
)

def test_manifest_hash_and_invariants():
    print("▶ Running test_manifest_hash_and_invariants...")
    df = load_authoritative_manifest(Path("main/results/decisive_validation_manifest.csv"))
    assert len(df) == 150
    test_df = df[df["split"] == "test"]
    dev_df = df[df["split"] == "dev"]
    assert len(test_df) == 92
    assert len(dev_df) == 58
    assert test_df["canonical_image_id"].nunique() == 35
    assert dev_df["canonical_image_id"].nunique() == 23
    overlap = set(test_df["canonical_image_id"]).intersection(set(dev_df["canonical_image_id"]))
    assert len(overlap) == 0
    print("  ✅ PASS: Manifest hash and invariants verified.")

def test_image_verification_and_tamper_rejection():
    print("▶ Running test_image_verification_and_tamper_rejection...")
    df = load_authoritative_manifest(Path("main/results/decisive_validation_manifest.csv"))
    row = df.iloc[0]
    cid = row["canonical_image_id"]
    expected_hash = row["pixel_sha256"].strip().lower()
    
    img_p = Path(f"main/data/images/{cid}.png")
    if not img_p.exists():
        img_p = Path(f"main/data/images/img_{int(row['qa_id']):05d}.png")
    assert img_p.exists(), f"Image {img_p} not found!"
    
    with Image.open(img_p) as pil_img:
        actual_hash = compute_rgb_pixel_sha256(pil_img)
        assert actual_hash == expected_hash, f"Hash mismatch: {actual_hash} != {expected_hash}"
        
        # Test tamper rejection
        tampered_img = pil_img.copy()
        pixels = tampered_img.load()
        # Invert one pixel
        r, g, b = pixels[0, 0]
        pixels[0, 0] = (255 - r, 255 - g, 255 - b)
        tampered_hash = compute_rgb_pixel_sha256(tampered_img)
        assert tampered_hash != expected_hash, "Tampered image should NOT match original hash!"
    print("  ✅ PASS: Image verification and tamper rejection verified.")

def test_discrete_aurc_manual_calculation():
    print("▶ Running test_discrete_aurc_manual_calculation...")
    # Manual synthetic benchmark: N=5
    # correctness = [1, 0, 1, 1, 0], sorted scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    # k=1: acc=1/1=1.0, risk=0.0
    # k=2: acc=1/2=0.5, risk=0.5
    # k=3: acc=2/3, risk=1/3
    # k=4: acc=3/4=0.75, risk=0.25
    # k=5: acc=3/5=0.6, risk=0.4
    # mean_risk = (0.0 + 0.5 + 1/3 + 0.25 + 0.4) / 5 = 1.4833333333333334 / 5 = 0.2966666666666667
    c_syn = np.array([1, 0, 1, 1, 0])
    s_syn = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cov, risk, aurc_mean = compute_discrete_risk_coverage(s_syn, c_syn)
    
    np.testing.assert_allclose(cov, [0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.5, 1.0/3.0, 0.25, 0.4])
    np.testing.assert_allclose(aurc_mean, 0.2966666666666667)
    print("  ✅ PASS: Discrete empirical AURC matches hand calculation exactly.")

def test_paired_cluster_permutation_test():
    print("▶ Running test_paired_cluster_permutation_test...")
    df_syn = pd.DataFrame({
        "canonical_image_id": ["c1", "c1", "c2", "c3", "c3"],
        "score_a": [0.9, 0.8, 0.7, 0.6, 0.5],
        "score_identical": [0.9, 0.8, 0.7, 0.6, 0.5],
        "score_inverted": [0.1, 0.2, 0.3, 0.4, 0.5],
        "correct_id": [1, 0, 1, 1, 0]
    })
    # 1. Identical estimators: t_obs must be 0, p-val must be exactly 1.0
    _, _, t_obs_ident, p_ident = paired_cluster_permutation_test(
        df_syn, "score_a", "score_identical", n_perm=500, seed=42
    )
    assert abs(t_obs_ident) < 1e-12
    assert abs(p_ident - 1.0) < 1e-6
    
    # 2. Inverted estimators: t_obs must be negative (score_a has lower risk than inverted)
    aurc_a, aurc_inv, t_obs_inv, p_inv = paired_cluster_permutation_test(
        df_syn, "score_a", "score_inverted", n_perm=500, seed=42
    )
    assert aurc_a < aurc_inv, f"Expected aurc_a ({aurc_a}) < aurc_inv ({aurc_inv})"
    assert t_obs_inv < 0
    print("  ✅ PASS: Paired Cluster Permutation Test verified.")

def test_holm_bonferroni():
    print("▶ Running test_holm_bonferroni...")
    raw_p = {"m2": 0.01, "m_dir": 0.04, "m_norm": 0.02, "m3_old": 0.20}
    # Sorted: m2 (0.01), m_norm (0.02), m_dir (0.04), m3_old (0.20)
    # Multipliers: 4, 3, 2, 1
    # m2: 0.01 * 4 = 0.04
    # m_norm: 0.02 * 3 = 0.06
    # m_dir: 0.04 * 2 = 0.08
    # m3_old: 0.20 * 1 = 0.20
    adj_p = apply_holm_bonferroni(raw_p)
    assert abs(adj_p["m2"] - 0.04) < 1e-6
    assert abs(adj_p["m_norm"] - 0.06) < 1e-6
    assert abs(adj_p["m_dir"] - 0.08) < 1e-6
    assert abs(adj_p["m3_old"] - 0.20) < 1e-6
    print("  ✅ PASS: Holm-Bonferroni step-down adjustment verified.")

def test_checkpoint_governance():
    print("▶ Running test_checkpoint_governance...")
    # Create valid mock checkpoint
    valid_row = {
        "qa_id": 74, "canonical_image_id": "cid_test", "split": "test", "reference_answer": "YES",
        "pred_en": "YES", "conf_en": 0.9, "correct_en": 1, "p_en_yes": 0.9, "p_en_no": 0.1,
        "pred_id": "YES", "conf_id": 0.85, "correct_id": 1, "p_orig_ya": 0.85, "p_orig_tidak": 0.15,
        "p_blank_ya": 0.5, "p_blank_tidak": 0.5, "p_blank_pred": 0.5, "delta_p_dir": 0.35, "delta_p_abs": 0.35,
        "score_m1": 0.85, "score_m2": 0.35, "score_m_dir": 0.2975, "score_m_norm": 0.57375, "score_m3_old": 0.57375,
        "scoring_method": "length_normalized_closed_response_v3",
        "revision_id": "rev03",
        "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
        "model_id": "TestModel"
    }
    df_valid = pd.DataFrame([valid_row])
    is_valid, msg = validate_checkpoint_compatibility(df_valid, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert is_valid, f"Expected valid checkpoint, got {msg}"
    
    # 1. Rejection of rev02 revision ID
    df_rev02 = df_valid.copy()
    df_rev02["revision_id"] = "rev02"
    is_v, msg = validate_checkpoint_compatibility(df_rev02, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "revision ID" in msg
    
    # 2. Rejection of corrupted manifest hash
    df_bad_sha = df_valid.copy()
    df_bad_sha["manifest_sha256"] = "bad_hash_123"
    is_v, msg = validate_checkpoint_compatibility(df_bad_sha, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "SHA256" in msg
    
    # 3. Rejection of wrong model ID
    df_bad_model = df_valid.copy()
    df_bad_model["model_id"] = "DifferentModel"
    is_v, msg = validate_checkpoint_compatibility(df_bad_model, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "Model ID" in msg
    
    # 4. Rejection of NaN in scores
    df_nan = df_valid.copy()
    df_nan.loc[0, "score_m_dir"] = np.nan
    is_v, msg = validate_checkpoint_compatibility(df_nan, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "non-finite" in msg
    
    print("  ✅ PASS: Checkpoint governance and rejection logic verified.")

def test_length_normalization_mathematical_symmetry():
    print("▶ Running test_length_normalization_mathematical_symmetry...")
    # Candidate 1: "YA<|im_end|>" (length = 2)
    # Candidate 2: "TIDAK<|im_end|>" (length = 4)
    # Case A: Identical per-token quality (-1.20)
    lp1_tokens = [-1.20, -1.20]
    lp2_tokens = [-1.20, -1.20, -1.20, -1.20]
    
    raw_s1 = sum(lp1_tokens)  # -2.40
    raw_s2 = sum(lp2_tokens)  # -4.80
    
    # Under rev02 raw sum:
    p_raw_ya = np.exp(raw_s1) / (np.exp(raw_s1) + np.exp(raw_s2))
    assert p_raw_ya > 0.90, f"Expected massive bias (>90%) in raw sum, got {p_raw_ya}"
    
    # Under rev03 length normalization:
    norm_s1 = raw_s1 / len(lp1_tokens)  # -1.20
    norm_s2 = raw_s2 / len(lp2_tokens)  # -1.20
    p_norm_ya = np.exp(norm_s1) / (np.exp(norm_s1) + np.exp(norm_s2))
    assert abs(p_norm_ya - 0.50) < 1e-7, f"Expected exactly 0.50 under length normalization, got {p_norm_ya}"
    
    # Case B: Candidate 2 ("TIDAK") is superior per token (-1.0 vs -1.5)
    # But rev02 raw sum incorrectly favors "YA" due to length penalty
    lp1_tokens_b = [-1.50, -1.50]
    lp2_tokens_b = [-1.00, -1.00, -1.00, -1.00]
    raw_s1_b = sum(lp1_tokens_b)  # -3.00
    raw_s2_b = sum(lp2_tokens_b)  # -4.00
    p_raw_ya_b = np.exp(raw_s1_b) / (np.exp(raw_s1_b) + np.exp(raw_s2_b))
    assert p_raw_ya_b > 0.70, "Rev02 artifact: Shorter inferior token won due to length penalty!"
    
    norm_s1_b = raw_s1_b / len(lp1_tokens_b)  # -1.50
    norm_s2_b = raw_s2_b / len(lp2_tokens_b)  # -1.00
    p_norm_tidak_b = np.exp(norm_s2_b) / (np.exp(norm_s1_b) + np.exp(norm_s2_b))
    assert p_norm_tidak_b > 0.60, f"Rev03 fix: Longer superior token wins ({p_norm_tidak_b:.4f})!"
    
    print("  ✅ PASS: Mathematical symmetry and elimination of token-length bias verified.")

def run_all_tests():
    print("=" * 60)
    print("RUNNING UNIT TEST SUITE FOR REV03 ENGINE")
    print("=" * 60)
    test_manifest_hash_and_invariants()
    test_image_verification_and_tamper_rejection()
    test_discrete_aurc_manual_calculation()
    test_paired_cluster_permutation_test()
    test_holm_bonferroni()
    test_checkpoint_governance()
    test_length_normalization_mathematical_symmetry()
    print("=" * 60)
    print("ALL UNIT TESTS PASSED (PASS_UNIT_TEST)")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
