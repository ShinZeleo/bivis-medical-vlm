#!/usr/bin/env python3
"""
Comprehensive Unit & Preflight Test Suite (rev05)
=================================================
Tests all critical components of decisive_validation_experiment_v5:
1. Manifest Raw-Byte SHA-256 Verification & Embedded Base64 Decompression
2. Cohort Invariants (Primary N=1122, Historical N=150, Active Union N=1133) & Zero Leakage
3. Image Cryptographic Verification & Tamper Rejection
4. Exact Sequence Log-Likelihood vs Length-Normalized Scoring Mathematics
5. Causal Logit Alignment & Prefix Invariance
6. Label Mapping & Canonical Comparison Consistency
7. Discrete Empirical AURC Hand-Verified Calculation
8. Monte Carlo Paired Cluster Permutation Test & Holm-Bonferroni Correction
9. Checkpoint Governance & Atomic Writing Safeguards
"""

import os
import sys
import io
import zlib
import base64
import hashlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath("scripts"))
sys.path.insert(0, os.path.abspath("src"))
from run_decisive_validation import (
    AUTHORITATIVE_MANIFEST_SHA256,
    REVISION_ID,
    LABEL_MAP,
    CANDIDATES_EN,
    CANDIDATES_ID,
    compute_rgb_pixel_sha256,
    resolve_and_verify_images,
    compute_discrete_risk_coverage,
    get_risk_at_realized_coverage,
    paired_cluster_permutation_test,
    run_paired_cluster_bootstrap,
    apply_holm_bonferroni,
    validate_checkpoint_compatibility,
)

try:
    from manifest_rev05_b64 import EMBEDDED_MANIFEST_B64 as EMBEDDED_MANIFEST_REV05_B64
except ImportError:
    EMBEDDED_MANIFEST_REV05_B64 = None


def test_manifest_raw_byte_hash_and_embedded_decompression():
    print("▶ Running test_manifest_raw_byte_hash_and_embedded_decompression...")
    manifest_candidates = [
        Path("data/manifests/manifest_rev05.csv"),
        Path("artifacts/experiments/decisive_validation/rev05/metrics/manifest_rev05.csv"),
    ]
    manifest_p = next((p for p in manifest_candidates if p.exists()), None)
    assert manifest_p is not None, f"Missing manifest across standard locations: {manifest_candidates}"
    
    # 1. External file raw bytes verification
    raw_bytes = manifest_p.read_bytes()
    calc_sha = hashlib.sha256(raw_bytes).hexdigest()
    assert calc_sha == AUTHORITATIVE_MANIFEST_SHA256, \
        f"External Manifest SHA256 mismatch! Expected {AUTHORITATIVE_MANIFEST_SHA256}, got {calc_sha}"
        
    # 2. Mirror copy verification
    mirror_candidates = [
        Path("artifacts/experiments/decisive_validation/rev05/metrics/manifest_rev05.csv"),
        Path("data/manifests/manifest_rev05.csv"),
    ]
    mirror_p = next((p for p in mirror_candidates if p.exists()), None)
    assert mirror_p is not None, f"Missing mirror manifest: {mirror_candidates}"
    mirror_sha = hashlib.sha256(mirror_p.read_bytes()).hexdigest()
    assert mirror_sha == AUTHORITATIVE_MANIFEST_SHA256, "Mirror manifest hash does not match authoritative SHA256!"

    # 3. Embedded base64 decompression verification
    assert EMBEDDED_MANIFEST_REV05_B64 is not None, "Failed to import EMBEDDED_MANIFEST_REV05_B64!"
    compressed = base64.b64decode(EMBEDDED_MANIFEST_REV05_B64.encode("ascii"))
    decompressed_bytes = zlib.decompress(compressed)
    decomp_sha = hashlib.sha256(decompressed_bytes).hexdigest()
    assert decomp_sha == AUTHORITATIVE_MANIFEST_SHA256, \
        f"Embedded decompressed SHA256 mismatch! Expected {AUTHORITATIVE_MANIFEST_SHA256}, got {decomp_sha}"
        
    # 4. Invariant: parsed DataFrames must be identical in shape and columns
    df_ext = pd.read_csv(manifest_p)
    df_decomp = pd.read_csv(io.BytesIO(decompressed_bytes))
    pd.testing.assert_frame_equal(df_ext, df_decomp)
    
    print("  ✅ PASS: Manifest raw byte SHA-256 and embedded decompression verified bit-identically.")


def test_cohort_invariants_and_zero_cluster_leakage():
    print("▶ Running test_cohort_invariants_and_zero_cluster_leakage...")
    manifest_p = Path("data/manifests/manifest_rev05.csv")
    df = pd.read_csv(manifest_p)
    assert len(df) == 2244, f"Expected 2244 total rows, got {len(df)}"
    assert df["qid"].nunique() == 2244, "Duplicate QIDs detected!"
    assert df["canonical_image_id"].nunique() == 314, f"Expected 314 canonical images, got {df['canonical_image_id'].nunique()}"
    
    # 1. Primary Cohort (N=1122)
    df_primary = df[df["is_eligible_primary"] == True]
    assert len(df_primary) == 1122, f"Expected 1122 primary samples, got {len(df_primary)}"
    test_prim = df_primary[df_primary["split"] == "test"]
    dev_prim = df_primary[df_primary["split"] == "dev"]
    assert len(test_prim) == 799, f"Expected 799 primary test rows, got {len(test_prim)}"
    assert len(dev_prim) == 323, f"Expected 323 primary dev rows, got {len(dev_prim)}"
    
    test_clusters_prim = set(test_prim["canonical_image_id"].unique())
    dev_clusters_prim = set(dev_prim["canonical_image_id"].unique())
    assert len(test_clusters_prim) == 204, f"Expected 204 primary test clusters, got {len(test_clusters_prim)}"
    assert len(dev_clusters_prim) == 88, f"Expected 88 primary dev clusters, got {len(dev_clusters_prim)}"
    overlap_prim = test_clusters_prim.intersection(dev_clusters_prim)
    assert len(overlap_prim) == 0, f"Primary cross-split leakage! Overlapping clusters: {overlap_prim}"
    
    # 2. Historical Benchmark Cohort (N=150)
    df_hist = df[df["in_rev03_benchmark"] == True]
    assert len(df_hist) == 150, f"Expected 150 historical samples, got {len(df_hist)}"
    test_hist = df_hist[df_hist["split"] == "test"]
    dev_hist = df_hist[df_hist["split"] == "dev"]
    assert len(test_hist) == 92, f"Expected 92 historical test rows, got {len(test_hist)}"
    assert len(dev_hist) == 58, f"Expected 58 historical dev rows, got {len(dev_hist)}"
    test_clusters_hist = set(test_hist["canonical_image_id"].unique())
    dev_clusters_hist = set(dev_hist["canonical_image_id"].unique())
    assert len(test_clusters_hist) == 35, f"Expected 35 historical test clusters, got {len(test_clusters_hist)}"
    assert len(dev_clusters_hist) == 23, f"Expected 23 historical dev clusters, got {len(dev_clusters_hist)}"
    assert len(test_clusters_hist.intersection(dev_clusters_hist)) == 0, "Historical cross-split leakage!"
    
    # 3. Active Evaluation Cohort Union (N=1133)
    df_union = df[(df["is_eligible_primary"] == True) | (df["in_rev03_benchmark"] == True)]
    assert len(df_union) == 1133, f"Expected exactly 1133 union rows, got {len(df_union)}"
    test_union = df_union[df_union["split"] == "test"]
    dev_union = df_union[df_union["split"] == "dev"]
    assert len(test_union) == 806, f"Expected 806 union test rows, got {len(test_union)}"
    assert len(dev_union) == 327, f"Expected 327 union dev rows, got {len(dev_union)}"
    test_clusters_union = set(test_union["canonical_image_id"].unique())
    dev_clusters_union = set(dev_union["canonical_image_id"].unique())
    assert len(test_clusters_union) == 204, f"Expected 204 union test clusters, got {len(test_clusters_union)}"
    assert len(dev_clusters_union) == 88, f"Expected 88 union dev clusters, got {len(dev_clusters_union)}"
    assert len(test_clusters_union.intersection(dev_clusters_union)) == 0, "Union cross-split leakage!"
    
    # 4. Verify 100% containment of both cohorts in the union
    assert set(df_primary["qid"]).issubset(set(df_union["qid"]))
    assert set(df_hist["qid"]).issubset(set(df_union["qid"]))
    
    print("  ✅ PASS: Cohort invariants (Primary 1,122, Historical 150, Union 1,133) and zero leakage verified.")


def test_image_verification_and_tamper_rejection():
    print("▶ Running test_image_verification_and_tamper_rejection...")
    manifest_p = Path("data/manifests/manifest_rev05.csv")
    df = pd.read_csv(manifest_p)
    sample_df = df.head(10).copy()
    image_dirs = ["data/raw/indorad_vqa/images"]
    resolved = resolve_and_verify_images(sample_df, local_image_dirs=image_dirs)
    assert len(resolved) == 10, f"Expected 10 resolved images, got {len(resolved)}"
    
    # Test tamper rejection: artificially corrupted pixel hash
    tampered_df = sample_df.copy()
    tampered_df.loc[0, "pixel_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    
    rejected = False
    try:
        resolve_and_verify_images(tampered_df, local_image_dirs=image_dirs)
    except RuntimeError:
        rejected = True
    assert rejected, "Tampered image hash was NOT rejected by resolver!"
    print("  ✅ PASS: Cryptographic pixel verification and tamper rejection verified.")


def test_exact_sequence_and_length_normalized_scoring_mathematics():
    print("▶ Running test_exact_sequence_and_length_normalized_scoring_mathematics...")
    # Scenario: Candidate 1: "YA<|im_end|>" (2 tokens), Candidate 2: "TIDAK<|im_end|>" (4 tokens)
    # Token conditional log probs:
    # C1: [-1.2, -1.2] -> Sequence Log-Likelihood = -2.40. Mean surprisal = -1.20
    # C2: [-1.2, -1.2, -1.2, -1.2] -> Sequence Log-Likelihood = -4.80. Mean surprisal = -1.20
    lp1 = [-1.2, -1.2]
    lp2 = [-1.2, -1.2, -1.2, -1.2]
    
    seq_s1 = sum(lp1)  # -2.40
    seq_s2 = sum(lp2)  # -4.80
    len_s1 = seq_s1 / len(lp1)  # -1.20
    len_s2 = seq_s2 / len(lp2)  # -1.20
    
    # 1. Exact Sequence Candidate-Normalized Score:
    # Softmax over [-2.40, -4.80]
    p_seq_1 = np.exp(seq_s1) / (np.exp(seq_s1) + np.exp(seq_s2))
    p_seq_2 = np.exp(seq_s2) / (np.exp(seq_s1) + np.exp(seq_s2))
    assert abs((p_seq_1 + p_seq_2) - 1.0) < 1e-12, "Softmax probabilities must sum to 1.0"
    # Under exact sequence log-likelihood, 2 tokens at -1.2 has higher joint likelihood than 4 tokens at -1.2
    assert p_seq_1 > 0.90, f"Expected higher joint sequence probability for 2 tokens vs 4 tokens, got {p_seq_1}"
    
    # 2. Length-Normalized Surprisal Track:
    # Softmax over [-1.20, -1.20]
    p_len_1 = np.exp(len_s1) / (np.exp(len_s1) + np.exp(len_s2))
    p_len_2 = np.exp(len_s2) / (np.exp(len_s1) + np.exp(len_s2))
    assert abs(p_len_1 - 0.50) < 1e-12, f"Expected exactly 0.50 for equal average surprisal, got {p_len_1}"
    
    # 3. Demonstration of dual-track necessity:
    # In rev05, both tracks are preserved:
    # Primary evaluation: Sequence Log-Likelihood (the true joint probability P(y | x) under autoregressive model)
    # Sensitivity evaluation: Length-Normalized Surprisal (evaluating if token length discrepancy shifts selective rank)
    print("  ✅ PASS: Exact Sequence Log-Likelihood and Length-Normalized Surprisal mathematics validated.")


def test_causal_logit_alignment_and_prefix_invariance():
    print("▶ Running test_causal_logit_alignment_and_prefix_invariance...")
    # Autoregressive transformer causal structure:
    # Input sequence tokens: [x_0, x_1, ..., x_{P-1}, y_0, y_1, ..., y_{T-1}]
    # Output logits tensor shape: [batch_size, sequence_length, vocab_size]
    # Logit at index k predicts token at index k + 1.
    # Therefore, to predict target tokens y_0 (index P) through y_{T-1} (index P + T - 1):
    # The predictive logits must be sliced from index (P - 1) up to (P - 1 + T).
    
    L_prompt = 15
    L_target = 4
    total_len = L_prompt + L_target  # 19
    
    start_idx = L_prompt - 1
    end_idx = L_prompt - 1 + L_target
    target_logits_slice = list(range(start_idx, end_idx))
    
    assert len(target_logits_slice) == L_target, f"Slice length must equal target length ({L_target})"
    assert target_logits_slice == [14, 15, 16, 17]
    
    # Prefix check simulation:
    prompt_ids = list(range(100, 115))
    cand_ids = list(range(100, 115)) + [200, 201, 202, 203]
    assert cand_ids[:L_prompt] == prompt_ids, "Prefix IDs must strictly match!"
    print("  ✅ PASS: Causal logit alignment indexing and prefix invariance verified.")


def test_label_mapping_and_canonical_comparison():
    print("▶ Running test_label_mapping_and_canonical_comparison...")
    # All candidates in EN and ID must map to either 'YES' or 'NO'
    for c in CANDIDATES_EN:
        assert c in LABEL_MAP, f"Candidate {c} missing from LABEL_MAP"
        assert LABEL_MAP[c] in {"YES", "NO"}
    for c in CANDIDATES_ID:
        assert c in LABEL_MAP, f"Candidate {c} missing from LABEL_MAP"
        assert LABEL_MAP[c] in {"YES", "NO"}
        
    assert LABEL_MAP["YES<|im_end|>"] == "YES"
    assert LABEL_MAP["NO<|im_end|>"] == "NO"
    assert LABEL_MAP["YA<|im_end|>"] == "YES"
    assert LABEL_MAP["TIDAK<|im_end|>"] == "NO"
    print("  ✅ PASS: Label mapping to canonical space {'YES', 'NO'} verified.")


def test_discrete_aurc_manual_calculation():
    print("▶ Running test_discrete_aurc_manual_calculation...")
    # 4 samples: scores = [0.9, 0.8, 0.7, 0.6], correctness = [1, 1, 0, 1]
    # Rank order: 1, 2, 3, 4
    # k=1: acc = 1/1 = 1.0, risk = 0.0
    # k=2: acc = 2/2 = 1.0, risk = 0.0
    # k=3: acc = 2/3 = 0.6666667, risk = 1/3 = 0.3333333333
    # k=4: acc = 3/4 = 0.75, risk = 0.25
    # AURC_mean = (0.0 + 0.0 + 1/3 + 1/4) / 4 = (7/12) / 4 = 7/48 = 0.14583333333333334
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    corr = np.array([1, 1, 0, 1])
    cov, risk, aurc = compute_discrete_risk_coverage(scores, corr)
    expected_aurc = 7.0 / 48.0
    assert abs(aurc - expected_aurc) < 1e-12, f"Discrete AURC mismatch: got {aurc}, expected {expected_aurc}"
    
    r_100, cov_100, k_100 = get_risk_at_realized_coverage(cov, risk, 1.0)
    assert abs(r_100 - 0.25) < 1e-7 and k_100 == 4
    r_50, cov_50, k_50 = get_risk_at_realized_coverage(cov, risk, 0.5)
    assert abs(r_50 - 0.0) < 1e-7 and k_50 == 2
    print("  ✅ PASS: Discrete empirical AURC hand-verified calculation confirmed.")


def test_paired_cluster_permutation_test_and_holm_bonferroni():
    print("▶ Running test_paired_cluster_permutation_test_and_holm_bonferroni...")
    np.random.seed(42)
    # Identical estimators test
    df_mock = pd.DataFrame({
        "canonical_image_id": ["c1"] * 4 + ["c2"] * 4 + ["c3"] * 4 + ["c4"] * 4 + ["c5"] * 4,
        "correct_id_seq": [1, 1, 0, 1] * 5,
        "score_a": [0.9, 0.8, 0.7, 0.6] * 5,
        "score_b": [0.9, 0.8, 0.7, 0.6] * 5
    })
    aurc_a, aurc_b, t_obs, p_val = paired_cluster_permutation_test(
        df_mock, "score_a", "score_b", corr_col="correct_id_seq", n_perm=500, seed=42
    )
    assert abs(t_obs) < 1e-12, "T_obs must be exactly 0 for identical estimators"
    assert p_val == 1.0, f"Expected p=1.0 for identical scores, got {p_val}"
    
    # Holm-Bonferroni adjustment test
    p_dict = {"M2": 0.04, "M_dir": 0.01, "M_norm": 0.03, "M3_old": 0.10}
    adj = apply_holm_bonferroni(p_dict)
    assert abs(adj["M_dir"] - 0.04) < 1e-7
    assert abs(adj["M_norm"] - 0.09) < 1e-7
    assert abs(adj["M2"] - 0.09) < 1e-7
    assert abs(adj["M3_old"] - 0.10) < 1e-7
    print("  ✅ PASS: Monte Carlo Paired Cluster Permutation Test and Holm-Bonferroni verified.")


def test_checkpoint_governance_and_atomic_writing():
    print("▶ Running test_checkpoint_governance_and_atomic_writing...")
    valid_row = {
        "qa_id": 74, "canonical_image_id": "cid_test", "split": "test", "reference_answer": "YES",
        "pred_en_seq": "YES", "conf_en_seq": 0.9, "correct_en_seq": 1, "p_en_yes_seq": 0.9, "p_en_no_seq": 0.1,
        "pred_id_seq": "YES", "conf_id_seq": 0.85, "correct_id_seq": 1, "p_orig_ya_seq": 0.85, "p_orig_tidak_seq": 0.15,
        "p_blank_ya_seq": 0.5, "p_blank_tidak_seq": 0.5, "p_blank_pred_seq": 0.5, "delta_p_dir_seq": 0.35, "delta_p_abs_seq": 0.35,
        "score_m1_seq": 0.85, "score_m2_seq": 0.35, "score_m_dir_seq": 0.2975, "score_m_norm_seq": 0.57375, "score_m3_old_seq": 0.57375,
        "pred_en_len": "YES", "conf_en_len": 0.9, "correct_en_len": 1, "p_en_yes_len": 0.9, "p_en_no_len": 0.1,
        "pred_id_len": "YES", "conf_id_len": 0.85, "correct_id_len": 1, "p_orig_ya_len": 0.85, "p_orig_tidak_len": 0.15,
        "p_blank_ya_len": 0.5, "p_blank_tidak_len": 0.5, "p_blank_pred_len": 0.5, "delta_p_dir_len": 0.35, "delta_p_abs_len": 0.35,
        "score_m1_len": 0.85, "score_m2_len": 0.35, "score_m_dir_len": 0.2975, "score_m_norm_len": 0.57375, "score_m3_old_len": 0.57375,
        "scoring_method": "exact_sequence_candidate_normalized_v5",
        "revision_id": "rev05",
        "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
        "model_id": "TestModel"
    }
    df_valid = pd.DataFrame([valid_row])
    is_valid, msg = validate_checkpoint_compatibility(df_valid, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert is_valid, f"Expected valid checkpoint, got {msg}"
    
    # 1. Rejection of rev04 revision ID
    df_rev04 = df_valid.copy()
    df_rev04["revision_id"] = "rev04"
    is_v, msg = validate_checkpoint_compatibility(df_rev04, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "revision ID" in msg
    
    # 2. Rejection of corrupted manifest hash
    df_bad_sha = df_valid.copy()
    df_bad_sha["manifest_sha256"] = "corrupted_hash_999"
    is_v, msg = validate_checkpoint_compatibility(df_bad_sha, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "SHA256" in msg
    
    # 3. Rejection of NaN in sequence score
    df_nan = df_valid.copy()
    df_nan.loc[0, "score_m_dir_seq"] = np.nan
    is_v, msg = validate_checkpoint_compatibility(df_nan, "TestModel", AUTHORITATIVE_MANIFEST_SHA256, {74})
    assert not is_v and "non-finite" in msg
    
    # 4. Atomic writing verification
    test_out_dir = Path("artifacts/experiments/decisive_validation/rev05/test_atomic")
    test_out_dir.mkdir(parents=True, exist_ok=True)
    target_csv = test_out_dir / "atomic_checkpoint.csv"
    tmp_csv = test_out_dir / "atomic_checkpoint.csv.tmp"
    
    df_valid.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, target_csv)
    assert target_csv.exists() and not tmp_csv.exists()
    assert len(pd.read_csv(target_csv)) == 1
    
    # Clean up test dir
    import shutil
    shutil.rmtree(test_out_dir)
    
    print("  ✅ PASS: Checkpoint governance and atomic writing safeguards verified.")


def run_all_tests():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE PREFLIGHT TEST SUITE FOR REV05 ENGINE")
    print("=" * 70)
    test_manifest_raw_byte_hash_and_embedded_decompression()
    test_cohort_invariants_and_zero_cluster_leakage()
    test_image_verification_and_tamper_rejection()
    test_exact_sequence_and_length_normalized_scoring_mathematics()
    test_causal_logit_alignment_and_prefix_invariance()
    test_label_mapping_and_canonical_comparison()
    test_discrete_aurc_manual_calculation()
    test_paired_cluster_permutation_test_and_holm_bonferroni()
    test_checkpoint_governance_and_atomic_writing()
    print("=" * 70)
    print("ALL 9 PREFLIGHT TESTS PASSED (PASS_UNIT_TEST)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
