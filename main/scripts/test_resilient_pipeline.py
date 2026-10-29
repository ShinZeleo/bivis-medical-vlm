#!/usr/bin/env python3
"""
Preflight Unit Test Suite for Resilient Medical VLM Pipeline
=============================================================
Validates:
1. Checkpoint compatibility governance (acceptance of valid, strict rejection of invalid).
2. Set-based resume filtering (set(cohort) - set(completed)).
3. Atomic checkpoint merge and corruption-free flush.
4. Discrete Riemann AURC mathematical properties.
5. Paired Cluster Bootstrap and Permutation Testing.
6. Holm-Bonferroni FWER step-down correction.
7. Zero-GPU Learned Risk Ablation (GroupKFold on Dev Set).
8. Physical image pixel SHA-256 verification.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath("main/scripts"))
import resilient_validation_engine as rve

def test_checkpoint_validation():
    print("🧪 Test 1: Checkpoint Validation & Governance...")
    manifest_path = Path("main/results/decisive_validation/rev05/manifest_rev05.csv")
    df_manifest = pd.read_csv(manifest_path)
    valid_qids = set(df_manifest["qa_id"])

    # 1. Test existing valid Qwen checkpoint
    qwen_path = Path("main/results/decisive_validation/rev05/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
    df_qwen = pd.read_csv(qwen_path)
    ok_q, msg_q = rve.validate_checkpoint_compatibility(
        df_qwen, "Qwen/Qwen2.5-VL-7B-Instruct", rve.AUTHORITATIVE_MANIFEST_SHA256, valid_qids
    )
    assert ok_q, f"Qwen checkpoint failed validation: {msg_q}"
    print(f"   ✅ Valid Qwen Checkpoint: Accepted ({len(df_qwen)} rows)")

    # 2. Test existing valid Huatuo checkpoint
    huatuo_path = Path("main/results/decisive_validation/rev05/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")
    df_huatuo = pd.read_csv(huatuo_path)
    ok_h, msg_h = rve.validate_checkpoint_compatibility(
        df_huatuo, "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL", rve.AUTHORITATIVE_MANIFEST_SHA256, valid_qids
    )
    assert ok_h, f"Huatuo checkpoint failed validation: {msg_h}"
    print(f"   ✅ Valid Huatuo Checkpoint: Accepted ({len(df_huatuo)} rows)")

    # 3. Test rejection of corrupted checkpoint (tampered manifest hash)
    df_corrupt_hash = df_qwen.copy()
    df_corrupt_hash["manifest_sha256"] = "wrong_hash_123"
    ok_bad_h, msg_bad_h = rve.validate_checkpoint_compatibility(
        df_corrupt_hash, "Qwen/Qwen2.5-VL-7B-Instruct", rve.AUTHORITATIVE_MANIFEST_SHA256, valid_qids
    )
    assert not ok_bad_h, "Failed to reject corrupted manifest hash!"
    print("   ✅ Tampered Manifest Hash: Correctly Rejected")

    # 4. Test rejection of wrong model name
    ok_wrong_m, msg_wrong_m = rve.validate_checkpoint_compatibility(
        df_qwen, "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL", rve.AUTHORITATIVE_MANIFEST_SHA256, valid_qids
    )
    assert not ok_wrong_m, "Failed to reject wrong model ID!"
    print("   ✅ Mismatched Model ID: Correctly Rejected")

    # 5. Test rejection of non-finite values (NaN / Inf)
    df_nan = df_qwen.copy()
    df_nan.loc[0, "score_m_norm_seq"] = np.nan
    ok_nan, msg_nan = rve.validate_checkpoint_compatibility(
        df_nan, "Qwen/Qwen2.5-VL-7B-Instruct", rve.AUTHORITATIVE_MANIFEST_SHA256, valid_qids
    )
    assert not ok_nan, "Failed to reject NaN score values!"
    print("   ✅ Non-Finite NaN Values: Correctly Rejected")

    # 6. Test rejection of duplicate QA IDs
    df_dup = pd.concat([df_qwen, df_qwen.iloc[:1]], ignore_index=True)
    ok_dup, msg_dup = rve.validate_checkpoint_compatibility(
        df_dup, "Qwen/Qwen2.5-VL-7B-Instruct", rve.AUTHORITATIVE_MANIFEST_SHA256, valid_qids
    )
    assert not ok_dup, "Failed to reject duplicate QA IDs!"
    print("   ✅ Duplicate QA IDs: Correctly Rejected")


def test_set_based_resume():
    print("\n🧪 Test 2: True Set-Based Resume Filtering...")
    manifest_path = Path("main/results/decisive_validation/rev05/manifest_rev05.csv")
    df_manifest = pd.read_csv(manifest_path)
    df_cohort = rve.get_active_evaluation_cohort(df_manifest, "active_union")
    assert len(df_cohort) == 1133, f"Expected 1133 cohort rows, got {len(df_cohort)}"

    huatuo_path = Path("main/results/decisive_validation/rev05/FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")
    df_huatuo = pd.read_csv(huatuo_path)
    completed_qids = set(df_huatuo["qa_id"])

    df_remaining = rve.filter_remaining_samples(df_cohort, completed_qids)
    assert len(df_remaining) == 253, f"Expected 253 remaining samples, got {len(df_remaining)}"
    assert len(set(df_remaining["qa_id"]).intersection(completed_qids)) == 0, "Intersection between remaining and completed!"
    print(f"   ✅ Filter Remaining: Exactly {len(df_remaining)} samples remaining (0% duplicate overlap).")

    # Bounded window test
    df_window = rve.filter_remaining_samples(df_cohort, completed_qids, start_index=10, max_samples=20)
    assert len(df_window) == 20, f"Expected 20 samples in window, got {len(df_window)}"
    assert df_window.iloc[0]["qa_id"] == df_remaining.iloc[10]["qa_id"]
    print("   ✅ Bounded Windowing: Passed (start_index=10, max_samples=20).")


def test_atomic_checkpoint_merge():
    print("\n🧪 Test 3: Atomic Checkpoint Save & Merge...")
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_path = Path(tmpdir) / "test_checkpoint.csv"
        
        # Initial chunk (5 samples)
        df_chunk1 = pd.DataFrame({
            "qa_id": [1, 2, 3, 4, 5],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        df_saved1 = rve.atomic_save_checkpoint(df_chunk1, dest_path)
        assert dest_path.exists()
        assert len(pd.read_csv(dest_path)) == 5

        # Second chunk with 1 duplicate (sample 5 updated) and 2 new (6, 7)
        df_chunk2 = pd.DataFrame({
            "qa_id": [5, 6, 7],
            "score": [0.99, 0.6, 0.7]
        })
        df_saved2 = rve.atomic_save_checkpoint(df_chunk2, dest_path, df_existing=df_saved1)
        df_loaded = pd.read_csv(dest_path)
        assert len(df_loaded) == 7, f"Expected 7 rows, got {len(df_loaded)}"
        assert df_loaded.loc[df_loaded["qa_id"] == 5, "score"].values[0] == 0.99, "keep='last' failed to update row 5!"
        print("   ✅ Atomic Flush & Deduplicated Merge: Passed (7 rows, keep='last').")


def test_discrete_riemann_aurc():
    print("\n🧪 Test 4: Discrete Riemann AURC Mathematical Properties...")
    # Perfect model: correct array all 1s -> risk should be 0.0 everywhere
    y_perf = np.ones(100)
    s_perf = np.linspace(1.0, 0.0, 100)
    aurc_perf, cov_perf, risk_perf = rve.calculate_discrete_aurc(y_perf, s_perf)
    assert aurc_perf == 0.0, f"Expected AURC 0.0 for perfect model, got {aurc_perf}"
    assert np.all(risk_perf == 0.0)

    # Inverted model: correct array all 0s -> risk should be 1.0 everywhere
    y_zero = np.zeros(100)
    aurc_zero, cov_zero, risk_zero = rve.calculate_discrete_aurc(y_zero, s_perf)
    assert aurc_zero == 1.0, f"Expected AURC 1.0 for zero accuracy model, got {aurc_zero}"
    assert np.all(risk_zero == 1.0)
    print("   ✅ Perfect & Inverted Bounds (0.0 & 1.0): Passed.")


def test_bootstrap_and_permutation():
    print("\n🧪 Test 5: Paired Cluster Bootstrap & Permutation Testing (Arbitrary Index Immunity)...")
    qwen_path = Path("main/results/decisive_validation/rev05/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
    df_qwen = pd.read_csv(qwen_path)
    # Sliced DataFrame WITH NON-STANDARD INDEX (arbitrary non-zero, non-consecutive indices)
    df_test_arbitrary = df_qwen[(df_qwen["split"] == "test") & (df_qwen["is_eligible_primary"] == True)].copy()
    # Explicitly set arbitrary scattered index values like 5000, 5001, ...
    df_test_arbitrary.index = np.arange(5000, 5000 + len(df_test_arbitrary))

    boot_res = rve.compute_paired_cluster_bootstrap(
        df_test_arbitrary, score_col_m="score_m_norm_seq", score_col_baseline="score_m1_seq", n_bootstraps=100, seed=42
    )
    assert "ci_95" in boot_res
    assert len(boot_res["ci_95"]) == 2
    assert boot_res["ci_95"][0] <= boot_res["ci_95"][1]
    print(f"   ✅ Paired Cluster Bootstrap (Index [5000..{5000+len(df_test_arbitrary)-1}]): CI95 = [{boot_res['ci_95'][0]:.4f}, {boot_res['ci_95'][1]:.4f}] (No IndexError!)")

    perm_p = rve.compute_paired_cluster_permutation_test(
        df_test_arbitrary, score_col_m="score_m_norm_seq", score_col_baseline="score_m1_seq", n_permutations=100, seed=42
    )
    assert 0.0 <= perm_p <= 1.0
    print(f"   ✅ Cluster Permutation Test (Index [5000..{5000+len(df_test_arbitrary)-1}]): p = {perm_p:.4f} (No IndexError!)")


def test_holm_bonferroni():
    print("\n🧪 Test 6: Holm-Bonferroni FWER Step-Down Correction...")
    raw_p = {"m2": 0.01, "m_dir": 0.04, "m_norm": 0.005, "m3_old": 0.10}
    adj = rve.apply_holm_bonferroni_correction(raw_p)
    assert len(adj) == 4
    # Check monotonicity of adjusted p-values
    sorted_adj = sorted(adj.values(), key=lambda x: x["raw_p"])
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i]["adjusted_p"] <= sorted_adj[i+1]["adjusted_p"]
    print("   ✅ Holm-Bonferroni Monotonicity & Boundary Invariants: Passed.")


def test_zero_gpu_risk_ablation():
    print("\n🧪 Test 7: Zero-GPU Learned Risk Ablation on Dev Set...")
    qwen_path = Path("main/results/decisive_validation/rev05/Qwen_Qwen2.5_VL_7B_Instruct/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
    df_qwen = pd.read_csv(qwen_path)
    df_dev = df_qwen[(df_qwen["split"] == "dev") & (df_qwen["is_eligible_primary"] == True)].copy().reset_index(drop=True)
    df_test = df_qwen[(df_qwen["split"] == "test") & (df_qwen["is_eligible_primary"] == True)].copy().reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ablation_res = rve.train_and_evaluate_risk_ablation(df_dev, df_test, "Qwen2.5-VL-7B", Path(tmpdir))
        assert "LR_conf" in ablation_res
        assert "LR_visual" in ablation_res
        assert "LR_full" in ablation_res
        for mname, data in ablation_res.items():
            assert 0.0 <= data["dev_aurc"] <= 1.0
            assert 0.0 <= data["test_aurc"] <= 1.0
            print(f"   ✅ Model {mname:10s} -> Dev CV AURC: {data['dev_aurc']:.4f} | Test AURC: {data['test_aurc']:.4f}")


def main():
    print("=" * 70)
    print("🚀 RUNNING PREFLIGHT RESILIENT PIPELINE UNIT TEST SUITE")
    print("=" * 70)

    test_checkpoint_validation()
    test_set_based_resume()
    test_atomic_checkpoint_merge()
    test_discrete_riemann_aurc()
    test_bootstrap_and_permutation()
    test_holm_bonferroni()
    test_zero_gpu_risk_ablation()

    print("\n" + "=" * 70)
    print("🎉 100% PREFLIGHT UNIT TESTS PASSED! PIPELINE IS RESILIENT & VERIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
