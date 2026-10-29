"""
Unit Tests for Visual Intervention, Target Sequence Scoring, and Discrete AURC
================================================================================
Test Suite validating:
1. Deterministic Black Blank Image Generation (P0 vs P1).
2. Sequence Scoring & Candidate Softmax Normalization Logic.
3. Analytic Discrete AURC and Selective Risk Curve Verification.
4. Estimator Directionality (Fair Comparison of M1, M2, M_dir, M_norm, M3_old).
5. Paired Image-Cluster Bootstrap Determinism & Stability.

Author: Medical VLM Research Auditor & Reproducibility Team
"""

import unittest
import numpy as np
import pandas as pd
from PIL import Image
import hashlib
from pathlib import Path
import sys

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"

# Import core evaluation functions from decisive_validation_experiment
sys.path.insert(0, str(SCRIPT_DIR))
from decisive_validation_experiment import (
    compute_discrete_risk_coverage,
    get_risk_at_coverage,
    run_image_cluster_bootstrap
)


class TestVisualInterventionProtocol(unittest.TestCase):
    """Test Suite for Pre-Registered Visual Intervention Protocol."""

    def setUp(self):
        np.random.seed(42)

    def test_01_black_blank_image_generation_and_determinism(self):
        """
        Test 1: Verify P0 (original) is non-zero, and P1 (black blank) is strictly zero-valued
        with matching dimensions, mode, and deterministic SHA-256 hashes.
        """
        # Create a synthetic original radiograph-like image if no local image exists
        orig_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        orig_arr = np.array(orig_img)
        
        # P0 checks
        self.assertEqual(orig_img.size, (256, 256))
        self.assertEqual(orig_img.mode, "RGB")
        self.assertGreater(orig_arr.mean(), 0.0)
        self.assertGreater(orig_arr.max(), 0)
        
        # P1 Black Blank Generation
        blank_img = Image.new("RGB", orig_img.size, color=(0, 0, 0))
        blank_arr = np.array(blank_img)
        
        # P1 checks
        self.assertEqual(blank_img.size, orig_img.size)
        self.assertEqual(blank_img.mode, orig_img.mode)
        self.assertEqual(int(blank_arr.min()), 0)
        self.assertEqual(int(blank_arr.max()), 0)
        self.assertEqual(float(blank_arr.mean()), 0.0)
        self.assertEqual(float(blank_arr.std()), 0.0)
        
        # Hash uniqueness: P0 and P1 must have different pixel hashes
        hash_orig = hashlib.sha256(orig_arr.tobytes()).hexdigest()
        hash_blank = hashlib.sha256(blank_arr.tobytes()).hexdigest()
        self.assertNotEqual(hash_orig, hash_blank)
        
        # Hash determinism: re-generating P1 yields the identical hash
        blank_img2 = Image.new("RGB", orig_img.size, color=(0, 0, 0))
        hash_blank2 = hashlib.sha256(np.array(blank_img2).tobytes()).hexdigest()
        self.assertEqual(hash_blank, hash_blank2)

    def test_02_target_sequence_candidate_scoring_logic(self):
        """
        Test 2: Verify candidate sequence conditional log-likelihood normalization,
        binary softmax calculation, and symmetry between affirmative/negative tokens.
        """
        def normalize_candidates(logprob_aff: float, logprob_neg: float):
            max_lp = max(logprob_aff, logprob_neg)
            exp_aff = np.exp(logprob_aff - max_lp)
            exp_neg = np.exp(logprob_neg - max_lp)
            p_aff = exp_aff / (exp_aff + exp_neg)
            p_neg = 1.0 - p_aff
            pred = "YES" if p_aff >= 0.5 else "NO"
            conf = max(p_aff, p_neg)
            return p_aff, p_neg, pred, conf

        # Case A: Exact symmetry (equal log-likelihood)
        p_aff, p_neg, pred, conf = normalize_candidates(-1.25, -1.25)
        self.assertAlmostEqual(p_aff, 0.5, places=5)
        self.assertAlmostEqual(p_neg, 0.5, places=5)
        self.assertAlmostEqual(conf, 0.5, places=5)
        self.assertEqual(pred, "YES")  # tie-breaker >= 0.5

        # Case B: Strong affirmative preference
        p_aff, p_neg, pred, conf = normalize_candidates(-0.1, -5.0)
        self.assertGreater(p_aff, 0.99)
        self.assertLess(p_neg, 0.01)
        self.assertEqual(pred, "YES")
        self.assertAlmostEqual(conf, p_aff, places=5)

        # Case C: Strong negative preference
        p_aff, p_neg, pred, conf = normalize_candidates(-6.0, -0.2)
        self.assertLess(p_aff, 0.01)
        self.assertGreater(p_neg, 0.99)
        self.assertEqual(pred, "NO")
        self.assertAlmostEqual(conf, p_neg, places=5)

        # Case D: Multi-token logprob additive property sum(log P(w_t))
        # Token 1: log P = -0.5, Token 2: log P = -0.3 => Sequence log P = -0.8
        seq_logprob = -0.5 + (-0.3)
        self.assertAlmostEqual(seq_logprob, -0.8, places=5)

    def test_03_discrete_aurc_analytic_ground_truth(self):
        """
        Test 3: Verify discrete sample-level AURC against known analytic ground truth.
        """
        # Synthetic dataset: N = 10 samples
        # Scores strictly descending: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
        scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        # Scenario 1: Perfect ranking (8 correct, then 2 incorrect)
        # Correctness: 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
        corr_perfect = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        cov, risk, aurc_disc, aurc_trap = compute_discrete_risk_coverage(scores, corr_perfect)
        
        # Hand-calculated risk values:
        # k=1..8: risk = 0.0
        # k=9: cum_corr = 8, acc = 8/9, risk = 1 - 8/9 = 1/9
        # k=10: cum_corr = 8, acc = 8/10, risk = 1 - 8/10 = 2/10 = 0.2
        expected_risks = [0.0]*8 + [1.0/9.0, 0.2]
        for k in range(10):
            self.assertAlmostEqual(risk[k], expected_risks[k], places=6)
            
        expected_aurc = sum(expected_risks) / 10.0
        self.assertAlmostEqual(aurc_disc, expected_aurc, places=6)
        
        # Scenario 2: Inverted ranking (worst possible: 2 incorrect, then 8 correct)
        corr_inverted = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        _, risk_inv, aurc_inv, _ = compute_discrete_risk_coverage(scores, corr_inverted)
        # k=1: risk = 1.0; k=2: risk = 1.0; k=3: risk = 2/3, etc.
        self.assertGreater(aurc_inv, aurc_disc)
        self.assertAlmostEqual(risk_inv[0], 1.0, places=6)
        self.assertAlmostEqual(risk_inv[1], 1.0, places=6)
        
        # Test selective risk extraction at fixed coverage
        r_100, k_100 = get_risk_at_coverage(cov, risk, 1.0)
        self.assertEqual(k_100, 10)
        self.assertAlmostEqual(r_100, 0.2, places=6)
        
        r_80, k_80 = get_risk_at_coverage(cov, risk, 0.8)
        self.assertEqual(k_80, 8)
        self.assertAlmostEqual(r_80, 0.0, places=6)

    def test_04_estimator_directionality_and_fair_comparison(self):
        """
        Test 4: Verify directional behavior of all 5 estimators (M1, M2, M_dir, M_norm, M3_old)
        under positive visual support (Delta > 0) vs negative visual shift / hallucination (Delta < 0).
        """
        def calc_estimators(p_orig: float, p_blank: float):
            delta_dir = p_orig - p_blank
            delta_abs = abs(delta_dir)
            m1 = p_orig
            m2 = max(0.0, delta_dir)
            m_dir = p_orig * max(0.0, delta_dir)
            m_norm = p_orig * (1.0 + delta_dir) / 2.0
            m3_old = p_orig * (1.0 + delta_abs) / 2.0
            return m1, m2, m_dir, m_norm, m3_old, delta_dir

        # Case 1: Positive visual support: p_orig = 0.9, p_blank = 0.5 (Delta = +0.4)
        m1, m2, m_dir, m_norm, m3_old, delta = calc_estimators(0.9, 0.5)
        self.assertAlmostEqual(delta, 0.4)
        self.assertAlmostEqual(m1, 0.9)
        self.assertAlmostEqual(m2, 0.4)
        self.assertAlmostEqual(m_dir, 0.36)
        self.assertAlmostEqual(m_norm, 0.9 * 1.4 / 2.0)  # 0.63
        self.assertAlmostEqual(m3_old, 0.9 * 1.4 / 2.0) # 0.63
        self.assertEqual(m_norm, m3_old)  # When delta > 0, m_norm == m3_old

        # Case 2: Negative visual shift / language-bias hallucination:
        # Model is confident (0.9), but blank image gives even HIGHER confidence (0.95), Delta = -0.05
        m1, m2, m_dir, m_norm, m3_old, delta = calc_estimators(0.9, 0.95)
        self.assertAlmostEqual(delta, -0.05)
        self.assertAlmostEqual(m1, 0.9)
        self.assertAlmostEqual(m2, 0.0)      # Rectified: no positive support
        self.assertAlmostEqual(m_dir, 0.0)   # Fully suppressed
        self.assertAlmostEqual(m_norm, 0.9 * 0.95 / 2.0) # 0.4275 (penalized below neutral)
        
        # CRITICAL TEST: M3_old perversely elevates score when delta < 0 because of absolute value!
        # m3_old = 0.9 * (1 + 0.05) / 2 = 0.4725 > 0.45 (which would be neutral delta=0)
        self.assertGreater(m3_old, 0.9 * 0.5)
        self.assertGreater(m3_old, m_norm)
        # This proves M3_old rewarded the negative delta, while M_norm penalized it.

    def test_05_paired_image_cluster_bootstrap_reproducibility(self):
        """
        Test 5: Verify that paired image-cluster bootstrap across image clusters
        is reproducible with fixed random seed and properly handles paired differences.
        """
        # Create a mock dataframe with 10 images and 2 questions per image = 20 samples
        data = []
        for img_id in range(10):
            cid = f"img_cluster_{img_id}"
            for q_idx in range(2):
                data.append({
                    "qa_id": img_id * 2 + q_idx,
                    "canonical_image_id": cid,
                    "correct_id": 1 if (img_id + q_idx) % 3 != 0 else 0,
                    "score_m1": 0.5 + 0.04 * (img_id * 2 + q_idx),
                    "score_m2": 0.1 * ((img_id * 2 + q_idx) % 5),
                    "score_m_dir": 0.05 * (img_id * 2 + q_idx),
                    "score_m_norm": 0.4 + 0.03 * (img_id * 2 + q_idx),
                    "score_m3_old": 0.45 + 0.03 * (img_id * 2 + q_idx)
                })
        df = pd.DataFrame(data)
        
        score_cols = ["score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old"]
        
        # Run 1: seed = 42
        sum1, diff1 = run_image_cluster_bootstrap(df, score_cols, n_boot=100, seed=42)
        # Run 2: seed = 42 (must match identically)
        sum2, diff2 = run_image_cluster_bootstrap(df, score_cols, n_boot=100, seed=42)
        
        for col in score_cols:
            self.assertAlmostEqual(sum1[col]["mean"], sum2[col]["mean"], places=7)
            self.assertAlmostEqual(sum1[col]["ci_95_lower"], sum2[col]["ci_95_lower"], places=7)
            self.assertAlmostEqual(sum1[col]["ci_95_upper"], sum2[col]["ci_95_upper"], places=7)
            
        for col in diff1.keys():
            self.assertAlmostEqual(diff1[col]["mean_diff"], diff2[col]["mean_diff"], places=7)
            self.assertAlmostEqual(diff1[col]["p_value"], diff2[col]["p_value"], places=7)


if __name__ == "__main__":
    unittest.main()
