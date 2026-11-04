"""
Unit tests for Sequence Log-Likelihood and Directional Visual Sensitivity Scoring.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
from vlm_research.evaluation.scoring import (
    compute_sequence_log_likelihood,
    compute_directional_visual_sensitivity
)

class TestScoring(unittest.TestCase):
    def test_sequence_log_likelihood_numpy(self):
        # 1 sample, 5 tokens, 4 vocab
        logits = np.array([[[1.0, 2.0, 0.5, 0.1],
                            [0.2, 3.0, 1.0, 0.4],
                            [2.5, 0.1, 0.2, 0.8],
                            [0.1, 0.2, 0.3, 4.0],
                            [1.0, 1.0, 1.0, 1.0]]])
        target_ids = np.array([1, 0])
        prefix_len = 2 # target starts at index prefix_len-1 = 1
        
        score = compute_sequence_log_likelihood(logits, target_ids, prefix_length=prefix_len, alpha=1.0)
        self.assertIsInstance(score, float)
        self.assertLess(score, 0.0) # log-probabilities are non-positive
        
    def test_directional_visual_sensitivity(self):
        # Test case: reliance on image
        p_orig = 0.85
        p_blank = 0.20
        delta_p = compute_directional_visual_sensitivity(p_orig, p_blank)
        self.assertAlmostEqual(delta_p, 0.65, places=5)
        
        # Test case: blindness / hallucination (higher on blank)
        p_orig_blind = 0.30
        p_blank_blind = 0.70
        delta_p_blind = compute_directional_visual_sensitivity(p_orig_blind, p_blank_blind)
        self.assertAlmostEqual(delta_p_blind, -0.40, places=5)

if __name__ == '__main__':
    unittest.main()
