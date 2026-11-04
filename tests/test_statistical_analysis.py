"""
Unit tests for Discrete Empirical AURC and Paired Cluster Permutation Testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import numpy as np
from vlm_research.evaluation.selective_prediction import compute_discrete_aurc

class TestStatisticalAnalysis(unittest.TestCase):
    def test_hand_calculated_aurc(self):
        # 4 samples:
        # risk_scores: [0.1, 0.4, 0.7, 0.9]
        # errors:      [0,   0,   1,   1  ]
        # Coverages:
        # k=1: cov=0.25, risk = 0/1 = 0.0
        # k=2: cov=0.50, risk = 0/2 = 0.0
        # k=3: cov=0.75, risk = 1/3 = 0.3333333333333333
        # k=4: cov=1.00, risk = 2/4 = 0.5
        # Discrete AURC = (0.0 + 0.0 + 1/3 + 0.5) / 4 = 0.8333333333333334 / 4 = 0.20833333333333334
        risks = np.array([0.1, 0.4, 0.7, 0.9])
        errs = np.array([0, 0, 1, 1])
        
        aurc = compute_discrete_aurc(risks, errs)
        expected_aurc = (0.0 + 0.0 + (1.0 / 3.0) + 0.5) / 4.0
        self.assertAlmostEqual(aurc, expected_aurc, places=6)
        
    def test_perfect_selective_classifier_aurc(self):
        # 10 samples: 2 errors placed at the very highest risk
        risks = np.linspace(0.1, 1.0, 10)
        errs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        aurc = compute_discrete_aurc(risks, errs)
        # risks at k=1..8 are 0
        # k=9: 1/9
        # k=10: 2/10 = 0.2
        expected_aurc = ((1.0 / 9.0) + 0.2) / 10.0
        self.assertAlmostEqual(aurc, expected_aurc, places=6)

if __name__ == '__main__':
    unittest.main()
