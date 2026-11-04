"""
Unit tests for Dataset Quality and Primary Cohort Invariants.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import pandas as pd
from pathlib import Path
from vlm_research.config import MANIFESTS_DIR, PROJECT_ROOT

class TestDataQuality(unittest.TestCase):
    def setUp(self):
        self.manifest_path = MANIFESTS_DIR / "indorad_vqa_final_manifest.csv"
        if not self.manifest_path.exists():
            self.manifest_path = PROJECT_ROOT / "data" / "manifests" / "manifest_rev05.csv"

    def test_cohort_size_invariants(self):
        self.assertTrue(self.manifest_path.exists(), f"Missing manifest at {self.manifest_path}")
        df = pd.read_csv(self.manifest_path)
        
        # Total rows in IndoRad-VQA raw cohort
        self.assertEqual(len(df), 2244)
        
        # Primary cohort invariant: exactly 1,122 clean yes/no pairs
        primary_cohort = df[df['is_eligible_primary'] == True]
        self.assertEqual(len(primary_cohort), 1122)
        
    def test_canonical_labels(self):
        df = pd.read_csv(self.manifest_path)
        primary = df[df['is_eligible_primary'] == True]
        unique_answers = set(primary['answer_en'].str.strip().str.upper().unique())
        self.assertEqual(unique_answers, {'YES', 'NO'}, f"Unexpected labels: {unique_answers}")

if __name__ == '__main__':
    unittest.main()
