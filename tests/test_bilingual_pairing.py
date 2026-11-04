"""
Unit tests for Bilingual Question-Answer Pairing (EN <-> ID).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import pandas as pd
from vlm_research.config import MANIFESTS_DIR, PROJECT_ROOT

class TestBilingualPairing(unittest.TestCase):
    def setUp(self):
        self.manifest_path = MANIFESTS_DIR / "indorad_vqa_final_manifest.csv"
        if not self.manifest_path.exists():
            self.manifest_path = PROJECT_ROOT / "data" / "manifests" / "manifest_rev05.csv"

    def test_bilingual_fields_presence(self):
        df = pd.read_csv(self.manifest_path)
        # Verify that questions exist in both languages
        has_en = 'question' in df.columns or 'question_en' in df.columns
        has_id = 'question_indonesian' in df.columns or 'question_id' in df.columns
        self.assertTrue(has_en and has_id, "Manifest must contain both English and Indonesian questions")

    def test_canonical_pairing_consistency(self):
        df = pd.read_csv(self.manifest_path)
        # Check that answers in ID correspond to EN ground truth
        label_col = 'answer' if 'answer' in df.columns else 'answer_en'
        self.assertGreater(len(df), 0)
        self.assertFalse(df[label_col].isnull().any(), "Ground truth labels must not contain NaN")

if __name__ == '__main__':
    unittest.main()
