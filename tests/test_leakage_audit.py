"""
Unit tests for Group-Aware Cluster Independence and Zero Patient Leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest
import pandas as pd
from vlm_research.config import MANIFESTS_DIR, PROJECT_ROOT

class TestLeakageAudit(unittest.TestCase):
    def setUp(self):
        self.manifest_path = MANIFESTS_DIR / "indorad_vqa_final_manifest.csv"
        if not self.manifest_path.exists():
            self.manifest_path = PROJECT_ROOT / "data" / "manifests" / "manifest_rev05.csv"

    def test_zero_cluster_leakage_between_dev_and_test(self):
        df = pd.read_csv(self.manifest_path)
        primary = df[df['is_eligible_primary'] == True]
        
        dev_clusters = set(primary[primary['split'] == 'dev']['image_id'].unique())
        test_clusters = set(primary[primary['split'] == 'test']['image_id'].unique())
        
        overlap = dev_clusters.intersection(test_clusters)
        self.assertEqual(len(overlap), 0, f"DATA LEAKAGE DETECTED! Overlapping clusters: {overlap}")
        
        # Verify cluster counts on primary cohort
        self.assertEqual(len(dev_clusters), 88, f"Expected 88 dev clusters, got {len(dev_clusters)}")
        self.assertEqual(len(test_clusters), 204, f"Expected 204 test clusters, got {len(test_clusters)}")

if __name__ == '__main__':
    unittest.main()
