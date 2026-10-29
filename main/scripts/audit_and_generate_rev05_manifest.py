#!/usr/bin/env python3
"""
IndoRad-VQA Full Dataset Audit & rev05 Manifest Builder
======================================================
Produces comprehensive dataset audit metrics, dual-track cohort accounting logs,
and the authoritative rev05 evaluation manifest.
"""

import os
import sys
import json
import zlib
import base64
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    print("=" * 75)
    print("INDORAD-VQA DATASET AUDIT & REV05 MANIFEST BUILDER")
    print("=" * 75)
    
    out_dir = Path("main/results/decisive_validation/rev05")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load authoritative source tables
    df_merged = pd.read_csv("main/results/manifest_rev04.csv")
    print(f"Loaded source manifest: {len(df_merged)} rows")
    
    # Ensure in_eval_cohort boolean column is explicitly defined
    # in_eval_cohort is the exact union of primary cohort and historical benchmark
    df_merged["in_eval_cohort"] = (df_merged["is_eligible_primary"] == True) | (df_merged["in_rev03_benchmark"] == True)
    
    # Ensure reference_answer is canonical UPPERCASE ("YES" or "NO")
    df_merged["reference_answer"] = df_merged["reference_answer"].str.strip().str.upper()
    
    # Validate canonical labels
    invalid_labels = df_merged[df_merged["is_yes_no"] == True]["reference_answer"].unique()
    assert set(invalid_labels) == {"YES", "NO"}, f"Unknown binary labels detected: {invalid_labels}"
    
    # Re-order columns logically
    cols = [
        "qid", "qa_id", "canonical_image_id", "image_id", "pixel_sha256", "raw_sha256",
        "image_name", "image_width", "image_height", "image_mode", "qa_per_image",
        "split", "question_en", "question_id", "answer_en", "answer_id", "reference_answer",
        "answer_type", "translation_status", "translation_error_type",
        "is_yes_no", "is_eligible_primary", "in_rev03_benchmark", "in_eval_cohort", "exclusion_reason"
    ]
    manifest_df = df_merged[cols].copy()
    
    # Save manifest_rev05.csv
    manifest_p = out_dir / "manifest_rev05.csv"
    manifest_df.to_csv(manifest_p, index=False)
    manifest_bytes = manifest_p.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"Saved: {manifest_p} ({len(manifest_df)} rows, SHA256: {manifest_sha256})")
    
    # Save mirror at main/results/manifest_rev05.csv
    root_manifest_p = Path("main/results/manifest_rev05.csv")
    manifest_df.to_csv(root_manifest_p, index=False)
    assert hashlib.sha256(root_manifest_p.read_bytes()).hexdigest() == manifest_sha256
    print(f"Saved mirror: {root_manifest_p}")
    
    # Create compressed base64 string for embedded self-contained execution
    compressed = zlib.compress(manifest_bytes, level=9)
    b64_str = base64.b64encode(compressed).decode("ascii")
    # Verify decompress
    decomp = zlib.decompress(base64.b64decode(b64_str.encode("ascii")))
    assert hashlib.sha256(decomp).hexdigest() == manifest_sha256
    
    with open("main/scripts/manifest_rev05_b64.py", "w", encoding="utf-8") as f:
        f.write(f'AUTHORITATIVE_MANIFEST_SHA256 = "{manifest_sha256}"\n')
        f.write(f'EMBEDDED_MANIFEST_B64 = "{b64_str}"\n')
    print("Saved: main/scripts/manifest_rev05_b64.py (Verified raw-byte decompress SHA-256)")
    
    # 2. Comprehensive Cohort & Split Audits
    # Primary Cohort (is_eligible_primary == True)
    df_prim = manifest_df[manifest_df["is_eligible_primary"] == True]
    prim_test = df_prim[df_prim["split"] == "test"]
    prim_dev = df_prim[df_prim["split"] == "dev"]
    prim_test_clusters = set(prim_test["canonical_image_id"].unique())
    prim_dev_clusters = set(prim_dev["canonical_image_id"].unique())
    prim_leakage = prim_test_clusters.intersection(prim_dev_clusters)
    
    # Historical Benchmark Cohort (in_rev03_benchmark == True)
    df_hist = manifest_df[manifest_df["in_rev03_benchmark"] == True]
    hist_test = df_hist[df_hist["split"] == "test"]
    hist_dev = df_hist[df_hist["split"] == "dev"]
    hist_test_clusters = set(hist_test["canonical_image_id"].unique())
    hist_dev_clusters = set(hist_dev["canonical_image_id"].unique())
    hist_leakage = hist_test_clusters.intersection(hist_dev_clusters)
    
    # Active Evaluation Cohort (in_eval_cohort == True)
    df_eval = manifest_df[manifest_df["in_eval_cohort"] == True]
    eval_test = df_eval[df_eval["split"] == "test"]
    eval_dev = df_eval[df_eval["split"] == "dev"]
    eval_test_clusters = set(eval_test["canonical_image_id"].unique())
    eval_dev_clusters = set(eval_dev["canonical_image_id"].unique())
    eval_leakage = eval_test_clusters.intersection(eval_dev_clusters)
    
    # Global Cluster Allocation Integrity across entire dataset (2244 rows)
    test_all_clusters = set(manifest_df[manifest_df["split"] == "test"]["canonical_image_id"].unique())
    dev_all_clusters = set(manifest_df[manifest_df["split"] == "dev"]["canonical_image_id"].unique())
    global_leakage = test_all_clusters.intersection(dev_all_clusters)
    
    cohort_audit = {
        "manifest_sha256": manifest_sha256,
        "total_manifest_rows": len(manifest_df),
        "total_unique_qids": manifest_df["qid"].nunique(),
        "total_unique_physical_images": manifest_df["canonical_image_id"].nunique(),
        "global_split_cluster_leakage": len(global_leakage),
        "cohorts": {
            "primary_cohort": {
                "definition": "is_eligible_primary == True (Clean Yes/No only)",
                "total_samples": len(df_prim),
                "test_samples": len(prim_test),
                "dev_samples": len(prim_dev),
                "test_clusters": len(prim_test_clusters),
                "dev_clusters": len(prim_dev_clusters),
                "cluster_leakage": len(prim_leakage),
                "ref_answer_distribution": df_prim["reference_answer"].value_counts().to_dict()
            },
            "historical_benchmark_cohort": {
                "definition": "in_rev03_benchmark == True (Authoritative rev03 150-sample set)",
                "total_samples": len(df_hist),
                "test_samples": len(hist_test),
                "dev_samples": len(hist_dev),
                "test_clusters": len(hist_test_clusters),
                "dev_clusters": len(hist_dev_clusters),
                "cluster_leakage": len(hist_leakage),
                "ref_answer_distribution": df_hist["reference_answer"].value_counts().to_dict()
            },
            "active_evaluation_cohort": {
                "definition": "in_eval_cohort == True (Exact Union of Primary and Historical Cohorts)",
                "total_evaluated_samples": len(df_eval),
                "test_samples": len(eval_test),
                "dev_samples": len(eval_dev),
                "test_clusters": len(eval_test_clusters),
                "dev_clusters": len(eval_dev_clusters),
                "cluster_leakage": len(eval_leakage)
            }
        }
    }
    
    (out_dir / "cohort_audit.json").write_text(json.dumps(cohort_audit, indent=2))
    print(f"Saved: {out_dir / 'cohort_audit.json'}")
    
    # 3. Save manifest_audit.json
    manifest_audit = {
        "manifest_file": "manifest_rev05.csv",
        "sha256": manifest_sha256,
        "row_count": len(manifest_df),
        "column_count": len(manifest_df.columns),
        "columns": manifest_df.columns.tolist(),
        "unique_canonical_images": manifest_df["canonical_image_id"].nunique(),
        "split_counts": manifest_df["split"].value_counts().to_dict(),
        "is_yes_no_counts": manifest_df["is_yes_no"].value_counts().to_dict(),
        "translation_status_counts": manifest_df["translation_status"].value_counts().to_dict(),
        "is_eligible_primary_counts": manifest_df["is_eligible_primary"].value_counts().to_dict(),
        "in_rev03_benchmark_counts": manifest_df["in_rev03_benchmark"].value_counts().to_dict(),
        "in_eval_cohort_counts": manifest_df["in_eval_cohort"].value_counts().to_dict(),
        "zero_cluster_leakage_verified": len(global_leakage) == 0
    }
    (out_dir / "manifest_audit.json").write_text(json.dumps(manifest_audit, indent=2))
    print(f"Saved: {out_dir / 'manifest_audit.json'}")
    
    # 4. Save scoring_validation_report.json (Methodological specifications)
    scoring_specs = {
        "primary_scoring_method": "exact_sequence_candidate_normalized_v5",
        "sensitivity_scoring_method": "length_normalized_surprisal_v5",
        "causal_logit_alignment": "logits[L_prompt - 1 : L_prompt - 1 + L_target]",
        "prefix_assertion_enforced": True,
        "candidates": {
            "english": ["YES<|im_end|>", "NO<|im_end|>"],
            "indonesian": ["YA<|im_end|>", "TIDAK<|im_end|>"]
        },
        "canonical_label_space": ["YES", "NO"],
        "mapping_rules": {
            "YES<|im_end|>": "YES",
            "NO<|im_end|>": "NO",
            "YA<|im_end|>": "YES",
            "TIDAK<|im_end|>": "NO"
        },
        "terminology_designation": "candidate-normalized sequence score (NOT calibrated posterior probability)"
    }
    (out_dir / "scoring_validation_report.json").write_text(json.dumps(scoring_specs, indent=2))
    print(f"Saved: {out_dir / 'scoring_validation_report.json'}")
    
    print("=" * 75)
    print("MANIFEST & AUDIT ARTIFACTS SUCCESSFULLY GENERATED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
