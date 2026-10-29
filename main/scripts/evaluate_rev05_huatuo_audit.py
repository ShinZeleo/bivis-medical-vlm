#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("main/scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import resilient_validation_engine as rve

RESULTS_ROOT = Path("main/results/decisive_validation/rev05")
huatuo_ckpt = RESULTS_ROOT / "FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL" / "live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv"
qwen_ckpt = RESULTS_ROOT / "Qwen_Qwen2.5_VL_7B_Instruct" / "live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv"

df_h = pd.read_csv(huatuo_ckpt)
df_q = pd.read_csv(qwen_ckpt)

print("=" * 85)
print("🏥 AUDIT REVISI rev05: HASIL INFERENSI LENGKAP HUATUOGPT-VISION-7B vs QWEN2.5-VL-7B")
print("=" * 85)
print(f"Total Sampel Huatuo : {len(df_h)} / 1133 (100% COMPLETE)")
print(f"Total Sampel Qwen   : {len(df_q)} / 1133 (100% COMPLETE)")
print()

# 1. RQ1: Bilingual Accuracy & Degradation
def evaluate_rq1(df, model_name):
    cohorts = [
        ("Primary Test (N=799, 204 Clusters)", df[(df["split"]=="test") & (df["is_eligible_primary"]==True)]),
        ("Primary Dev (N=323, 88 Clusters)", df[(df["split"]=="dev") & (df["is_eligible_primary"]==True)]),
        ("Historical Test (N=92, 35 Clusters)", df[(df["split"]=="test") & (df["in_rev03_benchmark"]==True)]),
        ("Historical Dev (N=58, 23 Clusters)", df[(df["split"]=="dev") & (df["in_rev03_benchmark"]==True)])
    ]
    records = []
    for label, sub in cohorts:
        acc_en_seq = sub["correct_en_seq"].mean() * 100.0
        acc_id_seq = sub["correct_id_seq"].mean() * 100.0
        gap_seq = acc_en_seq - acc_id_seq
        flip_seq = sub["flip_seq"].mean() * 100.0
        joint_seq = sub["joint_correct_seq"].mean() * 100.0
        
        acc_en_len = sub["correct_en_len"].mean() * 100.0
        acc_id_len = sub["correct_id_len"].mean() * 100.0
        gap_len = acc_en_len - acc_id_len
        
        records.append({
            "model": model_name, "cohort": label,
            "acc_en_seq": acc_en_seq, "acc_id_seq": acc_id_seq, "gap_seq": gap_seq,
            "flip_seq": flip_seq, "joint_seq": joint_seq,
            "acc_en_len": acc_en_len, "acc_id_len": acc_id_len, "gap_len": gap_len
        })
    return pd.DataFrame(records)

df_rq1_h = evaluate_rq1(df_h, "HuatuoGPT-Vision-7B")
df_rq1_q = evaluate_rq1(df_q, "Qwen2.5-VL-7B")
df_rq1_all = pd.concat([df_rq1_h, df_rq1_q], ignore_index=True)

print("--- [RQ1] BILINGUAL ACCURACY & DISAGREEMENT (SEQUENCE TRACK) ---")
print(df_rq1_all[["model", "cohort", "acc_en_seq", "acc_id_seq", "gap_seq", "flip_seq", "joint_seq"]].to_string(index=False))
print()

# 2. RQ2: Visual Intervention (P0 Original vs P1 Blank)
def evaluate_rq2(df, model_name):
    sub = df[(df["split"]=="test") & (df["is_eligible_primary"]==True)].reset_index(drop=True)
    dp_seq = sub["delta_p_dir_seq"].values
    mean_dp_seq = float(np.mean(dp_seq))
    pos_seq = float(np.mean(dp_seq > 0)) * 100.0
    neg_seq = float(np.mean(dp_seq < 0)) * 100.0
    zero_seq = float(np.mean(dp_seq == 0)) * 100.0
    
    dp_len = sub["delta_p_dir_len"].values
    mean_dp_len = float(np.mean(dp_len))
    
    return {
        "model": model_name,
        "mean_delta_p_seq": mean_dp_seq,
        "pos_reliance_seq": pos_seq,
        "neg_reliance_seq": neg_seq,
        "zero_reliance_seq": zero_seq,
        "mean_delta_p_len": mean_dp_len
    }

df_rq2 = pd.DataFrame([evaluate_rq2(df_h, "HuatuoGPT-Vision-7B"), evaluate_rq2(df_q, "Qwen2.5-VL-7B")])
print("--- [RQ2] MULTIMODAL VISUAL INTERVENTION (PRIMARY TEST N=799) ---")
print(df_rq2.to_string(index=False))
print()

# 3. RQ3: Selective Prediction & Discrete Riemann AURC
def evaluate_rq3(df, model_name):
    sub = df[(df["split"]=="test") & (df["is_eligible_primary"]==True)].reset_index(drop=True)
    corr = sub["correct_id_seq"].values
    score_cols = ["score_m1_seq", "score_m2_seq", "score_m_dir_seq", "score_m_norm_seq", "score_m3_old_seq"]
    
    rows = []
    base_aurc = None
    raw_p = {}
    
    for col in score_cols:
        aurc, cov, risk = rve.calculate_discrete_aurc(corr, sub[col].values)
        if col == "score_m1_seq":
            base_aurc = aurc
            delta = 0.0
            p_val = 1.0
        else:
            delta = aurc - base_aurc
            p_val = rve.compute_paired_cluster_permutation_test(sub, col, "score_m1_seq", n_permutations=1000, seed=42)
            raw_p[col] = p_val
            
        r100 = risk[int(len(corr)*1.0)-1]
        r90 = risk[int(len(corr)*0.9)-1]
        r80 = risk[int(len(corr)*0.8)-1]
        r70 = risk[int(len(corr)*0.7)-1]
        r50 = risk[int(len(corr)*0.5)-1]
        
        # Bootstrap 95% CI
        boot_res = rve.compute_paired_cluster_bootstrap(sub, col, "score_m1_seq", n_bootstraps=1000, seed=42)
        ci_lower, ci_upper = boot_res["ci_95"]
        
        rows.append({
            "model": model_name, "estimator": col.replace("_seq", ""),
            "aurc": aurc, "delta_vs_m1": delta,
            "ci_95_lower": ci_lower, "ci_95_upper": ci_upper,
            "perm_p_raw": p_val,
            "risk_cov_100": r100, "risk_cov_80": r80, "risk_cov_50": r50
        })
        
    adj_p = rve.apply_holm_bonferroni_correction(raw_p)
    for r in rows:
        est = r["estimator"] + "_seq"
        if est in adj_p:
            r["perm_p_holm"] = adj_p[est]["adjusted_p"]
        else:
            r["perm_p_holm"] = 1.0
            
    return pd.DataFrame(rows)

df_rq3_h = evaluate_rq3(df_h, "HuatuoGPT-Vision-7B")
df_rq3_q = evaluate_rq3(df_q, "Qwen2.5-VL-7B")
df_rq3_all = pd.concat([df_rq3_h, df_rq3_q], ignore_index=True)

print("--- [RQ3] SELECTIVE PREDICTION COMPARISON (PRIMARY TEST N=799, 204 CLUSTERS) ---")
cols_rq3 = ["model", "estimator", "aurc", "delta_vs_m1", "ci_95_lower", "ci_95_upper", "perm_p_raw", "perm_p_holm", "risk_cov_100", "risk_cov_80", "risk_cov_50"]
print(df_rq3_all[cols_rq3].to_string(index=False))
print()

# 4. Learned Risk Ablation (Dev N=323 -> Test N=799)
import tempfile
print("--- [RQ3 BONUS] LEARNED RISK ABLATION (DEV N=323 CV -> TEST N=799) ---")
with tempfile.TemporaryDirectory() as tmp:
    df_dev_h = df_h[(df_h["split"]=="dev") & (df_h["is_eligible_primary"]==True)].copy().reset_index(drop=True)
    df_test_h = df_h[(df_h["split"]=="test") & (df_h["is_eligible_primary"]==True)].copy().reset_index(drop=True)
    res_h = rve.train_and_evaluate_risk_ablation(df_dev_h, df_test_h, "HuatuoGPT-Vision-7B", Path(tmp))
    for mname, d in res_h.items():
        print(f"HuatuoGPT | {mname:10s} | Dev CV AURC: {d['dev_aurc']:.4f} | Test AURC: {d['test_aurc']:.4f} | Coefs: {d['coefficients']}")
    print()
    df_dev_q = df_q[(df_q["split"]=="dev") & (df_q["is_eligible_primary"]==True)].copy().reset_index(drop=True)
    df_test_q = df_q[(df_q["split"]=="test") & (df_q["is_eligible_primary"]==True)].copy().reset_index(drop=True)
    res_q = rve.train_and_evaluate_risk_ablation(df_dev_q, df_test_q, "Qwen2.5-VL-7B", Path(tmp))
    for mname, d in res_q.items():
        print(f"Qwen2.5   | {mname:10s} | Dev CV AURC: {d['dev_aurc']:.4f} | Test AURC: {d['test_aurc']:.4f} | Coefs: {d['coefficients']}")

print("=" * 85)
