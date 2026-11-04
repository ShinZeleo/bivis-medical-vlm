import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 80)
print("RUNNING TUGAS A: DIAGNOSTIK VALIDITAS LENGKAP PADA CHECKPOINT ASLI")
print("=" * 80)

# Paths to checkpoints
qwen_ckpt = Path("artifacts/experiments/decisive_validation/rev05/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
huatuo_ckpt = Path("artifacts/experiments/decisive_validation/rev05/predictions/huatuo/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")

assert qwen_ckpt.exists(), f"Missing {qwen_ckpt}"
assert huatuo_ckpt.exists(), f"Missing {huatuo_ckpt}"

df_qwen_full = pd.read_csv(qwen_ckpt)
df_huatuo_full = pd.read_csv(huatuo_ckpt)

# Filter Primary Test cohort
q_test = df_qwen_full[(df_qwen_full["split"] == "test") & (df_qwen_full["is_eligible_primary"] == True)].copy().reset_index(drop=True)
h_test = df_huatuo_full[(df_huatuo_full["split"] == "test") & (df_huatuo_full["is_eligible_primary"] == True)].copy().reset_index(drop=True)

# Filter Primary Dev cohort (for LR training)
q_dev = df_qwen_full[(df_qwen_full["split"] == "dev") & (df_qwen_full["is_eligible_primary"] == True)].copy().reset_index(drop=True)
h_dev = df_huatuo_full[(df_huatuo_full["split"] == "dev") & (df_huatuo_full["is_eligible_primary"] == True)].copy().reset_index(drop=True)

print(f"Loaded Qwen: dev N={len(q_dev)}, test N={len(q_test)}")
print(f"Loaded Huatuo: dev N={len(h_dev)}, test N={len(h_test)}")

# ------------------------------------------------------------------------------
# Wilson score interval
# ------------------------------------------------------------------------------
def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

# ------------------------------------------------------------------------------
# Discrete AURC function (notebook 10 reference)
# ------------------------------------------------------------------------------
def calculate_discrete_aurc(correct_array, score_array):
    N = len(correct_array)
    if N == 0:
        return 0.0, np.array([]), np.array([])
    sort_idx = np.argsort(-score_array, kind="mergesort")
    sorted_corr = correct_array[sort_idx]
    k_vals = np.arange(1, N + 1)
    coverages = k_vals / float(N)
    selective_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals.astype(float))
    discrete_aurc = float(np.mean(selective_risks))
    return discrete_aurc, coverages, selective_risks

def get_risk_at_coverage(cov_arr, risk_arr, target_cov):
    idx = int((np.abs(cov_arr - target_cov)).argmin())
    return float(risk_arr[idx])

# ------------------------------------------------------------------------------
# McNemar test
# ------------------------------------------------------------------------------
def run_mcnemar(corr_a, corr_b):
    # a: both correct, b: A correct & B incorrect, c: A incorrect & B correct, d: both incorrect
    b = int(np.sum((corr_a == 1) & (corr_b == 0)))
    c = int(np.sum((corr_a == 0) & (corr_b == 1)))
    a = int(np.sum((corr_a == 1) & (corr_b == 1)))
    d = int(np.sum((corr_a == 0) & (corr_b == 0)))
    total_discordant = b + c
    if total_discordant == 0:
        chi2_corr = 0.0
        p_val_exact = 1.0
    else:
        chi2_corr = (abs(b - c) - 1.0)**2 / float(total_discordant)
        p_val_exact = 2.0 * stats.binom.cdf(min(b, c), total_discordant, 0.5)
        p_val_exact = min(1.0, p_val_exact)
    return {"b (EN=1, ID=0)": b, "c (EN=0, ID=1)": c, "a (both=1)": a, "d (both=0)": d,
            "chi2_continuity": chi2_corr, "p_value_exact": p_val_exact}

# ------------------------------------------------------------------------------
# A1: PREDICTED-YES RATE, TPR, TNR, BALANCED ACCURACY, WILSON CI, MCNEMAR
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print(">>> A1: PREDICTED-YES RATE, TPR, TNR, BALANCED ACCURACY PER BAHASA & PER TRACK")
print("=" * 80)

a1_rows = []
mcnemar_rows = []

for mname, df in [("Qwen2.5-VL-7B", q_test), ("HuatuoGPT-Vision-7B", h_test)]:
    gt = df["reference_answer"].values # "YES" or "NO"
    n_total = len(df)
    n_gt_yes = int(np.sum(gt == "YES"))
    n_gt_no = int(np.sum(gt == "NO"))
    
    for track in ["seq", "len"]:
        pred_en = df[f"pred_en_{track}"].values
        pred_id = df[f"pred_id_{track}"].values
        corr_en = df[f"correct_en_{track}"].values
        corr_id = df[f"correct_id_{track}"].values
        
        # English
        pred_en_yes = int(np.sum(pred_en == "YES"))
        tp_en = int(np.sum((pred_en == "YES") & (gt == "YES")))
        fp_en = int(np.sum((pred_en == "YES") & (gt == "NO")))
        tn_en = int(np.sum((pred_en == "NO") & (gt == "NO")))
        fn_en = int(np.sum((pred_en == "NO") & (gt == "YES")))
        
        tpr_en = tp_en / n_gt_yes
        tnr_en = tn_en / n_gt_no
        bacc_en = (tpr_en + tnr_en) / 2.0
        acc_en = (tp_en + tn_en) / n_total
        
        ci_pyes_en = wilson_ci(pred_en_yes, n_total)
        ci_tpr_en = wilson_ci(tp_en, n_gt_yes)
        ci_tnr_en = wilson_ci(tn_en, n_gt_no)
        ci_acc_en = wilson_ci(tp_en + tn_en, n_total)
        
        a1_rows.append({
            "model": mname, "track": track, "lang": "EN",
            "N": n_total, "GT_YES": n_gt_yes, "GT_NO": n_gt_no,
            "Pred_YES": pred_en_yes, "Pred_YES_rate": pred_en_yes / n_total * 100.0,
            "Pred_YES_CI": (ci_pyes_en[0] * 100.0, ci_pyes_en[1] * 100.0),
            "TP": tp_en, "FP": fp_en, "TN": tn_en, "FN": fn_en,
            "Accuracy": acc_en * 100.0, "Acc_CI": (ci_acc_en[0] * 100.0, ci_acc_en[1] * 100.0),
            "TPR": tpr_en * 100.0, "TPR_CI": (ci_tpr_en[0] * 100.0, ci_tpr_en[1] * 100.0),
            "TNR": tnr_en * 100.0, "TNR_CI": (ci_tnr_en[0] * 100.0, ci_tnr_en[1] * 100.0),
            "Balanced_Acc": bacc_en * 100.0
        })
        
        # Indonesian
        pred_id_yes = int(np.sum(pred_id == "YES"))
        tp_id = int(np.sum((pred_id == "YES") & (gt == "YES")))
        fp_id = int(np.sum((pred_id == "YES") & (gt == "NO")))
        tn_id = int(np.sum((pred_id == "NO") & (gt == "NO")))
        fn_id = int(np.sum((pred_id == "NO") & (gt == "YES")))
        
        tpr_id = tp_id / n_gt_yes
        tnr_id = tn_id / n_gt_no
        bacc_id = (tpr_id + tnr_id) / 2.0
        acc_id = (tp_id + tn_id) / n_total
        
        ci_pyes_id = wilson_ci(pred_id_yes, n_total)
        ci_tpr_id = wilson_ci(tp_id, n_gt_yes)
        ci_tnr_id = wilson_ci(tn_id, n_gt_no)
        ci_acc_id = wilson_ci(tp_id + tn_id, n_total)
        
        a1_rows.append({
            "model": mname, "track": track, "lang": "ID",
            "N": n_total, "GT_YES": n_gt_yes, "GT_NO": n_gt_no,
            "Pred_YES": pred_id_yes, "Pred_YES_rate": pred_id_yes / n_total * 100.0,
            "Pred_YES_CI": (ci_pyes_id[0] * 100.0, ci_pyes_id[1] * 100.0),
            "TP": tp_id, "FP": fp_id, "TN": tn_id, "FN": fn_id,
            "Accuracy": acc_id * 100.0, "Acc_CI": (ci_acc_id[0] * 100.0, ci_acc_id[1] * 100.0),
            "TPR": tpr_id * 100.0, "TPR_CI": (ci_tpr_id[0] * 100.0, ci_tpr_id[1] * 100.0),
            "TNR": tnr_id * 100.0, "TNR_CI": (ci_tnr_id[0] * 100.0, ci_tnr_id[1] * 100.0),
            "Balanced_Acc": bacc_id * 100.0
        })
        
        # McNemar between EN and ID
        mcn = run_mcnemar(corr_en, corr_id)
        mcnemar_rows.append({
            "model": mname, "track": track, **mcn
        })

df_a1 = pd.DataFrame(a1_rows)
df_mcn = pd.DataFrame(mcnemar_rows)

print(df_a1[["model", "track", "lang", "Pred_YES_rate", "Accuracy", "TPR", "TNR", "Balanced_Acc"]].to_string(index=False))
print("\nMcNemar Test Results (EN vs ID):")
print(df_mcn.to_string(index=False))

