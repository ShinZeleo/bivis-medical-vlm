import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Set random seed
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print("EXECUTING COMPREHENSIVE VALIDITY DIAGNOSTICS SUITE (TASKS A1 - A5)")
print("=" * 80)

# Checkpoints
qwen_path = Path("artifacts/experiments/decisive_validation/rev05/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
huatuo_path = Path("artifacts/experiments/decisive_validation/rev05/predictions/huatuo/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")

df_qwen_raw = pd.read_csv(qwen_path)
df_huatuo_raw = pd.read_csv(huatuo_path)

# Cohorts
q_test = df_qwen_raw[(df_qwen_raw["split"] == "test") & (df_qwen_raw["is_eligible_primary"] == True)].copy().reset_index(drop=True)
q_dev  = df_qwen_raw[(df_qwen_raw["split"] == "dev") & (df_qwen_raw["is_eligible_primary"] == True)].copy().reset_index(drop=True)

h_test = df_huatuo_raw[(df_huatuo_raw["split"] == "test") & (df_huatuo_raw["is_eligible_primary"] == True)].copy().reset_index(drop=True)
h_dev  = df_huatuo_raw[(df_huatuo_raw["split"] == "dev") & (df_huatuo_raw["is_eligible_primary"] == True)].copy().reset_index(drop=True)

print(f"Primary Test: N = {len(q_test)}, Dev: N = {len(q_dev)}")

# Wilson score interval
def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

# Discrete Riemann AURC
def calculate_discrete_aurc(correct_array, score_array, tie_mode="stable", rng=None):
    N = len(correct_array)
    if N == 0:
        return 0.0, np.array([]), np.array([])
    if tie_mode == "stable":
        sort_idx = np.argsort(-score_array, kind="mergesort")
    elif tie_mode == "random":
        # Add tiny jitter to break ties randomly
        jitter = rng.uniform(-1e-12, 1e-12, size=N) if rng is not None else np.random.uniform(-1e-12, 1e-12, size=N)
        sort_idx = np.argsort(-(score_array + jitter), kind="mergesort")
    elif tie_mode == "worst":
        # Tied scores sorted with incorrect answers first
        sort_idx = np.lexsort((correct_array, -score_array))
    elif tie_mode == "best":
        # Tied scores sorted with correct answers first
        sort_idx = np.lexsort((-correct_array, -score_array))
    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode}")
        
    sorted_corr = correct_array[sort_idx]
    k_vals = np.arange(1, N + 1)
    coverages = k_vals / float(N)
    selective_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals.astype(float))
    discrete_aurc = float(np.mean(selective_risks))
    return discrete_aurc, coverages, selective_risks

def get_risk_at_coverage(cov_arr, risk_arr, target_cov):
    idx = int((np.abs(cov_arr - target_cov)).argmin())
    return float(risk_arr[idx])

# ==============================================================================
# SECTION A1: PREDICTED-YES RATE, TPR, TNR, BALANCED ACCURACY, WILSON CI, MCNEMAR
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION A1: PER-LANGUAGE & PER-TRACK METRICS (PRIMARY TEST N=799)")
print("=" * 80)

a1_results = []
mcnemar_results = []

for mname, df in [("Qwen2.5-VL-7B", q_test), ("HuatuoGPT-Vision-7B", h_test)]:
    gt = df["reference_answer"].values
    N = len(df)
    n_gt_yes = int(np.sum(gt == "YES"))
    n_gt_no = int(np.sum(gt == "NO"))
    
    for track in ["seq", "len"]:
        pred_en = df[f"pred_en_{track}"].values
        pred_id = df[f"pred_id_{track}"].values
        corr_en = df[f"correct_en_{track}"].values
        corr_id = df[f"correct_id_{track}"].values
        
        # English
        pyes_en = int(np.sum(pred_en == "YES"))
        tp_en = int(np.sum((pred_en == "YES") & (gt == "YES")))
        fp_en = int(np.sum((pred_en == "YES") & (gt == "NO")))
        tn_en = int(np.sum((pred_en == "NO") & (gt == "NO")))
        fn_en = int(np.sum((pred_en == "NO") & (gt == "YES")))
        
        acc_en = (tp_en + tn_en) / N
        tpr_en = tp_en / n_gt_yes
        tnr_en = tn_en / n_gt_no
        bacc_en = (tpr_en + tnr_en) / 2.0
        
        a1_results.append({
            "model": mname, "track": track, "lang": "EN",
            "N": N, "GT_YES": n_gt_yes, "GT_NO": n_gt_no,
            "Pred_YES": pyes_en, "Pred_YES_rate": pyes_en / N * 100,
            "Pred_YES_CI": [x * 100 for x in wilson_ci(pyes_en, N)],
            "TP": tp_en, "FP": fp_en, "TN": tn_en, "FN": fn_en,
            "Accuracy": acc_en * 100,
            "Acc_CI": [x * 100 for x in wilson_ci(tp_en + tn_en, N)],
            "TPR": tpr_en * 100,
            "TPR_CI": [x * 100 for x in wilson_ci(tp_en, n_gt_yes)],
            "TNR": tnr_en * 100,
            "TNR_CI": [x * 100 for x in wilson_ci(tn_en, n_gt_no)],
            "Balanced_Acc": bacc_en * 100
        })
        
        # Indonesian
        pyes_id = int(np.sum(pred_id == "YES"))
        tp_id = int(np.sum((pred_id == "YES") & (gt == "YES")))
        fp_id = int(np.sum((pred_id == "YES") & (gt == "NO")))
        tn_id = int(np.sum((pred_id == "NO") & (gt == "NO")))
        fn_id = int(np.sum((pred_id == "NO") & (gt == "YES")))
        
        acc_id = (tp_id + tn_id) / N
        tpr_id = tp_id / n_gt_yes
        tnr_id = tn_id / n_gt_no
        bacc_id = (tpr_id + tnr_id) / 2.0
        
        a1_results.append({
            "model": mname, "track": track, "lang": "ID",
            "N": N, "GT_YES": n_gt_yes, "GT_NO": n_gt_no,
            "Pred_YES": pyes_id, "Pred_YES_rate": pyes_id / N * 100,
            "Pred_YES_CI": [x * 100 for x in wilson_ci(pyes_id, N)],
            "TP": tp_id, "FP": fp_id, "TN": tn_id, "FN": fn_id,
            "Accuracy": acc_id * 100,
            "Acc_CI": [x * 100 for x in wilson_ci(tp_id + tn_id, N)],
            "TPR": tpr_id * 100,
            "TPR_CI": [x * 100 for x in wilson_ci(tp_id, n_gt_yes)],
            "TNR": tnr_id * 100,
            "TNR_CI": [x * 100 for x in wilson_ci(tn_id, n_gt_no)],
            "Balanced_Acc": bacc_id * 100
        })
        
        # McNemar
        b = int(np.sum((corr_en == 1) & (corr_id == 0)))
        c = int(np.sum((corr_en == 0) & (corr_id == 1)))
        a = int(np.sum((corr_en == 1) & (corr_id == 1)))
        d = int(np.sum((corr_en == 0) & (corr_id == 0)))
        tot_discord = b + c
        chi2_c = (abs(b - c) - 1.0)**2 / float(tot_discord) if tot_discord > 0 else 0.0
        p_exact = 2.0 * stats.binom.cdf(min(b, c), tot_discord, 0.5) if tot_discord > 0 else 1.0
        p_exact = min(1.0, p_exact)
        
        mcnemar_results.append({
            "model": mname, "track": track,
            "b (EN=1, ID=0)": b, "c (EN=0, ID=1)": c,
            "a (both=1)": a, "d (both=0)": d,
            "chi2_continuity": chi2_c, "p_value_exact": p_exact
        })

df_a1_out = pd.DataFrame(a1_results)
df_mcn_out = pd.DataFrame(mcnemar_results)

print(df_a1_out[["model", "track", "lang", "Pred_YES_rate", "Accuracy", "TPR", "TNR", "Balanced_Acc"]].to_string(index=False))
print("\nMcNemar Test Results (EN correct & ID incorrect vs vice-versa):")
print(df_mcn_out.to_string(index=False))

# ==============================================================================
# SECTION A2: ARTIFACT TOKEN PANJANG HUATUO & FLIP RATE 50.81%
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION A2: INVESTIGASI TOKEN PANJANG HUATUO & FLIP RATE 50.81%")
print("=" * 80)

# Check tokenizer
from transformers import AutoTokenizer
tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
tok_huatuo = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL", trust_remote_code=True)

candidate_tokens_info = []
for c in ["YES<|im_end|>", "NO<|im_end|>", "YA<|im_end|>", "TIDAK<|im_end|>"]:
    q_ids = tok_qwen.encode(c, add_special_tokens=False)
    h_ids = tok_huatuo.encode(c, add_special_tokens=False)
    candidate_tokens_info.append({
        "candidate": c,
        "qwen_ids": q_ids,
        "qwen_num_tokens": len(q_ids),
        "qwen_tokens": [tok_qwen.decode([i]) for i in q_ids],
        "huatuo_ids": h_ids,
        "huatuo_num_tokens": len(h_ids),
        "huatuo_tokens": [tok_huatuo.decode([i]) for i in h_ids]
    })
print(pd.DataFrame(candidate_tokens_info)[["candidate", "qwen_num_tokens", "qwen_tokens", "huatuo_num_tokens", "huatuo_tokens"]].to_string(index=False))

# 5 concrete samples
print("\n5 CONCRETE SAMPLES FROM HUATUO TEST SET:")
sample_subset = h_test.head(5)[["qa_id", "question_en", "question_id", "reference_answer", 
                               "pred_en_seq", "pred_id_seq", "p_orig_ya_seq", "p_orig_tidak_seq",
                               "pred_en_len", "pred_id_len", "p_orig_ya_len", "p_orig_tidak_len"]]
print(sample_subset.to_string(index=False))

# Flip breakdown
for mname, df in [("Qwen2.5-VL-7B", q_test), ("HuatuoGPT-Vision-7B", h_test)]:
    print(f"\n--- Model: {mname} ---")
    for tr in ["seq", "len"]:
        pen = df[f"pred_en_{tr}"]
        pid = df[f"pred_id_{tr}"]
        flips = (pen != pid)
        en_y_id_n = ((pen == "YES") & (pid == "NO")).sum()
        en_n_id_y = ((pen == "NO") & (pid == "YES")).sum()
        gap = (df[f"correct_en_{tr}"].mean() - df[f"correct_id_{tr}"].mean()) * 100
        print(f"Track {tr}:")
        print(f"  Accuracy EN: {df[f'correct_en_{tr}'].mean()*100:.2f}%, ID: {df[f'correct_id_{tr}'].mean()*100:.2f}%, Gap: {gap:.2f} pp")
        print(f"  Flip rate: {flips.sum()}/{len(df)} ({flips.mean()*100:.2f}%)")
        print(f"  Asymmetry: EN=YES -> ID=NO: {en_y_id_n} ({en_y_id_n/len(df)*100:.2f}%), EN=NO -> ID=YES: {en_n_id_y} ({en_n_id_y/len(df)*100:.2f}%)")
        print(f"  Predicted-YES rate: EN={np.mean(pen=='YES')*100:.2f}%, ID={np.mean(pid=='YES')*100:.2f}%")

# ==============================================================================
# SECTION A3: TABEL BASELINE PEMBANDING PADA TEST SET (N=799)
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION A3: TABEL BASELINE PEMBANDING PADA PRIMARY TEST SET")
print("=" * 80)

# Build Logistic Regression models on Dev set
def train_and_eval_baselines(df_dev, df_test, model_title):
    for df in [df_dev, df_test]:
        df["d_lang"] = (df["pred_en_seq"] != df["pred_id_seq"]).astype(int)
        df["error_id"] = (1 - df["correct_id_seq"]).astype(int)
        
    y_dev = df_dev["error_id"].values
    y_test_corr = df_test["correct_id_seq"].values
    
    # 1. M1 raw conf
    aurc_m1, cov_m1, risk_m1 = calculate_discrete_aurc(y_test_corr, df_test["score_m1_seq"].values)
    r70_m1 = get_risk_at_coverage(cov_m1, risk_m1, 0.7)
    acc70_m1 = (1.0 - r70_m1) * 100.0
    
    # 2. LR_visual
    feat_vis = ["score_m1_seq", "delta_p_dir_seq"]
    sc_vis = StandardScaler()
    X_dev_vis = sc_vis.fit_transform(df_dev[feat_vis].values)
    X_test_vis = sc_vis.transform(df_test[feat_vis].values)
    clf_vis = LogisticRegression(penalty="l2", C=1.0, random_state=42)
    clf_vis.fit(X_dev_vis, y_dev)
    risk_vis = clf_vis.predict_proba(X_test_vis)[:, 1]
    score_vis = 1.0 - risk_vis
    aurc_vis, cov_vis, r_vis = calculate_discrete_aurc(y_test_corr, score_vis)
    r70_vis = get_risk_at_coverage(cov_vis, r_vis, 0.7)
    acc70_vis = (1.0 - r70_vis) * 100.0
    
    # 3. LR_full
    feat_full = ["score_m1_seq", "delta_p_dir_seq", "d_lang"]
    sc_full = StandardScaler()
    X_dev_full = sc_full.fit_transform(df_dev[feat_full].values)
    X_test_full = sc_full.transform(df_test[feat_full].values)
    clf_full = LogisticRegression(penalty="l2", C=1.0, random_state=42)
    clf_full.fit(X_dev_full, y_dev)
    risk_full = clf_full.predict_proba(X_test_full)[:, 1]
    score_full = 1.0 - risk_full
    aurc_full, cov_full, r_full = calculate_discrete_aurc(y_test_corr, score_full)
    r70_full = get_risk_at_coverage(cov_full, r_full, 0.7)
    acc70_full = (1.0 - r70_full) * 100.0
    
    # 4. D_lang-only ranking: score = 1.0 - d_lang
    score_dlang = 1.0 - df_test["d_lang"].values
    aurc_dlang, cov_dlang, r_dlang = calculate_discrete_aurc(y_test_corr, score_dlang)
    r70_dlang = get_risk_at_coverage(cov_dlang, r_dlang, 0.7)
    acc70_dlang = (1.0 - r70_dlang) * 100.0
    
    # 5. "Jawaban EN + conf EN" (using English prediction as the system answer!)
    y_test_corr_en = df_test["correct_en_seq"].values
    score_en = df_test["conf_en_seq"].values
    aurc_en, cov_en, r_en = calculate_discrete_aurc(y_test_corr_en, score_en)
    r70_en = get_risk_at_coverage(cov_en, r_en, 0.7)
    acc70_en = (1.0 - r70_en) * 100.0
    acc100_en = df_test["correct_en_seq"].mean() * 100.0
    
    # 6. "Abstain iff D_lang=1" (Rule-based gate: only answer when EN and ID agree)
    agree_mask = (df_test["d_lang"] == 0)
    cov_gate = float(agree_mask.mean()) * 100.0
    acc_gate = float(df_test.loc[agree_mask, "correct_id_seq"].mean()) * 100.0
    
    return {
        "model": model_title,
        "Acc_EN_unabstained": acc100_en,
        "Acc_ID_unabstained": df_test["correct_id_seq"].mean() * 100.0,
        "M1_AURC": aurc_m1, "M1_Acc_at_70": acc70_m1,
        "LR_visual_AURC": aurc_vis, "LR_visual_Acc_at_70": acc70_vis,
        "LR_full_AURC": aurc_full, "LR_full_Acc_at_70": acc70_full,
        "D_lang_only_AURC": aurc_dlang, "D_lang_only_Acc_at_70": acc70_dlang,
        "EN_ans_AURC": aurc_en, "EN_ans_Acc_at_70": acc70_en,
        "Gate_Dlang_coverage": cov_gate, "Gate_Dlang_acc": acc_gate
    }

b_qwen = train_and_eval_baselines(q_dev, q_test, "Qwen2.5-VL-7B")
b_huatuo = train_and_eval_baselines(h_dev, h_test, "HuatuoGPT-Vision-7B")
df_baselines = pd.DataFrame([b_qwen, b_huatuo])
print(df_baselines.to_string(index=False))

# ==============================================================================
# SECTION A4: TIE-SHARE & AURC UNDER RANDOM TIE BREAKING (M2 & M_DIR)
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION A4: TIE-SHARE & AURC TIE-AWARE ANALYSIS (200 RUNS)")
print("=" * 80)

a4_rows = []
N_TIE_REPS = 200

for mname, df in [("Qwen2.5-VL-7B", q_test), ("HuatuoGPT-Vision-7B", h_test)]:
    y_corr = df["correct_id_seq"].values
    N = len(df)
    
    score_cols = ["score_m1_seq", "score_m2_seq", "score_m_dir_seq", "score_m_norm_seq"]
    
    for col in score_cols:
        s = df[col].values
        # Check ties
        unique_vals, counts = np.unique(s, return_counts=True)
        tied_counts = counts[counts > 1]
        n_tied_samples = int(np.sum(tied_counts))
        tie_share = (n_tied_samples / float(N)) * 100.0
        
        # Count specifically ties at 0.0
        n_zeros = int(np.sum(s == 0.0))
        zero_share = (n_zeros / float(N)) * 100.0
        
        # AURC stable
        aurc_stable, _, _ = calculate_discrete_aurc(y_corr, s, tie_mode="stable")
        
        # AURC worst & best
        aurc_worst, _, _ = calculate_discrete_aurc(y_corr, s, tie_mode="worst")
        aurc_best, _, _ = calculate_discrete_aurc(y_corr, s, tie_mode="best")
        
        # AURC random 200 reps
        rng_tie = np.random.RandomState(42)
        aurc_rand_list = []
        for _ in range(N_TIE_REPS):
            a_rand, _, _ = calculate_discrete_aurc(y_corr, s, tie_mode="random", rng=rng_tie)
            aurc_rand_list.append(a_rand)
            
        mean_rand = float(np.mean(aurc_rand_list))
        std_rand = float(np.std(aurc_rand_list))
        p2_5 = float(np.percentile(aurc_rand_list, 2.5))
        p97_5 = float(np.percentile(aurc_rand_list, 97.5))
        
        a4_rows.append({
            "model": mname, "estimator": col,
            "N": N, "tie_share_%": tie_share, "zero_share_%": zero_share,
            "aurc_stable": aurc_stable,
            "aurc_rand_mean": mean_rand,
            "aurc_rand_std": std_rand,
            "rand_ci_95": (p2_5, p97_5),
            "aurc_best": aurc_best,
            "aurc_worst": aurc_worst,
            "spread_best_worst": aurc_worst - aurc_best
        })

df_a4 = pd.DataFrame(a4_rows)
print(df_a4[["model", "estimator", "tie_share_%", "zero_share_%", "aurc_stable", "aurc_rand_mean", "aurc_rand_std", "spread_best_worst"]].to_string(index=False))

# ==============================================================================
# SECTION A5: PAIRED CLUSTER PERMUTATION TEST (B=10,000, HOLM FAMILY 4)
# ==============================================================================
print("\n" + "=" * 80)
print("SECTION A5: SIGNIFICANCE TESTING LR_FULL VS BASELINES (B=10,000)")
print("=" * 80)

def paired_cluster_permutation_test_exact(df, score_a, score_b, corr_col="correct_id_seq", 
                                          cluster_col="canonical_image_id", n_perm=10000, seed=42):
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    
    cluster_vals = df[cluster_col].to_numpy()
    clusters = np.unique(cluster_vals)
    n_clusters = len(clusters)
    cluster_to_pos = [np.where(cluster_vals == c)[0] for c in clusters]
    
    y = df[corr_col].to_numpy(dtype=float)
    s_a = df[score_a].to_numpy(dtype=float)
    s_b = df[score_b].to_numpy(dtype=float)
    
    obs_a, _, _ = calculate_discrete_aurc(y, s_a)
    obs_b, _, _ = calculate_discrete_aurc(y, s_b)
    obs_diff = abs(obs_a - obs_b)
    
    if obs_diff == 0.0:
        return 1.0, 0.0, 0
        
    extreme = 0
    # Vectorized loop across clusters
    for _ in range(n_perm):
        sa_p = s_a.copy()
        sb_p = s_b.copy()
        flips = rng.choice([-1, 1], size=n_clusters)
        swap_clusters = np.where(flips == -1)[0]
        if len(swap_clusters) > 0:
            swap_idx = np.concatenate([cluster_to_pos[c_i] for c_i in swap_clusters])
            sa_p[swap_idx], sb_p[swap_idx] = s_b[swap_idx], s_a[swap_idx]
            
        pa, _, _ = calculate_discrete_aurc(y, sa_p)
        pb, _, _ = calculate_discrete_aurc(y, sb_p)
        if abs(pa - pb) >= obs_diff - 1e-12:
            extreme += 1
            
    p_val = float((extreme + 1) / (n_perm + 1))
    return p_val, obs_diff, extreme

# Prepare scores for LR_conf, LR_visual, LR_full on test set
# Fit on Dev, score on Test
for df_d, df_t, mname in [(q_dev, q_test, "Qwen"), (h_dev, h_test, "Huatuo")]:
    y_d = df_d["error_id"].values
    
    # LR_conf
    sc_c = StandardScaler()
    X_d_c = sc_c.fit_transform(df_d[["score_m1_seq"]].values)
    X_t_c = sc_c.transform(df_t[["score_m1_seq"]].values)
    clf_c = LogisticRegression(penalty="l2", C=1.0, random_state=42).fit(X_d_c, y_d)
    df_t["score_lr_conf"] = 1.0 - clf_c.predict_proba(X_t_c)[:, 1]
    
    # LR_visual
    sc_v = StandardScaler()
    X_d_v = sc_v.fit_transform(df_d[["score_m1_seq", "delta_p_dir_seq"]].values)
    X_t_v = sc_v.transform(df_t[["score_m1_seq", "delta_p_dir_seq"]].values)
    clf_v = LogisticRegression(penalty="l2", C=1.0, random_state=42).fit(X_d_v, y_d)
    df_t["score_lr_visual"] = 1.0 - clf_v.predict_proba(X_t_v)[:, 1]
    
    # LR_full
    sc_f = StandardScaler()
    X_d_f = sc_f.fit_transform(df_d[["score_m1_seq", "delta_p_dir_seq", "d_lang"]].values)
    X_t_f = sc_f.transform(df_t[["score_m1_seq", "delta_p_dir_seq", "d_lang"]].values)
    clf_f = LogisticRegression(penalty="l2", C=1.0, random_state=42).fit(X_d_f, y_d)
    df_t["score_lr_full"] = 1.0 - clf_f.predict_proba(X_t_f)[:, 1]

tests = [
    ("Qwen2.5-VL-7B", q_test, "score_lr_full", "score_lr_conf", "LR_full vs LR_conf"),
    ("Qwen2.5-VL-7B", q_test, "score_lr_full", "score_lr_visual", "LR_full vs LR_visual"),
    ("HuatuoGPT-Vision-7B", h_test, "score_lr_full", "score_lr_conf", "LR_full vs LR_conf"),
    ("HuatuoGPT-Vision-7B", h_test, "score_lr_full", "score_lr_visual", "LR_full vs LR_visual")
]

a5_raw = []
for mname, df, col_a, col_b, label in tests:
    print(f"Running permutation test (B=10,000) for {mname}: {label}...")
    p_val, obs_diff, extreme = paired_cluster_permutation_test_exact(
        df, col_a, col_b, corr_col="correct_id_seq", cluster_col="canonical_image_id", n_perm=10000, seed=42
    )
    aurc_a, _, _ = calculate_discrete_aurc(df["correct_id_seq"].values, df[col_a].values)
    aurc_b, _, _ = calculate_discrete_aurc(df["correct_id_seq"].values, df[col_b].values)
    a5_raw.append({
        "model": mname, "comparison": label,
        "aurc_full": aurc_a, "aurc_baseline": aurc_b, "delta_aurc": aurc_a - aurc_b,
        "extreme_count": extreme, "n_perm": 10000,
        "p_raw": p_val
    })

# Apply Holm-Bonferroni (family of 4)
# Sort by p_raw ascending
a5_raw_sorted = sorted(a5_raw, key=lambda x: x["p_raw"])
m_fam = len(a5_raw_sorted)
running_max = 0.0
for rank, item in enumerate(a5_raw_sorted, 1):
    mult = m_fam - rank + 1
    p_adj = min(1.0, item["p_raw"] * mult)
    running_max = max(running_max, p_adj)
    item["holm_rank"] = rank
    item["multiplier"] = mult
    item["p_holm"] = min(1.0, running_max)

df_a5 = pd.DataFrame(a5_raw_sorted)
print("\n" + "=" * 80)
print("A5 RESULTS: B=10,000 PERMUTATION TESTS & HOLM-BONFERRONI CORRECTION")
print("=" * 80)
print(df_a5[["model", "comparison", "aurc_full", "aurc_baseline", "delta_aurc", "extreme_count", "p_raw", "multiplier", "p_holm"]].to_string(index=False))

print("\n" + "=" * 80)
print("ALL DIAGNOSTICS EXECUTED SUCCESSFULLY.")
print("=" * 80)
