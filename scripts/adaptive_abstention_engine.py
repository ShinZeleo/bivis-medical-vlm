#!/usr/bin/env python3
"""
Adaptive Abstention Engine (v11.0 — Reconciled & OOF-Calibrated)
================================================================
Implements risk-aware selective answering for Medical VLMs:
1. Calibrates decision threshold tau strictly on Development Set (N=323, 88 clusters)
   using Out-of-Fold (OOF) predictions via GroupKFold (K=5) grouped by canonical_image_id
   to ELIMINATE in-sample threshold optimism.
2. Evaluates Coverage-Targeted Grid: 50%, 60%, 70%, 80%, 90% on Primary Test Set (N=799, 204 clusters).
3. Evaluates Risk-Targeted Grid: 15%, 20%, 25% on Primary Test Set.
4. Generates reconciled, non-overclaiming official CSV artifacts:
   - adaptive_abstention_coverage_targeted.csv
   - adaptive_abstention_risk_targeted.csv
   - threshold_calibration_oof_vs_insample.csv
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


def compute_oof_and_insample_risk_dev(
    df_dev: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    random_seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Computes Out-of-Fold (OOF) risk predictions on Dev Set via GroupKFold (K=5)
    grouped by canonical_image_id to prevent in-sample threshold optimism.
    Also fits final models on 100% of Dev Set for Test Set inference.
    """
    groups_dev = df_dev["canonical_image_id"].values
    y_dev = df_dev["error_id"].values

    oof_risks = {}
    insample_risks = {}
    fitted_models = {}

    gkf = GroupKFold(n_splits=5)

    for mname, feats in feature_sets.items():
        if mname == "M1_direct":
            # Direct confidence: no training required
            risk_direct = 1.0 - df_dev["score_m1_seq"].values
            oof_risks[mname] = risk_direct
            insample_risks[mname] = risk_direct
            fitted_models[mname] = None
        else:
            # 1. Out-of-Fold predictions via GroupKFold
            oof_pred = np.zeros(len(df_dev))
            for tr_idx, val_idx in gkf.split(df_dev[feats].values, y_dev, groups=groups_dev):
                scaler_cv = StandardScaler()
                X_tr = scaler_cv.fit_transform(df_dev[feats].values[tr_idx])
                X_val = scaler_cv.transform(df_dev[feats].values[val_idx])

                clf_cv = LogisticRegression(penalty="l2", C=1.0, random_state=random_seed)
                clf_cv.fit(X_tr, y_dev[tr_idx])
                oof_pred[val_idx] = clf_cv.predict_proba(X_val)[:, 1]

            oof_risks[mname] = oof_pred

            # 2. Fit final model on 100% of Dev Set
            scaler_final = StandardScaler()
            X_dev_scaled = scaler_final.fit_transform(df_dev[feats].values)
            clf_final = LogisticRegression(penalty="l2", C=1.0, random_state=random_seed)
            clf_final.fit(X_dev_scaled, y_dev)

            insample_risks[mname] = clf_final.predict_proba(X_dev_scaled)[:, 1]
            fitted_models[mname] = (scaler_final, clf_final)

    return oof_risks, insample_risks, fitted_models


def calibrate_and_evaluate_coverage_targeted(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str,
    target_coverages: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90],
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coverage-Targeted Selective Answering:
    Calibrates threshold tau on Dev Set using:
      (a) Out-of-Fold (OOF) GroupKFold predictions (PRIMARY, UNBIASED)
      (b) In-Sample Dev predictions (COMPARATIVE ABLATION)
    Applies frozen thresholds to Test Set.
    """
    for df in [df_dev, df_test]:
        if "d_lang" not in df.columns:
            df["d_lang"] = (df["pred_en_seq"] != df["pred_id_seq"]).astype(int)
        if "error_id" not in df.columns:
            df["error_id"] = (1 - df["correct_id_seq"]).astype(int)

    feature_sets = {
        "M1_direct": None,
        "LR_conf": ["score_m1_seq"],
        "LR_visual": ["score_m1_seq", "delta_p_dir_seq"],
        "LR_full": ["score_m1_seq", "delta_p_dir_seq", "d_lang"]
    }

    oof_risks, insample_risks, fitted_models = compute_oof_and_insample_risk_dev(
        df_dev, feature_sets, random_seed=random_seed
    )

    records_eval = []
    records_comp = []

    for mname, feats in feature_sets.items():
        if mname == "M1_direct":
            test_risk = 1.0 - df_test["score_m1_seq"].values
        else:
            scaler_final, clf_final = fitted_models[mname]
            X_test_scaled = scaler_final.transform(df_test[feats].values)
            test_risk = clf_final.predict_proba(X_test_scaled)[:, 1]

        dev_oof_r = oof_risks[mname]
        dev_in_r = insample_risks[mname]

        for target_cov in target_coverages:
            # 1. Primary OOF Threshold
            tau_oof = float(np.percentile(dev_oof_r, target_cov * 100.0))
            ans_oof = (test_risk <= tau_oof)
            n_ans_oof = int(np.sum(ans_oof))
            n_tot = len(df_test)
            realized_cov_oof = float(n_ans_oof / n_tot)
            abstain_oof = float(1.0 - realized_cov_oof)
            n_err_oof = int(np.sum(df_test["error_id"].values[ans_oof])) if n_ans_oof > 0 else 0
            sel_risk_oof = float(n_err_oof / n_ans_oof) if n_ans_oof > 0 else 0.0
            sel_acc_oof = float(1.0 - sel_risk_oof)

            records_eval.append({
                "model": model_name,
                "estimator": mname,
                "calibration_method": "group_kfold_oof",
                "target_dev_coverage": target_cov,
                "tau_risk_threshold": tau_oof,
                "tau_score_threshold": 1.0 - tau_oof,
                "realized_test_coverage": realized_cov_oof,
                "abstention_rate": abstain_oof,
                "n_answered": n_ans_oof,
                "n_errors_answered": n_err_oof,
                "selective_risk": sel_risk_oof,
                "selective_accuracy": sel_acc_oof
            })

            # 2. In-Sample Comparison Threshold
            tau_in = float(np.percentile(dev_in_r, target_cov * 100.0))
            ans_in = (test_risk <= tau_in)
            n_ans_in = int(np.sum(ans_in))
            realized_cov_in = float(n_ans_in / n_tot)
            n_err_in = int(np.sum(df_test["error_id"].values[ans_in])) if n_ans_in > 0 else 0
            sel_risk_in = float(n_err_in / n_ans_in) if n_ans_in > 0 else 0.0

            records_comp.append({
                "model": model_name,
                "estimator": mname,
                "target_dev_coverage": target_cov,
                "tau_oof": tau_oof,
                "test_cov_oof": realized_cov_oof,
                "sel_risk_oof": sel_risk_oof,
                "tau_insample": tau_in,
                "test_cov_insample": realized_cov_in,
                "sel_risk_insample": sel_risk_in,
                "tau_diff_in_minus_oof": tau_in - tau_oof
            })

    return pd.DataFrame(records_eval), pd.DataFrame(records_comp)


def calibrate_and_evaluate_risk_targeted(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str,
    target_risks: List[float] = [0.15, 0.20, 0.25],
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Risk-Targeted Selective Answering via OOF Dev Calibration:
    Finds threshold on Dev Set using GroupKFold OOF predictions such that cumulative risk <= target_risk.
    Evaluates retained coverage and realized selective risk on Test Set.
    """
    for df in [df_dev, df_test]:
        if "d_lang" not in df.columns:
            df["d_lang"] = (df["pred_en_seq"] != df["pred_id_seq"]).astype(int)
        if "error_id" not in df.columns:
            df["error_id"] = (1 - df["correct_id_seq"]).astype(int)

    feature_sets = {
        "M1_direct": None,
        "LR_conf": ["score_m1_seq"],
        "LR_visual": ["score_m1_seq", "delta_p_dir_seq"],
        "LR_full": ["score_m1_seq", "delta_p_dir_seq", "d_lang"]
    }

    oof_risks, _, fitted_models = compute_oof_and_insample_risk_dev(
        df_dev, feature_sets, random_seed=random_seed
    )

    records = []

    for mname, feats in feature_sets.items():
        if mname == "M1_direct":
            test_risk = 1.0 - df_test["score_m1_seq"].values
        else:
            scaler_final, clf_final = fitted_models[mname]
            X_test_scaled = scaler_final.transform(df_test[feats].values)
            test_risk = clf_final.predict_proba(X_test_scaled)[:, 1]

        dev_oof_r = oof_risks[mname]

        # Order Dev samples by OOF predicted risk
        order_dev = np.argsort(dev_oof_r)
        sorted_dev_err = df_dev["error_id"].values[order_dev]
        sorted_dev_risk = dev_oof_r[order_dev]
        cum_err = np.cumsum(sorted_dev_err)
        cum_counts = np.arange(1, len(df_dev) + 1)
        cum_risk = cum_err / cum_counts

        for target_risk in target_risks:
            valid_k = np.where(cum_risk <= target_risk)[0]
            if len(valid_k) > 0:
                best_k = valid_k[-1]
                tau_risk = float(sorted_dev_risk[best_k])
                dev_achieved_risk = float(cum_risk[best_k])
                dev_cov = float((best_k + 1) / len(df_dev))
            else:
                best_k = 0
                tau_risk = float(sorted_dev_risk[0])
                dev_achieved_risk = float(cum_risk[0])
                dev_cov = float(1.0 / len(df_dev))

            ans_test = (test_risk <= tau_risk)
            n_ans = int(np.sum(ans_test))
            n_tot = len(df_test)
            test_cov = float(n_ans / n_tot)
            if n_ans > 0:
                act_risk = float(np.sum(df_test["error_id"].values[ans_test]) / n_ans)
                act_acc = float(1.0 - act_risk)
            else:
                act_risk = 0.0
                act_acc = 1.0

            records.append({
                "model": model_name,
                "estimator": mname,
                "calibration_method": "group_kfold_oof",
                "target_dev_risk": target_risk,
                "tau_risk_threshold": tau_risk,
                "dev_achieved_risk": dev_achieved_risk,
                "dev_retained_coverage": dev_cov,
                "realized_test_coverage": test_cov,
                "actual_test_risk": act_risk,
                "actual_test_selective_accuracy": act_acc,
                "risk_slippage": act_risk - target_risk
            })

    return pd.DataFrame(records)


def run_full_adaptive_abstention_pipeline():
    qwen_ckpt = Path("artifacts/experiments/decisive_validation/rev05/predictions/qwen/live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv")
    huatuo_ckpt = Path("artifacts/experiments/decisive_validation/rev05/predictions/huatuo/live_validation_checkpoint_FreedomIntelligence_HuatuoGPT_Vision_7B_Qwen2.5VL.csv")

    assert qwen_ckpt.exists(), f"Missing Qwen checkpoint: {qwen_ckpt}"
    assert huatuo_ckpt.exists(), f"Missing Huatuo checkpoint: {huatuo_ckpt}"

    cov_dfs = []
    comp_dfs = []
    risk_dfs = []

    for mtitle, p in [("Qwen2.5-VL-7B", qwen_ckpt), ("HuatuoGPT-Vision-7B", huatuo_ckpt)]:
        df = pd.read_csv(p)
        df_dev = df[(df["split"] == "dev") & (df["is_eligible_primary"] == True)].copy().reset_index(drop=True)
        df_test = df[(df["split"] == "test") & (df["is_eligible_primary"] == True)].copy().reset_index(drop=True)

        print(f"Executing OOF-Calibrated Adaptive Abstention: {mtitle} (Dev N={len(df_dev)}, Test N={len(df_test)})...")
        df_cov, df_comp = calibrate_and_evaluate_coverage_targeted(df_dev, df_test, mtitle)
        df_risk = calibrate_and_evaluate_risk_targeted(df_dev, df_test, mtitle)

        cov_dfs.append(df_cov)
        comp_dfs.append(df_comp)
        risk_dfs.append(df_risk)

    df_cov_all = pd.concat(cov_dfs, ignore_index=True)
    df_comp_all = pd.concat(comp_dfs, ignore_index=True)
    df_risk_all = pd.concat(risk_dfs, ignore_index=True)

    metrics_dir = Path("artifacts/experiments/decisive_validation/rev05/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_cov_p = metrics_dir / "adaptive_abstention_coverage_targeted.csv"
    out_comp_p = metrics_dir / "threshold_calibration_oof_vs_insample.csv"
    out_risk_p = metrics_dir / "adaptive_abstention_risk_targeted.csv"

    df_cov_all.to_csv(out_cov_p, index=False)
    df_comp_all.to_csv(out_comp_p, index=False)
    df_risk_all.to_csv(out_risk_p, index=False)

    print(f"\n✅ Saved: {out_cov_p} ({len(df_cov_all)} rows, GroupKFold OOF Calibrated)")
    print(f"✅ Saved: {out_comp_p} ({len(df_comp_all)} rows, OOF vs In-Sample Comparison)")
    print(f"✅ Saved: {out_risk_p} ({len(df_risk_all)} rows, Risk-Targeted with Slippage Tracking)")


if __name__ == "__main__":
    run_full_adaptive_abstention_pipeline()
