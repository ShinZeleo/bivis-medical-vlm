"""
Advanced Reliability & Abstention Metrics Module.
Implements ECE, Brier score, confidently-wrong rate, visual flip rate,
Odds Ratio with cluster-robust standard errors, out-of-fold threshold calibration,
and matched-coverage risk interpolation.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

from src.stats_engine import calculate_discrete_aurc, get_risk_at_coverage


def compute_calibration_metrics(
    conf_array: np.ndarray,
    correct_array: np.ndarray,
    n_bins: int = 15,
    conf_wrong_threshold: float = 0.90
) -> Dict[str, float]:
    """
    Computes:
    1. Expected Calibration Error (ECE) across n_bins equal-width bins in [0, 1].
    2. Brier Score: mean((conf - correct)^2).
    3. Confidently-Wrong Rate at threshold (default 0.90): P(correct=0 | conf >= threshold).
    4. Confidently-Wrong Share of Total: P(correct=0 and conf >= threshold).
    """
    conf = np.asarray(conf_array, dtype=float)
    corr = np.asarray(correct_array, dtype=float)
    N = len(conf)
    if N == 0:
        return {"ece": 0.0, "brier": 0.0, "conf_wrong_rate": 0.0, "conf_wrong_share": 0.0}
        
    # Brier score
    brier = float(np.mean((conf - corr) ** 2))
    
    # ECE with equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = np.mean(corr[mask])
            bin_conf = np.mean(conf[mask])
            ece += (count / float(N)) * abs(bin_acc - bin_conf)
            
    # Confidently wrong
    high_conf_mask = (conf >= conf_wrong_threshold)
    high_conf_count = int(np.sum(high_conf_mask))
    if high_conf_count > 0:
        wrong_high_conf = int(np.sum(corr[high_conf_mask] == 0))
        conf_wrong_rate = float(wrong_high_conf / high_conf_count)
        conf_wrong_share = float(wrong_high_conf / N)
    else:
        conf_wrong_rate = 0.0
        conf_wrong_share = 0.0
        
    return {
        "ece": float(ece),
        "brier": float(brier),
        "conf_wrong_rate": float(conf_wrong_rate),
        "conf_wrong_share": float(conf_wrong_share)
    }


def compute_visual_flip_rate(
    pred_orig: np.ndarray,
    pred_blank: np.ndarray
) -> Dict[str, float]:
    """Computes Visual Flip Rate between original image (P0) and blank canvas (P1)."""
    p0 = np.asarray(pred_orig)
    p1 = np.asarray(pred_blank)
    N = len(p0)
    if N == 0:
        return {"visual_flip_rate": 0.0, "count": 0}
    flips = (p0 != p1)
    rate = float(np.mean(flips)) * 100.0
    return {"visual_flip_rate": rate, "flip_count": int(np.sum(flips)), "total": N}


def compute_odds_ratio_dlang(
    d_lang: np.ndarray,
    error_id: np.ndarray,
    clusters: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes Odds Ratio of D_lang vs error_id:
    - Naive 2x2 contingency table & Fisher exact test.
    - Logistic Regression with Cluster-Robust Standard Errors.
    
    Note: On binary VQA, D_lang = 1 implies exactly one language failed.
    Thus, P(error_id=1 | D_lang=1) is mechanically bounded by the error distribution.
    """
    dl = np.asarray(d_lang, dtype=int)
    err = np.asarray(error_id, dtype=int)
    
    # 2x2 table: [[n_dl0_err0, n_dl0_err1], [n_dl1_err0, n_dl1_err1]]
    table = pd.crosstab(dl, err)
    odds_ratio_naive, p_fisher = stats.fisher_exact(table.values)
    
    res = {
        "odds_ratio": float(odds_ratio_naive),
        "odds_ratio_naive": float(odds_ratio_naive),
        "p_value": float(p_fisher),
        "p_fisher_exact": float(p_fisher),
        "table": table.to_dict()
    }
    
    # Cluster-robust logistic regression
    if clusters is not None:
        try:
            X = sm.add_constant(dl)
            model = Logit(err, X)
            # cluster-robust SE
            cluster_res = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": clusters})
            coef_dlang = cluster_res.params[1]
            se_dlang = cluster_res.bse[1]
            ci_low = cluster_res.conf_int()[1][0]
            ci_high = cluster_res.conf_int()[1][1]
            p_cluster = cluster_res.pvalues[1]
            
            res.update({
                "odds_ratio_cluster_robust": float(np.exp(coef_dlang)),
                "or_ci_95_lower": float(np.exp(ci_low)),
                "or_ci_95_upper": float(np.exp(ci_high)),
                "p_cluster_robust": float(p_cluster),
                "se_cluster_robust": float(se_dlang),
                "logit_coef": float(coef_dlang),
                "cluster_robust_se": float(se_dlang),
                "logit_pvalue": float(p_cluster)
            })
        except Exception as e:
            res["cluster_robust_error"] = str(e)
            
    return res


def run_oof_abstention_pipeline(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str] = ["score_m1_seq", "delta_p_dir_seq", "d_lang"],
    error_col: str = "error_id",
    cluster_col: str = "canonical_image_id",
    target_coverages: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90],
    n_splits: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Strict Out-of-Fold (OOF) Abstention Pipeline via GroupKFold (K=5).
    1. Fits scaler and logistic model on (K-1) folds, predicts OOF risk on holdout fold.
    2. Calibrates risk thresholds tau from OOF risk quantiles corresponding to target coverages.
    3. Fits final model on entire Dev set, applies tau to Test set.
    """
    df_dev = df_dev.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    groups_dev = df_dev[cluster_col].values
    y_dev = df_dev[error_col].values
    X_dev_raw = df_dev[feature_cols].values
    
    gkf = GroupKFold(n_splits=n_splits)
    oof_risks = np.zeros(len(df_dev), dtype=float)
    
    # Verify zero leakage across folds
    fold_splits = []
    for tr_idx, v_idx in gkf.split(X_dev_raw, y_dev, groups=groups_dev):
        train_clusters = set(groups_dev[tr_idx])
        val_clusters = set(groups_dev[v_idx])
        overlap = train_clusters.intersection(val_clusters)
        assert len(overlap) == 0, f"Critical leakage! Shared clusters in OOF fold: {overlap}"
        fold_splits.append((tr_idx, v_idx))
        
        sc_fold = StandardScaler()
        X_tr = sc_fold.fit_transform(X_dev_raw[tr_idx])
        X_v = sc_fold.transform(X_dev_raw[v_idx])
        clf_fold = LogisticRegression(C=1.0, random_state=seed)
        clf_fold.fit(X_tr, y_dev[tr_idx])
        oof_risks[v_idx] = clf_fold.predict_proba(X_v)[:, 1]
        
    # Calibrate thresholds from OOF quantiles
    calibrated_thresholds = {}
    for cov in target_coverages:
        # To retain top cov fraction of samples, risk must be <= tau
        tau = float(np.quantile(oof_risks, cov))
        calibrated_thresholds[cov] = tau
        
    # Fit final model on 100% dev
    sc_final = StandardScaler()
    X_dev_sc = sc_final.fit_transform(X_dev_raw)
    clf_final = LogisticRegression(C=1.0, random_state=seed)
    clf_final.fit(X_dev_sc, y_dev)
    
    # Predict on test
    X_test_raw = df_test[feature_cols].values
    X_test_sc = sc_final.transform(X_test_raw)
    test_risks = clf_final.predict_proba(X_test_sc)[:, 1]
    y_test_corr = 1 - df_test[error_col].values
    
    # Evaluate at calibrated thresholds
    test_results = []
    for cov, tau in calibrated_thresholds.items():
        retained_mask = (test_risks <= tau)
        actual_cov = float(np.mean(retained_mask)) * 100.0
        if np.sum(retained_mask) > 0:
            sel_acc = float(np.mean(y_test_corr[retained_mask])) * 100.0
            sel_risk = 100.0 - sel_acc
        else:
            sel_acc = 0.0
            sel_risk = 100.0
            
        test_results.append({
            "target_coverage": cov,
            "tau_oof": tau,
            "actual_coverage_test": actual_cov,
            "realized_coverage": actual_cov,
            "selective_acc_test": sel_acc,
            "selective_accuracy": sel_acc,
            "selective_risk_test": sel_risk,
            "selective_risk": sel_risk
        })
        
    return {
        "calibrated_thresholds": calibrated_thresholds,
        "oof_thresholds": calibrated_thresholds,
        "oof_risks": oof_risks,
        "test_risks": test_risks,
        "test_results": pd.DataFrame(test_results),
        "test_realized_coverages": {r["target_coverage"]: r["realized_coverage"] for r in test_results},
        "test_risks_dict": {r["target_coverage"]: r["selective_risk"] for r in test_results},
        "coefs": clf_final.coef_[0].tolist(),
        "intercept": float(clf_final.intercept_[0])
    }


def interpolate_risk_coverage(
    coverages: np.ndarray,
    risks: np.ndarray,
    target_coverages: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90]
) -> Dict[float, float]:
    """
    Interpolates selective risk at exact target coverages using linear interpolation.
    """
    interp_risks = {}
    for t_cov in target_coverages:
        # np.interp requires x to be increasing (coverages is increasing)
        r_val = float(np.interp(t_cov, coverages, risks))
        interp_risks[t_cov] = r_val
    return interp_risks
