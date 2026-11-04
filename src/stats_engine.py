"""
Statistical Evaluation Engine for Bilingual Medical VLM Selective Answering.
Authoritative implementation of discrete Riemann AURC, cluster-aware permutation tests,
cluster bootstrap confidence intervals, and Holm-Bonferroni corrections.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_discrete_aurc(
    correct_array: np.ndarray,
    score_array: np.ndarray,
    tie_mode: str = "stable",
    n_tie_reps: int = 100,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes Discrete Riemann Area Under the Risk-Coverage Curve (AURC).
    
    Parameters:
    -----------
    correct_array : np.ndarray
        Binary correctness array (1 for correct, 0 for incorrect).
    score_array : np.ndarray
        Confidence / selection scores (higher means more confident / lower risk).
    tie_mode : str
        Tie handling mode:
        - "stable": Standard mergesort (rev05 legacy).
        - "random": Expected AURC under random tie breaking (rev06 recommended default).
        - "worst": Pessimistic tie breaking (incorrect cases ranked first within tie).
        - "best": Optimistic tie breaking (correct cases ranked first within tie).
    n_tie_reps : int
        Number of random permutations when tie_mode == "random".
    rng : Optional[np.random.RandomState]
        Random generator for reproducibility.
        
    Returns:
    --------
    aurc : float
        Mean selective risk across all discrete coverages k/N.
    coverages : np.ndarray
        Array of coverages k/N from 1/N to 1.0.
    selective_risks : np.ndarray
        Array of selective risks at each coverage step.
    """
    correct_array = np.asarray(correct_array, dtype=float)
    score_array = np.asarray(score_array, dtype=float)
    N = len(correct_array)
    
    if N == 0:
        return 0.0, np.array([]), np.array([])
        
    k_vals = np.arange(1, N + 1, dtype=float)
    coverages = k_vals / float(N)
    
    if tie_mode == "stable":
        sort_idx = np.argsort(-score_array, kind="mergesort")
        sorted_corr = correct_array[sort_idx]
        selective_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals)
        aurc = float(np.mean(selective_risks))
        return aurc, coverages, selective_risks

    elif tie_mode == "worst":
        sort_idx = np.lexsort((correct_array, -score_array))
        sorted_corr = correct_array[sort_idx]
        selective_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals)
        aurc = float(np.mean(selective_risks))
        return aurc, coverages, selective_risks

    elif tie_mode == "best":
        sort_idx = np.lexsort((-correct_array, -score_array))
        sorted_corr = correct_array[sort_idx]
        selective_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals)
        aurc = float(np.mean(selective_risks))
        return aurc, coverages, selective_risks

    elif tie_mode == "random":
        if rng is None:
            rng = np.random.RandomState(42)
        if n_tie_reps == 1:
            jitter = rng.uniform(-1e-12, 1e-12, size=N)
            sort_idx = np.argsort(-(score_array + jitter), kind="mergesort")
            sorted_corr = correct_array[sort_idx]
            selective_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals)
            aurc = float(np.mean(selective_risks))
            return aurc, coverages, selective_risks
            
        aurc_list = []
        last_risks = None
        for _ in range(n_tie_reps):
            jitter = rng.uniform(-1e-12, 1e-12, size=N)
            sort_idx = np.argsort(-(score_array + jitter), kind="mergesort")
            sorted_corr = correct_array[sort_idx]
            s_risks = 1.0 - (np.cumsum(sorted_corr) / k_vals)
            aurc_list.append(float(np.mean(s_risks)))
            last_risks = s_risks
            
        mean_aurc = float(np.mean(aurc_list))
        return mean_aurc, coverages, last_risks
    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode}. Valid options: ['stable', 'random', 'worst', 'best']")


def get_risk_at_coverage(cov_arr: np.ndarray, risk_arr: np.ndarray, target_cov: float) -> float:
    """Finds nearest selective risk for a given target coverage."""
    if len(cov_arr) == 0 or len(risk_arr) == 0:
        return 0.0
    idx = int((np.abs(cov_arr - target_cov)).argmin())
    return float(risk_arr[idx])


def paired_cluster_permutation_test(
    df: pd.DataFrame,
    score_col_m: str,
    score_col_base: str = "score_m1_seq",
    corr_col: str = "correct_id_seq",
    cluster_col: str = "canonical_image_id",
    n_perm: int = 10000,
    seed: int = 42,
    tie_mode: str = "stable"
) -> float:
    """
    Vectorized cluster-level paired permutation test.
    Performs sign-flip across unique clusters (e.g., canonical_image_id).
    
    p = (extreme + 1) / (n_perm + 1)
    """
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    
    cluster_vals = df[cluster_col].to_numpy()
    clusters = np.unique(cluster_vals)
    n_clusters = len(clusters)
    cluster_to_pos = [np.where(cluster_vals == c)[0] for c in clusters]
    
    y = df[corr_col].to_numpy(dtype=float)
    s_m = df[score_col_m].to_numpy(dtype=float)
    s_base = df[score_col_base].to_numpy(dtype=float)
    
    obs_m, _, _ = calculate_discrete_aurc(y, s_m, tie_mode=tie_mode)
    obs_base, _, _ = calculate_discrete_aurc(y, s_base, tie_mode=tie_mode)
    obs_diff = abs(obs_m - obs_base)
    
    if obs_diff == 0.0:
        return 1.0
        
    extreme = 0
    for _ in range(n_perm):
        sm_p = s_m.copy()
        sb_p = s_base.copy()
        flips = rng.choice([-1, 1], size=n_clusters)
        swap_clusters = np.where(flips == -1)[0]
        if len(swap_clusters) > 0:
            swap_idx = np.concatenate([cluster_to_pos[c_i] for c_i in swap_clusters])
            sm_p[swap_idx], sb_p[swap_idx] = s_base[swap_idx], s_m[swap_idx]
            
        pm, _, _ = calculate_discrete_aurc(y, sm_p, tie_mode=tie_mode, n_tie_reps=1, rng=rng)
        pb, _, _ = calculate_discrete_aurc(y, sb_p, tie_mode=tie_mode, n_tie_reps=1, rng=rng)
        if abs(pm - pb) >= obs_diff - 1e-12:
            extreme += 1
            
    return float((extreme + 1) / (n_perm + 1))


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    score_cols: List[str],
    corr_col: str = "correct_id_seq",
    cluster_col: str = "canonical_image_id",
    n_boot: int = 1000,
    seed: int = 42,
    tie_mode: str = "stable"
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Cluster-level bootstrap for AURC estimates and paired differences relative to score_cols[0].
    """
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    
    cluster_vals = df[cluster_col].to_numpy()
    clusters = np.unique(cluster_vals)
    n_clusters = len(clusters)
    cluster_to_pos = {c: np.where(cluster_vals == c)[0] for c in clusters}
    
    y = df[corr_col].to_numpy(dtype=float)
    scores_dict = {col: df[col].to_numpy(dtype=float) for col in score_cols}
    base_col = score_cols[0]
    
    boot_aurcs = {col: [] for col in score_cols}
    boot_deltas = {col: [] for col in score_cols if col != base_col}
    
    for _ in range(n_boot):
        sampled_c = rng.choice(clusters, size=n_clusters, replace=True)
        samp_idx = np.concatenate([cluster_to_pos[c] for c in sampled_c])
        y_b = y[samp_idx]
        
        aurc_b = {}
        for col in score_cols:
            a, _, _ = calculate_discrete_aurc(y_b, scores_dict[col][samp_idx], tie_mode=tie_mode, n_tie_reps=1, rng=rng)
            boot_aurcs[col].append(a)
            aurc_b[col] = a
            
        for col in boot_deltas:
            boot_deltas[col].append(aurc_b[col] - aurc_b[base_col])
            
    summary = {}
    for col in score_cols:
        vals = np.array(boot_aurcs[col])
        summary[col] = {
            "mean": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5))
        }
        
    diff_summary = {}
    for col in boot_deltas:
        d_vals = np.array(boot_deltas[col])
        diff_summary[col] = {
            "delta_mean": float(np.mean(d_vals)),
            "delta_ci_lower": float(np.percentile(d_vals, 2.5)),
            "delta_ci_upper": float(np.percentile(d_vals, 97.5))
        }
        
    return summary, diff_summary


def apply_holm_bonferroni(p_vals_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Step-down Holm-Bonferroni correction with strict monotonicity preservation.
    """
    sorted_pairs = sorted(p_vals_dict.items(), key=lambda x: x[1])
    m = len(sorted_pairs)
    adj = {}
    running_max = 0.0
    for rank, (k, p) in enumerate(sorted_pairs, 1):
        multiplier = m - rank + 1
        adj_p = min(1.0, p * multiplier)
        running_max = max(running_max, adj_p)
        adj[k] = min(1.0, running_max)
    return adj


def wilson_ci(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    if k == 0:
        z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
        high = (z**2 / n) / (1.0 + z**2 / n)
        return 0.0, float(min(1.0, high))
    if k == n:
        z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
        low = 1.0 / (1.0 + z**2 / n)
        return float(max(0.0, low)), 1.0
        
    z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    p = k / float(n)
    denom = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p)) / n + (z**2) / (4.0 * (n**2)))) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def run_mcnemar(corr_a: np.ndarray, corr_b: np.ndarray) -> Dict[str, Union[int, float]]:
    """McNemar test with continuity correction and two-sided exact binomial p-value."""
    b = int(np.sum((corr_a == 1) & (corr_b == 0)))
    c = int(np.sum((corr_a == 0) & (corr_b == 1)))
    a = int(np.sum((corr_a == 1) & (corr_b == 1)))
    d = int(np.sum((corr_a == 0) & (corr_b == 0)))
    tot_discord = b + c
    if tot_discord == 0:
        chi2_c = 0.0
        p_exact = 1.0
    else:
        chi2_c = float((abs(b - c) - 1.0)**2 / float(tot_discord))
        p_exact = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), tot_discord, 0.5)))
    return {
        "b": b, "c": c, "a": a, "d": d,
        "chi2_continuity": chi2_c,
        "p_value_exact": p_exact
    }
