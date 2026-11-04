"""
Selective Prediction, Risk Estimation, and Area Under the Risk-Coverage Curve (AURC).
Formulates baseline confidence (M1), heuristic fusion (M_norm), and parametric risk modeling (LR).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def compute_discrete_aurc(
    risk_scores: np.ndarray,
    errors: np.ndarray
) -> float:
    """
    Computes exact discrete Riemann sum Area Under Risk-Coverage curve (AURC).
    
    AURC = (1 / N) * sum_{i=1}^N Risk(kappa_i)
    
    Args:
        risk_scores: Array of rejection/uncertainty scores (higher = more risky / higher error likelihood).
                     If confidence scores are provided (higher = more confident), invert them or negate them.
        errors: Binary 0/1 array where 1 indicates prediction error, 0 indicates correct.
        
    Returns:
        float: Empirical discrete AURC score in [0, 1].
    """
    n = len(risk_scores)
    if n == 0:
        return 0.0
        
    # Sort samples ascending by risk score (most confident first)
    order = np.argsort(risk_scores)
    sorted_errors = errors[order]
    
    # Cumulative error rate at each sample step
    cum_errors = np.cumsum(sorted_errors)
    sample_indices = np.arange(1, n + 1)
    empirical_risks = cum_errors / sample_indices
    
    return float(np.mean(empirical_risks))

def compute_risk_coverage_trajectory(
    risk_scores: np.ndarray,
    errors: np.ndarray,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates coverage and empirical risk points for plotting Risk-Coverage curves.
    
    Returns:
        Tuple of (coverages, risks) arrays.
    """
    n = len(risk_scores)
    order = np.argsort(risk_scores)
    sorted_errors = errors[order]
    
    coverages = np.linspace(1.0 / n, 1.0, num_points)
    risks = []
    
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        empirical_risk = np.mean(sorted_errors[:k])
        risks.append(empirical_risk)
        
    return coverages, np.array(risks)

def compute_heuristic_estimators(
    p_orig: float,
    delta_p_dir: float
) -> Dict[str, float]:
    """
    Computes selective rejection estimators:
    - M1: Raw confidence = p_orig
    - M2: Max-clamped visual sensitivity = max(0, delta_p_dir)
    - M_dir: Multiplicative = p_orig * max(0, delta_p_dir)
    - M_norm: Shifted normalized = p_orig * (1 + delta_p_dir) / 2
    """
    m1 = p_orig
    m2 = max(0.0, delta_p_dir)
    m_dir = p_orig * m2
    m_norm = p_orig * (1.0 + delta_p_dir) / 2.0
    
    return {
        'M1': m1,
        'M2': m2,
        'M_dir': m_dir,
        'M_norm': m_norm
    }
