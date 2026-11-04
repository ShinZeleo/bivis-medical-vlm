"""
Sequence Scoring and Likelihood Estimation for Autoregressive Multimodal Models.
Implements exact causal logit slicing, length normalization, and directional visual sensitivity.
"""

import math
from typing import Dict, List, Tuple, Optional, Any

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False

def compute_sequence_log_likelihood(
    logits: Any,
    target_ids: Any,
    prefix_length: int,
    alpha: float = 1.0
) -> float:
    """
    Computes length-normalized log-likelihood for candidate target tokens following a prompt prefix.
    Supports both PyTorch Tensors and NumPy arrays for CPU/lightweight testing.
    """
    if HAS_TORCH and isinstance(logits, torch.Tensor):
        seq_len = target_ids.size(0)
        if seq_len == 0:
            return 0.0
            
        start_idx = prefix_length - 1
        end_idx = start_idx + seq_len
        target_logits = logits[0, start_idx:end_idx, :]
        
        log_probs = F.log_softmax(target_logits, dim=-1)
        target_log_probs = log_probs[torch.arange(seq_len), target_ids]
        
        raw_log_likelihood = target_log_probs.sum().item()
        norm_factor = math.pow(seq_len, alpha)
        return raw_log_likelihood / norm_factor
    else:
        import numpy as np
        seq_len = len(target_ids)
        if seq_len == 0:
            return 0.0
        start_idx = prefix_length - 1
        end_idx = start_idx + seq_len
        target_logits = logits[0, start_idx:end_idx, :]
        # Softmax in numpy
        exp_logits = np.exp(target_logits - np.max(target_logits, axis=-1, keepdims=True))
        log_probs = np.log(exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))
        target_log_probs = [log_probs[i, target_ids[i]] for i in range(seq_len)]
        raw_log_likelihood = float(np.sum(target_log_probs))
        norm_factor = math.pow(seq_len, alpha)
        return raw_log_likelihood / norm_factor

def compute_directional_visual_sensitivity(
    p_orig: float,
    p_blank: float
) -> float:
    """
    Computes directional visual deletion sensitivity Delta P_dir.
    
    Delta P_dir = p_orig(y_hat) - p_blank(y_hat)
    Bounded in [-1.0, 1.0]. Positive values indicate visual reliance.
    Negative values indicate higher confidence on blind black control (severe visual hallucination).
    """
    return float(p_orig - p_blank)
