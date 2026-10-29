#!/usr/bin/env python3
"""
Resilient Medical VLM Decisive Validation Engine (rev05b / resilient)
======================================================================
Robust, timeout-resilient experiment engine for Medical Vision-Language Model
evaluation under Language Shift (IndoRad-VQA).

Key Enhancements for Timeout Resilience:
1. True Set-Based Resume Mechanism:
   - Identifies completed QA IDs via cryptographic set operations.
   - Filters out already-finished samples regardless of row ordering.
2. Independent Per-Model Execution & Checkpointing:
   - Fully decouples Model A (Qwen) and Model B (HuatuoGPT).
   - Dedicated output directories and atomic checkpoint files.
3. Micro-Batching & Atomic Flush:
   - Configurable batch size and checkpoint intervals (default: 5 samples).
   - Writes to `.tmp` and uses `os.replace` to prevent corrupted checkpoints.
4. Comprehensive Checkpoint Metadata Validation:
   - Strict rejection of cross-revision, wrong-model, corrupted, or non-finite checkpoints.
5. Zero-GPU Standalone Aggregator / Evaluator:
   - 100% CPU-only evaluation of checkpoints: Discrete Riemann AURC, cluster bootstrap,
     Monte Carlo cluster permutation test, and D_lang risk ablation.
"""

import os
import sys
import gc
import json
import hashlib
import io
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, List, Optional, Any, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image

# ==============================================================================
# 1. CONSTANTS, CONFIGURATION & FROZEN MANIFEST SIGNATURE
# ==============================================================================

REVISION_ID = "rev05"
REVISION_VARIANTS = {"rev05", "rev05b"}
SCORING_METHOD = "exact_sequence_candidate_normalized_v5"
AUTHORITATIVE_MANIFEST_SHA256 = "fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9"

RANDOM_SEED = 42

PROMPT_EN_TEMPLATE = "Answer only YES or NO.\nQuestion: {q}"
PROMPT_ID_TEMPLATE = "Jawab hanya dengan YA atau TIDAK.\nPertanyaan: {q}"

CANDIDATES_EN = ["YES<|im_end|>", "NO<|im_end|>"]
CANDIDATES_ID = ["YA<|im_end|>", "TIDAK<|im_end|>"]

LABEL_MAP = {
    "YES<|im_end|>": "YES",
    "NO<|im_end|>": "NO",
    "YA<|im_end|>": "YES",
    "TIDAK<|im_end|>": "NO"
}

DEFAULT_MODEL_1 = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_2 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

SCORE_COLUMNS_PRIMARY = [
    "score_m1_seq", "score_m2_seq", "score_m_dir_seq", "score_m_norm_seq", "score_m3_old_seq",
    "delta_p_dir_seq", "p_orig_ya_seq", "p_orig_tidak_seq", "p_blank_ya_seq", "p_blank_tidak_seq"
]

SCORE_COLUMNS_SENSITIVITY = [
    "score_m1_len", "score_m2_len", "score_m_dir_len", "score_m_norm_len", "score_m3_old_len",
    "delta_p_dir_len", "p_orig_ya_len", "p_orig_tidak_len", "p_blank_ya_len", "p_blank_tidak_len"
]

ALL_REQUIRED_SCORE_COLS = SCORE_COLUMNS_PRIMARY + SCORE_COLUMNS_SENSITIVITY

# ==============================================================================
# 2. IMAGE DISCOVERY, RESOLUTION & PIXEL CRYPTOGRAPHIC VERIFICATION
# ==============================================================================

def compute_rgb_pixel_sha256(pil_img: Image.Image) -> str:
    """Computes SHA256 hash of RGB raw bytes for strict pixel identity matching."""
    rgb = pil_img.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def resolve_and_verify_images(df_manifest: pd.DataFrame, 
                              local_image_dirs: list = None,
                              cache_dir: Path = Path("main/data/images")) -> dict:
    """
    Resolves every manifest row to its exact physical image on disk.
    Verifies image integrity using:
      1. Cryptographic pixel SHA256 (matching manifest['pixel_sha256'])
      2. Image dimensions (width, height)
    Returns: mapping {qa_id: Path_to_verified_image}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = [cache_dir]
    if local_image_dirs:
        search_dirs.extend([Path(d) for d in local_image_dirs if Path(d).exists()])
        
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for zf in kaggle_input.rglob("*.zip"):
            try:
                print(f"📦 Auto-extracting zip archive: {zf.name} to {cache_dir}...")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(cache_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not extract {zf.name}: {e}")
                
        for p in kaggle_input.rglob("*.png"):
            if p.parent not in search_dirs:
                search_dirs.append(p.parent)

    resolved_map = {}
    unique_hashes = df_manifest["pixel_sha256"].unique()
    print(f"🔍 Cryptographically verifying pixel SHA-256 for all {len(unique_hashes)} unique physical images...")
    
    disk_cache = {}
    for sdir in search_dirs:
        for p in sdir.glob("*.png"):
            fname = p.stem.lower()
            if len(fname) == 16:
                disk_cache[fname] = p
                
    for _, row in df_manifest.iterrows():
        qa_id = row["qa_id"]
        cid = str(row["canonical_image_id"]).strip().lower()
        expected_hash = str(row["pixel_sha256"]).strip().lower()
        expected_w = int(row["image_width"])
        expected_h = int(row["image_height"])
        
        found_path = None
        if cid in disk_cache:
            p = disk_cache[cid]
            try:
                with Image.open(p) as img:
                    if img.size == (expected_w, expected_h):
                        if compute_rgb_pixel_sha256(img) == expected_hash:
                            found_path = p
            except Exception:
                pass
                
        if found_path is None:
            for sdir in search_dirs:
                for cand_name in [f"{cid}.png", f"img_{int(qa_id):05d}.png", f"{qa_id}.png", f"{row.get('image_name', '')}"]:
                    if not cand_name:
                        continue
                    p = sdir / cand_name
                    if p.exists():
                        try:
                            with Image.open(p) as img:
                                if img.size == (expected_w, expected_h):
                                    if compute_rgb_pixel_sha256(img) == expected_hash:
                                        found_path = p
                                        break
                        except Exception:
                            pass
                if found_path is not None:
                    break
                    
        assert found_path is not None, f"❌ Failed to resolve verified image for QA ID {qa_id} (hash {expected_hash})"
        resolved_map[qa_id] = found_path
        
    print(f"✅ All {len(resolved_map)} images cryptographically verified with 0% tampering.")
    return resolved_map

# ==============================================================================
# 3. MANIFEST & COHORT SELECTION
# ==============================================================================

def get_active_evaluation_cohort(df_manifest: pd.DataFrame, 
                                 cohort_mode: str = "active_union") -> pd.DataFrame:
    """
    Selects the exact evaluation cohort based on research protocol.
    Modes:
      - 'active_union': Union of Primary Cohort (1122) and Historical Benchmark (150) -> 1,133 samples
      - 'primary_only': Primary Cohort only -> 1,122 samples (799 test / 323 dev)
      - 'benchmark_only': Historical Benchmark only -> 150 samples (92 test / 58 dev)
    """
    if cohort_mode == "active_union":
        mask = (df_manifest["is_eligible_primary"] == True) | (df_manifest["in_rev03_benchmark"] == True)
    elif cohort_mode == "primary_only":
        mask = df_manifest["is_eligible_primary"] == True
    elif cohort_mode == "benchmark_only":
        mask = df_manifest["in_rev03_benchmark"] == True
    else:
        raise ValueError(f"Unknown cohort mode: {cohort_mode}")
        
    df_cohort = df_manifest[mask].copy().sort_values("qa_id").reset_index(drop=True)
    return df_cohort

# ==============================================================================
# 4. STRICT CHECKPOINT METADATA VALIDATION & RESUME LOGIC
# ==============================================================================

def validate_checkpoint_compatibility(
    df_ckpt: pd.DataFrame,
    target_model_name: str,
    expected_manifest_sha256: str,
    valid_manifest_qa_ids: Set[int]
) -> Tuple[bool, str]:
    """
    Strictly validates that a candidate checkpoint file is 100% compatible with
    the current evaluation specification. Rejects corrupted or cross-revision files.
    """
    if df_ckpt is None or len(df_ckpt) == 0:
        return False, "Checkpoint is empty"
        
    required_cols = [
        "qa_id", "model_id", "revision_id", "manifest_sha256", "scoring_method",
        "pred_en_seq", "conf_en_seq", "pred_id_seq", "conf_id_seq",
        "p_blank_ya_seq", "p_blank_tidak_seq", "delta_p_dir_seq",
        "score_m1_seq", "score_m_norm_seq"
    ]
    missing_cols = [c for c in required_cols if c not in df_ckpt.columns]
    if missing_cols:
        return False, f"Missing required columns in checkpoint: {missing_cols}"
        
    # Check revision ID
    revs = set(df_ckpt["revision_id"].dropna().unique())
    if not revs.issubset(REVISION_VARIANTS):
        return False, f"Incompatible revision ID in checkpoint: {revs} (expected {REVISION_VARIANTS})"
        
    # Check manifest SHA256
    m_hashes = set(df_ckpt["manifest_sha256"].dropna().unique())
    if m_hashes != {expected_manifest_sha256}:
        return False, f"Manifest hash mismatch: {m_hashes} vs expected {expected_manifest_sha256}"
        
    # Check model ID
    m_ids = set(df_ckpt["model_id"].dropna().unique())
    if m_ids != {target_model_name}:
        return False, f"Model ID mismatch: {m_ids} vs target {target_model_name}"
        
    # Check scoring method
    s_methods = set(df_ckpt["scoring_method"].dropna().unique())
    if s_methods != {SCORING_METHOD}:
        return False, f"Scoring method mismatch: {s_methods} vs expected {SCORING_METHOD}"
        
    # Check for NaN / Inf in score columns
    for col in ALL_REQUIRED_SCORE_COLS:
        if col in df_ckpt.columns:
            vals = df_ckpt[col].values
            if not np.isfinite(vals).all():
                nan_count = int(np.isnan(vals).sum())
                inf_count = int(np.isinf(vals).sum())
                return False, f"Non-finite values in {col}: {nan_count} NaNs, {inf_count} Infs"
                
    # Check duplicates
    dup_count = df_ckpt["qa_id"].duplicated().sum()
    if dup_count > 0:
        return False, f"Checkpoint contains {dup_count} duplicate QA IDs"
        
    # Check extraneous QA IDs
    ckpt_qids = set(df_ckpt["qa_id"])
    extraneous = ckpt_qids - valid_manifest_qa_ids
    if extraneous:
        return False, f"Checkpoint contains {len(extraneous)} extraneous QA IDs not in manifest"
        
    return True, f"Valid checkpoint containing {len(df_ckpt)} completed samples"


def filter_remaining_samples(
    df_cohort: pd.DataFrame,
    completed_qids: Set[int],
    start_index: int = 0,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    True set-based resume mechanism.
    Filters out all QA IDs that have already been evaluated, and applies optional bounded windowing.
    """
    df_remaining = df_cohort[~df_cohort["qa_id"].isin(completed_qids)].copy().reset_index(drop=True)
    
    if start_index > 0:
        df_remaining = df_remaining.iloc[start_index:].reset_index(drop=True)
    if max_samples is not None and max_samples > 0:
        df_remaining = df_remaining.iloc[:max_samples].reset_index(drop=True)
        
    return df_remaining


def atomic_save_checkpoint(
    df_new_records: pd.DataFrame,
    checkpoint_path: Path,
    df_existing: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Atomically saves merged checkpoint via temporary file replacement.
    Guarantees zero file corruption on timeout or preemption.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    if df_existing is not None and len(df_existing) > 0:
        df_merged = pd.concat([df_existing, df_new_records], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=["qa_id"], keep="last").reset_index(drop=True)
    else:
        df_merged = df_new_records.drop_duplicates(subset=["qa_id"], keep="last").reset_index(drop=True)
        
    tmp_path = checkpoint_path.parent / f"{checkpoint_path.stem}.tmp_{os.getpid()}.csv"
    df_merged.to_csv(tmp_path, index=False)
    shutil.move(tmp_path, checkpoint_path)
    return df_merged

# ==============================================================================
# 5. CANDIDATE SCORING ENGINE (EXACT SEQUENCE & LENGTH-NORMALIZED SURPRISAL)
# ==============================================================================

def extract_target_loglikelihood_v5(
    model, processor, image_path: Path, prompt_text: str, 
    candidate_strings: List[str], target_device: Any
) -> Tuple[dict, Tuple[float, float], Tuple[float, float]]:
    """
    Evaluates conditional log-likelihood of exact candidate target strings
    given multimodal context (Image + Prompt).
    Slices causal logits [L_prompt - 1 : L_prompt - 1 + L_cand].
    """
    import torch
    import torch.nn.functional as F
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path), "min_pixels": 256 * 28 * 28, "max_pixels": 512 * 512},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    prompt_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    prompt_inputs = processor(
        text=[prompt_str],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(target_device)
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    cand_stats = {}
    for cand in candidate_strings:
        full_text = prompt_str + cand
        inputs = processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(target_device)
        
        # Verify prefix invariance to ensure zero boundary slippage
        assert torch.equal(inputs.input_ids[0, :prompt_len], prompt_inputs.input_ids[0]), \
            f"Tokenizer boundary prefix mismatch! prompt_len={prompt_len}, inputs_len={inputs.input_ids.shape[1]}"
            
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            
        target_ids = inputs.input_ids[0, prompt_len:].tolist()
        if len(target_ids) == 0:
            cand_stats[cand] = {"raw_sum": -999.0, "mean_lp": -999.0, "target_ids": [], "token_lps": []}
            del inputs, outputs
            continue
            
        target_logits = logits[prompt_len - 1 : prompt_len - 1 + len(target_ids)]
        log_probs = F.log_softmax(target_logits, dim=-1)
        token_lps = [log_probs[i, target_ids[i]].item() for i in range(len(target_ids))]
        raw_sum = float(sum(token_lps))
        mean_lp = float(raw_sum / len(target_ids))
        
        cand_stats[cand] = {
            "raw_sum": raw_sum,
            "mean_lp": mean_lp,
            "target_ids": target_ids,
            "token_lps": token_lps
        }
        del inputs, outputs
        
    del prompt_inputs
    
    c1, c2 = candidate_strings[0], candidate_strings[1]
    
    # Primary rev05: Candidate-normalized exact sequence score via logsumexp
    s1_seq = cand_stats[c1]["raw_sum"]
    s2_seq = cand_stats[c2]["raw_sum"]
    max_seq = max(s1_seq, s2_seq)
    exp_seq1 = np.exp(s1_seq - max_seq)
    exp_seq2 = np.exp(s2_seq - max_seq)
    p1_seq = float(exp_seq1 / (exp_seq1 + exp_seq2))
    p2_seq = float(1.0 - p1_seq)
    
    # Sensitivity rev05: Length-normalized relative score via logsumexp
    s1_norm = cand_stats[c1]["mean_lp"]
    s2_norm = cand_stats[c2]["mean_lp"]
    max_norm = max(s1_norm, s2_norm)
    exp_norm1 = np.exp(s1_norm - max_norm)
    exp_norm2 = np.exp(s2_norm - max_norm)
    p1_norm = float(exp_norm1 / (exp_norm1 + exp_norm2))
    p2_norm = float(1.0 - p1_norm)
    
    return cand_stats, (p1_seq, p2_seq), (p1_norm, p2_norm)


def score_binary_candidates_v5(
    model, processor, image_path: Path, prompt_text: str, 
    candidate_strings: List[str], target_device: Any
) -> Tuple[dict, dict, dict]:
    """
    Scores binary candidate strings with explicit canonical label mapping.
    """
    c_aff, c_neg = candidate_strings[0], candidate_strings[1]
    assert c_aff in LABEL_MAP and c_neg in LABEL_MAP, f"Unmapped candidate string in: {candidate_strings}"
    label_aff = LABEL_MAP[c_aff]
    label_neg = LABEL_MAP[c_neg]
    
    cand_stats, (p_aff_seq, p_neg_seq), (p_aff_norm, p_neg_norm) = extract_target_loglikelihood_v5(
        model, processor, image_path, prompt_text, candidate_strings, target_device
    )
    
    pred_seq = label_aff if p_aff_seq >= 0.5 else label_neg
    conf_seq = p_aff_seq if p_aff_seq >= 0.5 else p_neg_seq
    
    pred_norm = label_aff if p_aff_norm >= 0.5 else label_neg
    conf_norm = p_aff_norm if p_aff_norm >= 0.5 else p_neg_norm
    
    seq_dict = {"pred": pred_seq, "conf": conf_seq, "p_aff": p_aff_seq, "p_neg": p_neg_seq}
    norm_dict = {"pred": pred_norm, "conf": conf_norm, "p_aff": p_aff_norm, "p_neg": p_neg_norm}
    
    return seq_dict, norm_dict, cand_stats

# ==============================================================================
# 6. DISCRETE RIEMANN AURC & STATISTICAL TESTING ENGINE (ZERO-GPU)
# ==============================================================================

def calculate_discrete_aurc(correct_array: np.ndarray, score_array: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Exact Discrete Riemann AURC (Geifman & El-Yaniv, NeurIPS 2017).
    Sorts descending by reliability score S.
    """
    N = len(correct_array)
    if N == 0:
        return 0.0, np.array([]), np.array([])
        
    sort_idx = np.argsort(-score_array, kind="mergesort")
    sorted_corr = correct_array[sort_idx]
    
    cum_correct = np.cumsum(sorted_corr)
    k_vals = np.arange(1, N + 1)
    
    coverages = k_vals / float(N)
    selective_risks = 1.0 - (cum_correct / k_vals.astype(float))
    
    discrete_aurc = float(np.mean(selective_risks))
    return discrete_aurc, coverages, selective_risks


def compute_paired_cluster_bootstrap(
    df_eval: pd.DataFrame, 
    score_col_m: str, 
    score_col_baseline: str = "score_m1_seq",
    correct_col: str = "correct_id_seq",
    cluster_col: str = "canonical_image_id",
    n_bootstraps: int = 1000, 
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes Paired Image-Cluster Bootstrap (B=1000) for AURC and Delta AURC.
    Resamples unique physical image clusters with replacement.
    Guarantees 100% positional index alignment regardless of df_eval.index.
    """
    df_eval = df_eval.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    cluster_vals = df_eval[cluster_col].to_numpy()
    clusters = np.unique(cluster_vals)
    n_clusters = len(clusters)
    
    cluster_to_indices = {cid: np.where(cluster_vals == cid)[0] for cid in clusters}
    
    y_corr = df_eval[correct_col].to_numpy(dtype=float)
    s_m = df_eval[score_col_m].to_numpy(dtype=float)
    s_base = df_eval[score_col_baseline].to_numpy(dtype=float)
    
    obs_aurc_m, _, _ = calculate_discrete_aurc(y_corr, s_m)
    obs_aurc_base, _, _ = calculate_discrete_aurc(y_corr, s_base)
    obs_delta = obs_aurc_m - obs_aurc_base
    
    boot_aurc_m = []
    boot_aurc_base = []
    boot_deltas = []
    
    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        
        y_b = y_corr[sampled_indices]
        sm_b = s_m[sampled_indices]
        sbase_b = s_base[sampled_indices]
        
        a_m, _, _ = calculate_discrete_aurc(y_b, sm_b)
        a_base, _, _ = calculate_discrete_aurc(y_b, sbase_b)
        
        boot_aurc_m.append(a_m)
        boot_aurc_base.append(a_base)
        boot_deltas.append(a_m - a_base)
        
    boot_deltas = np.array(boot_deltas)
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    
    return {
        "observed_aurc_m": obs_aurc_m,
        "observed_aurc_base": obs_aurc_base,
        "observed_delta_aurc": obs_delta,
        "ci_95": [ci_lower, ci_upper],
        "bootstrap_mean_delta": float(np.mean(boot_deltas)),
        "bootstrap_std_delta": float(np.std(boot_deltas))
    }


def compute_paired_cluster_permutation_test(
    df_eval: pd.DataFrame, 
    score_col_m: str, 
    score_col_baseline: str = "score_m1_seq",
    correct_col: str = "correct_id_seq",
    cluster_col: str = "canonical_image_id",
    n_permutations: int = 1000, 
    seed: int = 42
) -> float:
    """
    Exact Paired Image-Cluster Permutation Test under H0: E[Delta AURC] = 0.
    Applies Rademacher sign-flips eps_c in {-1, +1} at the cluster level.
    Guarantees 100% positional index alignment regardless of df_eval.index.
    """
    df_eval = df_eval.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    cluster_vals = df_eval[cluster_col].to_numpy()
    clusters = np.unique(cluster_vals)
    cluster_to_indices = {cid: np.where(cluster_vals == cid)[0] for cid in clusters}
    
    y_corr = df_eval[correct_col].to_numpy(dtype=float)
    s_m = df_eval[score_col_m].to_numpy(dtype=float)
    s_base = df_eval[score_col_baseline].to_numpy(dtype=float)
    
    obs_aurc_m, _, _ = calculate_discrete_aurc(y_corr, s_m)
    obs_aurc_base, _, _ = calculate_discrete_aurc(y_corr, s_base)
    obs_diff = abs(obs_aurc_m - obs_aurc_base)
    
    if obs_diff == 0.0:
        return 1.0
        
    count_extreme = 0
    for _ in range(n_permutations):
        s_m_perm = s_m.copy()
        s_base_perm = s_base.copy()
        
        flips = rng.choice([-1, 1], size=len(clusters))
        for cid, flip in zip(clusters, flips):
            if flip == -1:
                idx = cluster_to_indices[cid]
                s_m_perm[idx], s_base_perm[idx] = s_base[idx], s_m[idx]
                
        p_m, _, _ = calculate_discrete_aurc(y_corr, s_m_perm)
        p_base, _, _ = calculate_discrete_aurc(y_corr, s_base_perm)
        perm_diff = abs(p_m - p_base)
        
        if perm_diff >= obs_diff:
            count_extreme += 1
            
    p_value = float((count_extreme + 1) / (n_permutations + 1))
    return p_value


def apply_holm_bonferroni_correction(raw_p_values: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER <= 0.05).
    """
    sorted_comparisons = sorted(raw_p_values.items(), key=lambda x: x[1])
    m = len(sorted_comparisons)
    adjusted_results = {}
    
    running_max_p = 0.0
    for rank, (name, p_val) in enumerate(sorted_comparisons):
        multiplier = m - rank
        adj_p = min(1.0, p_val * multiplier)
        running_max_p = max(running_max_p, adj_p)
        adj_p = running_max_p
        
        adjusted_results[name] = {
            "raw_p": float(p_val),
            "adjusted_p": float(adj_p),
            "rank": rank + 1,
            "significant_at_05": bool(adj_p < 0.05)
        }
    return adjusted_results

# ==============================================================================
# 7. RISK ABLATION ENGINE (D_LANG & LOGISTIC REGRESSION ON DEV SET)
# ==============================================================================

def train_and_evaluate_risk_ablation(
    df_dev: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Trains learned risk models strictly on Dev Set (N=323, 88 clusters)
    using GroupKFold (K=5) grouped by canonical_image_id.
    Evaluates cleanly on Test Set (N=799, 204 clusters) with 0% data leakage.
    
    Models:
      - LR-conf: Features [p_orig_ID]
      - LR-visual: Features [p_orig_ID, delta_p_dir_ID]
      - LR-full: Features [p_orig_ID, delta_p_dir_ID, d_lang]
      
    Score Orientation: S_LR = 1.0 - P(error_ID == 1)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for df in [df_dev, df_test]:
        if "d_lang" not in df.columns:
            df["d_lang"] = (df["pred_en_seq"] != df["pred_id_seq"]).astype(int)
        if "error_id" not in df.columns:
            df["error_id"] = (1 - df["correct_id_seq"]).astype(int)
            
    # Feature sets
    feature_sets = {
        "LR_conf": ["score_m1_seq"],
        "LR_visual": ["score_m1_seq", "delta_p_dir_seq"],
        "LR_full": ["score_m1_seq", "delta_p_dir_seq", "d_lang"]
    }
    
    results = {}
    fitted_models = {}
    
    gkf = GroupKFold(n_splits=5)
    groups_dev = df_dev["canonical_image_id"].values
    y_dev = df_dev["error_id"].values
    
    print(f"📊 Training risk ablation models for {model_name.split('/')[-1]} on Dev Set (N={len(df_dev)})...")
    
    for mname, feats in feature_sets.items():
        X_dev_raw = df_dev[feats].values
        X_test_raw = df_test[feats].values
        
        # Cross-validation on Dev Set
        oof_risk_preds = np.zeros(len(df_dev))
        for train_idx, val_idx in gkf.split(X_dev_raw, y_dev, groups=groups_dev):
            scaler_cv = StandardScaler()
            X_tr = scaler_cv.fit_transform(X_dev_raw[train_idx])
            X_val = scaler_cv.transform(X_dev_raw[val_idx])
            
            clf = LogisticRegression(penalty="l2", C=1.0, random_state=RANDOM_SEED)
            clf.fit(X_tr, y_dev[train_idx])
            oof_risk_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
            
        dev_scores = 1.0 - oof_risk_preds
        dev_aurc, _, _ = calculate_discrete_aurc(df_dev["correct_id_seq"].values, dev_scores)
        
        # Train on 100% of Dev Set
        final_scaler = StandardScaler()
        X_dev_scaled = final_scaler.fit_transform(X_dev_raw)
        X_test_scaled = final_scaler.transform(X_test_raw)
        
        final_clf = LogisticRegression(penalty="l2", C=1.0, random_state=RANDOM_SEED)
        final_clf.fit(X_dev_scaled, y_dev)
        
        test_risk_preds = final_clf.predict_proba(X_test_scaled)[:, 1]
        test_scores = 1.0 - test_risk_preds
        test_aurc, _, _ = calculate_discrete_aurc(df_test["correct_id_seq"].values, test_scores)
        
        results[mname] = {
            "dev_aurc": dev_aurc,
            "test_aurc": test_aurc,
            "features": feats,
            "coefficients": {f: float(c) for f, c in zip(feats, final_clf.coef_[0])},
            "intercept": float(final_clf.intercept_[0]),
            "test_scores": test_scores
        }
        fitted_models[mname] = {"clf": final_clf, "scaler": final_scaler}
        
    return results

# ==============================================================================
# 8. SELF-HEALING EMBEDDED CHECKPOINT DISCOVERY
# ==============================================================================

def find_existing_checkpoint(model_name: str, candidate_dirs: List[Path] = None) -> Optional[Path]:
    """
    Discovers existing checkpoint file across working and input directories.
    """
    model_slug = model_name.replace("/", "_").replace("-", "_")
    target_filename = f"live_validation_checkpoint_{model_slug}.csv"
    
    search_paths = [
        Path.cwd() / "main" / "results" / "decisive_validation" / "rev05" / model_slug / target_filename,
        Path.cwd() / "main" / "results" / "results 08_decisive_validation_rev05" / "decisive_validation" / "rev05" / model_slug / target_filename,
        Path("/kaggle/working/decisive_validation_rev05") / model_slug / target_filename,
        Path("/kaggle/working") / target_filename
    ]
    if candidate_dirs:
        for cd in candidate_dirs:
            search_paths.append(cd / model_slug / target_filename)
            search_paths.append(cd / target_filename)
            
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob(f"*{model_slug}*.csv"):
            if "checkpoint" in p.name.lower():
                search_paths.insert(0, p)
                
    for sp in search_paths:
        if sp.exists() and sp.is_file() and sp.stat().st_size > 1000:
            return sp
            
    return None
