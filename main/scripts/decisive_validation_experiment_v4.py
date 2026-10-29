#!/usr/bin/env python3
"""
Medical VLM Decisive Validation Experiment Engine (rev04)
=========================================================
Authoritative, statistically sound, and fully reproducible experiment engine
for Medical Vision-Language Model evaluation under Language Shift (IndoRad-VQA).

Key Methodological Standards in rev04:
1. Dual Evaluation Tracks:
   - Track 1 (Primary Test Cohort): N=799 QA pairs across 204 unique physical image clusters.
   - Track 2 (Historical Benchmark): N=92 QA pairs across 35 unique physical image clusters (rev03 cohort).
   - Calibration Set: N=323 Dev QA pairs across 88 clusters (used for protocol decisions, test set kept untouched).
2. Symmetrical Closed Candidate Scoring (<|im_end|>) with Causal Logit Verification:
   - Slices causal logits [L_prompt - 1 : L_prompt - 1 + L_target].
   - Extracts both Raw Sequence Log-Likelihood AND Length-Normalized Surprisal.
   - Normalizes via stable logsumexp into Relative Candidate Scores.
3. Visual Intervention (P0 Original vs P1 Black Blank):
   - Computes Directional Visual Reliance: Delta P_dir = P(y_hat | P0) - P(y_hat | P1) on identical predicted label.
4. Selective Prediction & Discrete AURC (Geifman & El-Yaniv, NeurIPS 2017):
   - Evaluates 5 estimators: M1 (confidence), M2 (visual-only), M_dir (multiplicative), M_norm (linear), M3_old (absolute).
   - Realized risk reported at 100%, 90%, 80%, 70%, 50% coverage.
5. Exact Paired Cluster Bootstrap & Monte Carlo Cluster Permutation Test:
   - Rademacher cluster-level sign flips under H0: E[Delta AURC] = 0 with Holm-Bonferroni correction.
6. Strict Checkpoint Governance & Zero Data Leakage.
"""

import os
import sys
import gc
import json
import hashlib
import io
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# ==============================================================================
# 1. CONSTANTS, CONFIGURATION & FROZEN MANIFEST SIGNATURE
# ==============================================================================

REVISION_ID = "rev04"
SCORING_METHOD = "length_normalized_closed_candidate_v4"
AUTHORITATIVE_MANIFEST_SHA256 = "bcb03bfafa0c329b73ab3b5ec2bb6f471daa1df9218582efa232e9dac74d3484"

RANDOM_SEED = 42

PROMPT_EN_TEMPLATE = "Answer only YES or NO.\nQuestion: {q}"
PROMPT_ID_TEMPLATE = "Jawab hanya dengan YA atau TIDAK.\nPertanyaan: {q}"

CANDIDATES_EN = ["YES<|im_end|>", "NO<|im_end|>"]
CANDIDATES_ID = ["YA<|im_end|>", "TIDAK<|im_end|>"]

DEFAULT_MODEL_1 = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_2 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

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
    Falls back to self-healing Parquet retrieval from Hugging Face if missing.
    Returns: mapping {qa_id: Path_to_verified_image}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = [cache_dir]
    if local_image_dirs:
        search_dirs.extend([Path(d) for d in local_image_dirs if Path(d).exists()])
        
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # 1. Auto-extract any zip archives in /kaggle/input into cache_dir
        for zf in kaggle_input.rglob("*.zip"):
            try:
                print(f"📦 Auto-extracting zip archive: {zf.name} to {cache_dir}...")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(cache_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not extract {zf.name}: {e}")
                
        # 2. Discover all directories containing PNG images
        for p in kaggle_input.rglob("*.png"):
            if p.parent not in search_dirs:
                search_dirs.append(p.parent)

    resolved_map = {}
    missing_rows = []
    
    unique_hashes = df_manifest["pixel_sha256"].unique()
    print(f"🔍 Cryptographically verifying pixel SHA-256 for all {len(unique_hashes)} unique physical images...")
    
    # Pre-index search dirs by canonical image ID
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
        # Fast path via pre-indexed canonical ID
        if cid in disk_cache:
            p = disk_cache[cid]
            try:
                with Image.open(p) as img:
                    if img.size == (expected_w, expected_h):
                        if compute_rgb_pixel_sha256(img) == expected_hash:
                            found_path = p
            except Exception:
                pass
                
        # Search path through directory candidates
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
                    
        if found_path is not None:
            resolved_map[qa_id] = found_path
        else:
            missing_rows.append(row)
            
    # Self-healing fallback via Hugging Face Parquet
    if missing_rows:
        print(f"⚠️ {len(missing_rows)} samples missing locally. Initiating self-healing download via Hugging Face Parquet...")
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
            
            # Map split names
            splits_needed = set()
            for r in missing_rows:
                s = str(r["split"]).lower()
                splits_needed.add("test" if "test" in s else "train")
                
            for sname in splits_needed:
                fname = f"data/{sname}-00000-of-00001.parquet"
                print(f"   Downloading Parquet split: {fname}...")
                local_pq = hf_hub_download(repo_id="Lab-IS/IndoRad-VQA", filename=fname, repo_type="dataset")
                table = pq.read_table(local_pq)
                df_pq = table.to_pandas()
                
                for r in missing_rows:
                    qid = r["qa_id"]
                    cid = str(r["canonical_image_id"]).strip().lower()
                    exp_h = str(r["pixel_sha256"]).strip().lower()
                    match = df_pq[df_pq["qid"] == qid]
                    if len(match) == 0:
                        continue
                    raw_bytes = match.iloc[0]["image"].get("bytes", None)
                    if raw_bytes:
                        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                            if compute_rgb_pixel_sha256(pil_img) == exp_h:
                                sp = cache_dir / f"{cid}.png"
                                pil_img.save(sp)
                                resolved_map[qid] = sp
        except Exception as e:
            print(f"❌ Parquet retrieval failed: {e}")
            
    unresolved = [r["qa_id"] for _, r in df_manifest.iterrows() if r["qa_id"] not in resolved_map]
    if unresolved:
        raise RuntimeError(f"Cryptographic verification failed: {len(unresolved)} images missing or tampered! Unresolved QIDs: {unresolved[:10]}")
        
    print(f"✅ Cryptographic Image Verification: 100% PASS ({len(resolved_map)}/{len(df_manifest)} images verified).")
    return resolved_map

# ==============================================================================
# 3. SELECTIVE PREDICTION & DISCRETE AURC MATHEMATICAL FORMULATION
# ==============================================================================

def compute_discrete_risk_coverage(scores: np.ndarray, correctness: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Sample-level discrete selective prediction evaluation (Geifman & El-Yaniv, NeurIPS 2017).
    Ranks samples by reliability score descending: S_(1) >= S_(2) >= ... >= S_(N).
    Deterministic tie handling via mergesort.
    Evaluates empirical selective risk r_k = 1 - (1/k) sum_{j=1}^k Correct_(j) for k=1..N.
    Returns:
      cov: discrete coverage grid [1/N, 2/N, ..., 1.0]
      risk: empirical selective risk array at each discrete coverage
      aurc_mean: exact mean empirical risk across all N evaluation points
    """
    scores = np.asarray(scores, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    n = len(scores)
    assert n > 0, "Empty cohort cannot compute risk-coverage!"
    
    # Stable sort descending
    order = np.argsort(-scores, kind="mergesort")
    sorted_corr = correctness[order]
    
    k = np.arange(1, n + 1)
    cov = k / n
    acc = np.cumsum(sorted_corr) / k
    risk = 1.0 - acc
    aurc_mean = float(np.mean(risk))
    return cov, risk, aurc_mean


def get_risk_at_realized_coverage(cov_arr: np.ndarray, risk_arr: np.ndarray, target_cov: float) -> Tuple[float, float, int]:
    """Finds selective risk, realized coverage, and sample count at closest discrete coverage level."""
    idx = int((np.abs(cov_arr - target_cov)).argmin())
    return float(risk_arr[idx]), float(cov_arr[idx]), int(idx + 1)


def paired_cluster_permutation_test(df: pd.DataFrame, score_col_a: str, score_col_b: str,
                                    corr_col: str = "correct_id", cluster_col: str = "canonical_image_id",
                                    n_perm: int = 1000, seed: int = 42) -> Tuple[float, float, float, float]:
    """
    Monte Carlo Paired Image-Cluster Permutation Test under H0: E[Delta AURC] = 0.
    For each permutation replicate, draws independent Rademacher sign flips eps_c in {-1, +1}
    for each unique physical image cluster. Swaps estimator scores for all QA pairs in cluster c if eps_c == -1.
    Computes exact two-sided finite-sample p-value:
      p = (1 + sum(I(|T_perm| >= |T_obs|))) / (n_perm + 1)
    """
    rng = np.random.default_rng(seed)
    corr = df[corr_col].to_numpy(dtype=float)
    scores_a = df[score_col_a].to_numpy(dtype=float)
    scores_b = df[score_col_b].to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()
    
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_indices = [np.where(clusters == c)[0] for c in unique_clusters]
    
    _, _, aurc_a = compute_discrete_risk_coverage(scores_a, corr)
    _, _, aurc_b = compute_discrete_risk_coverage(scores_b, corr)
    t_obs = float(aurc_a - aurc_b)
    abs_t_obs = abs(t_obs)
    
    count_extreme = 0
    for _ in range(n_perm):
        flips = rng.choice([-1, 1], size=n_clusters)
        perm_a = scores_a.copy()
        perm_b = scores_b.copy()
        for idxs, f in zip(cluster_indices, flips):
            if f == -1:
                perm_a[idxs] = scores_b[idxs]
                perm_b[idxs] = scores_a[idxs]
                
        _, _, perm_aurc_a = compute_discrete_risk_coverage(perm_a, corr)
        _, _, perm_aurc_b = compute_discrete_risk_coverage(perm_b, corr)
        t_perm = perm_aurc_a - perm_aurc_b
        if abs(t_perm) >= abs_t_obs - 1e-12:
            count_extreme += 1
            
    p_val = float((1.0 + count_extreme) / (n_perm + 1.0))
    return aurc_a, aurc_b, t_obs, p_val


def run_paired_cluster_bootstrap(df: pd.DataFrame, score_cols: list, corr_col: str = "correct_id",
                                cluster_col: str = "canonical_image_id", n_boot: int = 1000, seed: int = 42) -> Tuple[dict, dict]:
    """
    Paired Image-Cluster Bootstrap (resampling unique physical images with replacement).
    Handles multiple selections of the same cluster properly by concatenating cluster blocks.
    Generates 95% Percentile Confidence Intervals for individual AURCs and paired differences Delta AURC.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)
    cluster_dfs = {c: df[df[cluster_col] == c] for c in unique_clusters}
    
    boot_aurcs = {col: [] for col in score_cols}
    boot_diffs = {col: [] for col in score_cols if col != "score_m1"}
    
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([cluster_dfs[c] for c in sampled_clusters], ignore_index=True)
        corr = boot_df[corr_col].values
        
        aurc_m1 = None
        for col in score_cols:
            _, _, aurc = compute_discrete_risk_coverage(boot_df[col].values, corr)
            boot_aurcs[col].append(aurc)
            if col == "score_m1":
                aurc_m1 = aurc
                
        for col in boot_diffs.keys():
            boot_diffs[col].append(boot_aurcs[col][-1] - aurc_m1)
            
    summary = {}
    for col in score_cols:
        vals = np.array(boot_aurcs[col])
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_95_lower": float(np.percentile(vals, 2.5)),
            "ci_95_upper": float(np.percentile(vals, 97.5))
        }
        
    diff_summary = {}
    for col in boot_diffs.keys():
        d_vals = np.array(boot_diffs[col])
        diff_summary[col] = {
            "delta_mean": float(np.mean(d_vals)),
            "delta_std": float(np.std(d_vals)),
            "delta_ci_95_lower": float(np.percentile(d_vals, 2.5)),
            "delta_ci_95_upper": float(np.percentile(d_vals, 97.5))
        }
        
    return summary, diff_summary


def apply_holm_bonferroni(p_dict: dict) -> dict:
    """Applies Holm-Bonferroni step-down correction to a dictionary of raw p-values."""
    sorted_items = sorted(p_dict.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adj_p = {}
    running_max = 0.0
    for rank, (k, p) in enumerate(sorted_items, 1):
        mult = m - rank + 1
        val = min(1.0, p * mult)
        running_max = max(running_max, val)
        adj_p[k] = min(1.0, running_max)
    return adj_p

# ==============================================================================
# 4. STRICT CHECKPOINT GOVERNANCE & RESUME INTEGRITY
# ==============================================================================

def validate_checkpoint_compatibility(df_ckpt: pd.DataFrame, target_model: str, 
                                      expected_manifest_sha: str, expected_qa_ids: set) -> Tuple[bool, str]:
    """
    Validates that a checkpoint strictly matches rev04 specifications.
    Rejects any stale, cross-revision, corrupted, or non-finite rows.
    """
    if df_ckpt is None or len(df_ckpt) == 0:
        return False, "Checkpoint file is empty."
        
    required_cols = [
        "qa_id", "canonical_image_id", "split", "reference_answer",
        "pred_en", "conf_en", "correct_en", "p_en_yes", "p_en_no",
        "pred_id", "conf_id", "correct_id", "p_orig_ya", "p_orig_tidak",
        "p_blank_ya", "p_blank_tidak", "p_blank_pred", "delta_p_dir", "delta_p_abs",
        "score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old",
        "scoring_method", "revision_id", "manifest_sha256", "model_id"
    ]
    missing = [c for c in required_cols if c not in df_ckpt.columns]
    if missing:
        return False, f"Missing required checkpoint columns: {missing}"
        
    # Check revision ID
    revs = df_ckpt["revision_id"].unique()
    if len(revs) != 1 or revs[0] != REVISION_ID:
        return False, f"Incompatible revision ID in checkpoint: {revs} (expected {REVISION_ID})"
        
    # Check manifest SHA256
    shas = df_ckpt["manifest_sha256"].unique()
    if len(shas) != 1 or shas[0] != expected_manifest_sha:
        return False, f"Manifest SHA256 mismatch in checkpoint: {shas} (expected {expected_manifest_sha})"
        
    # Check target model ID
    models = df_ckpt["model_id"].unique()
    if len(models) != 1 or models[0] != target_model:
        return False, f"Model ID mismatch in checkpoint: {models} (expected {target_model})"
        
    # Check non-finite scores
    score_cols = ["score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old", "delta_p_dir"]
    for sc in score_cols:
        if not np.isfinite(df_ckpt[sc].values).all():
            return False, f"Checkpoint contains non-finite (NaN or Inf) values in column {sc}!"
            
    # Check QA IDs subset
    ckpt_qids = set(df_ckpt["qa_id"])
    invalid_qids = ckpt_qids - expected_qa_ids
    if invalid_qids:
        return False, f"Checkpoint contains extraneous QIDs not in evaluation manifest: {len(invalid_qids)} samples"
        
    return True, f"Valid checkpoint containing {len(df_ckpt)} completed rows."

# ==============================================================================
# 5. CANDIDATE SCORING ENGINE (rev04: CLOSED CANDIDATE LOG-LIKELIHOOD)
# ==============================================================================

def extract_target_loglikelihood_v4(model, processor, image_path: Path, prompt_text: str, 
                                   candidate_strings: List[str], target_device: Any) -> Tuple[dict, Tuple[float, float], Tuple[float, float]]:
    """
    Evaluates conditional log-likelihood of exact candidate target strings
    given multimodal context (Image + Prompt).
    
    Mathematical Implementation:
    1. Tokenizes prompt multimodal template. Identifies prompt length L_prompt.
    2. Appends each candidate string to create full text.
    3. Verifies prefix identity to eliminate boundary slippage.
    4. Slices causal logits corresponding to candidate sequence:
       target_logits = logits[L_prompt - 1 : L_prompt - 1 + L_cand].
    5. Calculates both:
       - Raw sequence log-likelihood: sum_{t=1}^L log P(y_t | ...)
       - Length-normalized mean log-likelihood: (1/L) sum_{t=1}^L log P(y_t | ...)
    6. Produces relative candidate scores via stable log-sum-exp softmax.
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
        
        # Verify prefix match to guarantee zero boundary slippage
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
    
    # Softmax over length-normalized log-likelihoods (Primary rev04 relative candidate score)
    s1_norm = cand_stats[c1]["mean_lp"]
    s2_norm = cand_stats[c2]["mean_lp"]
    max_norm = max(s1_norm, s2_norm)
    exp_norm1 = np.exp(s1_norm - max_norm)
    exp_norm2 = np.exp(s2_norm - max_norm)
    p1_norm = float(exp_norm1 / (exp_norm1 + exp_norm2))
    p2_norm = float(1.0 - p1_norm)
    
    # Softmax over raw sequence log-likelihoods (Historical comparison)
    s1_raw = cand_stats[c1]["raw_sum"]
    s2_raw = cand_stats[c2]["raw_sum"]
    max_raw = max(s1_raw, s2_raw)
    exp_raw1 = np.exp(s1_raw - max_raw)
    exp_raw2 = np.exp(s2_raw - max_raw)
    p1_raw = float(exp_raw1 / (exp_raw1 + exp_raw2))
    p2_raw = float(1.0 - p1_raw)
    
    return cand_stats, (p1_norm, p2_norm), (p1_raw, p2_raw)


def score_binary_candidates_v4(model, processor, image_path: Path, prompt_text: str, 
                               candidate_strings: List[str], target_device: Any) -> Tuple[str, float, float, float, dict, float, float]:
    """
    Scores binary candidate strings.
    Candidate 0: Affirmative ("YES<|im_end|>" / "YA<|im_end|>")
    Candidate 1: Negative ("NO<|im_end|>" / "TIDAK<|im_end|>")
    Returns:
      pred_label: "YES" if p_aff >= 0.5 else "NO"
      conf: max(p_aff, p_neg) (relative candidate score)
      p_aff, p_neg: normalized relative probabilities
      cand_stats: full per-token logprob details
      p_aff_raw, p_neg_raw: raw sequence softmax probabilities
    """
    cand_stats, (p_aff, p_neg), (p_aff_raw, p_neg_raw) = extract_target_loglikelihood_v4(
        model, processor, image_path, prompt_text, candidate_strings, target_device
    )
    if p_aff >= 0.5:
        pred_label = "YES"
        conf = p_aff
    else:
        pred_label = "NO"
        conf = p_neg
    return pred_label, conf, p_aff, p_neg, cand_stats, p_aff_raw, p_neg_raw

# ==============================================================================
# 6. MODEL PIPELINE EXECUTOR & CHECKPOINTING
# ==============================================================================

def execute_model_evaluation_v4(model_name: str, df_eval: pd.DataFrame, image_map: dict,
                               results_root: Path, perturb_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """
    Executes full evaluation pipeline for a specified vision-language model.
    Evaluates:
      1. Condition P0 (Original Image) in English (Prompt + YES/NO).
      2. Condition P0 (Original Image) in Indonesian (Prompt + YA/TIDAK).
      3. Condition P1 (Pure Black Canvas) in Indonesian (Prompt + YA/TIDAK).
    Calculates all 5 selective prediction estimators.
    Periodically flushes to checkpoint CSV.
    """
    import torch
    from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig
    import transformers

    model_slug = model_name.replace("/", "_").replace("-", "_")
    model_out_dir = results_root / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = model_out_dir / f"live_validation_checkpoint_{model_slug}.csv"
    
    # Checkpoint recovery check
    completed_qa_ids = set()
    results = []
    if checkpoint_csv.exists():
        try:
            df_existing = pd.read_csv(checkpoint_csv)
            is_valid, msg = validate_checkpoint_compatibility(
                df_existing, model_name, AUTHORITATIVE_MANIFEST_SHA256, set(df_eval["qa_id"])
            )
            if is_valid:
                print(f"🔄 Checkpoint verified: resuming {len(df_existing)} completed samples for {model_name}...")
                results = df_existing.to_dict("records")
                completed_qa_ids = set(df_existing["qa_id"])
            else:
                print(f"⚠️ Checkpoint rejected ({msg}). Initializing clean evaluation...")
                checkpoint_csv.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ Checkpoint unreadable ({e}). Initializing clean evaluation...")
            checkpoint_csv.unlink(missing_ok=True)

    if len(completed_qa_ids) == len(df_eval):
        print(f"✅ All {len(df_eval)} samples already evaluated in checkpoint. Skipping inference.")
        return pd.DataFrame(results), model_out_dir

    print(f"\n🚀 Loading Model: {model_name} (4-bit NF4 Quantization)...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch_list = getattr(config, "architectures", [])
    model_class = None
    for arch_name in arch_list:
        if hasattr(transformers, arch_name):
            model_class = getattr(transformers, arch_name)
            break
    if model_class is None:
        from transformers import AutoModelForVision2Seq
        model_class = AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = model_class.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.eval()
    
    if hasattr(model, "device"):
        target_dev = model.device
    else:
        target_dev = next(model.parameters()).device
    print(f"   Loaded onto: {target_dev}")

    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"Evaluating {model_name.split('/')[-1]}"):
        qa_id = row["qa_id"]
        if qa_id in completed_qa_ids:
            continue
            
        cid = str(row["canonical_image_id"])
        img_id = str(row.get("image_id", cid))
        split_val = str(row.get("split", "test"))
        trans_stat = str(row.get("translation_status", "CLEAN"))
        q_en = str(row["question_en"])
        q_id = str(row["question_id"])
        ref_ans = str(row["reference_answer"]).strip().upper()
        orig_img_path = image_map[qa_id]
        
        # Condition P1: Pure Black Blank Canvas
        blank_img_path = perturb_dir / f"blank_{cid}.png"
        if not blank_img_path.exists():
            with Image.open(orig_img_path) as orig_pil:
                blank_pil = Image.new("RGB", orig_pil.size, color=(0, 0, 0))
                blank_pil.save(blank_img_path)
                
        # RQ1: Bilingual Scoring on P0 (Original Image)
        prompt_en = PROMPT_EN_TEMPLATE.format(q=q_en)
        pred_en, conf_en, p_en_yes, p_en_no, _, p_en_yes_raw, p_en_no_raw = score_binary_candidates_v4(
            model, processor, orig_img_path, prompt_en, CANDIDATES_EN, target_dev
        )
        corr_en = int(pred_en == ref_ans)
        
        prompt_id = PROMPT_ID_TEMPLATE.format(q=q_id)
        pred_id, conf_id, p_orig_ya, p_orig_tidak, _, p_orig_ya_raw, p_orig_tidak_raw = score_binary_candidates_v4(
            model, processor, orig_img_path, prompt_id, CANDIDATES_ID, target_dev
        )
        corr_id = int(pred_id == ref_ans)
        flip = int(pred_en != pred_id)
        joint_corr = int(corr_en and corr_id)
        
        # RQ2: Visual Intervention Scoring on P1 (Black Canvas)
        # Evaluated on the SAME predicted label pred_id
        _, _, p_blank_ya, p_blank_tidak, _, p_blank_ya_raw, p_blank_tidak_raw = score_binary_candidates_v4(
            model, processor, blank_img_path, prompt_id, CANDIDATES_ID, target_dev
        )
        p_blank_pred = p_blank_ya if pred_id == "YES" else p_blank_tidak
        delta_p_dir = conf_id - p_blank_pred
        delta_p_abs = abs(conf_id - p_blank_pred)
        
        # RQ3: 5 Selective Prediction Estimators
        score_m1 = conf_id
        score_m2 = max(0.0, delta_p_dir)
        score_m_dir = conf_id * max(0.0, delta_p_dir)
        score_m_norm = conf_id * (1.0 + delta_p_dir) / 2.0
        score_m3_old = conf_id * (1.0 + delta_p_abs) / 2.0
        
        results.append({
            "qa_id": qa_id,
            "canonical_image_id": cid,
            "image_id": img_id,
            "split": split_val,
            "translation_status": trans_stat,
            "is_yes_no": bool(row.get("is_yes_no", True)),
            "in_rev03_benchmark": bool(row.get("in_rev03_benchmark", False)),
            "model_id": model_name,
            "device": str(target_dev),
            "question_en": q_en,
            "question_id": q_id,
            "reference_answer": ref_ans,
            "pred_en": pred_en,
            "conf_en": conf_en,
            "correct_en": corr_en,
            "p_en_yes": p_en_yes,
            "p_en_no": p_en_no,
            "pred_id": pred_id,
            "conf_id": conf_id,
            "correct_id": corr_id,
            "p_orig_ya": p_orig_ya,
            "p_orig_tidak": p_orig_tidak,
            "flip": flip,
            "joint_correct": joint_corr,
            "p_en_yes_raw": p_en_yes_raw,
            "p_en_no_raw": p_en_no_raw,
            "p_orig_ya_raw": p_orig_ya_raw,
            "p_orig_tidak_raw": p_orig_tidak_raw,
            "p_blank_ya": p_blank_ya,
            "p_blank_tidak": p_blank_tidak,
            "p_blank_pred": p_blank_pred,
            "delta_p_dir": delta_p_dir,
            "delta_p_abs": delta_p_abs,
            "score_m1": score_m1,
            "score_m2": score_m2,
            "score_m_dir": score_m_dir,
            "score_m_norm": score_m_norm,
            "score_m3_old": score_m3_old,
            "blank_method": "PIL.Image.new('RGB', size, (0,0,0))",
            "scoring_method": SCORING_METHOD,
            "revision_id": REVISION_ID,
            "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        completed_qa_ids.add(qa_id)
        if len(results) % 10 == 0 or len(results) == len(df_eval):
            pd.DataFrame(results).to_csv(checkpoint_csv, index=False)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(checkpoint_csv, index=False)
    print(f"💾 Checkpoint saved: {checkpoint_csv.name} ({len(df_res)} rows)")
    return df_res, model_out_dir
