"""
Decisive Validation Experiment Engine (rev03)
==============================================
Project: Medical Vision-Language Model Reliability Under Language Shift
Dataset: IndoRad-VQA Decisive Evaluation Benchmark
Author: Medical VLM Research Auditor & Reproducibility Team
Revision: rev03

Fixes & Improvements in rev03:
1. Symmetrical Candidate Scoring: Full closed response scoring with generation termination (<|im_end|>).
2. Length Normalization: Solves the geometric token-penalty asymmetry between English (1 token)
   and Indonesian ('Ya' 1 token vs 'Tidak' 2-3 tokens).
3. Symmetrical Prompting: Removes Indonesian preamble asymmetry, enforcing exact 1-to-1 syntactic parallelism.
4. Exact Paired Cluster Permutation Test: Replaces ad-hoc empirical bootstrap tail shortcuts with a
   mathematically sound cluster-level sign-flip permutation test under H0: E[Delta AURC] = 0.
5. Multiplicity Control: Applies Holm-Bonferroni step-down correction across the 4 candidate estimators.
6. Discrete AURC & Risk-Coverage Rigor: Evaluates Geifman & El-Yaniv discrete mean risk and realizes
   exact coverage levels (100%, 90%, 80%, 70%, 50%).
7. Strict Checkpoint Governance: Enforces revision_id == "rev03", manifest SHA-256, and finite score checks.
"""

import os
import sys
import zlib
import base64
import hashlib
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# REVISION CONSTANTS & AUTHORITATIVE INVARIANTS
# ==============================================================================
REVISION_ID = "rev03"
SCORING_METHOD = "length_normalized_closed_response_v3"
AUTHORITATIVE_MANIFEST_SHA256 = "988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52"

# 1-to-1 Syntactically Parallel Prompts
PROMPT_EN_TEMPLATE = "Answer only YES or NO.\nQuestion: {q}"
PROMPT_ID_TEMPLATE = "Jawab hanya dengan YA atau TIDAK.\nPertanyaan: {q}"

# Complete Closed Responses with Generation Termination Token
CANDIDATES_EN = ["YES<|im_end|>", "NO<|im_end|>"]
CANDIDATES_ID = ["YA<|im_end|>", "TIDAK<|im_end|>"]

DEFAULT_MODEL_1 = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_2 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

# Embedded Authoritative Manifest (Bit-identical to decisive_validation_manifest.csv)
EMBEDDED_MANIFEST_B64 = (
    "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"
    "ajP78Yf/BxzgFts="
)


# ==============================================================================
# 1. CRYPTOGRAPHIC MANIFEST & MULTI-TIER IMAGE RESOLVER
# ==============================================================================

def unpack_embedded_manifest() -> bytes:
    """Decompresses and returns the authoritative manifest CSV bytes."""
    raw = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64.encode("ascii")))
    computed_hash = hashlib.sha256(raw).hexdigest()
    if computed_hash != AUTHORITATIVE_MANIFEST_SHA256:
        raise ValueError(
            f"Corrupted embedded manifest! Expected SHA256 {AUTHORITATIVE_MANIFEST_SHA256}, "
            f"got {computed_hash}"
        )
    return raw


def load_authoritative_manifest(manifest_path: Path = None, 
                                fallback_out_dir: Path = None, 
                                allow_embedded_fallback: bool = True) -> pd.DataFrame:
    """
    Loads and strictly validates the authoritative frozen evaluation manifest.
    Asserts:
      1. Manifest file matches authoritative SHA256 exactly.
      2. Exactly 150 QA pairs.
      3. Exactly 92 test QA pairs across 35 image clusters.
      4. Exactly 58 dev QA pairs across 23 image clusters.
      5. Zero cross-split image cluster overlap.
    """
    df_manifest = None
    loaded_from = None
    
    if manifest_path is not None and Path(manifest_path).exists():
        p = Path(manifest_path)
        content = p.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        if file_hash == AUTHORITATIVE_MANIFEST_SHA256:
            import io
            df_manifest = pd.read_csv(io.BytesIO(content))
            loaded_from = str(p)
        else:
            print(f"⚠️ Warning: File at {p} does not match authoritative SHA256! (Got: {file_hash})")

    if df_manifest is None:
        search_paths = [
            Path("main/results/decisive_validation_manifest.csv"),
            Path("results/decisive_validation_manifest.csv"),
            Path("./decisive_validation_manifest.csv"),
            Path("/kaggle/input/indorad-vqa-research/main/results/decisive_validation_manifest.csv"),
            Path("/kaggle/input/indorad-vqa-research/results/decisive_validation_manifest.csv"),
        ]
        for sp in search_paths:
            if sp.exists():
                content = sp.read_bytes()
                if hashlib.sha256(content).hexdigest() == AUTHORITATIVE_MANIFEST_SHA256:
                    import io
                    df_manifest = pd.read_csv(io.BytesIO(content))
                    loaded_from = str(sp)
                    break
                    
    if df_manifest is None and allow_embedded_fallback:
        print("ℹ️ Using verified embedded authoritative manifest.")
        import io
        raw_bytes = unpack_embedded_manifest()
        df_manifest = pd.read_csv(io.BytesIO(raw_bytes))
        loaded_from = "EMBEDDED_AUTHENTICATED_MANIFEST"
        if fallback_out_dir is not None:
            fallback_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = fallback_out_dir / "decisive_validation_manifest.csv"
            if not out_file.exists():
                out_file.write_bytes(raw_bytes)
                print(f"💾 Written authoritative manifest to: {out_file}")

    if df_manifest is None:
        raise FileNotFoundError(
            f"Could not load authoritative manifest matching SHA256 {AUTHORITATIVE_MANIFEST_SHA256}."
        )

    # Invariant assertions
    assert len(df_manifest) == 150, f"Expected 150 rows, got {len(df_manifest)}"
    test_df = df_manifest[df_manifest["split"] == "test"]
    dev_df = df_manifest[df_manifest["split"] == "dev"]
    assert len(test_df) == 92, f"Expected 92 test QA pairs, got {len(test_df)}"
    assert len(dev_df) == 58, f"Expected 58 dev QA pairs, got {len(dev_df)}"
    
    test_clusters = set(test_df["canonical_image_id"].unique())
    dev_clusters = set(dev_df["canonical_image_id"].unique())
    assert len(test_clusters) == 35, f"Expected 35 test clusters, got {len(test_clusters)}"
    assert len(dev_clusters) == 23, f"Expected 23 dev clusters, got {len(dev_clusters)}"
    overlap = test_clusters.intersection(dev_clusters)
    assert len(overlap) == 0, f"Cross-split cluster leakage detected! Overlapping clusters: {overlap}"

    print(f"🔒 Authoritative Manifest Verified ({loaded_from})")
    print(f"   SHA256: {AUTHORITATIVE_MANIFEST_SHA256}")
    print(f"   Invariants: 150 QA pairs (92 Test / 58 Dev) across 58 unique physical image clusters (35 Test / 23 Dev).")
    return df_manifest


def compute_rgb_pixel_sha256(pil_img: Image.Image) -> str:
    """Decodes image into RGB bytes and computes SHA256 over raw pixel bytes."""
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
        import zipfile
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
    
    print("\n🔍 Verifying cryptographic image identity for all 150 QA pairs...")
    for _, row in df_manifest.iterrows():
        qa_id = row["qa_id"]
        cid = str(row["canonical_image_id"])
        expected_hash = str(row["pixel_sha256"]).strip().lower()
        expected_w = int(row["image_width"])
        expected_h = int(row["image_height"])
        
        found_path = None
        for sdir in search_dirs:
            for cand_name in [f"{cid}.png", f"img_{int(qa_id):05d}.png", f"{qa_id}.png", f"{row.get('image_name', '')}"]:
                if not cand_name:
                    continue
                p = sdir / cand_name
                if p.exists():
                    try:
                        with Image.open(p) as img:
                            if img.size == (expected_w, expected_h):
                                pix_hash = compute_rgb_pixel_sha256(img)
                                if pix_hash == expected_hash:
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

    if missing_rows:
        print(f"⚠️ {len(missing_rows)} images missing from local search dirs. Triggering self-healing Parquet download...")
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
            
            splits_to_fetch = set(r["split"] for r in missing_rows)
            for split_name in splits_to_fetch:
                fname = f"data/{split_name}-00000-of-00001.parquet"
                print(f"   Downloading {fname} from Hugging Face Hub (Lab-IS/IndoRad-VQA)...")
                local_pq = hf_hub_download(repo_id="Lab-IS/IndoRad-VQA", filename=fname, repo_type="dataset")
                table = pq.read_table(local_pq)
                df_pq = table.to_pandas()
                
                for r in missing_rows:
                    if r["split"] != split_name:
                        continue
                    qa_id = r["qa_id"]
                    cid = str(r["canonical_image_id"])
                    expected_hash = str(r["pixel_sha256"]).strip().lower()
                    
                    match = df_pq[df_pq["qid"] == qa_id]
                    if len(match) == 0:
                        continue
                    img_dict = match.iloc[0]["image"]
                    raw_bytes = img_dict.get("bytes", None)
                    if raw_bytes is None:
                        continue
                        
                    import io
                    with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                        pix_hash = compute_rgb_pixel_sha256(pil_img)
                        if pix_hash == expected_hash:
                            save_path = cache_dir / f"{cid}.png"
                            pil_img.save(save_path)
                            resolved_map[qa_id] = save_path
                            print(f"   ✅ Self-healed & cached: qa_id {qa_id} -> {save_path.name}")
        except Exception as e:
            print(f"❌ Self-healing failed: {e}")

    # Final assertion
    unresolved = [r["qa_id"] for _, r in df_manifest.iterrows() if r["qa_id"] not in resolved_map]
    if unresolved:
        raise RuntimeError(
            f"Failed to cryptographically resolve {len(unresolved)} images! QA IDs: {unresolved[:10]}... "
            f"Searched {len(search_dirs)} directories."
        )

    print(f"✅ Cryptographic Image Verification: 100% PASS (150/150 images verified by pixel SHA256).")
    return resolved_map


# ==============================================================================
# 2. STATISTICAL & SELECTIVE PREDICTION EVALUATION ENGINE
# ==============================================================================

def compute_discrete_risk_coverage(scores: np.ndarray, correctness: np.ndarray):
    """
    Standard sample-level discrete selective prediction evaluation (Geifman & El-Yaniv, NeurIPS 2017).
    Ranks samples by reliability score descending: S_(1) >= S_(2) >= ... >= S_(N).
    Evaluates empirical selective risk r_k = 1 - (1/k) sum_{j=1}^k Correct_(j) for k=1..N.
    Returns:
      cov: discrete coverage grid [1/N, 2/N, ..., 1.0]
      risk: empirical selective risk array at each discrete coverage
      aurc_mean: exact Riemann sum / mean empirical risk across all N evaluation points
    """
    scores = np.asarray(scores)
    correctness = np.asarray(correctness, dtype=float)
    n = len(scores)
    
    order = np.argsort(-scores, kind="mergesort")
    sorted_corr = correctness[order]
    
    k = np.arange(1, n + 1)
    cov = k / n
    cum_correct = np.cumsum(sorted_corr)
    acc = cum_correct / k
    risk = 1.0 - acc
    
    aurc_mean = float(np.mean(risk))
    return cov, risk, aurc_mean


def get_risk_at_realized_coverage(cov_arr: np.ndarray, risk_arr: np.ndarray, target_cov: float):
    """Finds selective risk and sample count at closest discrete coverage level."""
    idx = int((np.abs(cov_arr - target_cov)).argmin())
    return float(risk_arr[idx]), float(cov_arr[idx]), int(idx + 1)


def paired_cluster_permutation_test(df: pd.DataFrame, score_col_a: str, score_col_b: str,
                                    corr_col: str = "correct_id", cluster_col: str = "canonical_image_id",
                                    n_perm: int = 1000, seed: int = 42) -> tuple:
    """
    Exact Paired Image-Cluster Permutation Test under H0: E[Delta AURC] = 0.
    For each permutation replicate, draws independent Rademacher sign flips eps_c in {-1, +1}
    for each unique image cluster. Swaps estimator scores for all QA pairs in cluster c if eps_c == -1.
    Computes exact two-sided finite-sample p-value.
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
                                cluster_col: str = "canonical_image_id", n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Paired Image-Cluster Bootstrap (resampling unique physical images with replacement).
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
        diff_vals = np.array(boot_diffs[col])
        diff_summary[col] = {
            "mean_diff": float(np.mean(diff_vals)),
            "ci_95_lower": float(np.percentile(diff_vals, 2.5)),
            "ci_95_upper": float(np.percentile(diff_vals, 97.5))
        }
        
    return summary, diff_summary


def apply_holm_bonferroni(p_dict: dict) -> dict:
    """Applies Holm-Bonferroni step-down correction to a dict of raw p-values."""
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
# 3. CHECKPOINT & REVISION GOVERNANCE
# ==============================================================================

def validate_checkpoint_compatibility(df_ckpt: pd.DataFrame, model_name: str, 
                                      expected_manifest_sha256: str, expected_qa_ids: set) -> tuple:
    """
    Strict validation for rev03 checkpoint compatibility.
    Rejects incompatible models, manifests, revision IDs, or corrupted rows.
    """
    required_cols = [
        "qa_id", "canonical_image_id", "split", "reference_answer",
        "pred_en", "conf_en", "correct_en", "p_en_yes", "p_en_no",
        "pred_id", "conf_id", "correct_id", "p_orig_ya", "p_orig_tidak",
        "p_blank_ya", "p_blank_tidak", "p_blank_pred", "delta_p_dir", "delta_p_abs",
        "score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old",
        "scoring_method", "revision_id", "manifest_sha256"
    ]
    for c in required_cols:
        if c not in df_ckpt.columns:
            return False, f"Missing required column: {c}"
            
    rev_vals = df_ckpt["revision_id"].dropna().unique()
    if len(rev_vals) == 0 or rev_vals[0] != REVISION_ID:
        return False, f"Incompatible revision ID! Expected {REVISION_ID}, got {rev_vals}"
        
    sha_vals = df_ckpt["manifest_sha256"].dropna().unique()
    if len(sha_vals) == 0 or sha_vals[0] != expected_manifest_sha256:
        return False, f"Manifest SHA256 mismatch! Expected {expected_manifest_sha256}, got {sha_vals}"
        
    model_vals = df_ckpt["model_id"].dropna().unique()
    if len(model_vals) == 0 or model_vals[0] != model_name:
        return False, f"Model ID mismatch! Expected {model_name}, got {model_vals}"
        
    ckpt_qa_ids = set(df_ckpt["qa_id"].tolist())
    if not ckpt_qa_ids.issubset(expected_qa_ids):
        extra = ckpt_qa_ids - expected_qa_ids
        return False, f"Found unexpected QA IDs in checkpoint: {extra}"
        
    numeric_cols = ["conf_en", "conf_id", "p_orig_ya", "p_orig_tidak", "score_m1", "score_m_dir", "score_m_norm"]
    for nc in numeric_cols:
        if not np.all(np.isfinite(df_ckpt[nc].values)):
            return False, f"Found non-finite values (NaN/Inf) in column: {nc}"
            
    return True, "Valid rev03 checkpoint"


# ==============================================================================
# 4. TARGET SEQUENCE SCORING WITH EOS & LENGTH NORMALIZATION (REV03)
# ==============================================================================

def extract_target_loglikelihood_v3(model, processor, image_path, prompt_text, candidate_strings, target_device):
    """
    Evaluates conditional log-likelihood of exact candidate target strings
    given the multimodal context (Image + Prompt).
    
    Features in rev03:
    - Appends generation termination (<|im_end|>) to candidates.
    - Causal logit slicing verified with prefix assertion.
    - Computes both raw sequence logprob AND length-normalized log-likelihood.
    - Softmax over length-normalized log-likelihoods using stable logsumexp.
    
    Returns:
      cand_stats: dict mapping cand -> {raw_sum, mean_lp, target_ids, token_lps}
      p_rel_norm: relative probabilities via length-normalized log-likelihood
      p_rel_raw: relative probabilities via raw sequence log-likelihood
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
    
    # Softmax over length-normalized log-likelihoods (Primary rev03 method)
    s1_norm = cand_stats[c1]["mean_lp"]
    s2_norm = cand_stats[c2]["mean_lp"]
    max_norm = max(s1_norm, s2_norm)
    exp_norm1 = np.exp(s1_norm - max_norm)
    exp_norm2 = np.exp(s2_norm - max_norm)
    p1_norm = float(exp_norm1 / (exp_norm1 + exp_norm2))
    p2_norm = float(1.0 - p1_norm)
    
    # Softmax over raw logprobs (for historical ablation & transparency)
    s1_raw = cand_stats[c1]["raw_sum"]
    s2_raw = cand_stats[c2]["raw_sum"]
    max_raw = max(s1_raw, s2_raw)
    exp_raw1 = np.exp(s1_raw - max_raw)
    exp_raw2 = np.exp(s2_raw - max_raw)
    p1_raw = float(exp_raw1 / (exp_raw1 + exp_raw2))
    p2_raw = float(1.0 - p1_raw)
    
    return cand_stats, (p1_norm, p2_norm), (p1_raw, p2_raw)


def score_binary_candidates_v3(model, processor, image_path, prompt_text, candidate_strings, target_device):
    """
    Runs sequence scoring under rev03 and returns predictions and scores.
    candidate_strings: [affirmatory, negative], e.g. ["YES<|im_end|>", "NO<|im_end|>"]
    Returns: (pred_label, confidence, p_aff, p_neg, cand_stats, p_aff_raw, p_neg_raw)
    """
    cand_stats, (p_aff, p_neg), (p_aff_raw, p_neg_raw) = extract_target_loglikelihood_v3(
        model, processor, image_path, prompt_text, candidate_strings, target_device
    )
    if p_aff >= 0.5:
        pred_label = "YES"
        confidence = p_aff
    else:
        pred_label = "NO"
        confidence = p_neg
        
    return pred_label, confidence, p_aff, p_neg, cand_stats, p_aff_raw, p_neg_raw


# ==============================================================================
# 5. PIPELINE EXECUTION FOR SINGLE MODEL (REV03)
# ==============================================================================

def execute_model_evaluation_v3(model_name: str, df_eval: pd.DataFrame, image_map: dict, 
                                results_root: Path, perturb_dir: Path, target_device_str: str = "auto"):
    """
    Executes full evaluation on Model: RQ1 bilingual on P0, RQ2 visual intervention P1, RQ3 estimators.
    """
    import gc
    import time
    import torch
    from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig
    import transformers
    from tqdm.auto import tqdm
    
    safe_name = model_name.replace("/", "_").replace("-", "_")
    model_out_dir = results_root / safe_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_csv = model_out_dir / f"live_validation_checkpoint_{safe_name}.csv"
    expected_qa_ids = set(df_eval["qa_id"].tolist())
    
    completed_qa_ids = set()
    results = []
    
    if checkpoint_csv.exists():
        try:
            df_old = pd.read_csv(checkpoint_csv)
            is_valid, msg = validate_checkpoint_compatibility(
                df_old, model_name, AUTHORITATIVE_MANIFEST_SHA256, expected_qa_ids
            )
            if is_valid:
                completed_qa_ids = set(df_old["qa_id"].tolist())
                results = df_old.to_dict(orient="records")
                print(f"🔄 Resuming from valid {REVISION_ID} checkpoint: {len(completed_qa_ids)}/{len(df_eval)} done.")
            else:
                print(f"⚠️ Incompatible checkpoint detected ({msg}).")
                backup_p = model_out_dir / f"incompatible_checkpoint_backup_{int(time.time())}.csv"
                checkpoint_csv.rename(backup_p)
                print(f"   Backed up to: {backup_p.name}")
        except Exception as e:
            print(f"⚠️ Error reading checkpoint: {e}. Starting fresh.")
            
    if len(completed_qa_ids) >= len(df_eval):
        print(f"✅ Model [{model_name}] already 100% completed in {REVISION_ID}.")
        return pd.DataFrame(results), model_out_dir

    print(f"\n📦 Inspecting Configuration for: {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch_list = getattr(config, "architectures", [])
    print(f"   Declared Architecture(s): {arch_list} | Model Type: {config.model_type}")
    
    model_class = None
    for arch_name in arch_list:
        if hasattr(transformers, arch_name):
            model_class = getattr(transformers, arch_name)
            break
            
    if model_class is None:
        try:
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_class = AutoModelForVision2Seq
            
    print(f"   Resolved Model Class: {model_class.__name__}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = model_class.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=target_device_str,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.eval()
    target_dev = next(model.parameters()).device
    print(f"✅ Loaded {model_name} ({model_class.__name__}) onto: {target_dev}")

    # Main evaluation loop
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
        
        # 1. P1 Deterministic Black Blank
        blank_img_path = perturb_dir / f"blank_{cid}.png"
        if not blank_img_path.exists():
            with Image.open(orig_img_path) as orig_pil:
                blank_pil = Image.new("RGB", orig_pil.size, color=(0, 0, 0))
                blank_pil.save(blank_img_path)
                
        # 2. RQ1: Bilingual Evaluation on P0
        prompt_en = PROMPT_EN_TEMPLATE.format(q=q_en)
        pred_en, conf_en, p_en_yes, p_en_no, stats_en, p_en_yes_raw, p_en_no_raw = score_binary_candidates_v3(
            model, processor, orig_img_path, prompt_en, CANDIDATES_EN, target_dev
        )
        corr_en = int(pred_en == ref_ans)
        
        prompt_id = PROMPT_ID_TEMPLATE.format(q=q_id)
        pred_id, conf_id, p_orig_ya, p_orig_tidak, stats_id, p_orig_ya_raw, p_orig_tidak_raw = score_binary_candidates_v3(
            model, processor, orig_img_path, prompt_id, CANDIDATES_ID, target_dev
        )
        corr_id = int(pred_id == ref_ans)
        
        flip = int(pred_en != pred_id)
        joint_corr = int(corr_en and corr_id)
        
        # 3. RQ2: Visual Intervention P1 on Indonesian
        _, _, p_blank_ya, p_blank_tidak, _, p_blank_ya_raw, p_blank_tidak_raw = score_binary_candidates_v3(
            model, processor, blank_img_path, prompt_id, CANDIDATES_ID, target_dev
        )
        p_blank_pred = p_blank_ya if pred_id == "YES" else p_blank_tidak
        delta_p_dir = conf_id - p_blank_pred
        delta_p_abs = abs(conf_id - p_blank_pred)
        
        # 4. RQ3: 5 Selective Prediction Estimators
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
            "model_id": model_name,
            "device": str(target_dev),
            "question_en": q_en,
            "question_id": q_id,
            "reference_answer": ref_ans,
            # RQ1 Bilingual (Length-Normalized Probabilities)
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
            # Raw Probabilities (for auditability & ablation)
            "p_en_yes_raw": p_en_yes_raw,
            "p_en_no_raw": p_en_no_raw,
            "p_orig_ya_raw": p_orig_ya_raw,
            "p_orig_tidak_raw": p_orig_tidak_raw,
            # RQ2 Visual Intervention
            "p_blank_ya": p_blank_ya,
            "p_blank_tidak": p_blank_tidak,
            "p_blank_pred": p_blank_pred,
            "delta_p_dir": delta_p_dir,
            "delta_p_abs": delta_p_abs,
            # RQ3 Estimators
            "score_m1": score_m1,
            "score_m2": score_m2,
            "score_m_dir": score_m_dir,
            "score_m_norm": score_m_norm,
            "score_m3_old": score_m3_old,
            # Metadata & Auditing
            "blank_method": "PIL.Image.new('RGB', size, (0,0,0))",
            "scoring_method": SCORING_METHOD,
            "revision_id": REVISION_ID,
            "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        completed_qa_ids.add(qa_id)
        
        # Periodic checkpointing
        if len(results) % 5 == 0 or len(results) == len(df_eval):
            df_ckpt_curr = pd.DataFrame(results)
            df_ckpt_curr.to_csv(checkpoint_csv, index=False)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    df_final = pd.DataFrame(results)
    df_final.to_csv(checkpoint_csv, index=False)
    print(f"💾 Checkpoint saved ({len(df_final)} rows): {checkpoint_csv.name}")
    return df_final, model_out_dir


# ==============================================================================
# 6. ARTIFACT GENERATION & STATISTICAL REPORTING ENGINE
# ==============================================================================

def analyze_and_export_results_v3(df_all: pd.DataFrame, model_name: str, out_dir: Path):
    """
    Computes all decisive validation metrics for Test-Only (N=92) and Full Benchmark (N=150).
    Performs Paired Cluster Permutation Tests and Cluster Bootstrap CIs.
    Exports plots and CSV reports.
    """
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    safe_name = model_name.replace("/", "_").replace("-", "_")
    score_cols = ["score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old"]
    
    subsets = {
        "test_only": df_all[df_all["split"] == "test"],
        "full_benchmark": df_all
    }
    
    exported_files = []
    
    for sub_name, df_sub in subsets.items():
        n_samples = len(df_sub)
        n_clusters = df_sub["canonical_image_id"].nunique()
        print(f"\n📊 [{safe_name}] Analyzing {sub_name} (N={n_samples}, Clusters={n_clusters})...")
        
        # 1. RQ1 Metrics
        acc_en = float(df_sub["correct_en"].mean())
        acc_id = float(df_sub["correct_id"].mean())
        lang_gap = (acc_en - acc_id) * 100.0
        flip_rate = float(df_sub["flip"].mean()) * 100.0
        joint_acc = float(df_sub["joint_correct"].mean()) * 100.0
        
        pred_dist_en = df_sub["pred_en"].value_counts().to_dict()
        pred_dist_id = df_sub["pred_id"].value_counts().to_dict()
        ref_dist = df_sub["reference_answer"].value_counts().to_dict()
        
        # 2. RQ2 Visual Intervention Metrics
        delta_p = df_sub["delta_p_dir"].values
        mean_delta_p = float(np.mean(delta_p))
        median_delta_p = float(np.median(delta_p))
        prop_pos = float(np.mean(delta_p > 0)) * 100.0
        prop_neg = float(np.mean(delta_p < 0)) * 100.0
        prop_zero = float(np.mean(delta_p == 0)) * 100.0
        
        # 3. RQ3 Selective Prediction & Exact Paired Permutation Test
        estimator_rows = []
        raw_p_values = {}
        perm_details = {}
        
        # Cluster Bootstrap CIs
        boot_sum, boot_diff = run_paired_cluster_bootstrap(df_sub, score_cols, n_boot=1000, seed=42)
        
        for col in score_cols:
            cov, risk, aurc = compute_discrete_risk_coverage(df_sub[col].values, df_sub["correct_id"].values)
            r_100, _, _ = get_risk_at_realized_coverage(cov, risk, 1.0)
            r_90, _, _ = get_risk_at_realized_coverage(cov, risk, 0.9)
            r_80, _, _ = get_risk_at_realized_coverage(cov, risk, 0.8)
            r_70, _, _ = get_risk_at_realized_coverage(cov, risk, 0.7)
            r_50, _, _ = get_risk_at_realized_coverage(cov, risk, 0.5)
            
            if col != "score_m1":
                _, _, t_obs, p_val = paired_cluster_permutation_test(
                    df_sub, col, "score_m1", n_perm=1000, seed=42
                )
                raw_p_values[col] = p_val
                perm_details[col] = (t_obs, p_val)
            else:
                t_obs = 0.0
                p_val = 1.0
                
            estimator_rows.append({
                "subset": sub_name,
                "estimator": col,
                "aurc_discrete_mean": aurc,
                "ci_95_lower": boot_sum[col]["ci_95_lower"],
                "ci_95_upper": boot_sum[col]["ci_95_upper"],
                "delta_aurc_vs_m1": (aurc - estimator_rows[0]["aurc_discrete_mean"]) if col != "score_m1" else 0.0,
                "delta_ci_95_lower": boot_diff[col]["ci_95_lower"] if col != "score_m1" else 0.0,
                "delta_ci_95_upper": boot_diff[col]["ci_95_upper"] if col != "score_m1" else 0.0,
                "perm_p_value_raw": p_val,
                "risk_cov_100": r_100,
                "risk_cov_90": r_90,
                "risk_cov_80": r_80,
                "risk_cov_70": r_70,
                "risk_cov_50": r_50,
            })
            
        # Apply Holm-Bonferroni correction across the 4 candidate estimators
        adj_p_dict = apply_holm_bonferroni(raw_p_values)
        for r in estimator_rows:
            est = r["estimator"]
            r["perm_p_value_holm"] = adj_p_dict.get(est, 1.0 if est == "score_m1" else 1.0)
            
        df_est = pd.DataFrame(estimator_rows)
        csv_est_path = out_dir / f"estimator_comparison_{safe_name}_{sub_name}.csv"
        df_est.to_csv(csv_est_path, index=False)
        exported_files.append(csv_est_path)
        
        # 4. Generate Publication-Grade Risk-Coverage Curves
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        color_map = {
            "score_m1": ("#1f77b4", "M1: Confidence-Only (p_orig)", "-"),
            "score_m2": ("#ff7f0e", "M2: Visual-Only max(0, Delta)", "--"),
            "score_m_dir": ("#2ca02c", "M_dir: p_orig * max(0, Delta)", "-"),
            "score_m_norm": ("#9467bd", "M_norm: p_orig * (1 + Delta)/2", "-."),
            "score_m3_old": ("#d62728", "M3_old: Historical Unsigned", ":")
        }
        for col in score_cols:
            cov, risk, aurc = compute_discrete_risk_coverage(df_sub[col].values, df_sub["correct_id"].values)
            color, label, linestyle = color_map[col]
            p_holm = df_est.loc[df_est["estimator"] == col, "perm_p_value_holm"].values[0]
            p_str = f" | p_adj={p_holm:.3f}" if col != "score_m1" else " [Base]"
            ax.plot(cov, risk, label=f"{label} (AURC={aurc:.4f}{p_str})", color=color, linestyle=linestyle, linewidth=2.0)
            
        ax.set_title(f"Selective Risk-Coverage Curves ({model_name.split('/')[-1]})\nSubset: {sub_name} (N={n_samples}, Clusters={n_clusters})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Coverage (Fraction of Accepted Samples)", fontsize=11)
        ax.set_ylabel("Selective Empirical Risk (1 - Accuracy)", fontsize=11)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(-0.02, 0.65)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        fig.tight_layout()
        plot_p = out_dir / f"risk_coverage_curve_{safe_name}_{sub_name}.png"
        fig.savefig(plot_p)
        plt.close(fig)
        exported_files.append(plot_p)

    return exported_files


# ==============================================================================
# 7. CROSS-MODEL SYNTHESIS & METADATA EXPORT
# ==============================================================================

def build_cross_model_summary(results_root: Path, model_dirs: list):
    """Aggregates test-only results across both models into unified synthesis tables."""
    dfs = []
    for md in model_dirs:
        test_est_csv = list(md.glob("estimator_comparison_*_test_only.csv"))
        if test_est_csv:
            df = pd.read_csv(test_est_csv[0])
            df["model"] = md.name
            dfs.append(df)
            
    if dfs:
        df_cross = pd.concat(dfs, ignore_index=True)
        cross_csv = results_root / "final_cross_model_comparison_test_only.csv"
        df_cross.to_csv(cross_csv, index=False)
        print(f"📊 Saved cross-model comparison: {cross_csv.name}")
        return cross_csv
    return None


def export_reproducibility_zip(results_root: Path, zip_name: str = "decisive_validation_rev03_results_final.zip") -> Path:
    """Zips all files in results directory for one-click Kaggle download."""
    import zipfile
    zip_path = results_root / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_root):
            for file in files:
                p = Path(root) / file
                if p != zip_path:
                    arcname = p.relative_to(results_root)
                    zipf.write(p, arcname)
    print(f"📦 Successfully created reproducibility archive: {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    return zip_path
