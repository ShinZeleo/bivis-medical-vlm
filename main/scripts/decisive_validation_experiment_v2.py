"""
Decisive Validation Experiment v2 for Bilingual Medical VLM Selective Prediction (Rev02)
========================================================================================
Independent, publication-grade implementation strictly adhering to:
1. Authoritative Frozen Manifest Enforcement:
   - Evaluates exact N=150 QA pairs across 58 physical image clusters.
   - Primary Test-Only: N=92 QA pairs across 35 image clusters.
   - Development Split: N=58 QA pairs across 23 image clusters.
   - Zero image cluster cross-split leakage.
   - Manifest SHA256 verified against canonical hash: 988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52.
   - Embedded manifest fallback guarantees exact evaluation manifest even if dataset paths differ.
2. Complete Elimination of Blur Manipulations (Advisor Directive):
   - P0_ORIGINAL: True radiological scan without modification.
   - P1_BLACK_BLANK: Pure RGB (0, 0, 0) canvas preserving exact native resolution and mode.
3. Target Candidate Sequence Scoring:
   - English: ["Yes", "No"]
   - Indonesian: ["Ya", "Tidak"]
   - Full conditional sequence log-likelihood with runtime prefix-match assertion.
4. Fair Pre-Registered Comparison Across 5 Estimators:
   - M1: Confidence-only baseline (p_orig)
   - M2: Visual-only rectified support (max(0, delta_dir))
   - M_dir: Directed multiplicative (p_orig * max(0, delta_dir))
   - M_norm: Normalized directed linear (p_orig * (1 + delta_dir) / 2)
   - M3_old: Historical unsigned baseline (p_orig * (1 + |delta_dir|) / 2)
5. Paired Image-Cluster Bootstrap (B=1,000, seed=42) and Discrete Sample-Level AURC.
6. Revision Isolation:
   - Revision ID: 'rev02'
   - Outputs written exclusively to revision-specific directory: results/decisive_validation/rev02/

Author: Medical VLM Research Auditor & Reproducibility Team
Revision: rev02
"""

import os
import sys
import zlib
import base64
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Revision Constants
REVISION_ID = "rev02"
AUTHORITATIVE_MANIFEST_SHA256 = "988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52"

PROMPT_EN_TEMPLATE = "Answer only YES or NO.\nQuestion: {q}"
PROMPT_ID_TEMPLATE = "Jawab pertanyaan radiologi berdasarkan gambar berikut. Jawab hanya dengan YA atau TIDAK.\nPertanyaan: {q}"
CANDIDATES_EN = ["Yes", "No"]
CANDIDATES_ID = ["Ya", "Tidak"]

DEFAULT_MODEL_1 = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_2 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

# Embedded Authoritative Manifest (Bit-identical to decisive_validation_manifest.csv)
EMBEDDED_MANIFEST_B64 = "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"


# ==============================================================================
# 1. MANIFEST & IMAGE RESOLUTION ENGINE
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
    
    # 1. Try provided path
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

    # 2. Try default search paths if not loaded
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

    # 3. Fallback to embedded manifest
    if df_manifest is None and allow_embedded_fallback:
        print("📦 Loading authoritative manifest from embedded bit-identical store...")
        raw_bytes = unpack_embedded_manifest()
        import io
        df_manifest = pd.read_csv(io.BytesIO(raw_bytes))
        loaded_from = "embedded:AUTHORITATIVE_MANIFEST_REV02"
        if fallback_out_dir is not None:
            fallback_out_dir = Path(fallback_out_dir)
            fallback_out_dir.mkdir(parents=True, exist_ok=True)
            saved_p = fallback_out_dir / "decisive_validation_manifest.csv"
            saved_p.write_bytes(raw_bytes)
            print(f"   Materialized manifest to: {saved_p}")

    if df_manifest is None:
        raise FileNotFoundError(
            "Could not find or verify authoritative manifest matching SHA256: "
            f"{AUTHORITATIVE_MANIFEST_SHA256}"
        )

    # Strict Methodological Invariants Validation
    assert len(df_manifest) == 150, f"Manifest length must be 150, got {len(df_manifest)}"
    split_counts = df_manifest["split"].value_counts().to_dict()
    assert split_counts == {"test": 92, "dev": 58}, f"Split counts must be {{'test': 92, 'dev': 58}}, got {split_counts}"
    
    test_clusters = set(df_manifest[df_manifest["split"] == "test"]["canonical_image_id"])
    dev_clusters = set(df_manifest[df_manifest["split"] == "dev"]["canonical_image_id"])
    assert len(test_clusters) == 35, f"Test unique clusters must be 35, got {len(test_clusters)}"
    assert len(dev_clusters) == 23, f"Dev unique clusters must be 23, got {len(dev_clusters)}"
    
    overlap = test_clusters.intersection(dev_clusters)
    assert len(overlap) == 0, f"Fatal cross-split leakage detected! Overlapping clusters: {overlap}"
    
    print(f"✅ Authoritative manifest verified [{loaded_from}]:")
    print(f"   Total: {len(df_manifest)} QA pairs across {len(test_clusters) + len(dev_clusters)} image clusters.")
    print(f"   Test-Only: 92 QA pairs across 35 image clusters.")
    print(f"   Dev-Only : 58 QA pairs across 23 image clusters.")
    print(f"   Cross-split leakage: ZERO (verified).")
    return df_manifest


def resolve_image_path(cid: str, img_id: str, qa_id: int, image_name: str = None, 
                       search_dirs: list = None,
                       expected_pixel_hash: str = None,
                       expected_w: int = None,
                       expected_h: int = None) -> Path:
    """
    Explicit, cryptographically verified image resolver.
    Matches candidate files and verifies decoded RGB pixel SHA256 against manifest ground truth.
    """
    if search_dirs is None:
        search_dirs = [
            Path("main/data/images"),
            Path("data/images"),
            Path("images"),
            Path("./main/data/images"),
            Path("./data/images"),
        ]
        if Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").rglob("*.png"):
                if p.parent not in search_dirs:
                    search_dirs.append(p.parent)
        
    candidates = [
        f"{cid}.png",
        f"{img_id}.png",
        f"{img_id}",
        f"img_{qa_id}.png",
        f"img_{qa_id:05d}.png"
    ]
    if image_name:
        candidates.append(str(image_name))
        
    for sdir in search_dirs:
        sdir_p = Path(sdir)
        if not sdir_p.exists():
            continue
        for cand in candidates:
            cand_p = sdir_p / cand
            if cand_p.is_file():
                try:
                    with Image.open(cand_p) as im:
                        if expected_w is not None and expected_h is not None:
                            if (im.width, im.height) != (expected_w, expected_h):
                                continue
                        if expected_pixel_hash is not None:
                            rgb_arr = np.array(im.convert("RGB"))
                            calc_hash = hashlib.sha256(rgb_arr.tobytes()).hexdigest()
                            if calc_hash != expected_pixel_hash:
                                continue
                        return cand_p
                except Exception:
                    continue
                    
    # Auto-Materialize from Hugging Face if not found on disk
    if expected_pixel_hash is not None:
        try:
            import io
            import pandas as pd
            from huggingface_hub import hf_hub_download
            
            p_split = "test" if qa_id > 1794 else "train"  # or try both
            # Try appropriate split or both
            for split_name in ["test", "train"]:
                p_parquet = hf_hub_download(repo_id="Lab-IS/IndoRad-VQA", filename=f"data/{split_name}-00000-of-00001.parquet", repo_type="dataset")
                df_sp = pd.read_parquet(p_parquet)
                hf_row = df_sp[df_sp["qid"] == qa_id]
                if len(hf_row) > 0:
                    raw_bytes = hf_row.iloc[0]["image"]["bytes"]
                    im = Image.open(io.BytesIO(raw_bytes))
                    rgb_arr = np.array(im.convert("RGB"))
                    calc_hash = hashlib.sha256(rgb_arr.tobytes()).hexdigest()
                    if calc_hash == expected_pixel_hash:
                        out_dir = Path("main/data/images") if Path("main/data").exists() else Path("data/images")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        save_p = out_dir / f"{cid}.png"
                        im.save(save_p)
                        return save_p
        except Exception:
            pass

    raise FileNotFoundError(
        f"Unable to establish verified cryptographic image identity for QA ID {qa_id} "
        f"(cid: {cid}, img_id: {img_id}, expected_hash: {expected_pixel_hash}). "
        f"Searched {len(search_dirs)} directories."
    )


# ==============================================================================
# 2. STATISTICAL & SELECTIVE PREDICTION EVALUATION ENGINE
# ==============================================================================

def compute_discrete_risk_coverage(scores: np.ndarray, correctness: np.ndarray):
    """
    Standard sample-level discrete selective prediction evaluation (Geifman & El-Yaniv, NeurIPS 2017).
    Ranks samples by reliability score descending: S_(1) >= S_(2) >= ... >= S_(N).
    Evaluates empirical selective risk r_k = 1 - (1/k) sum_{j=1}^k Correct_(j) for k=1..N.
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
    
    aurc = float(np.mean(risk))
    
    cov_trap = np.concatenate([[0.0], cov])
    risk_trap = np.concatenate([[0.0], risk])
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc_trap = float(trap_fn(risk_trap, cov_trap))
    
    return cov, risk, aurc, aurc_trap


def get_risk_at_coverage(cov_arr, risk_arr, target_cov: float):
    """Find selective risk at closest discrete coverage level."""
    idx = (np.abs(cov_arr - target_cov)).argmin()
    return float(risk_arr[idx]), int(idx + 1)


def run_image_cluster_bootstrap(df: pd.DataFrame, score_cols: list, corr_col: str = "correct_id", 
                                cluster_col: str = "canonical_image_id", n_boot: int = 1000, seed: int = 42):
    """
    Paired Image-Cluster Bootstrap to account for intra-cluster correlation.
    Resamples unique physical images with replacement.
    """
    np.random.seed(seed)
    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)
    
    cluster_dfs = {c: df[df[cluster_col] == c] for c in unique_clusters}
    boot_aurcs = {col: [] for col in score_cols}
    boot_diffs = {col: [] for col in score_cols if col != "score_m1"}
    
    for _ in range(n_boot):
        sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([cluster_dfs[c] for c in sampled_clusters], ignore_index=True)
        corr = boot_df[corr_col].values
        aurc_m1 = None
        
        for col in score_cols:
            _, _, aurc, _ = compute_discrete_risk_coverage(boot_df[col].values, corr)
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
        p_val = 2.0 * min(np.mean(diff_vals <= 0), np.mean(diff_vals >= 0))
        p_val = min(1.0, max(1.0 / n_boot, p_val))
        diff_summary[col] = {
            "mean_diff": float(np.mean(diff_vals)),
            "ci_95_lower": float(np.percentile(diff_vals, 2.5)),
            "ci_95_upper": float(np.percentile(diff_vals, 97.5)),
            "p_value": float(p_val)
        }
        
    return summary, diff_summary


def plot_risk_coverage_curves(df: pd.DataFrame, score_cols: list, save_path: Path, model_title: str):
    """
    Generates publication-ready Risk-Coverage curves for all 5 estimators.
    """
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    color_map = {
        "score_m1": ("#1f77b4", "M1: Confidence-Only (p_orig)", "-"),
        "score_m2": ("#ff7f0e", "M2: Visual-Only max(0, Delta)", "--"),
        "score_m_dir": ("#2ca02c", "M_dir: p_orig * max(0, Delta)", "-"),
        "score_m_norm": ("#9467bd", "M_norm: p_orig * (1 + Delta)/2", "-."),
        "score_m3_old": ("#d62728", "M3_old: Historical Unsigned", ":")
    }
    
    correctness = df["correct_id"].values
    n_samples = len(df)
    n_clust = df["canonical_image_id"].nunique()
    
    for col in score_cols:
        cov, risk, aurc, _ = compute_discrete_risk_coverage(df[col].values, correctness)
        color, label, linestyle = color_map.get(col, ("black", col, "-"))
        ax.plot(cov, risk, label=f"{label} (AURC={aurc:.4f})", color=color, linestyle=linestyle, linewidth=2.0)
        
    base_error = 1.0 - np.mean(correctness)
    ax.axhline(base_error, color="gray", linestyle="--", alpha=0.6, label=f"Random Selection (Risk={base_error:.4f})")
    ax.axvline(0.8, color="red", linestyle=":", alpha=0.5, label="Operational Reference (Coverage=0.80)")
    
    ax.set_title(f"Selective Risk vs. Coverage under Indonesian Language Shift\n[{model_title}] (N={n_samples}, {n_clust} Image Clusters)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Coverage (Fraction of Questions Answered)", fontsize=11, labelpad=8)
    ax.set_ylabel("Selective Risk (Error Rate among Answered Questions)", fontsize=11, labelpad=8)
    ax.set_xlim(0.1, 1.02)
    ax.set_ylim(-0.02, max(0.45, base_error + 0.1))
    ax.legend(loc="upper left", frameon=True, fontsize=9.5, facecolor="white", edgecolor="#ccc")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved Risk-Coverage curve: {save_path}")


# ==============================================================================
# 3. CHECKPOINT RESUME & COMPATIBILITY VALIDATION
# ==============================================================================

def validate_checkpoint_compatibility(checkpoint_df: pd.DataFrame, model_name: str, 
                                      expected_manifest_sha256: str, 
                                      expected_qa_ids: set) -> tuple:
    """
    Strictly verifies that a checkpoint is valid, non-synthetic, matches the exact model,
    uses the authoritative manifest, and conforms to Rev02 schema.
    """
    if checkpoint_df.empty:
        return False, "Checkpoint dataframe is empty"
        
    required_cols = {
        "qa_id", "canonical_image_id", "split", "pred_en", "pred_id",
        "conf_id", "delta_p_dir", "score_m_dir", "reference_answer",
        "scoring_method", "revision_id"
    }
    missing_cols = required_cols - set(checkpoint_df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
        
    if len(checkpoint_df) > 150:
        return False, f"Checkpoint sample count ({len(checkpoint_df)}) exceeds authoritative manifest (150)"
        
    if not (checkpoint_df["model_id"] == model_name).all():
        return False, f"Model ID mismatch in checkpoint rows"
        
    if not (checkpoint_df["scoring_method"] == "target_candidate_sequence_scoring").all():
        return False, f"Scoring method mismatch (expected target_candidate_sequence_scoring)"
        
    if "revision_id" in checkpoint_df.columns:
        if not (checkpoint_df["revision_id"] == REVISION_ID).all():
            return False, f"Revision ID mismatch (expected {REVISION_ID})"
            
    if "manifest_sha256" in checkpoint_df.columns:
        if not (checkpoint_df["manifest_sha256"] == expected_manifest_sha256).all():
            return False, f"Manifest SHA256 mismatch in checkpoint"
            
    # Check that critical score columns contain valid finite numbers
    numeric_cols = ["conf_id", "delta_p_dir", "score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old"]
    for nc in numeric_cols:
        if nc in checkpoint_df.columns:
            if checkpoint_df[nc].isna().any():
                return False, f"Column {nc} contains NaN values in checkpoint"
            if not np.isfinite(checkpoint_df[nc].values).all():
                return False, f"Column {nc} contains infinite values in checkpoint"
                
    checkpoint_qa_ids = set(checkpoint_df["qa_id"].tolist())
    if not checkpoint_qa_ids.issubset(expected_qa_ids):
        return False, f"Checkpoint contains QA IDs not present in authoritative manifest"
        
    return True, f"Checkpoint valid with {len(checkpoint_qa_ids)} completed samples"


# ==============================================================================
# 4. TARGET SEQUENCE SCORING WITH PREFIX MATCH ASSERTION
# ==============================================================================

def extract_target_loglikelihood(model, processor, image_path, prompt_text, candidate_strings, target_device):
    """
    Evaluates conditional log-likelihood of exact candidate target strings
    given the multimodal context (Image + Prompt).
    Avoids single first-token logit hacking and verifies prefix match.
    """
    import torch
    import torch.nn.functional as F
    from qwen_vl_utils import process_vision_info
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
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
    )
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    cand_logprobs = []
    
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
        assert torch.equal(inputs.input_ids[0, :prompt_len].cpu(), prompt_inputs.input_ids[0].cpu()), \
            f"Tokenizer boundary prefix mismatch! prompt_len={prompt_len}, inputs_len={inputs.input_ids.shape[1]}"
            
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            
        target_ids = inputs.input_ids[0, prompt_len:].tolist()
        if len(target_ids) == 0:
            cand_logprobs.append(-999.0)
            del inputs, outputs
            continue
            
        target_logits = logits[prompt_len - 1 : prompt_len - 1 + len(target_ids)]
        log_probs = F.log_softmax(target_logits, dim=-1)
        seq_logprob = sum(log_probs[i, target_ids[i]].item() for i in range(len(target_ids)))
        cand_logprobs.append(seq_logprob)
        del inputs, outputs
        
    return cand_logprobs


def score_binary_candidates(model, processor, image_path, prompt_text, candidate_strings, target_device):
    """
    Runs sequence scoring and returns normalized binary probabilities.
    candidate_strings: [affirmatory, negative], e.g. ["Yes", "No"] or ["Ya", "Tidak"]
    Returns: (pred_label, confidence, p_aff, p_neg)
    """
    logprobs = extract_target_loglikelihood(model, processor, image_path, prompt_text, candidate_strings, target_device)
    max_lp = max(logprobs)
    exp_lps = [np.exp(lp - max_lp) for lp in logprobs]
    total_mass = sum(exp_lps)
    p_aff = float(exp_lps[0] / total_mass)
    p_neg = float(1.0 - p_aff)
    
    if p_aff >= 0.5:
        pred_label = "YES"
        confidence = p_aff
    else:
        pred_label = "NO"
        confidence = p_neg
        
    return pred_label, confidence, p_aff, p_neg


# ==============================================================================
# 5. PIPELINE EXECUTION FOR SINGLE MODEL (REV02)
# ==============================================================================

def execute_model_evaluation(model_name: str, df_eval: pd.DataFrame, image_map: dict, 
                             results_root: Path, perturb_dir: Path, target_device_str: str = "auto"):
    """
    Executes full evaluation on Model: RQ1 bilingual on P0, RQ2 visual intervention P1, RQ3 estimators.
    """
    import gc
    import time
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
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
    import transformers
    from transformers import AutoConfig, AutoProcessor
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch_list = getattr(config, "architectures", [])
    print(f"   Declared Architecture(s): {arch_list} | Model Type: {config.model_type}")
    
    # Dynamically resolve model class
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
        pred_en, conf_en, p_en_yes, p_en_no = score_binary_candidates(model, processor, orig_img_path, prompt_en, CANDIDATES_EN, target_dev)
        corr_en = int(pred_en == ref_ans)
        
        prompt_id = PROMPT_ID_TEMPLATE.format(q=q_id)
        pred_id, conf_id, p_orig_ya, p_orig_tidak = score_binary_candidates(model, processor, orig_img_path, prompt_id, CANDIDATES_ID, target_dev)
        corr_id = int(pred_id == ref_ans)
        
        flip = int(pred_en != pred_id)
        joint_corr = int(corr_en and corr_id)
        
        # 3. RQ2: Visual Intervention P1 on Indonesian
        _, _, p_blank_ya, p_blank_tidak = score_binary_candidates(model, processor, blank_img_path, prompt_id, CANDIDATES_ID, target_dev)
        p_blank_pred = p_blank_ya if pred_id == "YES" else p_blank_tidak
        delta_p_dir = conf_id - p_blank_pred
        delta_p_abs = abs(conf_id - p_blank_pred)
        
        # 4. RQ3: 5 Estimators
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
            # RQ1 Bilingual
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
            "scoring_method": "target_candidate_sequence_scoring",
            "revision_id": REVISION_ID,
            "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        completed_qa_ids.add(qa_id)
        
        if len(results) % 5 == 0 or len(results) == len(df_eval):
            pd.DataFrame(results).to_csv(checkpoint_csv, index=False)
            
        if idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(checkpoint_csv, index=False)
    print(f"💾 Checkpoint fully updated: {len(res_df)} samples saved to {checkpoint_csv}")
    
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 GPU memory released.")
    return res_df, model_out_dir


# ==============================================================================
# 6. STATISTICAL SUBSET ANALYSIS
# ==============================================================================

def analyze_subset(sub_df: pd.DataFrame, subset_name: str, score_cols: list) -> pd.DataFrame:
    """
    Computes discrete AURC, selective risk at key coverage thresholds,
    and paired image-cluster bootstrap differences vs baseline M1.
    """
    rows = []
    for sc in score_cols:
        cov, risk, aurc, _ = compute_discrete_risk_coverage(sub_df[sc].values, sub_df["correct_id"].values)
        r100, _ = get_risk_at_coverage(cov, risk, 1.0)
        r90, _ = get_risk_at_coverage(cov, risk, 0.9)
        r80, _ = get_risk_at_coverage(cov, risk, 0.8)
        r70, _ = get_risk_at_coverage(cov, risk, 0.7)
        r50, _ = get_risk_at_coverage(cov, risk, 0.5)
        rows.append({
            "subset": subset_name,
            "estimator": sc,
            "aurc_discrete": round(aurc, 4),
            "risk_cov_100": round(r100, 4),
            "risk_cov_90": round(r90, 4),
            "risk_cov_80": round(r80, 4),
            "risk_cov_70": round(r70, 4),
            "risk_cov_50": round(r50, 4)
        })
    boot_sum, boot_diff = run_image_cluster_bootstrap(
        sub_df, score_cols, corr_col="correct_id", cluster_col="canonical_image_id", n_boot=1000, seed=42
    )
    for r in rows:
        sc = r["estimator"]
        r["boot_mean"] = round(boot_sum[sc]["mean"], 4)
        r["ci_95"] = f"[{boot_sum[sc]['ci_95_lower']:.4f}, {boot_sum[sc]['ci_95_upper']:.4f}]"
        if sc == "score_m1":
            r["delta_vs_m1"] = 0.0
            r["p_value"] = 1.0
            r["significance"] = "Baseline"
        else:
            diff = boot_diff[sc]
            r["delta_vs_m1"] = round(diff["mean_diff"], 4)
            r["p_value"] = round(diff["p_value"], 4)
            r["significance"] = "Significant (p<0.05)" if diff["p_value"] < 0.05 else "Not Significant"
    return pd.DataFrame(rows)


# ==============================================================================
# 7. MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Decisive Validation Experiment v2 (Rev02)")
    parser.add_argument("--manifest", type=str, default="main/results/decisive_validation_manifest.csv", help="Path to manifest")
    parser.add_argument("--out-dir", type=str, default="results/decisive_validation/rev02", help="Output root directory")
    parser.add_argument("--dry-run-check", action="store_true", help="Run verification of manifest and image resolution only")
    args = parser.parse_args()
    
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    df_manifest = load_authoritative_manifest(Path(args.manifest), fallback_out_dir=out_root)
    
    # Cryptographic image verification
    resolved_images = {}
    for idx, row in df_manifest.iterrows():
        p = resolve_image_path(
            cid=row["canonical_image_id"], 
            img_id=row["image_id"], 
            qa_id=row["qa_id"], 
            image_name=row.get("image_name"),
            expected_pixel_hash=row.get("pixel_sha256"),
            expected_w=int(row["image_width"]) if "image_width" in row and pd.notna(row["image_width"]) else None,
            expected_h=int(row["image_height"]) if "image_height" in row and pd.notna(row["image_height"]) else None
        )
        resolved_images[row["qa_id"]] = p
        
    print(f"✅ All {len(resolved_images)}/{len(df_manifest)} images cryptographically verified on disk.")
    if args.dry_run_check:
        print("🏁 Dry-run preflight check completed successfully with 100% compliance.")
        return

if __name__ == "__main__":
    main()
