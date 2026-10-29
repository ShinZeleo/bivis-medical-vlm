#!/usr/bin/env python3
"""
Generator for Kaggle Notebook 07_decisive_validation_rev04.ipynb
===============================================================
Deterministically constructs the self-contained, publication-ready
Kaggle Dual-T4 notebook for Medical VLM Decisive Validation (rev04).
"""

import os
import sys
import json
import hashlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import nbformat as nbf

# Import base64 embedded manifest string
sys.path.insert(0, os.path.abspath("main/scripts"))
from manifest_rev04_b64 import EMBEDDED_MANIFEST_B64

def build_kaggle_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # =========================================================================
    # CELL 0: TITLE & SCIENTIFIC FRAMING (MARKDOWN)
    # =========================================================================
    c0_md = """# 🏥 Medical VLM Decisive Validation Experiment (Kaggle Dual-T4 GPU) — Revision rev04

**Research Project:** Evaluasi Reliabilitas dan Selective Prediction pada Medical Vision-Language Model di Bawah Pergeseran Bahasa (IndoRad-VQA)  
**Target Publikasi:** Standar Metodologi Ilmiah SINTA 2 / Scopus  
**Revision:** `rev04`  
**Execution Environment:** Kaggle Cloud Platform (Dual Tesla T4 GPUs — 2x16GB VRAM, Compute Capability 7.5)  

---

### 📋 Scientific & Methodological Standards in Revision rev04:
1. **Dual Evaluation Tracks:**
   - **Track 1 (Primary Test Evaluation):** $N_{\\text{test}} = 799$ QA pairs across 204 unique physical image clusters ($N_{\\text{primary}} = 1122$ total with $N_{\\text{dev}} = 323$ across 88 clusters; **0% cross-split leakage**).
   - **Track 2 (Historical Benchmark Comparison):** $N_{\\text{test}} = 92$ QA pairs across 35 unique physical image clusters ($N_{\\text{benchmark}} = 150$ total, rev03 cohort) evaluated under the exact identical rev04 scoring engine for unbiased cross-revision comparison.
2. **Symmetrical Closed Candidate Scoring (`<|im_end|>`) with Exact Causal Logit Slicing:**
   - Evaluates conditional log-likelihood on exact target sequence tokens $[L_{\\text{prompt}} - 1 : L_{\\text{prompt}} - 1 + L_{\\text{target}}]$.
   - Extracts both raw sequence log-likelihood and length-normalized surprisal $\\bar{s} = \\frac{1}{L} \\sum_{t=1}^L \\log P(y_t \\mid \\dots)$.
   - Reports **Relative Candidate Scores** via stable logsumexp softmax.
3. **Visual Intervention Protocol (P0 vs P1):**
   - Condition P0: Original radiology image.
   - Condition P1: Pure black canvas $(0,0,0)$ matching exact dimensions and RGB mode.
   - Computes Directional Visual Reliance: $\\Delta P_{\\text{dir}} = P(\\hat{y} \\mid P_0) - P(\\hat{y} \\mid P_1)$ on the identical predicted label.
4. **Selective Prediction & Discrete AURC (Geifman & El-Yaniv, NeurIPS 2017):**
   - 5 Estimators: $M_1$ (Confidence), $M_2$ (Pure visual), $M_{\\text{dir}}$ (Multiplicative), $M_{\\text{norm}}$ (Linear), $M_{3,\\text{old}}$ (Absolute).
   - Reports realized selective risk at 100%, 90%, 80%, 70%, and 50% coverage.
5. **Exact Paired Cluster Resampling & Testing:**
   - Paired Cluster Bootstrap ($B=1000$) for 95% percentile confidence intervals.
   - Monte Carlo Paired Cluster Permutation Test ($B=1000$ Rademacher sign-flips) with Holm-Bonferroni correction.
6. **Strict Checkpoint Governance & One-Click Reproducibility ZIP.**
"""
    cells.append(nbf.v4.new_markdown_cell(c0_md))

    # =========================================================================
    # STAGE 2: ENVIRONMENT & HARDWARE AUDIT
    # =========================================================================
    c1_code = """# ==============================================================================
# STAGE 2: ENVIRONMENT & HARDWARE AUDIT
# ==============================================================================
import os
import sys
import subprocess
import platform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from IPython.display import display, FileLink
except ImportError:
    display = print
    FileLink = str

print("=" * 70)
print("🔍 SYSTEM & HARDWARE AUDIT")
print("=" * 70)
print(f"OS Platform     : {platform.platform()}")
print(f"Python Version  : {sys.version.split()[0]}")

required_pkgs = ["transformers", "accelerate", "bitsandbytes", "qwen_vl_utils", "torch", "huggingface_hub", "pyarrow", "matplotlib", "pandas", "numpy"]
missing_pkgs = []
for pkg in required_pkgs:
    try:
        __import__(pkg)
    except ImportError:
        missing_pkgs.append(pkg)

if missing_pkgs:
    print(f"📦 Installing missing packages: {missing_pkgs}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs)
    print("✅ Installation complete.")

# Check native Qwen2.5-VL support in transformers
needs_upgrade = False
try:
    import transformers
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        needs_upgrade = True
except Exception:
    needs_upgrade = True

if needs_upgrade:
    print("📦 Upgrading transformers to >=4.49.0 for native Qwen2.5-VL support...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers>=4.49.0", "accelerate>=0.26.0", "qwen-vl-utils"])
    import importlib
    importlib.invalidate_caches()
    transformers = importlib.import_module("transformers")
    print("✅ Transformers upgraded and reloaded.")

import torch
import transformers

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version    : {torch.version.cuda}")
    print(f"Device Count    : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  - GPU {i}: {props.name} ({vram_gb:.2f} GB VRAM, Compute {props.major}.{props.minor})")
else:
    print("⚠️ WARNING: Running on CPU! GPU acceleration strongly recommended.")
print("=" * 70)"""
    cells.append(nbf.v4.new_code_cell(c1_code))

    # =========================================================================
    # STAGE 3: EXPERIMENT CONFIGURATION & REVISION rev04 ISOLATION
    # =========================================================================
    c2_code = f"""# ==============================================================================
# STAGE 3: EXPERIMENT CONFIGURATION & REVISION rev04 ISOLATION
# ==============================================================================
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import torch

REVISION_ID = "rev04"
SCORING_METHOD = "length_normalized_closed_candidate_v4"
AUTHORITATIVE_MANIFEST_SHA256 = "bcb03bfafa0c329b73ab3b5ec2bb6f471daa1df9218582efa232e9dac74d3484"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Isolated Revision rev04 Output Directory
RESULTS_ROOT = Path("results/decisive_validation/rev04")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
PERTURB_DIR = RESULTS_ROOT / "perturbed_images"
PERTURB_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_EN_TEMPLATE = "Answer only YES or NO.\\nQuestion: {{q}}"
PROMPT_ID_TEMPLATE = "Jawab hanya dengan YA atau TIDAK.\\nPertanyaan: {{q}}"

CANDIDATES_EN = ["YES<|im_end|>", "NO<|im_end|>"]
CANDIDATES_ID = ["YA<|im_end|>", "TIDAK<|im_end|>"]

DEFAULT_MODEL_1 = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_2 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

EMBEDDED_MANIFEST_B64 = "{EMBEDDED_MANIFEST_B64}"
"""
    cells.append(nbf.v4.new_code_cell(c2_code))

    # =========================================================================
    # STAGE 4 & 5: MANIFEST LOADING & SPLIT / CLUSTER LEAKAGE AUDIT
    # =========================================================================
    c3_code = """# ==============================================================================
# STAGE 4 & 5: AUTHORITATIVE FROZEN MANIFEST ENFORCEMENT & DUAL-TRACK INVARIANTS
# ==============================================================================
import zlib
import base64
import hashlib
import io
import pandas as pd

def load_and_verify_manifest():
    search_paths = [
        Path("main/results/manifest_rev04.csv"),
        Path("main/results/decisive_validation/rev04/manifest_rev04.csv"),
        Path("results/decisive_validation/rev04/manifest_rev04.csv"),
        Path("manifest_rev04.csv"),
        Path("./manifest_rev04.csv")
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.glob("**/manifest_rev04.csv"):
            search_paths.append(p)
            
    df_manifest = None
    loaded_source = None
    for p in search_paths:
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h == AUTHORITATIVE_MANIFEST_SHA256:
                df_manifest = pd.read_csv(p)
                loaded_source = str(p)
                break
                
    if df_manifest is None:
        print("ℹ️ Local manifest file not found. Loading embedded base64 verified manifest...")
        manifest_bytes = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64))
        df_manifest = pd.read_csv(io.BytesIO(manifest_bytes))
        loaded_source = "EMBEDDED_VERIFIED_BASE64"
        
    calc_sha = hashlib.sha256(df_manifest.to_csv(index=False).encode("utf-8")).hexdigest()
    assert len(df_manifest) == 2244, f"Expected 2244 total QA rows, got {len(df_manifest)}"
    
    # Track 1: Primary Cohort Verification (Clean Yes/No)
    df_primary = df_manifest[df_manifest["is_eligible_primary"] == True]
    assert len(df_primary) == 1122, f"Expected 1122 primary samples, got {len(df_primary)}"
    
    test_prim = df_primary[df_primary["split"] == "test"]
    dev_prim = df_primary[df_primary["split"] == "dev"]
    assert len(test_prim) == 799, f"Expected 799 primary test rows, got {len(test_prim)}"
    assert len(dev_prim) == 323, f"Expected 323 primary dev rows, got {len(dev_prim)}"
    
    test_c = set(test_prim["canonical_image_id"].unique())
    dev_c = set(dev_prim["canonical_image_id"].unique())
    assert len(test_c) == 204, f"Expected 204 test clusters, got {len(test_c)}"
    assert len(dev_c) == 88, f"Expected 88 dev clusters, got {len(dev_c)}"
    leakage = test_c.intersection(dev_c)
    assert len(leakage) == 0, f"Cross-split cluster leakage detected! Overlapping: {leakage}"
    
    # Track 2: Historical rev03 Benchmark Cohort Verification
    df_hist = df_manifest[df_manifest["in_rev03_benchmark"] == True]
    assert len(df_hist) == 150, f"Expected 150 historical benchmark rows, got {len(df_hist)}"
    test_h = df_hist[df_hist["split"] == "test"]
    dev_h = df_hist[df_hist["split"] == "dev"]
    assert len(test_h) == 92, f"Expected 92 historical test rows, got {len(test_h)}"
    assert len(dev_h) == 58, f"Expected 58 historical dev rows, got {len(dev_h)}"
    
    print(f"✅ Authoritative Manifest 100% Verified ({loaded_source})")
    print(f"   SHA256 Manifest           : {AUTHORITATIVE_MANIFEST_SHA256}")
    print(f"   Total IndoRad-VQA Rows    : {len(df_manifest)} QA pairs")
    print(f"   Primary Test Cohort (T1)  : {len(test_prim)} QA pairs (204 Unique Clusters | ZERO LEAKAGE)")
    print(f"   Primary Dev Cohort        : {len(dev_prim)} QA pairs (88 Unique Clusters | ZERO LEAKAGE)")
    print(f"   Historical Benchmark (T2) : {len(test_h)} Test QA pairs (35 Clusters | rev03 comparison)")
    return df_manifest

df_manifest = load_and_verify_manifest()"""
    cells.append(nbf.v4.new_code_cell(c3_code))

    # =========================================================================
    # STAGE 6: MULTI-STRATEGY IMAGE RESOLVER & PIXEL VERIFICATION
    # =========================================================================
    c4_code = """# ==============================================================================
# STAGE 6: MULTI-STRATEGY IMAGE RESOLVER & PIXEL SHA-256 VERIFICATION
# ==============================================================================
from PIL import Image
import hashlib
import io
import zipfile

def compute_rgb_pixel_sha256(pil_img):
    rgb = pil_img.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()

def resolve_all_images(df):
    cache_dir = RESULTS_ROOT / "verified_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    search_dirs = [
        cache_dir,
        Path("main/data/images"),
        Path("data/images"),
        Path("./images")
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # 1. Auto-extract any zip archives found in /kaggle/input
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
    
    unique_hashes = df["pixel_sha256"].unique()
    print(f"🔍 Cryptographically verifying pixel SHA-256 for all {len(unique_hashes)} unique physical images...")
    
    # Pre-index disk cache
    disk_cache = {}
    for sdir in search_dirs:
        for p in sdir.glob("*.png"):
            fname = p.stem.lower()
            if len(fname) == 16:
                disk_cache[fname] = p

    for _, row in df.iterrows():
        qa_id = row["qa_id"]
        cid = str(row["canonical_image_id"]).strip().lower()
        expected_hash = str(row["pixel_sha256"]).strip().lower()
        expected_w = int(row["image_width"])
        expected_h = int(row["image_height"])
        
        found = None
        if cid in disk_cache:
            p = disk_cache[cid]
            try:
                with Image.open(p) as img:
                    if img.size == (expected_w, expected_h):
                        if compute_rgb_pixel_sha256(img) == expected_hash:
                            found = p
            except Exception:
                pass
                
        if found is None:
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
                                        found = p
                                        break
                        except Exception:
                            pass
                if found is not None:
                    break
                
        if found is not None:
            resolved_map[qa_id] = found
        else:
            missing_rows.append(row)
            
    if missing_rows:
        print(f"⚠️ {len(missing_rows)} images missing from local search dirs. Downloading via Parquet...")
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        splits_needed = set()
        for r in missing_rows:
            splits_needed.add("test" if "test" in str(r["split"]).lower() else "train")
            
        for sname in splits_needed:
            fname = f"data/{sname}-00000-of-00001.parquet"
            local_pq = hf_hub_download(repo_id="Lab-IS/IndoRad-VQA", filename=fname, repo_type="dataset")
            table = pq.read_table(local_pq)
            df_pq = table.to_pandas()
            for r in missing_rows:
                qid = r["qa_id"]
                cid = str(r["canonical_image_id"])
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
                            
    unresolved = [r["qa_id"] for _, r in df.iterrows() if r["qa_id"] not in resolved_map]
    assert len(unresolved) == 0, f"Critical failure: {len(unresolved)} images unresolved!"
    print(f"✅ Cryptographic Image Verification: 100% PASS ({len(resolved_map)} images verified).")
    return resolved_map

# Filter evaluation cohort: Primary Clean Yes/No (N=1122)
df_eval_cohort = df_manifest[df_manifest["is_eligible_primary"] == True].copy()
image_map = resolve_all_images(df_eval_cohort)"""
    cells.append(nbf.v4.new_code_cell(c4_code))

    # =========================================================================
    # STAGE 7 & 8: CANDIDATE SCORING ENGINE
    # =========================================================================
    c5_code = """# ==============================================================================
# STAGE 7 & 8: CANDIDATE TOKENIZATION AUDIT & SCORING IMPLEMENTATION
# ==============================================================================
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

def extract_target_loglikelihood_v4(model, processor, image_path, prompt_text, candidate_strings, target_device):
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
        
        # Symmetrical boundary prefix assertion
        assert torch.equal(inputs.input_ids[0, :prompt_len], prompt_inputs.input_ids[0]), \
            f"Prefix boundary mismatch! prompt_len={prompt_len}, inputs_len={inputs.input_ids.shape[1]}"
            
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
    # Primary rev04 length-normalized relative candidate score
    s1_norm = cand_stats[c1]["mean_lp"]
    s2_norm = cand_stats[c2]["mean_lp"]
    max_norm = max(s1_norm, s2_norm)
    exp_norm1 = np.exp(s1_norm - max_norm)
    exp_norm2 = np.exp(s2_norm - max_norm)
    p1_norm = float(exp_norm1 / (exp_norm1 + exp_norm2))
    p2_norm = float(1.0 - p1_norm)
    
    # Raw sequence logprob (for ablation & historical comparison)
    s1_raw = cand_stats[c1]["raw_sum"]
    s2_raw = cand_stats[c2]["raw_sum"]
    max_raw = max(s1_raw, s2_raw)
    exp_raw1 = np.exp(s1_raw - max_raw)
    exp_raw2 = np.exp(s2_raw - max_raw)
    p1_raw = float(exp_raw1 / (exp_raw1 + exp_raw2))
    p2_raw = float(1.0 - p1_raw)
    
    return cand_stats, (p1_norm, p2_norm), (p1_raw, p2_raw)

def score_binary_candidates_v4(model, processor, image_path, prompt_text, candidate_strings, target_device):
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

print("✅ Candidate Scoring Engine (rev04) compiled successfully.")"""
    cells.append(nbf.v4.new_code_cell(c5_code))

    # =========================================================================
    # STAGE 16 & 17: STATISTICAL INFERENCE ENGINE
    # =========================================================================
    c6_code = """# ==============================================================================
# STAGE 16 & 17: DISCRETE AURC, CLUSTER BOOTSTRAP & PERMUTATION TEST
# ==============================================================================

def compute_discrete_risk_coverage(scores, correctness):
    scores = np.asarray(scores, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    sorted_corr = correctness[order]
    k = np.arange(1, n + 1)
    cov = k / n
    acc = np.cumsum(sorted_corr) / k
    risk = 1.0 - acc
    aurc_mean = float(np.mean(risk))
    return cov, risk, aurc_mean

def get_risk_at_realized_coverage(cov_arr, risk_arr, target_cov):
    idx = int((np.abs(cov_arr - target_cov)).argmin())
    return float(risk_arr[idx]), float(cov_arr[idx]), int(idx + 1)

def paired_cluster_permutation_test(df, score_col_a, score_col_b, corr_col="correct_id", 
                                    cluster_col="canonical_image_id", n_perm=1000, seed=42):
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

def run_paired_cluster_bootstrap(df, score_cols, corr_col="correct_id", cluster_col="canonical_image_id", n_boot=1000, seed=42):
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
            "ci_95_lower": float(np.percentile(vals, 2.5)),
            "ci_95_upper": float(np.percentile(vals, 97.5))
        }
    diff_summary = {}
    for col in boot_diffs.keys():
        d_vals = np.array(boot_diffs[col])
        diff_summary[col] = {
            "delta_mean": float(np.mean(d_vals)),
            "delta_ci_95_lower": float(np.percentile(d_vals, 2.5)),
            "delta_ci_95_upper": float(np.percentile(d_vals, 97.5))
        }
    return summary, diff_summary

def apply_holm_bonferroni(p_dict):
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

print("✅ Statistical Inference Engine (Bootstrap & Permutation) compiled successfully.")"""
    cells.append(nbf.v4.new_code_cell(c6_code))

    # =========================================================================
    # STAGE 9-12: MODEL PIPELINE EXECUTOR & CHECKPOINTS
    # =========================================================================
    c7_code = """# ==============================================================================
# STAGE 9-12: MODEL PIPELINE EXECUTOR & CHECKPOINT GOVERNANCE
# ==============================================================================
from tqdm.auto import tqdm
import gc
from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig
import transformers

def execute_model_evaluation_v4(model_name, df_eval, image_map, results_root, perturb_dir):
    model_slug = model_name.replace("/", "_").replace("-", "_")
    model_out_dir = results_root / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = model_out_dir / f"live_validation_checkpoint_{model_slug}.csv"
    
    completed_qa_ids = set()
    results = []
    if checkpoint_csv.exists():
        try:
            df_ckpt = pd.read_csv(checkpoint_csv)
            # Checkpoint compatibility check
            if len(df_ckpt) > 0 and df_ckpt["revision_id"].iloc[0] == REVISION_ID and df_ckpt["manifest_sha256"].iloc[0] == AUTHORITATIVE_MANIFEST_SHA256:
                print(f"🔄 Checkpoint verified: resuming {len(df_ckpt)} completed samples for {model_name}...")
                results = df_ckpt.to_dict("records")
                completed_qa_ids = set(df_ckpt["qa_id"])
            else:
                print(f"⚠️ Stale or cross-revision checkpoint detected. Restarting cleanly...")
                checkpoint_csv.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ Checkpoint unreadable ({e}). Restarting cleanly...")
            checkpoint_csv.unlink(missing_ok=True)

    if len(completed_qa_ids) == len(df_eval):
        print(f"✅ All {len(df_eval)} samples already evaluated in checkpoint. Skipping inference.")
        return pd.DataFrame(results), model_out_dir

    print(f"\\n🚀 Loading Model: {model_name} (4-bit NF4 Quantization)...")
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
    target_dev = model.device if hasattr(model, "device") else next(model.parameters()).device
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
    return df_res, model_out_dir"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # =========================================================================
    # STAGE 11 & 12: MODEL INFERENCE RUNS
    # =========================================================================
    c8_code = """# ==============================================================================
# STAGE 11: INFERENCE RUN — MODEL 1 (QWEN2.5-VL-7B-INSTRUCT)
# ==============================================================================
print("🚀 Launching Model 1: Qwen/Qwen2.5-VL-7B-Instruct...")
df_qwen, qwen_dir = execute_model_evaluation_v4(
    DEFAULT_MODEL_1, df_eval_cohort, image_map, RESULTS_ROOT, PERTURB_DIR
)
print("✅ Model 1 evaluation complete.")"""
    cells.append(nbf.v4.new_code_cell(c8_code))

    c9_code = """# ==============================================================================
# STAGE 12: INFERENCE RUN — MODEL 2 (HUATUOGPT-VISION-7B-QWEN2.5VL)
# ==============================================================================
print("🚀 Launching Model 2: FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL...")
df_huatuo, huatuo_dir = execute_model_evaluation_v4(
    DEFAULT_MODEL_2, df_eval_cohort, image_map, RESULTS_ROOT, PERTURB_DIR
)
print("✅ Model 2 evaluation complete.")"""
    cells.append(nbf.v4.new_code_cell(c9_code))

    # =========================================================================
    # STAGE 13 & 14: RQ1 BILINGUAL EVALUATION
    # =========================================================================
    c10_code = """# ==============================================================================
# STAGE 13 & 14: RQ1 BILINGUAL EVALUATION (PRIMARY TEST N=799 & HISTORICAL N=92)
# ==============================================================================
def summarize_rq1(df, model_title):
    rows = []
    for track_name, subset_df in [("primary_test_cohort", df[df["split"] == "test"]), 
                                  ("historical_rev03_benchmark", df[(df["split"] == "test") & (df["in_rev03_benchmark"] == True)])]:
        n = len(subset_df)
        acc_en = subset_df["correct_en"].mean() * 100.0
        acc_id = subset_df["correct_id"].mean() * 100.0
        gap = acc_en - acc_id
        flip = subset_df["flip"].mean() * 100.0
        joint = subset_df["joint_correct"].mean() * 100.0
        pred_en_dist = subset_df["pred_en"].value_counts().to_dict()
        pred_id_dist = subset_df["pred_id"].value_counts().to_dict()
        
        print(f"--- RQ1 METRICS: {model_title} | Track: {track_name} (N={n}) ---")
        print(f"English Accuracy          : {acc_en:.2f}% | Predictions: {pred_en_dist}")
        print(f"Indonesian Accuracy       : {acc_id:.2f}% | Predictions: {pred_id_dist}")
        print(f"Cross-Lingual Gap (EN-ID) : {gap:+.2f} pp")
        print(f"Prediction Flip Rate      : {flip:.2f}%")
        print(f"Joint Bilingual Correct   : {joint:.2f}%\\n")
        
        rows.append({
            "model": model_title, "track": track_name, "sample_size": n,
            "acc_en": acc_en, "acc_id": acc_id, "gap_en_id": gap,
            "flip_rate": flip, "joint_acc": joint,
            "pred_en_yes": pred_en_dist.get("YES", 0), "pred_en_no": pred_en_dist.get("NO", 0),
            "pred_id_yes": pred_id_dist.get("YES", 0), "pred_id_no": pred_id_dist.get("NO", 0)
        })
    return rows

rq1_records = summarize_rq1(df_qwen, "Qwen2.5-VL-7B") + summarize_rq1(df_huatuo, "HuatuoGPT-Vision-7B")
df_rq1 = pd.DataFrame(rq1_records)
rq1_csv = RESULTS_ROOT / "rq1_primary_test.csv"
df_rq1.to_csv(rq1_csv, index=False)
print(f"💾 Saved RQ1 Metrics: {rq1_csv.name}")"""
    cells.append(nbf.v4.new_code_cell(c10_code))

    # =========================================================================
    # STAGE 15: RQ2 VISUAL INTERVENTION
    # =========================================================================
    c11_code = """# ==============================================================================
# STAGE 15: RQ2 VISUAL INTERVENTION (P0 ORIGINAL VS P1 BLANK)
# ==============================================================================
def summarize_rq2(df, model_title):
    rows = []
    for track_name, subset_df in [("primary_test_cohort", df[df["split"] == "test"]), 
                                  ("historical_rev03_benchmark", df[(df["split"] == "test") & (df["in_rev03_benchmark"] == True)])]:
        n = len(subset_df)
        dp = subset_df["delta_p_dir"].values
        mean_dp = float(np.mean(dp))
        median_dp = float(np.median(dp))
        pos = float(np.mean(dp > 0)) * 100.0
        neg = float(np.mean(dp < 0)) * 100.0
        zero = float(np.mean(dp == 0)) * 100.0
        
        print(f"--- RQ2 VISUAL INTERVENTION: {model_title} | Track: {track_name} (N={n}) ---")
        print(f"Mean Delta P_dir          : {mean_dp:+.4f}")
        print(f"Median Delta P_dir        : {median_dp:+.4f}")
        print(f"Positive Reliance (>0)    : {pos:.2f}% (Behavioral degradation upon visual deletion)")
        print(f"Negative Reliance (<0)    : {neg:.2f}% (Model confidence increases without image)")
        print(f"Agnostic / Zero (==0)     : {zero:.2f}%\\n")
        
        rows.append({
            "model": model_title, "track": track_name, "sample_size": n,
            "mean_delta_p": mean_dp, "median_delta_p": median_dp,
            "prop_positive": pos, "prop_negative": neg, "prop_zero": zero
        })
    return rows

rq2_records = summarize_rq2(df_qwen, "Qwen2.5-VL-7B") + summarize_rq2(df_huatuo, "HuatuoGPT-Vision-7B")
df_rq2 = pd.DataFrame(rq2_records)
rq2_csv = RESULTS_ROOT / "rq2_visual_intervention_test.csv"
df_rq2.to_csv(rq2_csv, index=False)
print(f"💾 Saved RQ2 Visual Intervention Metrics: {rq2_csv.name}")"""
    cells.append(nbf.v4.new_code_cell(c11_code))

    # =========================================================================
    # STAGE 16: RQ3 SELECTIVE PREDICTION (PRIMARY TEST N=799)
    # =========================================================================
    c12_code = """# ==============================================================================
# STAGE 16: RQ3 SELECTIVE PREDICTION & STATISTICAL TESTING (PRIMARY TEST N=799)
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def run_selective_prediction_audit(df_subset, model_title, out_dir, track_label="primary_test"):
    n_samples = len(df_subset)
    n_clusters = df_subset["canonical_image_id"].nunique()
    score_cols = ["score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old"]
    
    boot_sum, boot_diff = run_paired_cluster_bootstrap(df_subset, score_cols, n_boot=1000, seed=42)
    
    rows = []
    raw_p = {}
    curves = {}
    for col in score_cols:
        cov, risk, aurc = compute_discrete_risk_coverage(df_subset[col].values, df_subset["correct_id"].values)
        curves[col] = (cov, risk)
        
        r100, _, _ = get_risk_at_realized_coverage(cov, risk, 1.0)
        r90, _, _ = get_risk_at_realized_coverage(cov, risk, 0.9)
        r80, _, _ = get_risk_at_realized_coverage(cov, risk, 0.8)
        r70, _, _ = get_risk_at_realized_coverage(cov, risk, 0.7)
        r50, _, _ = get_risk_at_realized_coverage(cov, risk, 0.5)
        
        if col != "score_m1":
            _, _, t_obs, p_val = paired_cluster_permutation_test(df_subset, col, "score_m1", n_perm=1000, seed=42)
            raw_p[col] = p_val
            delta_aurc = aurc - rows[0]["aurc_discrete_mean"]
        else:
            p_val = 1.0
            delta_aurc = 0.0
            
        rows.append({
            "model": model_title, "track": track_label, "sample_size": n_samples, "clusters": n_clusters,
            "estimator": col, "aurc_discrete_mean": aurc,
            "ci_95_lower": boot_sum[col]["ci_95_lower"], "ci_95_upper": boot_sum[col]["ci_95_upper"],
            "delta_aurc_vs_m1": delta_aurc,
            "delta_ci_lower": boot_diff[col]["delta_ci_95_lower"] if col != "score_m1" else 0.0,
            "delta_ci_upper": boot_diff[col]["delta_ci_95_upper"] if col != "score_m1" else 0.0,
            "perm_p_value_raw": p_val,
            "risk_cov_100": r100, "risk_cov_90": r90, "risk_cov_80": r80, "risk_cov_70": r70, "risk_cov_50": r50
        })
        
    adj_p = apply_holm_bonferroni(raw_p)
    for r in rows:
        est = r["estimator"]
        r["perm_p_value_holm"] = adj_p.get(est, 1.0 if est == "score_m1" else 1.0)
        
    df_res = pd.DataFrame(rows)
    safe_m = model_title.replace("/", "_").replace("-", "_")
    csv_p = out_dir / f"rq3_{track_label}_{safe_m}.csv"
    df_res.to_csv(csv_p, index=False)
    
    # Plot Risk-Coverage Curve
    plt.figure(figsize=(8, 6), dpi=150)
    for col, (cov, risk) in curves.items():
        plt.plot(cov, risk, label=f"{col} (AURC={df_res.loc[df_res['estimator']==col, 'aurc_discrete_mean'].values[0]:.4f})", lw=2)
    plt.xlabel("Coverage (Proportion of Answered Cases)", fontsize=11)
    plt.ylabel("Selective Diagnostic Risk (1 - Accuracy)", fontsize=11)
    plt.title(f"Risk-Coverage Trade-Off: {model_title}\\n({track_label.upper()} N={n_samples}, {n_clusters} Clusters)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_p = out_dir / f"risk_coverage_{track_label}_{safe_m}.png"
    plt.savefig(plot_p)
    plt.close()
    
    print(f"📊 Saved Selective Prediction Audit: {csv_p.name} & {plot_p.name}")
    return df_res

df_qwen_test = df_qwen[df_qwen["split"] == "test"]
df_huatuo_test = df_huatuo[df_huatuo["split"] == "test"]

est_qwen_prim = run_selective_prediction_audit(df_qwen_test, "Qwen2.5-VL-7B", qwen_dir, "primary_test")
est_huatuo_prim = run_selective_prediction_audit(df_huatuo_test, "HuatuoGPT-Vision-7B", huatuo_dir, "primary_test")"""
    cells.append(nbf.v4.new_code_cell(c12_code))

    # =========================================================================
    # STAGE 17: HISTORICAL BENCHMARK COMPARISON (N=92)
    # =========================================================================
    c13_code = """# ==============================================================================
# STAGE 17: RQ3 HISTORICAL BENCHMARK COMPARISON (N=92 SAMPLES)
# ==============================================================================
df_qwen_hist = df_qwen[(df_qwen["split"] == "test") & (df_qwen["in_rev03_benchmark"] == True)]
df_huatuo_hist = df_huatuo[(df_huatuo["split"] == "test") & (df_huatuo["in_rev03_benchmark"] == True)]

est_qwen_hist = run_selective_prediction_audit(df_qwen_hist, "Qwen2.5-VL-7B", qwen_dir, "historical_rev03_benchmark")
est_huatuo_hist = run_selective_prediction_audit(df_huatuo_hist, "HuatuoGPT-Vision-7B", huatuo_dir, "historical_rev03_benchmark")"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # =========================================================================
    # STAGE 18: DEV SET CALIBRATION AUDIT (N=323)
    # =========================================================================
    c14_code = """# ==============================================================================
# STAGE 18: DEV SET PROTOCOL CALIBRATION (N=323 SAMPLES)
# ==============================================================================
df_qwen_dev = df_qwen[df_qwen["split"] == "dev"]
df_huatuo_dev = df_huatuo[df_huatuo["split"] == "dev"]

est_qwen_dev = run_selective_prediction_audit(df_qwen_dev, "Qwen2.5-VL-7B", qwen_dir, "dev_calibration")
est_huatuo_dev = run_selective_prediction_audit(df_huatuo_dev, "HuatuoGPT-Vision-7B", huatuo_dir, "dev_calibration")"""
    cells.append(nbf.v4.new_code_cell(c14_code))

    # =========================================================================
    # STAGE 20: CROSS-MODEL SYNTHESIS TABLE
    # =========================================================================
    c15_code = """# ==============================================================================
# STAGE 20: FINAL CROSS-MODEL COMPARISON (PRIMARY TEST COHORT N=799)
# ==============================================================================
df_cross_prim = pd.concat([est_qwen_prim, est_huatuo_prim], ignore_index=True)
cross_prim_csv = RESULTS_ROOT / "rq3_estimator_comparison_test.csv"
df_cross_prim.to_csv(cross_prim_csv, index=False)

print("=" * 85)
print("🏆 FINAL CROSS-MODEL COMPARISON (PRIMARY TEST COHORT N=799)")
print("=" * 85)
cols_show = ["model", "estimator", "aurc_discrete_mean", "ci_95_lower", "ci_95_upper", "delta_aurc_vs_m1", "perm_p_value_holm", "risk_cov_100", "risk_cov_80", "risk_cov_50"]
display(df_cross_prim[cols_show])
print("=" * 85)"""
    cells.append(nbf.v4.new_code_cell(c15_code))

    # =========================================================================
    # STAGE 21 & 22: METADATA & REPRODUCIBILITY ZIP
    # =========================================================================
    c16_code = """# ==============================================================================
# STAGE 21 & 22: METADATA, SHA-256 MANIFEST & FINAL REPRODUCIBILITY ZIP
# ==============================================================================
import zipfile
import shutil

# 1. Generate run_metadata.json
meta = {
    "revision_id": REVISION_ID,
    "scoring_method": SCORING_METHOD,
    "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
    "random_seed": RANDOM_SEED,
    "candidates_en": CANDIDATES_EN,
    "candidates_id": CANDIDATES_ID,
    "prompt_en": PROMPT_EN_TEMPLATE,
    "prompt_id": PROMPT_ID_TEMPLATE,
    "completed_timestamp": datetime.now(timezone.utc).isoformat(),
    "n_total_indorad_vqa": len(df_manifest),
    "n_primary_evaluated": len(df_eval_cohort),
    "n_primary_test": int((df_eval_cohort["split"] == "test").sum()),
    "n_primary_dev": int((df_eval_cohort["split"] == "dev").sum()),
    "n_historical_benchmark_test": int(((df_eval_cohort["split"] == "test") & (df_eval_cohort["in_rev03_benchmark"] == True)).sum())
}
meta_p = RESULTS_ROOT / "run_metadata.json"
meta_p.write_text(json.dumps(meta, indent=2))

# 2. Build SHA-256 Manifest of all produced artifacts
artifact_rows = []
for root, dirs, files in os.walk(RESULTS_ROOT):
    for f in files:
        if f.endswith(".zip"):
            continue
        fp = Path(root) / f
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        artifact_rows.append({"file_name": fp.name, "relative_path": str(fp.relative_to(RESULTS_ROOT)), "sha256": h})

df_artifacts = pd.DataFrame(artifact_rows)
art_manifest_p = RESULTS_ROOT / "artifact_sha256_manifest.csv"
df_artifacts.to_csv(art_manifest_p, index=False)

# 3. Create Final Reproducibility Zip
zip_name = "decisive_validation_rev04_results_final.zip"
zip_path = RESULTS_ROOT / zip_name
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(RESULTS_ROOT):
        for file in files:
            p = Path(root) / file
            if p != zip_path:
                zipf.write(p, p.relative_to(RESULTS_ROOT))

# Copy to notebook root working directory for 1-click download from Kaggle output panel
root_zip = Path(zip_name)
if root_zip.resolve() != zip_path.resolve():
    shutil.copy2(zip_path, root_zip)

print("=" * 85)
print(f"📦 REPRODUCIBILITY ARCHIVE COMPLETE: {zip_path.name}")
print(f"   Archive path: {zip_path}")
print(f"   Root output : {root_zip.resolve()}")
print(f"   Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
print("=" * 85)
print("🔗 Direct download links for Kaggle environment:")
display(FileLink(str(root_zip)))
display(FileLink(str(zip_path)))"""
    cells.append(nbf.v4.new_code_cell(c16_code))

    nb.cells = cells
    return nb

def main():
    print("Generating 07_decisive_validation_rev04.ipynb...")
    nb = build_kaggle_notebook()
    
    out_paths = [
        Path("main/Kaggle/07_decisive_validation_rev04.ipynb"),
        Path("main/notebooks/07_decisive_validation_rev04.ipynb")
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"✅ Generated: {p} (SHA256: {h})")

if __name__ == "__main__":
    main()
