#!/usr/bin/env python3
"""
Generator for Kaggle Notebook 08_decisive_validation_rev05.ipynb
===============================================================
Deterministically constructs the self-contained, publication-ready
Kaggle Dual-T4 notebook for Medical VLM Decisive Validation (rev05).

Enhancements in rev05:
1. Exact Sequence Log-Likelihood Primary Scoring with Length-Normalized Sensitivity Track
2. Union Cohort Evaluation (N=1,133) covering 100% of Primary (1,122) and Historical (150)
3. Raw-Byte Decompression Manifest Verification (fb55fd75...)
4. Strict Atomic Checkpointing (.tmp -> os.replace)
5. Comprehensive Discrete AURC, Cluster Bootstrap & Permutation Testing
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
from manifest_rev05_b64 import EMBEDDED_MANIFEST_B64

def build_kaggle_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # =========================================================================
    # CELL 0: TITLE & SCIENTIFIC FRAMING (MARKDOWN)
    # =========================================================================
    c0_md = """# 🏥 Medical VLM Decisive Validation Experiment (Kaggle Dual-T4 GPU) — Revision rev05

**Research Project:** Selective Prediction for Bilingual Medical Vision-Language Models Under Language Shift: Mitigating Overconfidence via Multimodal Sensitivity Analysis (IndoRad-VQA)  
**Target Publikasi:** Standar Metodologi Ilmiah SINTA 2 / Scopus  
**Revision:** `rev05`  
**Execution Environment:** Kaggle Cloud Platform (Dual Tesla T4 GPUs — 2x16GB VRAM, Compute Capability 7.5)  

---

### 📋 Scientific & Methodological Enhancements in Revision rev05:
1. **Foundational Exact Sequence Log-Likelihood Scoring:**
   - Evaluates exact causal continuation log-likelihood: $\\sum_{t=1}^L \\log P(y_t \\mid y_{<t}, X)$ on exact causal logits $[L_{\\text{prompt}} - 1 : L_{\\text{prompt}} - 1 + L_{\\text{target}}]$.
   - Computes **Candidate-Normalized Sequence Score** via stable logsumexp softmax over closed candidate space.
   - Retains **Length-Normalized Surprisal** ($\\frac{1}{L} \\sum \\log P$) as an explicit sensitivity analysis track.
2. **Unified Evaluation Cohort ($N_{\\text{union}} = 1,133$):**
   - Evaluates the union of Primary Cohort ($N=1,122$, 799 test / 323 dev) and Historical Benchmark ($N=150$, 92 test / 58 dev).
   - Guarantees **0 dropped samples** in historical comparisons and **0% cross-split cluster leakage** across 314 physical images.
3. **Cryptographic Image Pixel Verification (314 Canonical Images):**
   - Verifies RGB pixel byte SHA-256 for every image before inference, with automatic fallback retrieval.
4. **Visual Intervention Protocol (P0 vs P1):**
   - Condition P0: Original radiology image.
   - Condition P1: Pure black canvas $(0,0,0)$ matching exact dimensions and RGB mode.
   - Directional Visual Reliance: $\\Delta P_{\\text{dir}} = P(\\hat{y} \\mid P_0) - P(\\hat{y} \\mid P_1)$ on the identical predicted label.
5. **Selective Prediction & Discrete Empirical AURC (Geifman & El-Yaniv, NeurIPS 2017):**
   - 5 Estimators: $M_1$ (Confidence), $M_2$ (Pure visual reliance), $M_{\\text{dir}}$ (Multiplicative), $M_{\\text{norm}}$ (Linear), $M_{3,\\text{old}}$ (Absolute).
   - Discrete empirical trapezoidal AURC and realized risk at 100%, 90%, 80%, 70%, 50% coverage.
6. **Paired Cluster Bootstrap & Permutation Testing ($B=1,000$):**
   - Cluster-level Rademacher sign-flip permutation tests under $H_0: \\mathbb{E}[\\Delta \\text{AURC}] = 0$ with Holm-Bonferroni FWER control.
7. **Atomic Checkpointing & One-Click Reproducibility ZIP.**
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
    # STAGE 3: EXPERIMENT CONFIGURATION & REVISION rev05 ISOLATION
    # =========================================================================
    c2_code = f"""# ==============================================================================
# STAGE 3: EXPERIMENT CONFIGURATION & REVISION rev05 ISOLATION
# ==============================================================================
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import torch

REVISION_ID = "rev05"
SCORING_METHOD = "exact_sequence_candidate_normalized_v5"
AUTHORITATIVE_MANIFEST_SHA256 = "fb55fd75378b1c03abcbeb9772fa27d6ab81dd9abdfca4735c03438507c19ee9"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Isolated Revision rev05 Output Directory
RESULTS_ROOT = Path("results/decisive_validation/rev05")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
PERTURB_DIR = RESULTS_ROOT / "perturbed_images"
PERTURB_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_EN_TEMPLATE = "Answer only YES or NO.\\nQuestion: {{q}}"
PROMPT_ID_TEMPLATE = "Jawab hanya dengan YA atau TIDAK.\\nPertanyaan: {{q}}"

CANDIDATES_EN = ["YES<|im_end|>", "NO<|im_end|>"]
CANDIDATES_ID = ["YA<|im_end|>", "TIDAK<|im_end|>"]

LABEL_MAP = {{
    "YES<|im_end|>": "YES",
    "NO<|im_end|>": "NO",
    "YA<|im_end|>": "YES",
    "TIDAK<|im_end|>": "NO"
}}

DEFAULT_MODEL_1 = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_2 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

EMBEDDED_MANIFEST_B64 = "{EMBEDDED_MANIFEST_B64}"
"""
    cells.append(nbf.v4.new_code_cell(c2_code))

    # =========================================================================
    # STAGE 4 & 5: MANIFEST LOADING & DUAL-TRACK COHORT INVARIANTS
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
        Path("main/results/manifest_rev05.csv"),
        Path("main/results/decisive_validation/rev05/manifest_rev05.csv"),
        Path("results/decisive_validation/rev05/manifest_rev05.csv"),
        Path("manifest_rev05.csv"),
        Path("./manifest_rev05.csv")
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.glob("**/manifest_rev05.csv"):
            search_paths.append(p)
            
    df_manifest = None
    loaded_source = None
    for p in search_paths:
        if p.exists():
            raw_bytes = p.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() == AUTHORITATIVE_MANIFEST_SHA256:
                df_manifest = pd.read_csv(io.BytesIO(raw_bytes))
                loaded_source = str(p)
                break
                
    if df_manifest is None:
        print("ℹ️ Local manifest file not found. Decompressing embedded base64 verified manifest...")
        decompressed_bytes = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64))
        calc_sha = hashlib.sha256(decompressed_bytes).hexdigest()
        assert calc_sha == AUTHORITATIVE_MANIFEST_SHA256, \
            f"Embedded manifest hash mismatch! Expected {AUTHORITATIVE_MANIFEST_SHA256}, got {calc_sha}"
        df_manifest = pd.read_csv(io.BytesIO(decompressed_bytes))
        loaded_source = "EMBEDDED_VERIFIED_BASE64"
        
    assert len(df_manifest) == 2244, f"Expected 2244 total QA rows, got {len(df_manifest)}"
    assert df_manifest["qid"].nunique() == 2244, "Duplicate QIDs detected!"
    assert df_manifest["canonical_image_id"].nunique() == 314, "Expected 314 canonical images!"
    
    # 1. Primary Cohort Verification (Clean Yes/No, N=1122)
    df_primary = df_manifest[df_manifest["is_eligible_primary"] == True]
    assert len(df_primary) == 1122, f"Expected 1122 primary samples, got {len(df_primary)}"
    test_prim = df_primary[df_primary["split"] == "test"]
    dev_prim = df_primary[df_primary["split"] == "dev"]
    assert len(test_prim) == 799, f"Expected 799 primary test rows, got {len(test_prim)}"
    assert len(dev_prim) == 323, f"Expected 323 primary dev rows, got {len(dev_prim)}"
    test_c_prim = set(test_prim["canonical_image_id"].unique())
    dev_c_prim = set(dev_prim["canonical_image_id"].unique())
    assert len(test_c_prim) == 204, f"Expected 204 primary test clusters, got {len(test_c_prim)}"
    assert len(dev_c_prim) == 88, f"Expected 88 primary dev clusters, got {len(dev_c_prim)}"
    assert len(test_c_prim.intersection(dev_c_prim)) == 0, "Primary cross-split leakage detected!"
    
    # 2. Historical rev03 Benchmark Cohort Verification (N=150)
    df_hist = df_manifest[df_manifest["in_rev03_benchmark"] == True]
    assert len(df_hist) == 150, f"Expected 150 historical benchmark rows, got {len(df_hist)}"
    test_h = df_hist[df_hist["split"] == "test"]
    dev_h = df_hist[df_hist["split"] == "dev"]
    assert len(test_h) == 92, f"Expected 92 historical test rows, got {len(test_h)}"
    assert len(dev_h) == 58, f"Expected 58 historical dev rows, got {len(dev_h)}"
    test_c_h = set(test_h["canonical_image_id"].unique())
    dev_c_h = set(dev_h["canonical_image_id"].unique())
    assert len(test_c_h) == 35, f"Expected 35 historical test clusters, got {len(test_c_h)}"
    assert len(dev_c_h) == 23, f"Expected 23 historical dev clusters, got {len(dev_c_h)}"
    assert len(test_c_h.intersection(dev_c_h)) == 0, "Historical cross-split leakage detected!"
    
    # 3. Active Evaluation Cohort Union (N=1133)
    df_union = df_manifest[(df_manifest["is_eligible_primary"] == True) | (df_manifest["in_rev03_benchmark"] == True)]
    assert len(df_union) == 1133, f"Expected 1133 union rows, got {len(df_union)}"
    test_u = df_union[df_union["split"] == "test"]
    dev_u = df_union[df_union["split"] == "dev"]
    assert len(test_u) == 806, f"Expected 806 union test rows, got {len(test_u)}"
    assert len(dev_u) == 327, f"Expected 327 union dev rows, got {len(dev_u)}"
    assert len(set(test_u["canonical_image_id"]).intersection(set(dev_u["canonical_image_id"]))) == 0, "Union cross-split leakage!"
    
    print(f"✅ Authoritative Manifest 100% Cryptographically Verified ({loaded_source})")
    print(f"   SHA256 Manifest           : {AUTHORITATIVE_MANIFEST_SHA256}")
    print(f"   Total IndoRad-VQA Rows    : {len(df_manifest)} QA pairs (314 Images)")
    print(f"   Active Evaluation Union   : {len(df_union)} QA pairs (806 Test / 327 Dev | ZERO LEAKAGE)")
    print(f"   Primary Test Cohort       : {len(test_prim)} QA pairs (204 Clusters | 0% Leakage)")
    print(f"   Primary Dev Cohort        : {len(dev_prim)} QA pairs (88 Clusters | 0% Leakage)")
    print(f"   Historical Benchmark Test : {len(test_h)} QA pairs (35 Clusters | rev03 Exact Comparison)")
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

# Active Evaluation Union Cohort: N=1133 (Primary N=1122 + Historical Benchmark N=150)
df_eval_cohort = df_manifest[(df_manifest["is_eligible_primary"] == True) | (df_manifest["in_rev03_benchmark"] == True)].copy()
image_map = resolve_all_images(df_eval_cohort)"""
    cells.append(nbf.v4.new_code_cell(c4_code))

    # =========================================================================
    # STAGE 7 & 8: CANDIDATE SCORING ENGINE (REV05 DUAL-TRACK)
    # =========================================================================
    c5_code = """# ==============================================================================
# STAGE 7 & 8: CANDIDATE TOKENIZATION AUDIT & REV05 DUAL-TRACK SCORING ENGINE
# ==============================================================================
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

def extract_target_loglikelihood_v5(model, processor, image_path, prompt_text, candidate_strings, target_device):
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
        
        # Strict prefix invariance assertion
        assert torch.equal(inputs.input_ids[0, :prompt_len], prompt_inputs.input_ids[0]), \
            f"Prefix boundary mismatch! prompt_len={prompt_len}, inputs_len={inputs.input_ids.shape[1]}"
            
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            
        target_ids = inputs.input_ids[0, prompt_len:].tolist()
        if len(target_ids) == 0:
            cand_stats[cand] = {"seq_logprob": -999.0, "mean_surprisal": -999.0, "target_ids": [], "token_lps": []}
            del inputs, outputs
            continue
            
        # Exact causal alignment: logits at index t predict token at index t + 1
        target_logits = logits[prompt_len - 1 : prompt_len - 1 + len(target_ids)]
        log_probs = F.log_softmax(target_logits, dim=-1)
        token_lps = [log_probs[i, target_ids[i]].item() for i in range(len(target_ids))]
        seq_logprob = float(sum(token_lps))
        mean_surprisal = float(seq_logprob / len(target_ids))
        
        cand_stats[cand] = {
            "seq_logprob": seq_logprob,
            "mean_surprisal": mean_surprisal,
            "target_ids": target_ids,
            "token_lps": token_lps
        }
        del inputs, outputs
        
    del prompt_inputs
    
    c1, c2 = candidate_strings[0], candidate_strings[1]
    
    # Track 1: Foundational Exact Sequence Candidate-Normalized Score
    s1_seq = cand_stats[c1]["seq_logprob"]
    s2_seq = cand_stats[c2]["seq_logprob"]
    max_seq = max(s1_seq, s2_seq)
    exp_seq1 = np.exp(s1_seq - max_seq)
    exp_seq2 = np.exp(s2_seq - max_seq)
    p1_seq = float(exp_seq1 / (exp_seq1 + exp_seq2))
    p2_seq = float(1.0 - p1_seq)
    
    # Track 2: Length-Normalized Surprisal Sensitivity Track
    s1_len = cand_stats[c1]["mean_surprisal"]
    s2_len = cand_stats[c2]["mean_surprisal"]
    max_len = max(s1_len, s2_len)
    exp_len1 = np.exp(s1_len - max_len)
    exp_len2 = np.exp(s2_len - max_len)
    p1_len = float(exp_len1 / (exp_len1 + exp_len2))
    p2_len = float(1.0 - p1_len)
    
    return cand_stats, (p1_seq, p2_seq), (p1_len, p2_len)

def score_binary_candidates_v5(model, processor, image_path, prompt_text, candidate_strings, target_device):
    cand_stats, (p_aff_seq, p_neg_seq), (p_aff_len, p_neg_len) = extract_target_loglikelihood_v5(
        model, processor, image_path, prompt_text, candidate_strings, target_device
    )
    # Sequence Track decision
    c_aff, c_neg = candidate_strings[0], candidate_strings[1]
    lbl_aff = LABEL_MAP[c_aff]
    lbl_neg = LABEL_MAP[c_neg]
    
    if p_aff_seq >= 0.5:
        pred_label_seq = lbl_aff
        conf_seq = p_aff_seq
    else:
        pred_label_seq = lbl_neg
        conf_seq = p_neg_seq
        
    # Length-Normalized Track decision
    if p_aff_len >= 0.5:
        pred_label_len = lbl_aff
        conf_len = p_aff_len
    else:
        pred_label_len = lbl_neg
        conf_len = p_neg_len
        
    return {
        "pred_seq": pred_label_seq, "conf_seq": conf_seq, "p_aff_seq": p_aff_seq, "p_neg_seq": p_neg_seq,
        "pred_len": pred_label_len, "conf_len": conf_len, "p_aff_len": p_aff_len, "p_neg_len": p_neg_len,
        "cand_stats": cand_stats
    }

print("✅ Candidate Scoring Engine (rev05 Dual-Track) compiled successfully.")"""
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

def paired_cluster_permutation_test(df, score_col_a, score_col_b, corr_col="correct_id_seq", 
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

def run_paired_cluster_bootstrap(df, score_cols, corr_col="correct_id_seq", cluster_col="canonical_image_id", n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)
    cluster_dfs = {c: df[df[cluster_col] == c] for c in unique_clusters}
    
    baseline_col = score_cols[0]
    boot_aurcs = {col: [] for col in score_cols}
    boot_diffs = {col: [] for col in score_cols if col != baseline_col}
    
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([cluster_dfs[c] for c in sampled_clusters], ignore_index=True)
        corr = boot_df[corr_col].values
        
        aurc_base = None
        for col in score_cols:
            _, _, aurc = compute_discrete_risk_coverage(boot_df[col].values, corr)
            boot_aurcs[col].append(aurc)
            if col == baseline_col:
                aurc_base = aurc
                
        for col in boot_diffs.keys():
            boot_diffs[col].append(boot_aurcs[col][-1] - aurc_base)
            
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
    # STAGE 9-12: MODEL PIPELINE EXECUTOR & CHECKPOINTS (ATOMIC REV05)
    # =========================================================================
    c7_code = """# ==============================================================================
# STAGE 9-12: MODEL PIPELINE EXECUTOR & STRICT ATOMIC CHECKPOINT GOVERNANCE
# ==============================================================================
from tqdm.auto import tqdm
import gc
from transformers import AutoProcessor, AutoConfig, BitsAndBytesConfig
import transformers

def execute_model_evaluation_v5(model_name, df_eval, image_map, results_root, perturb_dir):
    model_slug = model_name.replace("/", "_").replace("-", "_")
    model_out_dir = results_root / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = model_out_dir / f"live_validation_checkpoint_{model_slug}.csv"
    
    completed_qa_ids = set()
    results = []
    if checkpoint_csv.exists():
        try:
            df_ckpt = pd.read_csv(checkpoint_csv)
            # Strict Checkpoint Governance
            if (len(df_ckpt) > 0 and 
                df_ckpt["revision_id"].iloc[0] == REVISION_ID and 
                df_ckpt["manifest_sha256"].iloc[0] == AUTHORITATIVE_MANIFEST_SHA256 and
                np.isfinite(df_ckpt["score_m_dir_seq"]).all()):
                print(f"🔄 Checkpoint verified: resuming {len(df_ckpt)} completed samples for {model_name}...")
                results = df_ckpt.to_dict("records")
                completed_qa_ids = set(df_ckpt["qa_id"])
            else:
                print(f"⚠️ Stale or invalid checkpoint detected. Restarting cleanly...")
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
        
        # Condition P1: Pure Black Canvas
        blank_img_path = perturb_dir / f"blank_{cid}.png"
        if not blank_img_path.exists():
            with Image.open(orig_img_path) as orig_pil:
                blank_pil = Image.new("RGB", orig_pil.size, color=(0, 0, 0))
                blank_pil.save(blank_img_path)
                
        # 1. English Evaluation on P0 (Original Image)
        prompt_en = PROMPT_EN_TEMPLATE.format(q=q_en)
        res_en = score_binary_candidates_v5(model, processor, orig_img_path, prompt_en, CANDIDATES_EN, target_dev)
        
        # 2. Indonesian Evaluation on P0 (Original Image)
        prompt_id = PROMPT_ID_TEMPLATE.format(q=q_id)
        res_id_orig = score_binary_candidates_v5(model, processor, orig_img_path, prompt_id, CANDIDATES_ID, target_dev)
        
        # 3. Indonesian Evaluation on P1 (Blank Image)
        res_id_blank = score_binary_candidates_v5(model, processor, blank_img_path, prompt_id, CANDIDATES_ID, target_dev)
        
        # --- TRACK 1: SEQUENCE LOG-LIKELIHOOD CALCULATIONS ---
        corr_en_seq = int(res_en["pred_seq"] == ref_ans)
        corr_id_seq = int(res_id_orig["pred_seq"] == ref_ans)
        flip_seq = int(res_en["pred_seq"] != res_id_orig["pred_seq"])
        joint_corr_seq = int(corr_en_seq and corr_id_seq)
        
        p_blank_pred_seq = res_id_blank["p_aff_seq"] if res_id_orig["pred_seq"] == "YES" else res_id_blank["p_neg_seq"]
        delta_p_dir_seq = res_id_orig["conf_seq"] - p_blank_pred_seq
        delta_p_abs_seq = abs(res_id_orig["conf_seq"] - p_blank_pred_seq)
        
        score_m1_seq = res_id_orig["conf_seq"]
        score_m2_seq = max(0.0, delta_p_dir_seq)
        score_m_dir_seq = res_id_orig["conf_seq"] * max(0.0, delta_p_dir_seq)
        score_m_norm_seq = res_id_orig["conf_seq"] * (1.0 + delta_p_dir_seq) / 2.0
        score_m3_old_seq = res_id_orig["conf_seq"] * (1.0 + delta_p_abs_seq) / 2.0
        
        # --- TRACK 2: LENGTH-NORMALIZED SURPRISAL CALCULATIONS ---
        corr_en_len = int(res_en["pred_len"] == ref_ans)
        corr_id_len = int(res_id_orig["pred_len"] == ref_ans)
        flip_len = int(res_en["pred_len"] != res_id_orig["pred_len"])
        joint_corr_len = int(corr_en_len and corr_id_len)
        
        p_blank_pred_len = res_id_blank["p_aff_len"] if res_id_orig["pred_len"] == "YES" else res_id_blank["p_neg_len"]
        delta_p_dir_len = res_id_orig["conf_len"] - p_blank_pred_len
        delta_p_abs_len = abs(res_id_orig["conf_len"] - p_blank_pred_len)
        
        score_m1_len = res_id_orig["conf_len"]
        score_m2_len = max(0.0, delta_p_dir_len)
        score_m_dir_len = res_id_orig["conf_len"] * max(0.0, delta_p_dir_len)
        score_m_norm_len = res_id_orig["conf_len"] * (1.0 + delta_p_dir_len) / 2.0
        score_m3_old_len = res_id_orig["conf_len"] * (1.0 + delta_p_abs_len) / 2.0
        
        results.append({
            "qa_id": qa_id,
            "canonical_image_id": cid,
            "image_id": img_id,
            "split": split_val,
            "translation_status": trans_stat,
            "is_yes_no": bool(row.get("is_yes_no", True)),
            "in_rev03_benchmark": bool(row.get("in_rev03_benchmark", False)),
            "is_eligible_primary": bool(row.get("is_eligible_primary", False)),
            "model_id": model_name,
            "device": str(target_dev),
            "question_en": q_en,
            "question_id": q_id,
            "reference_answer": ref_ans,
            
            # Sequence Track Columns
            "pred_en_seq": res_en["pred_seq"],
            "conf_en_seq": res_en["conf_seq"],
            "correct_en_seq": corr_en_seq,
            "p_en_yes_seq": res_en["p_aff_seq"],
            "p_en_no_seq": res_en["p_neg_seq"],
            "pred_id_seq": res_id_orig["pred_seq"],
            "conf_id_seq": res_id_orig["conf_seq"],
            "correct_id_seq": corr_id_seq,
            "p_orig_ya_seq": res_id_orig["p_aff_seq"],
            "p_orig_tidak_seq": res_id_orig["p_neg_seq"],
            "flip_seq": flip_seq,
            "joint_correct_seq": joint_corr_seq,
            "p_blank_ya_seq": res_id_blank["p_aff_seq"],
            "p_blank_tidak_seq": res_id_blank["p_neg_seq"],
            "p_blank_pred_seq": p_blank_pred_seq,
            "delta_p_dir_seq": delta_p_dir_seq,
            "delta_p_abs_seq": delta_p_abs_seq,
            "score_m1_seq": score_m1_seq,
            "score_m2_seq": score_m2_seq,
            "score_m_dir_seq": score_m_dir_seq,
            "score_m_norm_seq": score_m_norm_seq,
            "score_m3_old_seq": score_m3_old_seq,
            
            # Length-Normalized Track Columns
            "pred_en_len": res_en["pred_len"],
            "conf_en_len": res_en["conf_len"],
            "correct_en_len": corr_en_len,
            "p_en_yes_len": res_en["p_aff_len"],
            "p_en_no_len": res_en["p_neg_len"],
            "pred_id_len": res_id_orig["pred_len"],
            "conf_id_len": res_id_orig["conf_len"],
            "correct_id_len": corr_id_len,
            "p_orig_ya_len": res_id_orig["p_aff_len"],
            "p_orig_tidak_len": res_id_orig["p_neg_len"],
            "flip_len": flip_len,
            "joint_correct_len": joint_corr_len,
            "p_blank_ya_len": res_id_blank["p_aff_len"],
            "p_blank_tidak_len": res_id_blank["p_neg_len"],
            "p_blank_pred_len": p_blank_pred_len,
            "delta_p_dir_len": delta_p_dir_len,
            "delta_p_abs_len": delta_p_abs_len,
            "score_m1_len": score_m1_len,
            "score_m2_len": score_m2_len,
            "score_m_dir_len": score_m_dir_len,
            "score_m_norm_len": score_m_norm_len,
            "score_m3_old_len": score_m3_old_len,
            
            "blank_method": "PIL.Image.new('RGB', size, (0,0,0))",
            "scoring_method": SCORING_METHOD,
            "revision_id": REVISION_ID,
            "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        completed_qa_ids.add(qa_id)
        
        # Atomic Checkpoint Writing via Temporary File
        if len(results) % 10 == 0 or len(results) == len(df_eval):
            tmp_ckpt = checkpoint_csv.with_suffix(".csv.tmp")
            pd.DataFrame(results).to_csv(tmp_ckpt, index=False)
            os.replace(tmp_ckpt, checkpoint_csv)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    df_res = pd.DataFrame(results)
    tmp_ckpt = checkpoint_csv.with_suffix(".csv.tmp")
    df_res.to_csv(tmp_ckpt, index=False)
    os.replace(tmp_ckpt, checkpoint_csv)
    print(f"💾 Checkpoint saved atomically: {checkpoint_csv.name} ({len(df_res)} rows)")
    return df_res, model_out_dir"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # =========================================================================
    # STAGE 11 & 12: MODEL INFERENCE RUNS
    # =========================================================================
    c8_code = """# ==============================================================================
# STAGE 11: INFERENCE RUN — MODEL 1 (QWEN2.5-VL-7B-INSTRUCT)
# ==============================================================================
print("🚀 Launching Model 1: Qwen/Qwen2.5-VL-7B-Instruct...")
df_qwen, qwen_dir = execute_model_evaluation_v5(
    DEFAULT_MODEL_1, df_eval_cohort, image_map, RESULTS_ROOT, PERTURB_DIR
)
print("✅ Model 1 evaluation complete.")"""
    cells.append(nbf.v4.new_code_cell(c8_code))

    c9_code = """# ==============================================================================
# STAGE 12: INFERENCE RUN — MODEL 2 (HUATUOGPT-VISION-7B-QWEN2.5VL)
# ==============================================================================
print("🚀 Launching Model 2: FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL...")
df_huatuo, huatuo_dir = execute_model_evaluation_v5(
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
    # Evaluate Primary Test (N=799) and Historical Benchmark (N=92)
    cohorts = [
        ("primary_test_cohort", df[(df["split"] == "test") & (df["is_eligible_primary"] == True)]),
        ("historical_rev03_benchmark", df[(df["split"] == "test") & (df["in_rev03_benchmark"] == True)])
    ]
    for track_name, subset_df in cohorts:
        n = len(subset_df)
        acc_en_seq = subset_df["correct_en_seq"].mean() * 100.0
        acc_id_seq = subset_df["correct_id_seq"].mean() * 100.0
        gap_seq = acc_en_seq - acc_id_seq
        flip_seq = subset_df["flip_seq"].mean() * 100.0
        joint_seq = subset_df["joint_correct_seq"].mean() * 100.0
        
        acc_en_len = subset_df["correct_en_len"].mean() * 100.0
        acc_id_len = subset_df["correct_id_len"].mean() * 100.0
        gap_len = acc_en_len - acc_id_len
        
        print(f"--- RQ1 METRICS: {model_title} | Track: {track_name} (N={n}) ---")
        print(f"Sequence Track (Primary):")
        print(f"  English Accuracy          : {acc_en_seq:.2f}%")
        print(f"  Indonesian Accuracy       : {acc_id_seq:.2f}%")
        print(f"  Cross-Lingual Gap (EN-ID) : {gap_seq:+.2f} pp")
        print(f"  Prediction Flip Rate      : {flip_seq:.2f}%")
        print(f"  Joint Bilingual Correct   : {joint_seq:.2f}%")
        print(f"Length-Normalized Track (Sensitivity):")
        print(f"  English Accuracy          : {acc_en_len:.2f}%")
        print(f"  Indonesian Accuracy       : {acc_id_len:.2f}%")
        print(f"  Cross-Lingual Gap (EN-ID) : {gap_len:+.2f} pp\\n")
        
        rows.append({
            "model": model_title, "track": track_name, "sample_size": n,
            "acc_en_seq": acc_en_seq, "acc_id_seq": acc_id_seq, "gap_en_id_seq": gap_seq,
            "flip_rate_seq": flip_seq, "joint_acc_seq": joint_seq,
            "acc_en_len": acc_en_len, "acc_id_len": acc_id_len, "gap_en_id_len": gap_len
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
    cohorts = [
        ("primary_test_cohort", df[(df["split"] == "test") & (df["is_eligible_primary"] == True)]),
        ("historical_rev03_benchmark", df[(df["split"] == "test") & (df["in_rev03_benchmark"] == True)])
    ]
    for track_name, subset_df in cohorts:
        n = len(subset_df)
        dp_seq = subset_df["delta_p_dir_seq"].values
        mean_dp_seq = float(np.mean(dp_seq))
        pos_seq = float(np.mean(dp_seq > 0)) * 100.0
        neg_seq = float(np.mean(dp_seq < 0)) * 100.0
        
        dp_len = subset_df["delta_p_dir_len"].values
        mean_dp_len = float(np.mean(dp_len))
        
        print(f"--- RQ2 VISUAL INTERVENTION: {model_title} | Track: {track_name} (N={n}) ---")
        print(f"Sequence Track (Primary):")
        print(f"  Mean Delta P_dir          : {mean_dp_seq:+.4f}")
        print(f"  Positive Reliance (>0)    : {pos_seq:.2f}% (Degraded upon visual deletion)")
        print(f"  Negative Reliance (<0)    : {neg_seq:.2f}% (Confidence increased without image)")
        print(f"Length-Normalized Track (Sensitivity):")
        print(f"  Mean Delta P_dir          : {mean_dp_len:+.4f}\\n")
        
        rows.append({
            "model": model_title, "track": track_name, "sample_size": n,
            "mean_delta_p_seq": mean_dp_seq, "prop_pos_seq": pos_seq, "prop_neg_seq": neg_seq,
            "mean_delta_p_len": mean_dp_len
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
    
    # 1. Primary Sequence Track
    score_cols_seq = ["score_m1_seq", "score_m2_seq", "score_m_dir_seq", "score_m_norm_seq", "score_m3_old_seq"]
    boot_sum_seq, boot_diff_seq = run_paired_cluster_bootstrap(
        df_subset, score_cols_seq, corr_col="correct_id_seq", n_boot=1000, seed=42
    )
    
    rows = []
    raw_p_seq = {}
    curves_seq = {}
    base_aurc_seq = None
    
    for col in score_cols_seq:
        cov, risk, aurc = compute_discrete_risk_coverage(df_subset[col].values, df_subset["correct_id_seq"].values)
        curves_seq[col] = (cov, risk)
        if col == "score_m1_seq":
            base_aurc_seq = aurc
            p_val = 1.0
            delta_aurc = 0.0
        else:
            _, _, _, p_val = paired_cluster_permutation_test(
                df_subset, col, "score_m1_seq", corr_col="correct_id_seq", n_perm=1000, seed=42
            )
            raw_p_seq[col] = p_val
            delta_aurc = aurc - base_aurc_seq
            
        r100, _, _ = get_risk_at_realized_coverage(cov, risk, 1.0)
        r90, _, _ = get_risk_at_realized_coverage(cov, risk, 0.9)
        r80, _, _ = get_risk_at_realized_coverage(cov, risk, 0.8)
        r70, _, _ = get_risk_at_realized_coverage(cov, risk, 0.7)
        r50, _, _ = get_risk_at_realized_coverage(cov, risk, 0.5)
        
        rows.append({
            "model": model_title, "track": track_label, "scoring_track": "sequence_logprob",
            "sample_size": n_samples, "clusters": n_clusters,
            "estimator": col, "aurc_discrete_mean": aurc,
            "ci_95_lower": boot_sum_seq[col]["ci_95_lower"], "ci_95_upper": boot_sum_seq[col]["ci_95_upper"],
            "delta_aurc_vs_m1": delta_aurc,
            "delta_ci_lower": boot_diff_seq[col]["delta_ci_95_lower"] if col != "score_m1_seq" else 0.0,
            "delta_ci_upper": boot_diff_seq[col]["delta_ci_95_upper"] if col != "score_m1_seq" else 0.0,
            "perm_p_value_raw": p_val,
            "risk_cov_100": r100, "risk_cov_90": r90, "risk_cov_80": r80, "risk_cov_70": r70, "risk_cov_50": r50
        })
        
    adj_p_seq = apply_holm_bonferroni(raw_p_seq)
    for r in rows:
        est = r["estimator"]
        r["perm_p_value_holm"] = adj_p_seq.get(est, 1.0)
        
    # 2. Length-Normalized Sensitivity Track
    score_cols_len = ["score_m1_len", "score_m2_len", "score_m_dir_len", "score_m_norm_len", "score_m3_old_len"]
    base_aurc_len = None
    for col in score_cols_len:
        cov, risk, aurc = compute_discrete_risk_coverage(df_subset[col].values, df_subset["correct_id_len"].values)
        if col == "score_m1_len":
            base_aurc_len = aurc
            delta_aurc = 0.0
        else:
            delta_aurc = aurc - base_aurc_len
        rows.append({
            "model": model_title, "track": track_label, "scoring_track": "length_normalized_surprisal",
            "sample_size": n_samples, "clusters": n_clusters,
            "estimator": col, "aurc_discrete_mean": aurc,
            "ci_95_lower": np.nan, "ci_95_upper": np.nan,
            "delta_aurc_vs_m1": delta_aurc,
            "delta_ci_lower": np.nan, "delta_ci_upper": np.nan,
            "perm_p_value_raw": np.nan, "perm_p_value_holm": np.nan,
            "risk_cov_100": np.nan, "risk_cov_90": np.nan, "risk_cov_80": np.nan, "risk_cov_70": np.nan, "risk_cov_50": np.nan
        })
        
    df_res = pd.DataFrame(rows)
    safe_m = model_title.replace("/", "_").replace("-", "_")
    csv_p = out_dir / f"rq3_{track_label}_{safe_m}.csv"
    df_res.to_csv(csv_p, index=False)
    
    # Plot Risk-Coverage Curve (Sequence Track)
    plt.figure(figsize=(8, 6), dpi=150)
    for col, (cov, risk) in curves_seq.items():
        aurc_val = df_res.loc[(df_res['estimator']==col) & (df_res['scoring_track']=='sequence_logprob'), 'aurc_discrete_mean'].values[0]
        plt.plot(cov, risk, label=f"{col.replace('_seq','')} (AURC={aurc_val:.4f})", lw=2)
    plt.xlabel("Coverage (Proportion of Evaluated Cases)", fontsize=11)
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

# Filter Primary Test Cohort (N=799)
df_qwen_test = df_qwen[(df_qwen["split"] == "test") & (df_qwen["is_eligible_primary"] == True)]
df_huatuo_test = df_huatuo[(df_huatuo["split"] == "test") & (df_huatuo["is_eligible_primary"] == True)]

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
df_qwen_dev = df_qwen[(df_qwen["split"] == "dev") & (df_qwen["is_eligible_primary"] == True)]
df_huatuo_dev = df_huatuo[(df_huatuo["split"] == "dev") & (df_huatuo["is_eligible_primary"] == True)]

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
print("🏆 FINAL CROSS-MODEL COMPARISON (PRIMARY TEST COHORT N=799 — SEQUENCE TRACK)")
print("=" * 85)
df_seq_show = df_cross_prim[df_cross_prim["scoring_track"] == "sequence_logprob"]
cols_show = ["model", "estimator", "aurc_discrete_mean", "ci_95_lower", "ci_95_upper", "delta_aurc_vs_m1", "perm_p_value_holm", "risk_cov_100", "risk_cov_80", "risk_cov_50"]
display(df_seq_show[cols_show])
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
    "n_active_union_evaluated": len(df_eval_cohort),
    "n_primary_cohort_total": int((df_manifest["is_eligible_primary"] == True).sum()),
    "n_primary_test": int(((df_manifest["split"] == "test") & (df_manifest["is_eligible_primary"] == True)).sum()),
    "n_primary_dev": int(((df_manifest["split"] == "dev") & (df_manifest["is_eligible_primary"] == True)).sum()),
    "n_historical_benchmark_test": int(((df_manifest["split"] == "test") & (df_manifest["in_rev03_benchmark"] == True)).sum()),
    "n_historical_benchmark_dev": int(((df_manifest["split"] == "dev") & (df_manifest["in_rev03_benchmark"] == True)).sum())
}
meta_p = RESULTS_ROOT / "run_metadata.json"
meta_p.write_text(json.dumps(meta, indent=2))

# 2. Build SHA-256 Manifest of all produced artifacts
artifact_rows = []
for root, dirs, files in os.walk(RESULTS_ROOT):
    for f in files:
        if f.endswith(".zip") or f.endswith(".tmp"):
            continue
        fp = Path(root) / f
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        artifact_rows.append({"file_name": fp.name, "relative_path": str(fp.relative_to(RESULTS_ROOT)), "sha256": h})

df_artifacts = pd.DataFrame(artifact_rows)
art_manifest_p = RESULTS_ROOT / "artifact_sha256_manifest.csv"
df_artifacts.to_csv(art_manifest_p, index=False)

# 3. Create Final Reproducibility Zip
zip_name = "decisive_validation_rev05_results_final.zip"
zip_path = RESULTS_ROOT / zip_name
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(RESULTS_ROOT):
        for file in files:
            p = Path(root) / file
            if p != zip_path and not p.name.endswith(".tmp"):
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
    print("Generating 08_decisive_validation_rev05.ipynb...")
    nb = build_kaggle_notebook()
    
    out_paths = [
        Path("main/Kaggle/08_decisive_validation_rev05.ipynb"),
        Path("main/notebooks/08_decisive_validation_rev05.ipynb")
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"✅ Generated: {p} (SHA256: {h})")

if __name__ == "__main__":
    main()
