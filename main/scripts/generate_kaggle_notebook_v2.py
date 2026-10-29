"""
Generator script for 05_decisive_validation_revised.ipynb (Revision rev02)
Creates an independent, publication-grade Jupyter Notebook for Kaggle Dual T4 GPU execution.
Conforms strictly to Thesis Advisor Directives and Rigorous Methodological Audit Standards.
"""

import sys
import zlib
import base64
import hashlib
from pathlib import Path
import nbformat as nbf

# Read authoritative manifest to embed in notebook
MANIFEST_PATH = Path("main/results/decisive_validation_manifest.csv")
raw_manifest = MANIFEST_PATH.read_bytes()
AUTHORITATIVE_MANIFEST_SHA256 = hashlib.sha256(raw_manifest).hexdigest()
EMBEDDED_MANIFEST_B64 = base64.b64encode(zlib.compress(raw_manifest, 9)).decode("ascii")


def build_kaggle_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
        },
        "accelerator": "GPU"
    }

    cells = []

    # =========================================================================
    # CELL 0: TITLE & METHODOLOGICAL DIRECTIVES (REV02)
    # =========================================================================
    c0_md = """# 🏥 Medical VLM Decisive Validation Experiment (Kaggle Dual-T4 GPU) — Revision rev02
**Research Project:** Evaluasi Reliabilitas dan Selective Prediction pada Medical Vision-Language Model di Bawah Pergeseran Bahasa (IndoRad-VQA)  
**Author:** Imam (Undergraduate CS Thesis)  
**Revision:** `rev02` (Strict Manifest Enforcement & Zero-Leakage Test-Only Split)  
**Environment:** Kaggle Dual Tesla T4 (2x 16GB VRAM, 4-bit NF4 Quantization)  

---

## 📌 Metodologi & Arahan Resmi Dosen Pembimbing
1. **Peniadaan Blur Total:** Sesuai arahan dosen pembimbing, manipulasi blur dieliminasi secara permanen. Hanya dua kondisi visual yang dievaluasi:
   - `P0_ORIGINAL`: Citra radiologi asli tanpa manipulasi.
   - `P1_BLACK_BLANK`: Kanvas hitam murni `RGB (0, 0, 0)` berdimensi dan mode identik dengan citra asli.
2. **Target Candidate Sequence Scoring:** Menghindari *single-token logit hacking*. Menghitung *conditional sequence log-likelihood* secara eksak dengan runtime prefix-match assertion untuk:
   - **English:** `["Yes", "No"]`
   - **Indonesian:** `["Ya", "Tidak"]`
3. **Komparasi Fair 5 Estimator:**
   - $M_1$: Confidence-only baseline ($p_{\\text{orig}}$)
   - $M_2$: Visual-only ($\max(0, \\Delta_{\\text{dir}})$)
   - $M_{\\text{dir}}$: Directed Multiplicative ($p_{\\text{orig}} \\times \\max(0, \\Delta_{\\text{dir}})$)
   - $M_{\\text{norm}}$: Normalized Directed Linear ($p_{\\text{orig}} \\times \\frac{1 + \\Delta_{\\text{dir}}}{2}$)
   - $M_3^{\\text{old}}$: Historical Unsigned Baseline ($p_{\\text{orig}} \\times \\frac{1 + |\\Delta_{\\text{dir}}|}{2}$)
4. **Disiplin Split Data (Zero Leakage, Authoritative Frozen Manifest):**
   - **Primary Test-Only Subset:** $N = 92$ pasangan QA (35 klaster citra fisik unik). Evaluasi utama bebas dari data dev.
   - **Full Benchmark Sensitivity Set:** $N = 150$ pasangan QA (58 klaster citra fisik unik). Memuat split test (92) + dev (58) untuk uji sensitivitas ukuran sampel.
   - **Bit-Identical Manifest Verification:** Manifest divalidasi langsung terhadap SHA256 otoritatif (`988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52`).
   - Tidak ada re-downloading dari Hugging Face atau pembuatan split ad-hoc.
5. **Isolasi Revisi (`rev02`):**
   - Seluruh luaran disimpan di direktori terpisah: `results/decisive_validation/rev02/`.
   - Checkpoint lama yang tidak kompatibel diidentifikasi dan diisolasi secara otomatis."""
    cells.append(nbf.v4.new_markdown_cell(c0_md))

    # =========================================================================
    # CELL 1: ENVIRONMENT & HARDWARE AUDIT
    # =========================================================================
    c1_code = """# ==============================================================================
# CELL 1: ENVIRONMENT & HARDWARE AUDIT
# ==============================================================================
import os
import sys
import subprocess
import platform

# Ensure UTF-8 stdout encoding in any terminal or Jupyter kernel
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Robust display fallback
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

# Ensure critical libraries are installed
required_pkgs = ["transformers", "accelerate", "bitsandbytes", "qwen_vl_utils", "torch", "huggingface_hub", "pyarrow"]
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
    # CELL 2: DIRECTORY STRUCTURE & REVISION ISOLATION SETUP (REV02)
    # =========================================================================
    c2_code = """# ==============================================================================
# CELL 2: REVISION REV02 PATHS & ISOLATION
# ==============================================================================
from pathlib import Path
from datetime import datetime, timezone
import json

REVISION_ID = "rev02"
AUTHORITATIVE_MANIFEST_SHA256 = "988f57a6512223337bb5ab3b192f1802e38bfec2f6b262f51607381faa74bf52"

# Workspace setup
WORKSPACE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
PROJECT_DIR = WORKSPACE / "indorad_validation_rev02"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Revision-specific output directory
RESULTS_DIR = WORKSPACE / "results" / "decisive_validation" / "rev02"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PERTURB_DIR = DATA_DIR / "scratch_perturbations_rev02"
PERTURB_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Active Revision    : {REVISION_ID}")
print(f"📂 Project Root       : {PROJECT_DIR.resolve()}")
print(f"📂 Data Directory     : {DATA_DIR.resolve()}")
print(f"📂 Results Directory  : {RESULTS_DIR.resolve()}")
print(f"📂 Perturbations Dir  : {PERTURB_DIR.resolve()}")"""
    cells.append(nbf.v4.new_code_cell(c2_code))

    # =========================================================================
    # CELL 3: PROJECT EXTRACTION & DATASET UNZIPPING
    # =========================================================================
    c3_code = """# ==============================================================================
# CELL 3: DATASET UNZIPPING & SEARCH
# ==============================================================================
import zipfile

# Search for any dataset zip across Kaggle input directories
zip_candidates = []
if Path("/kaggle/input").exists():
    zip_candidates.extend(list(Path("/kaggle/input").rglob("*.zip")))
if Path("main/Kaggle/indorad-vqa-research.zip").exists():
    zip_candidates.append(Path("main/Kaggle/indorad-vqa-research.zip"))

if zip_candidates:
    for src_zip in zip_candidates:
        print(f"📦 Extracting dataset zip: {src_zip} -> {DATA_DIR}...")
        try:
            with zipfile.ZipFile(src_zip, 'r') as zf:
                zf.extractall(DATA_DIR)
            print(f"✅ Unzipped {src_zip.name} successfully.")
        except Exception as e:
            print(f"⚠️ Could not extract {src_zip}: {e}")
else:
    print("ℹ️ No zip archives found in input. Using existing disk directories.")"""
    cells.append(nbf.v4.new_code_cell(c3_code))

    # =========================================================================
    # CELL 4: AUTHORITATIVE FROZEN MANIFEST LOADING WITH BIT-IDENTICAL STORE
    # =========================================================================
    c4_code = f"""# ==============================================================================
# CELL 4: AUTHORITATIVE FROZEN MANIFEST ENFORCEMENT & INTEGRITY AUDIT
# ==============================================================================
import io
import zlib
import base64
import hashlib
import numpy as np
import pandas as pd

AUTHORITATIVE_MANIFEST_SHA256 = "{AUTHORITATIVE_MANIFEST_SHA256}"
EMBEDDED_MANIFEST_B64 = "{EMBEDDED_MANIFEST_B64}"

manifest_search_paths = [
    DATA_DIR / "decisive_validation_manifest.csv",
    DATA_DIR / "results" / "decisive_validation_manifest.csv",
    Path("/kaggle/input/indorad-vqa-research/main/results/decisive_validation_manifest.csv"),
    Path("/kaggle/input/indorad-vqa-research/results/decisive_validation_manifest.csv"),
    Path("./main/results/decisive_validation_manifest.csv"),
    Path("./results/decisive_validation_manifest.csv")
]

manifest_path = None
df_eval = None

for p in manifest_search_paths:
    if p.exists():
        raw_b = p.read_bytes()
        if hashlib.sha256(raw_b).hexdigest() == AUTHORITATIVE_MANIFEST_SHA256:
            manifest_path = p
            df_eval = pd.read_csv(io.BytesIO(raw_b))
            print(f"✅ Loaded authoritative manifest from disk: {{p}}")
            break
        else:
            print(f"⚠️ Notice: Found {{p}} but SHA256 did not match authoritative hash. Skipping.")

# If not found on disk with matching hash, unpack bit-identical embedded store
if df_eval is None:
    print("📦 Restoring authoritative manifest from embedded bit-identical store...")
    raw_b = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64.encode("ascii")))
    assert hashlib.sha256(raw_b).hexdigest() == AUTHORITATIVE_MANIFEST_SHA256, "Embedded manifest corrupted!"
    df_eval = pd.read_csv(io.BytesIO(raw_b))
    manifest_path = DATA_DIR / "decisive_validation_manifest.csv"
    manifest_path.write_bytes(raw_b)
    print(f"✅ Successfully materialized authoritative manifest to: {{manifest_path}}")

# Strict Invariants Validation (Zero Compromise)
assert len(df_eval) == 150, f"Manifest length must be 150, got {{len(df_eval)}}"
split_counts = df_eval['split'].value_counts().to_dict()
assert split_counts == {{'test': 92, 'dev': 58}}, f"Split counts must be {{'test': 92, 'dev': 58}}, got {{split_counts}}"

test_df = df_eval[df_eval['split'] == 'test']
dev_df = df_eval[df_eval['split'] == 'dev']
test_clusters = set(test_df['canonical_image_id'])
dev_clusters = set(dev_df['canonical_image_id'])

assert len(test_clusters) == 35, f"Test unique clusters must be 35, got {{len(test_clusters)}}"
assert len(dev_clusters) == 23, f"Dev unique clusters must be 23, got {{len(dev_clusters)}}"

leakage = test_clusters.intersection(dev_clusters)
assert len(leakage) == 0, f"Fatal cross-split leakage detected: {{leakage}}"

print("\\n" + "=" * 70)
print("📊 AUTHORITATIVE FROZEN MANIFEST AUDIT REPORT (REV02)")
print("=" * 70)
print(f"  - Total QA Pairs            : {{len(df_eval)}}")
print(f"  - Total Unique Image Clusters: {{len(test_clusters) + len(dev_clusters)}}")
print(f"  - Primary Test-Only Subset   : {{len(test_df)}} QA pairs across {{len(test_clusters)}} image clusters")
print(f"  - Development Subset         : {{len(dev_df)}} QA pairs across {{len(dev_clusters)}} image clusters")
print(f"  - Cross-Split Image Leakage  : 0 clusters (OVERLAP ZERO VERIFIED)")
print(f"  - Manifest SHA256 Verification: PASS ({{AUTHORITATIVE_MANIFEST_SHA256}})")
print("=" * 70)

display(df_eval[['qa_id', 'canonical_image_id', 'image_id', 'split', 'translation_status', 'question_en', 'question_id', 'reference_answer']].head(5))"""
    cells.append(nbf.v4.new_code_cell(c4_code))

    # =========================================================================
    # CELL 5: ROBUST IMAGE RESOLVER (CANONICAL & DATASET CONVENTIONS)
    # =========================================================================
    c5_code = """# ==============================================================================
# CELL 5: MULTI-STRATEGY IMAGE RESOLVER & CRYPTOGRAPHIC VERIFICATION
# ==============================================================================
import io
from PIL import Image

search_image_dirs = [
    DATA_DIR / "images",
    DATA_DIR / "data" / "images",
    IMAGES_DIR,
    Path("./main/data/images"),
    Path("./data/images")
]

# Dynamically scan /kaggle/input for any directories containing png files
if Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").rglob("*.png"):
        if p.parent not in search_image_dirs:
            search_image_dirs.append(p.parent)

def resolve_single_image(cid: str, img_id: str, qa_id: int, image_name: str, 
                         expected_pixel_hash: str, expected_w: int, expected_h: int) -> Path:
    candidates = [
        f"{cid}.png",
        f"{img_id}.png",
        f"{img_id}",
        f"img_{qa_id}.png",
        f"img_{qa_id:05d}.png"
    ]
    if image_name:
        candidates.append(str(image_name))
        
    for sdir in search_image_dirs:
        if not sdir.exists():
            continue
        for cand in candidates:
            cand_p = sdir / cand
            if cand_p.is_file():
                try:
                    with Image.open(cand_p) as im:
                        if (im.width, im.height) != (expected_w, expected_h):
                            continue
                        rgb_arr = np.array(im.convert("RGB"))
                        calc_hash = hashlib.sha256(rgb_arr.tobytes()).hexdigest()
                        if calc_hash == expected_pixel_hash:
                            return cand_p
                except Exception:
                    continue
    raise FileNotFoundError(
        f"Could not establish cryptographic identity for QA ID {qa_id} (cid={cid}, expected_hash={expected_pixel_hash})"
    )

resolved_map = {}
missing_images = []

for idx, row in df_eval.iterrows():
    qid = row['qa_id']
    cid = str(row['canonical_image_id'])
    img_id = str(row.get('image_id', cid))
    iname = str(row.get('image_name', ''))
    exp_hash = str(row['pixel_sha256'])
    exp_w = int(row['image_width'])
    exp_h = int(row['image_height'])
    try:
        resolved_p = resolve_single_image(cid, img_id, qid, iname, exp_hash, exp_w, exp_h)
        resolved_map[qid] = resolved_p
    except FileNotFoundError:
        missing_images.append((qid, cid, img_id, iname, exp_hash))

# Automatic Self-Healing Materialization: If any images are missing from disk,
# fetch the exact QA rows from the authoritative Hugging Face repository (Lab-IS/IndoRad-VQA)
# and verify pixel hash cryptographically before saving to disk.
if missing_images:
    print(f"⚠️ Notice: {len(missing_images)} / {len(df_eval)} images not found in local disk paths.")
    print("🌐 Materializing images directly from authoritative Hugging Face repository (Lab-IS/IndoRad-VQA)...")
    try:
        from huggingface_hub import hf_hub_download
        
        p_train = hf_hub_download(repo_id="Lab-IS/IndoRad-VQA", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        p_test = hf_hub_download(repo_id="Lab-IS/IndoRad-VQA", filename="data/test-00000-of-00001.parquet", repo_type="dataset")
        
        df_tr = pd.read_parquet(p_train)
        df_te = pd.read_parquet(p_test)
        df_all_hf = pd.concat([df_tr, df_te], ignore_index=True)
        
        recovered_count = 0
        for qid, cid, img_id, iname, exp_hash in missing_images:
            hf_row = df_all_hf[df_all_hf["qid"] == qid]
            if len(hf_row) > 0:
                raw_bytes = hf_row.iloc[0]["image"]["bytes"]
                im = Image.open(io.BytesIO(raw_bytes))
                rgb_arr = np.array(im.convert("RGB"))
                calc_hash = hashlib.sha256(rgb_arr.tobytes()).hexdigest()
                if calc_hash == exp_hash:
                    save_path = IMAGES_DIR / f"{cid}.png"
                    im.save(save_path)
                    resolved_map[qid] = save_path
                    recovered_count += 1
                else:
                    print(f"❌ Hash mismatch for QID {qid} from HF: expected {exp_hash[:16]}, got {calc_hash[:16]}")
                    
        print(f"✅ Successfully materialized and verified {recovered_count} images from Hugging Face.")
        missing_images = [m for m in missing_images if m[0] not in resolved_map]
    except Exception as e:
        print(f"❌ Error during Hugging Face image materialization: {e}")

if missing_images:
    print(f"❌ FATAL ERROR: {len(missing_images)} / {len(df_eval)} images failed cryptographic identity verification!")
    for m in missing_images[:5]:
        print(f"   Failed QA ID {m[0]}: cid={m[1]}, expected_hash={m[4]}")
    raise RuntimeError(f"{len(missing_images)} images failed cryptographic verification. Stopping execution.")

print(f"✅ Cryptographic Image Identity Verified: All {len(resolved_map)} / {len(df_eval)} images match pixel SHA256 and dimensions perfectly.")"""
    cells.append(nbf.v4.new_code_cell(c5_code))

    # =========================================================================
    # CELL 6: CORE EVALUATION & STATISTICAL FUNCTIONS WITH PREFIX ASSERTION
    # =========================================================================
    c6_code = """# ==============================================================================
# CELL 6: CORE EVALUATION & STATISTICAL FUNCTIONS
# ==============================================================================
import torch
import torch.nn.functional as F

def extract_target_loglikelihood(model, processor, image_path, prompt_text, candidate_strings, target_device):
    \"\"\"
    Evaluates conditional log-likelihood of exact candidate target strings
    (e.g. 'Yes' vs 'No' or 'Ya' vs 'Tidak') given the multimodal context (Image + Prompt).
    Avoids single first-token logit hacking and includes prefix-match assertion.
    \"\"\"
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
        assert torch.equal(inputs.input_ids[0, :prompt_len].cpu(), prompt_inputs.input_ids[0].cpu()), \\
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
    \"\"\"
    Runs sequence scoring and returns normalized binary probabilities.
    candidate_strings: [affirmatory, negative], e.g. ["Yes", "No"] or ["Ya", "Tidak"]
    Returns: (pred_label, confidence, p_aff, p_neg)
    \"\"\"
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


def compute_discrete_risk_coverage(scores: np.ndarray, correctness: np.ndarray):
    \"\"\"
    Sample-level discrete selective prediction evaluation (Geifman & El-Yaniv, NeurIPS 2017).
    Ranks samples by reliability score descending: S_(1) >= S_(2) >= ... >= S_(N).
    Evaluates empirical selective risk r_k = 1 - (1/k) sum_{j=1}^k Correct_(j) for k=1..N.
    \"\"\"
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
    \"\"\"Find selective risk at closest discrete coverage level.\"\"\"
    idx = (np.abs(cov_arr - target_cov)).argmin()
    return float(risk_arr[idx]), int(idx + 1)


def run_image_cluster_bootstrap(df: pd.DataFrame, score_cols: list, corr_col: str = "correct_id", 
                                cluster_col: str = "canonical_image_id", n_boot: int = 1000, seed: int = 42):
    \"\"\"
    Paired Image-Cluster Bootstrap to account for intra-cluster correlation.
    Resamples unique physical images with replacement.
    \"\"\"
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

print("✅ Core Evaluation & Bootstrap Functions defined.")"""
    cells.append(nbf.v4.new_code_cell(c6_code))

    # =========================================================================
    # CELL 7: RISK-COVERAGE PLOTTING FUNCTION
    # =========================================================================
    c7_code = """# ==============================================================================
# CELL 7: RISK-COVERAGE PLOTTING ENGINE
# ==============================================================================
import matplotlib.pyplot as plt

def plot_risk_coverage_curves(df: pd.DataFrame, score_cols: list, save_path: Path, model_title: str):
    \"\"\"
    Generates publication-ready Risk-Coverage curves for all 5 estimators.
    \"\"\"
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
    
    ax.set_title(f"Selective Risk vs. Coverage under Indonesian Language Shift\\n[{model_title}] (N={n_samples}, {n_clust} Image Clusters)", fontsize=12, fontweight="bold", pad=12)
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

print("✅ Plotting Engine Ready.")"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # =========================================================================
    # CELL 8: PIPELINE EXECUTOR WITH STRICT REV02 RESUME & COMPATIBILITY
    # =========================================================================
    c8_code = """# ==============================================================================
# CELL 8: PIPELINE EXECUTOR WITH STRICT REV02 RESUME
# ==============================================================================
import gc
import time
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from tqdm.auto import tqdm

PROMPT_EN_TEMPLATE = "Answer only YES or NO.\\nQuestion: {q}"
PROMPT_ID_TEMPLATE = "Jawab pertanyaan radiologi berdasarkan gambar berikut. Jawab hanya dengan YA atau TIDAK.\\nPertanyaan: {q}"
CANDIDATES_EN = ["Yes", "No"]
CANDIDATES_ID = ["Ya", "Tidak"]

def execute_model_evaluation(model_name: str, df_input: pd.DataFrame, image_map: dict):
    safe_name = model_name.replace("/", "_").replace("-", "_")
    model_out_dir = RESULTS_DIR / safe_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_csv = model_out_dir / f"live_validation_checkpoint_{safe_name}.csv"
    expected_qa_ids = set(df_input["qa_id"].tolist())
    
    completed_qa_ids = set()
    results = []
    
    # Strict Checkpoint Validation & Resume Logic
    if checkpoint_csv.exists():
        try:
            df_done = pd.read_csv(checkpoint_csv)
            is_model_match = ("model_id" in df_done.columns and (df_done["model_id"] == model_name).all())
            is_method_match = ("scoring_method" in df_done.columns and (df_done["scoring_method"] == "target_candidate_sequence_scoring").all())
            is_rev_match = ("revision_id" in df_done.columns and (df_done["revision_id"] == REVISION_ID).all())
            is_hash_match = ("manifest_sha256" in df_done.columns and (df_done["manifest_sha256"] == AUTHORITATIVE_MANIFEST_SHA256).all())
            required_cols = {"qa_id", "canonical_image_id", "pred_en", "pred_id", "conf_id", "delta_p_dir", "score_m_dir", "reference_answer", "split"}
            has_required_cols = required_cols.issubset(set(df_done.columns))
            qa_subset_match = set(df_done["qa_id"]).issubset(expected_qa_ids) if "qa_id" in df_done.columns else False
            row_count_valid = len(df_done) <= len(df_input)
            
            numeric_cols = ["p_en_yes", "p_en_no", "p_orig_ya", "p_orig_tidak", "p_blank_ya", "p_blank_tidak", "delta_p_dir"]
            has_finite_scores = all(
                c in df_done.columns and df_done[c].notna().all() and np.isfinite(df_done[c]).all()
                for c in numeric_cols
            )
            
            if (is_model_match and is_method_match and is_rev_match and is_hash_match 
                and has_required_cols and qa_subset_match and row_count_valid and has_finite_scores):
                completed_qa_ids = set(df_done["qa_id"].tolist())
                results = df_done.to_dict(orient="records")
                print(f"🔄 Resuming from verified {REVISION_ID} live checkpoint: {len(completed_qa_ids)} / {len(df_input)} samples already evaluated.")
            else:
                reasons = []
                if not is_model_match: reasons.append("model mismatch")
                if not is_method_match: reasons.append("scoring method mismatch")
                if not is_rev_match: reasons.append("revision mismatch")
                if not is_hash_match: reasons.append("manifest hash mismatch")
                if not has_required_cols: reasons.append("missing required columns")
                if not qa_subset_match: reasons.append("QA IDs not a valid subset")
                if not row_count_valid: reasons.append(f"row count {len(df_done)} exceeds manifest {len(df_input)}")
                if not has_finite_scores: reasons.append("non-finite/null numeric scores")
                print(f"⚠️ Incompatible checkpoint detected ({', '.join(reasons)}).")
                backup_csv = model_out_dir / f"incompatible_checkpoint_backup_{int(time.time())}.csv"
                checkpoint_csv.rename(backup_csv)
                print(f"   Old checkpoint backed up to: {backup_csv.name}")
                completed_qa_ids = set()
                results = []
                print(f"🚀 Starting fresh {REVISION_ID} evaluation on {len(df_input)} samples.")
        except Exception as e:
            print(f"⚠️ Error inspecting checkpoint: {e}. Starting fresh.")
            completed_qa_ids = set()
            results = []
    else:
        completed_qa_ids = set()
        results = []
        print(f"🚀 Starting fresh {REVISION_ID} evaluation on {len(df_input)} samples.")
        
    # Check if already 100% complete
    if len(completed_qa_ids) >= len(df_input):
        print(f"✅ Model [{model_name}] already completed. Proceeding directly to statistical analysis.")
        return pd.DataFrame(results), model_out_dir
        
    print(f"\\n📦 Inspecting Configuration for: {model_name}...")
    import transformers
    from transformers import AutoConfig, AutoProcessor
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch_list = getattr(config, "architectures", [])
    print(f"   Declared Architecture(s): {arch_list} | Model Type: {getattr(config, 'model_type', 'unknown')}")
    
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
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.eval()
    target_dev = next(model.parameters()).device
    print(f"✅ Loaded {model_name} ({model_class.__name__}) onto device: {target_dev}")
    
    # Rigorous Empirical Smoke Test on First Pending Sample
    smoke_row = None
    for _, r in df_input.iterrows():
        if r["qa_id"] not in completed_qa_ids:
            smoke_row = r
            break
            
    if smoke_row is not None:
        print("\\n" + "=" * 70)
        print("🧪 RUNNING EMPIRICAL SMOKE TEST & CLOCK TIMING (QA ID:", smoke_row["qa_id"], ")")
        print("=" * 70)
        test_img_p = image_map[smoke_row["qa_id"]]
        cid = str(smoke_row["canonical_image_id"])
        
        test_blank_p = PERTURB_DIR / f"blank_{cid}.png"
        if not test_blank_p.exists():
            with Image.open(test_img_p) as orig_pil:
                blank_pil = Image.new("RGB", orig_pil.size, color=(0, 0, 0))
                blank_pil.save(test_blank_p)
            
        p_en = PROMPT_EN_TEMPLATE.format(q=smoke_row["question_en"])
        p_id = PROMPT_ID_TEMPLATE.format(q=smoke_row["question_id"])
        
        t0 = time.time()
        test_pred_en, test_conf_en, test_p_en_yes, test_p_en_no = score_binary_candidates(model, processor, test_img_p, p_en, CANDIDATES_EN, target_dev)
        test_pred_id, test_conf_id, test_p_orig_ya, test_p_orig_tidak = score_binary_candidates(model, processor, test_img_p, p_id, CANDIDATES_ID, target_dev)
        _, _, test_p_blank_ya, test_p_blank_tidak = score_binary_candidates(model, processor, test_blank_p, p_id, CANDIDATES_ID, target_dev)
        t_elapsed = time.time() - t0
        
        test_p_blank_pred = test_p_blank_ya if test_pred_id == "YES" else test_p_blank_tidak
        test_delta_dir = test_conf_id - test_p_blank_pred
        
        print(f"  [Condition P0 - EN] Pred: {test_pred_en} | Conf: {test_conf_en:.4f} (P_yes={test_p_en_yes:.4f}, P_no={test_p_en_no:.4f})")
        print(f"  [Condition P0 - ID] Pred: {test_pred_id} | Conf: {test_conf_id:.4f} (P_ya={test_p_orig_ya:.4f}, P_tidak={test_p_orig_tidak:.4f})")
        print(f"  [Condition P1 - ID] Blank Pred: P_blank(ya)={test_p_blank_ya:.4f}, P_blank(tidak)={test_p_blank_tidak:.4f}")
        print(f"  [Visual Support]   P_blank(pred)={test_p_blank_pred:.4f} -> Delta P_dir = {test_delta_dir:+.4f}")
        print(f"  [Clock Timing]     Measured smoke test runtime: {t_elapsed:.2f} seconds per sample.")
        
        n_rem = len(df_input) - len(completed_qa_ids)
        est_mins = (n_rem * t_elapsed) / 60.0
        print(f"  [Runtime Estimate] Projected time for {n_rem} remaining samples: {est_mins:.1f} minutes ({est_mins/60.0:.2f} hours)")
        print("=" * 70)
        print("✅ Smoke test passed! Tensor shapes, prefix alignment, and timing verified.\\n")

    # Main Evaluation Loop
    for idx, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"Evaluating {model_name.split('/')[-1]}"):
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
        
        # 1. Generate Deterministic Black-Blank Image P1
        blank_img_path = PERTURB_DIR / f"blank_{cid}.png"
        if not blank_img_path.exists():
            with Image.open(orig_img_path) as orig_pil:
                blank_pil = Image.new("RGB", orig_pil.size, color=(0, 0, 0))
                blank_pil.save(blank_img_path)
            
        # 2. RQ1: Bilingual Evaluation on P0 (Original)
        prompt_en = PROMPT_EN_TEMPLATE.format(q=q_en)
        pred_en, conf_en, p_en_yes, p_en_no = score_binary_candidates(model, processor, orig_img_path, prompt_en, CANDIDATES_EN, target_dev)
        corr_en = int(pred_en == ref_ans)
        
        prompt_id = PROMPT_ID_TEMPLATE.format(q=q_id)
        pred_id, conf_id, p_orig_ya, p_orig_tidak = score_binary_candidates(model, processor, orig_img_path, prompt_id, CANDIDATES_ID, target_dev)
        corr_id = int(pred_id == ref_ans)
        
        flip = int(pred_en != pred_id)
        joint_corr = int(corr_en and corr_id)
        
        # 3. RQ2: Visual Intervention P1 (Black-Blank) on Indonesian Question
        _, _, p_blank_ya, p_blank_tidak = score_binary_candidates(model, processor, blank_img_path, prompt_id, CANDIDATES_ID, target_dev)
        p_blank_pred = p_blank_ya if pred_id == "YES" else p_blank_tidak
        delta_p_dir = conf_id - p_blank_pred
        delta_p_abs = abs(conf_id - p_blank_pred)
        
        # 4. RQ3: Compute 5 Pre-registered Estimators
        score_m1 = conf_id
        score_m2 = max(0.0, delta_p_dir)
        score_m_dir = conf_id * max(0.0, delta_p_dir)
        score_m_norm = conf_id * (1.0 + delta_p_dir) / 2.0
        score_m3_old = conf_id * (1.0 + delta_p_abs) / 2.0
        
        row_record = {
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
            # Methodological Metadata
            "blank_method": "PIL.Image.new('RGB', size, (0,0,0))",
            "scoring_method": "target_candidate_sequence_scoring",
            "revision_id": REVISION_ID,
            "manifest_sha256": AUTHORITATIVE_MANIFEST_SHA256,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        results.append(row_record)
        completed_qa_ids.add(qa_id)
        
        # Save checkpoint periodically (every 5 samples)
        if len(results) % 5 == 0 or len(results) == len(df_input):
            pd.DataFrame(results).to_csv(checkpoint_csv, index=False)
            
        # Memory cleanup
        if idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Final flush to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(checkpoint_csv, index=False)
    print(f"💾 Checkpoint saved: {len(res_df)} samples to {checkpoint_csv}")
    
    # Free VRAM before next model
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 GPU memory released.")
    
    return res_df, model_out_dir

print("✅ Pipeline Engine Defined.")"""
    cells.append(nbf.v4.new_code_cell(c8_code))

    # =========================================================================
    # CELL 9: MODEL 1 EVALUATION (QWEN2.5-VL-7B-INSTRUCT)
    # =========================================================================
    c9_code = """# ==============================================================================
# CELL 9: 🚀 RUNNING MODEL 1: Qwen2.5-VL-7B-Instruct
# ==============================================================================
MODEL_1_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

df_results_m1, out_dir_m1 = execute_model_evaluation(
    model_name=MODEL_1_NAME,
    df_input=df_eval,
    image_map=resolved_map
)

print(f"\\n📊 Summary Statistics for [{MODEL_1_NAME}] (Full Benchmark N={len(df_results_m1)}):")
print(f"  - English Accuracy      : {df_results_m1['correct_en'].mean()*100:.2f}%")
print(f"  - Indonesian Accuracy   : {df_results_m1['correct_id'].mean()*100:.2f}%")
print(f"  - Language Gap          : {(df_results_m1['correct_en'].mean() - df_results_m1['correct_id'].mean())*100:+.2f}%")
print(f"  - Flip Rate             : {df_results_m1['flip'].mean()*100:.2f}%")
print(f"  - Joint Correctness     : {df_results_m1['joint_correct'].mean()*100:.2f}%")
print(f"  - Mean Delta P (Directed): {df_results_m1['delta_p_dir'].mean():+.4f}")"""
    cells.append(nbf.v4.new_code_cell(c9_code))

    # =========================================================================
    # CELL 10: MODEL 1 STATISTICAL ANALYSIS & PLOTTING (SPLIT-AWARE)
    # =========================================================================
    c10_code = """# ==============================================================================
# CELL 10: MODEL 1 STATISTICAL ANALYSIS & PLOTTING (TEST-ONLY & SENSITIVITY)
# ==============================================================================
score_cols = ["score_m1", "score_m2", "score_m_dir", "score_m_norm", "score_m3_old"]

def analyze_subset(sub_df, subset_name):
    rows = []
    for sc in score_cols:
        cov, risk, aurc, aurc_trap = compute_discrete_risk_coverage(sub_df[sc].values, sub_df["correct_id"].values)
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

# 1. Primary Analysis: Test-Only Subset (Exact N=92, 35 clusters)
df_test_m1 = df_results_m1[df_results_m1["split"] == "test"].copy()
assert len(df_test_m1) == 92, f"Expected 92 test samples, got {len(df_test_m1)}"
assert df_test_m1['canonical_image_id'].nunique() == 35, f"Expected 35 test clusters, got {df_test_m1['canonical_image_id'].nunique()}"

print(f"📊 Primary Analysis: Test-Only Subset (N={len(df_test_m1)}, {df_test_m1['canonical_image_id'].nunique()} clusters)")
df_table_test_m1 = analyze_subset(df_test_m1, "Test-Only (N=92)")
display(df_table_test_m1)

# 2. Sensitivity Analysis: Full Benchmark Set (Exact N=150, 58 clusters)
print(f"\\n📊 Sensitivity Analysis: Full Benchmark Set (N={len(df_results_m1)}, {df_results_m1['canonical_image_id'].nunique()} clusters)")
df_table_all_m1 = analyze_subset(df_results_m1, "Full-Benchmark (N=150)")
display(df_table_all_m1)

# Combine and save tables
df_final_m1 = pd.concat([df_table_test_m1, df_table_all_m1], ignore_index=True)
comp_path_m1 = out_dir_m1 / "estimator_comparison_qwen7b.csv"
df_final_m1.to_csv(comp_path_m1, index=False)
print(f"💾 Saved estimator comparison: {comp_path_m1}")

# 3. Plot Risk-Coverage Curves for Test-Only Subset
fig_path_m1 = out_dir_m1 / "risk_coverage_curve_qwen7b_test.png"
plot_risk_coverage_curves(df_test_m1, score_cols, fig_path_m1, "Qwen2.5-VL-7B-Instruct (Test-Only N=92)")"""
    cells.append(nbf.v4.new_code_cell(c10_code))

    # =========================================================================
    # CELL 11: MODEL 2 EVALUATION (HUATUOGPT-VISION-7B)
    # =========================================================================
    c11_code = """# ==============================================================================
# CELL 11: 🚀 RUNNING MODEL 2: HuatuoGPT-Vision-7B
# ==============================================================================
MODEL_2_NAME = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

df_results_m2, out_dir_m2 = execute_model_evaluation(
    model_name=MODEL_2_NAME,
    df_input=df_eval,
    image_map=resolved_map
)

print(f"\\n📊 Summary Statistics for [{MODEL_2_NAME}] (Full Benchmark N={len(df_results_m2)}):")
print(f"  - English Accuracy      : {df_results_m2['correct_en'].mean()*100:.2f}%")
print(f"  - Indonesian Accuracy   : {df_results_m2['correct_id'].mean()*100:.2f}%")
print(f"  - Language Gap          : {(df_results_m2['correct_en'].mean() - df_results_m2['correct_id'].mean())*100:+.2f}%")
print(f"  - Flip Rate             : {df_results_m2['flip'].mean()*100:.2f}%")
print(f"  - Joint Correctness     : {df_results_m2['joint_correct'].mean()*100:.2f}%")
print(f"  - Mean Delta P (Directed): {df_results_m2['delta_p_dir'].mean():+.4f}")"""
    cells.append(nbf.v4.new_code_cell(c11_code))

    # =========================================================================
    # CELL 12: MODEL 2 STATISTICAL ANALYSIS & PLOTTING (SPLIT-AWARE)
    # =========================================================================
    c12_code = """# ==============================================================================
# CELL 12: MODEL 2 STATISTICAL ANALYSIS & PLOTTING (TEST-ONLY & SENSITIVITY)
# ==============================================================================
df_test_m2 = df_results_m2[df_results_m2["split"] == "test"].copy()
assert len(df_test_m2) == 92, f"Expected 92 test samples, got {len(df_test_m2)}"
assert df_test_m2['canonical_image_id'].nunique() == 35, f"Expected 35 test clusters, got {df_test_m2['canonical_image_id'].nunique()}"

print(f"📊 Primary Analysis: Test-Only Subset (N={len(df_test_m2)}, {df_test_m2['canonical_image_id'].nunique()} clusters)")
df_table_test_m2 = analyze_subset(df_test_m2, "Test-Only (N=92)")
display(df_table_test_m2)

print(f"\\n📊 Sensitivity Analysis: Full Benchmark Set (N={len(df_results_m2)}, {df_results_m2['canonical_image_id'].nunique()} clusters)")
df_table_all_m2 = analyze_subset(df_results_m2, "Full-Benchmark (N=150)")
display(df_table_all_m2)

df_final_m2 = pd.concat([df_table_test_m2, df_table_all_m2], ignore_index=True)
comp_path_m2 = out_dir_m2 / "estimator_comparison_huatuo7b.csv"
df_final_m2.to_csv(comp_path_m2, index=False)
print(f"💾 Saved estimator comparison: {comp_path_m2}")

fig_path_m2 = out_dir_m2 / "risk_coverage_curve_huatuo7b_test.png"
plot_risk_coverage_curves(df_test_m2, score_cols, fig_path_m2, "HuatuoGPT-Vision-7B (Test-Only N=92)")"""
    cells.append(nbf.v4.new_code_cell(c12_code))

    # =========================================================================
    # CELL 13: SIDE-BY-SIDE COMPARISON & CROSS-MODEL LEADERBOARD
    # =========================================================================
    c13_code = """# ==============================================================================
# CELL 13: 📊 FINAL THESIS COMPARATIVE LEADERBOARD (RQ1 - RQ3)
# ==============================================================================
# Primary evaluations strictly on Test-Only (N=92, 35 clusters)
t1 = df_results_m1[df_results_m1["split"] == "test"]
t2 = df_results_m2[df_results_m2["split"] == "test"]

assert len(t1) == 92, f"Model 1 test count must be 92, got {len(t1)}"
assert len(t2) == 92, f"Model 2 test count must be 92, got {len(t2)}"

st1 = df_table_test_m1.set_index("estimator")
st2 = df_table_test_m2.set_index("estimator")

comparative_rows = [
    {
        "Research Question": "RQ1: English Accuracy (Test-Only N=92)",
        "Qwen2.5-VL-7B": f"{t1['correct_en'].mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{t2['correct_en'].mean()*100:.2f}%",
        "Interpretation": "Performa pada radiograf asli dalam bahasa sumber (EN)"
    },
    {
        "Research Question": "RQ1: Indonesian Accuracy (Test-Only N=92)",
        "Qwen2.5-VL-7B": f"{t1['correct_id'].mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{t2['correct_id'].mean()*100:.2f}%",
        "Interpretation": "Performa pada radiograf asli di bawah translasi (ID)"
    },
    {
        "Research Question": "RQ1: Language Gap (EN - ID)",
        "Qwen2.5-VL-7B": f"{(t1['correct_en'].mean() - t1['correct_id'].mean())*100:+.2f}%",
        "HuatuoGPT-Vision-7B": f"{(t2['correct_en'].mean() - t2['correct_id'].mean())*100:+.2f}%",
        "Interpretation": "Penurunan akurasi akibat translasi bahasa (positif = degradasi)"
    },
    {
        "Research Question": "RQ1: Cross-Lingual Flip Rate",
        "Qwen2.5-VL-7B": f"{t1['flip'].mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{t2['flip'].mean()*100:.2f}%",
        "Interpretation": "Proporsi sampel di mana jawaban model berubah antara EN dan ID"
    },
    {
        "Research Question": "RQ1: Joint Bilingual Correctness",
        "Qwen2.5-VL-7B": f"{t1['joint_correct'].mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{t2['joint_correct'].mean()*100:.2f}%",
        "Interpretation": "Konsistensi menjawab benar pada kedua bahasa sekaligus"
    },
    {
        "Research Question": "RQ2: Visual Sensitivity (Mean Delta P_dir)",
        "Qwen2.5-VL-7B": f"{t1['delta_p_dir'].mean():+.4f}",
        "HuatuoGPT-Vision-7B": f"{t2['delta_p_dir'].mean():+.4f}",
        "Interpretation": "Dukungan visual terarah (P0 Original vs P1 Black-Blank)"
    },
    {
        "Research Question": "RQ3: M1 Confidence-Only AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m1', 'aurc_discrete']:.4f}",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m1', 'aurc_discrete']:.4f}",
        "Interpretation": "Baseline selective risk menggunakan kepercayaan asli"
    },
    {
        "Research Question": "RQ3: M_dir Directed Multiplicative AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m_dir', 'aurc_discrete']:.4f} (p={st1.loc['score_m_dir', 'p_value']:.4f})",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m_dir', 'aurc_discrete']:.4f} (p={st2.loc['score_m_dir', 'p_value']:.4f})",
        "Interpretation": "Estimator utama terdaftar: conf * max(0, Delta)"
    },
    {
        "Research Question": "RQ3: M_norm Normalized Linear AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m_norm', 'aurc_discrete']:.4f} (p={st1.loc['score_m_norm', 'p_value']:.4f})",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m_norm', 'aurc_discrete']:.4f} (p={st2.loc['score_m_norm', 'p_value']:.4f})",
        "Interpretation": "Estimator linear terarah: conf * (1 + Delta) / 2"
    },
    {
        "Research Question": "RQ3: M2 Visual-Only AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m2', 'aurc_discrete']:.4f} (p={st1.loc['score_m2', 'p_value']:.4f})",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m2', 'aurc_discrete']:.4f} (p={st2.loc['score_m2', 'p_value']:.4f})",
        "Interpretation": "Estimator visual terisolasi: max(0, Delta)"
    },
    {
        "Research Question": "RQ3: M3_old Historical Unsigned AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m3_old', 'aurc_discrete']:.4f} (p={st1.loc['score_m3_old', 'p_value']:.4f})",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m3_old', 'aurc_discrete']:.4f} (p={st2.loc['score_m3_old', 'p_value']:.4f})",
        "Interpretation": "Baseline lama tidak terarah: conf * (1 + |Delta|) / 2"
    }
]

df_comparative = pd.DataFrame(comparative_rows)
print("\\n" + "=" * 90)
print("🏆 FINAL EMPIRICAL COMPARISON TABLE (TEST-ONLY N=92, 35 CLUSTERS)")
print("=" * 90)
display(df_comparative)

comp_save_path = RESULTS_DIR / "final_cross_model_comparison_test_only.csv"
df_comparative.to_csv(comp_save_path, index=False)
print(f"\\n💾 Saved final cross-model comparative table to: {comp_save_path}")"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # =========================================================================
    # CELL 14: PACKAGING & ARCHIVE CREATION (REV02)
    # =========================================================================
    c14_code = """# ==============================================================================
# CELL 14: PACKAGING & ARCHIVING RESULTS (REV02)
# ==============================================================================
import shutil
import zipfile
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
bundle_zip_name = f"decisive_validation_rev02_results_{timestamp}.zip"
bundle_zip_path = WORKSPACE / bundle_zip_name

print(f"📦 Packaging all rev02 results from {RESULTS_DIR} into: {bundle_zip_path}...")
with zipfile.ZipFile(bundle_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(RESULTS_DIR):
        for file in files:
            full_p = Path(root) / file
            rel_p = full_p.relative_to(RESULTS_DIR)
            zipf.write(full_p, arcname=str(Path("rev02") / rel_p))

zip_size_mb = bundle_zip_path.stat().st_size / (1024**2)
print(f"✅ Created downloadable archive: {bundle_zip_name} ({zip_size_mb:.2f} MB)")
print(f"📍 Location on Kaggle: {bundle_zip_path.resolve()}")

# Also create fixed-name copy for convenience
fixed_zip = WORKSPACE / "decisive_validation_rev02_latest.zip"
shutil.copy2(bundle_zip_path, fixed_zip)

try:
    from IPython.display import FileLink, display
    print("\\n🔗 Clickable Download Links (Direct Browser Download):")
    display(FileLink(bundle_zip_name))
    display(FileLink("decisive_validation_rev02_latest.zip"))
except Exception as e:
    pass

print("\\n💡 PANDUAN MENGUNDUH SELURUH FILE OUTPUT DI KAGGLE:")
print("1. Cara 1 (Langsung): Klik link 'FileLink' di atas untuk mengunduh langsung dari browser.")
print("2. Cara 2 (Output Sidebar): Di panel sebelah kanan layar Kaggle, buka tab 'Output'.")
print(f"   - Temukan '{bundle_zip_name}' atau 'decisive_validation_rev02_latest.zip'.")
print("   - Klik tombol titik tiga (...) di samping file, lalu pilih 'Download'.")
print("   - Anda juga dapat mengunduh individual file CSV atau PNG di dalam folder 'results/'.")
print("3. Ekstrak file zip ke direktori lokal proyek:")
print("   c:\\\\.PUSAT\\\\Project\\\\Kuliah\\\\VLM\\\\results\\\\decisive_validation\\\\")"""
    cells.append(nbf.v4.new_code_cell(c14_code))

    nb.cells = cells
    return nb


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path("main/Kaggle")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "05_decisive_validation_revised.ipynb"
    
    nb = build_kaggle_notebook()
    with open(out_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"[OK] Successfully built Kaggle notebook at: {out_nb_path}")
    print(f"Total cells: {len(nb.cells)}")
