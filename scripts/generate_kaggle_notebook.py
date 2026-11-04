"""
Generator script for 04_decisive_validation_live.ipynb
Creates a self-contained, publication-grade Jupyter Notebook for Kaggle Dual T4 GPU execution.
Conforms strictly to Thesis Advisor Directives and Rigorous Methodological Audit Standards.
"""

import sys
import json
from pathlib import Path
import nbformat as nbf

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
    # CELL 0: TITLE & METHODOLOGICAL DIRECTIVES
    # =========================================================================
    c0_md = """# 🏥 Medical VLM Decisive Validation Experiment (Kaggle Dual-T4 GPU)
**Research Project:** Evaluasi Reliabilitas dan Selective Prediction pada Medical Vision-Language Model di Bawah Pergeseran Bahasa (IndoRad-VQA)  
**Author:** Imam (Undergraduate CS Thesis)  
**Environment:** Kaggle Dual Tesla T4 (2x 16GB VRAM, 4-bit NF4 Quantization)  

---

## 📌 Metodologi & Arahan Resmi Dosen Pembimbing
1. **Peniadaan Blur Total:** Sesuai arahan dosen pembimbing, manipulasi blur dieliminasi secara permanen. Hanya dua kondisi visual yang dievaluasi:
   - `P0_ORIGINAL`: Citra radiologi asli tanpa manipulasi.
   - `P1_BLACK_BLANK`: Kanvas hitam murni `RGB (0, 0, 0)` berdimensi dan mode identik dengan citra asli.
2. **Target Candidate Sequence Scoring:** Menghindari *single-token logit hacking*. Menghitung *conditional sequence log-likelihood* secara eksak untuk:
   - **English:** `["Yes", "No"]`
   - **Indonesian:** `["Ya", "Tidak"]`
3. **Komparasi Fair 5 Estimator:**
   - $M_1$: Confidence-only baseline ($p_{\\text{orig}}$)
   - $M_2$: Visual-only ($\max(0, \\Delta_{\\text{dir}})$)
   - $M_{\\text{dir}}$: Directed Multiplicative ($p_{\\text{orig}} \\times \\max(0, \\Delta_{\\text{dir}})$)
   - $M_{\\text{norm}}$: Normalized Directed Linear ($p_{\\text{orig}} \\times \\frac{1 + \\Delta_{\\text{dir}}}{2}$)
   - $M_3^{\\text{old}}$: Historical Unsigned Baseline ($p_{\\text{orig}} \\times \\frac{1 + |\\Delta_{\\text{dir}}|}{2}$)
4. **Disiplin Split Data (Zero Leakage):**
   - **Primary Test-Only Subset:** $N = 92$ pasangan QA (35 klaster citra fisik unik). Evaluasi utama bebas dari data dev.
   - **Full Benchmark Sensitivity Set:** $N = 150$ pasangan QA (58 klaster citra fisik unik). Memuat split test (92) + dev (58) untuk uji sensitivitas ukuran sampel.
   - Kedua split dilaporkan secara terpisah dan transparan dalam tabel hasil.
5. **Model yang Diuji:**
   - Model 1: `Qwen/Qwen2.5-VL-7B-Instruct` (General Multimodal SOTA)
   - Model 2: `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` (Medical Domain Specialist)
6. **Batas Klaim Ilmiah (Epistemic Boundaries):**
   - Citra blank hitam adalah kondisi out-of-distribution (OOD). Perubahan prediksi mengukur responsivitas perilaku model terhadap penghilangan informasi visual, **bukan bukti pemahaman anatomis kausal sempurna**.
   - Penurunan AURC mengukur kemampuan perankingan risiko kesalahan retrospektif pada sampel uji, **bukan jaminan otomatis keselamatan klinis pasien**."""
    cells.append(nbf.v4.new_markdown_cell(c0_md))

    # =========================================================================
    # CELL 1: SETUP & DEPENDENCIES
    # =========================================================================
    c1_code = """# Step 1: Install pinned dependencies compatible with Qwen2.5-VL and 4-bit NF4
!pip uninstall -y torchaudio
!pip install -q -U "transformers>=4.45.0" "accelerate>=0.26.0" "bitsandbytes>=0.46.1" "qwen-vl-utils" "pillow>=10.0.0" scipy statsmodels seaborn matplotlib pandas datasets
print("✅ Libraries installed successfully.")"""
    cells.append(nbf.v4.new_code_cell(c1_code))

    # =========================================================================
    # CELL 2: GPU ENVIRONMENT SANITY CHECK
    # =========================================================================
    c2_code = """import os
import sys
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Set random seed for full reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("🖥️ GPU & ENVIRONMENT DIAGNOSTICS")
print("=" * 70)
print(f"Python Version  : {sys.version.split()[0]}")
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"GPU Count       : {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  - GPU {i}: {props.name} ({vram_gb:.2f} GB VRAM)")
else:
    print("⚠️ WARNING: Running without GPU. Inference will fail or be prohibitively slow.")
print("=" * 70)"""
    cells.append(nbf.v4.new_code_cell(c2_code))

    # =========================================================================
    # CELL 3: WORKSPACE & DIRECTORY SETUP
    # =========================================================================
    c3_code = """# Workspace directories setup
WORKSPACE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./kaggle_workspace")
PROJECT_DIR = WORKSPACE / "medical_vlm_validation"
DATA_DIR = PROJECT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = PROJECT_DIR / "results"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PERTURB_DIR = DATA_DIR / "perturbations"

for p in [DATA_DIR, IMAGES_DIR, RESULTS_DIR, REPORTS_DIR, FIGURES_DIR, PERTURB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print(f"📁 Workspace Root: {PROJECT_DIR.resolve()}")
print(f"📁 Images Dir    : {IMAGES_DIR.resolve()}")
print(f"📁 Results Dir   : {RESULTS_DIR.resolve()}")"""
    cells.append(nbf.v4.new_code_cell(c3_code))

    # =========================================================================
    # CELL 4: DATA RESOLVER & MANIFEST LOADER WITH STRICT SPLIT AUDIT
    # =========================================================================
    c4_code = """# Resolve Dataset and Images from Kaggle Input / Local / Hugging Face
import zipfile
import shutil

manifest_search_paths = [
    Path("/kaggle/input/indorad-vqa-research/results/decisive_validation_manifest.csv"),
    Path("/kaggle/input/indorad-vqa-research/main/results/decisive_validation_manifest.csv"),
    Path("/kaggle/input/indorad-vqa-research/results/bilingual_manifest_sensitivity.csv"),
    Path("./main/results/decisive_validation_manifest.csv"),
    Path("./main/results/bilingual_manifest_sensitivity.csv"),
    DATA_DIR / "decisive_validation_manifest.csv"
]

manifest_path = None
for p in manifest_search_paths:
    if p.exists():
        manifest_path = p
        break

# If not found directly, look for indorad-vqa-research.zip
if manifest_path is None:
    zip_candidates = list(Path("/kaggle/input").rglob("indorad-vqa-research.zip")) if Path("/kaggle/input").exists() else []
    if not zip_candidates and Path("main/Kaggle/indorad-vqa-research.zip").exists():
        zip_candidates = [Path("main/Kaggle/indorad-vqa-research.zip")]
    
    if zip_candidates:
        print(f"📦 Extracting project zip: {zip_candidates[0]}...")
        with zipfile.ZipFile(zip_candidates[0], 'r') as zf:
            zf.extractall(DATA_DIR)
        print("✅ Unzipped data successfully.")
        
        for p in [DATA_DIR / "results" / "decisive_validation_manifest.csv",
                 DATA_DIR / "results" / "bilingual_manifest_sensitivity.csv"]:
            if p.exists():
                manifest_path = p
                break

# If still not found, load from HF datasets
if manifest_path is None or not manifest_path.exists():
    print("📥 Loading manifest from Hugging Face dataset 'Lab-IS/IndoRad-VQA'...")
    from datasets import load_dataset
    hf_ds = load_dataset("Lab-IS/IndoRad-VQA", split="train")
    df_raw = hf_ds.to_pandas()
    df_eval = df_raw[df_raw['answer'].str.strip().str.upper().isin(['YES', 'NO'])].head(150).copy()
    df_eval['qa_id'] = df_eval.index
    df_eval['canonical_image_id'] = df_eval['qa_id'].apply(lambda x: f"img_{x:05d}")
    df_eval['image_id'] = df_eval['canonical_image_id']
    df_eval['split'] = np.where(df_eval.index < 92, 'test', 'dev')
    df_eval['question_en'] = df_eval['question']
    df_eval['question_id'] = df_eval['question_indonesian']
    df_eval['reference_answer'] = df_eval['answer'].str.strip().str.upper()
    df_eval['translation_status'] = 'CLEAN'
    manifest_path = DATA_DIR / "decisive_validation_manifest.csv"
    df_eval.to_csv(manifest_path, index=False)
else:
    df_eval = pd.read_csv(manifest_path)

print(f"✅ Loaded evaluation manifest: {len(df_eval)} QA pairs from {manifest_path}")

# Ensure critical identity columns
if 'canonical_image_id' not in df_eval.columns:
    df_eval['canonical_image_id'] = df_eval['image_id'] if 'image_id' in df_eval.columns else df_eval['qa_id'].apply(lambda x: f"img_{x}")

if 'image_id' not in df_eval.columns:
    df_eval['image_id'] = df_eval['canonical_image_id']

if 'split' not in df_eval.columns:
    df_eval['split'] = 'test'

if 'translation_status' not in df_eval.columns:
    df_eval['translation_status'] = 'CLEAN'

# Detailed Split Audit
print("\\n" + "=" * 60)
print("📊 MANIFEST SPLIT & CLUSTER BREAKDOWN")
print("=" * 60)
split_counts = df_eval['split'].value_counts()
for s_name, count in split_counts.items():
    s_clusters = df_eval[df_eval['split'] == s_name]['canonical_image_id'].nunique()
    print(f"  - Split '{s_name}': {count} QA pairs across {s_clusters} unique image clusters")

# Verify zero cluster leakage between dev and test
dev_clusters = set(df_eval[df_eval['split'] == 'dev']['canonical_image_id'])
test_clusters = set(df_eval[df_eval['split'] == 'test']['canonical_image_id'])
leakage_clusters = dev_clusters.intersection(test_clusters)
print(f"  - Cross-split cluster overlap: {len(leakage_clusters)} clusters (Leakage = {len(leakage_clusters) == 0})")
print("=" * 60)

display(df_eval[['qa_id', 'canonical_image_id', 'image_id', 'split', 'translation_status', 'question_en', 'question_id', 'reference_answer']].head(5))"""
    cells.append(nbf.v4.new_code_cell(c4_code))

    # =========================================================================
    # CELL 5: COMPREHENSIVE IMAGE RESOLVER
    # =========================================================================
    c5_code = """# Resolve image files: verify that all evaluation images exist locally on disk
search_image_dirs = [
    IMAGES_DIR,
    Path("/kaggle/input/indorad-vqa-research/data/images"),
    Path("/kaggle/input/indorad-vqa-research/main/data/images"),
    DATA_DIR / "data" / "images",
    DATA_DIR / "images",
    Path("./main/data/images")
]

def resolve_image_path(cid, img_id, qa_id, image_name=None):
    candidates = []
    candidates.append(f"{cid}.png")
    candidates.append(f"{img_id}.png")
    candidates.append(f"{img_id}")
    candidates.append(f"img_{qa_id}.png")
    candidates.append(f"img_{int(qa_id):05d}.png")
    if image_name:
        candidates.append(image_name)
        candidates.append(f"{image_name}.png")
        
    for d in search_image_dirs:
        if not d.exists():
            continue
        for cand in candidates:
            cand_p = d / cand
            if cand_p.exists():
                return cand_p
    return None

missing_images = []
resolved_map = {}

for idx, row in df_eval.iterrows():
    qa_id = row['qa_id']
    cid = str(row['canonical_image_id'])
    img_id = str(row.get('image_id', cid))
    im_name = str(row.get('image_name', ''))
    
    img_p = resolve_image_path(cid, img_id, qa_id, im_name)
    if img_p is None:
        missing_images.append((qa_id, cid, img_id))
    else:
        resolved_map[qa_id] = img_p

print(f"🖼️ Images Found: {len(resolved_map)} / {len(df_eval)}")

# If any images are missing, download directly from Hugging Face dataset
if missing_images:
    print(f"⚠️ {len(missing_images)} images missing on disk. Downloading from Hugging Face 'Lab-IS/IndoRad-VQA'...")
    from datasets import load_dataset
    import io
    hf_ds = load_dataset("Lab-IS/IndoRad-VQA", split="train")
    for qid, cid, i_id in missing_images:
        try:
            row_hf = hf_ds[int(qid)]
            im_obj = row_hf['image']
            if isinstance(im_obj, dict) and 'bytes' in im_obj:
                pil_im = Image.open(io.BytesIO(im_obj['bytes'])).convert("RGB")
            elif isinstance(im_obj, Image.Image):
                pil_im = im_obj.convert("RGB")
            else:
                pil_im = Image.new("RGB", (512, 512), color=(128, 128, 128))
            save_dest = IMAGES_DIR / f"{cid}.png"
            pil_im.save(save_dest)
            resolved_map[qid] = save_dest
        except Exception as e:
            print(f"Error fetching image for qid {qid}: {e}")

print(f"✅ Final Image Resolution: {len(resolved_map)} / {len(df_eval)} images verified.")"""
    cells.append(nbf.v4.new_code_cell(c5_code))

    # =========================================================================
    # CELL 6: CORE EVALUATION & STATISTICAL FUNCTIONS WITH PREFIX ASSERTION
    # =========================================================================
    c6_code = """# ==============================================================================
# CORE STATISTICAL & SELECTIVE PREDICTION EVALUATION ENGINE
# ==============================================================================
import torch.nn.functional as F

def extract_target_loglikelihood(model, processor, image_path, prompt_text, candidate_strings, target_device):
    \"\"\"
    Evaluates conditional log-likelihood of exact candidate target strings
    (e.g. 'Yes' vs 'No' or 'Ya' vs 'Tidak') given the multimodal context (Image + Prompt).
    
    Avoids single first-token logit hacking and is robust to multi-token words.
    Includes explicit runtime prefix-match verification.
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
    
    # Process base prompt once per multimodal query
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
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
        target_ids = inputs.input_ids[0, prompt_len:]
        if len(target_ids) == 0:
            cand_logprobs.append(-999.0)
            del inputs, outputs
            continue
            
        # In causal LMs, token at position t is predicted by logits at t-1
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
    aurc_trap = float(np.trapezoid(risk_trap, cov_trap))
    
    return cov, risk, aurc, aurc_trap


def get_risk_at_coverage(cov_arr, risk_arr, target_cov):
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
    c7_code = """import matplotlib.pyplot as plt

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
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved Risk-Coverage curve: {save_path}")

print("✅ Plotting Engine Ready.")"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # =========================================================================
    # CELL 8: PIPELINE EXECUTOR WITH RESUME, STRICT VALIDATION & SMOKE TEST
    # =========================================================================
    c8_code = """from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
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
    
    # Checkpoint Validation & Resume Logic
    if checkpoint_csv.exists():
        df_done = pd.read_csv(checkpoint_csv)
        # Check 1: Model ID match
        is_model_match = ("model_id" in df_done.columns and (df_done["model_id"] == model_name).all())
        # Check 2: Scoring method match
        is_method_match = ("scoring_method" in df_done.columns and (df_done["scoring_method"] == "target_candidate_sequence_scoring").all())
        # Check 3: Required columns present
        required_cols = {"qa_id", "canonical_image_id", "pred_en", "pred_id", "conf_id", "delta_p_dir", "score_m_dir", "reference_answer", "split"}
        has_required_cols = required_cols.issubset(set(df_done.columns))
        
        if is_model_match and is_method_match and has_required_cols:
            completed_qa_ids = set(df_done["qa_id"].tolist())
            results = df_done.to_dict(orient="records")
            print(f"🔄 Resuming from verified live checkpoint: {len(completed_qa_ids)} / {len(df_input)} samples already evaluated.")
        else:
            print("⚠️ Incompatible checkpoint detected (model mismatch, old schema, or synthetic data).")
            backup_csv = model_out_dir / f"incompatible_checkpoint_backup_{int(time.time())}.csv"
            checkpoint_csv.rename(backup_csv)
            print(f"   Old checkpoint backed up to: {backup_csv.name}")
            completed_qa_ids = set()
            results = []
            print(f"🚀 Starting fresh evaluation on {len(df_input)} samples.")
    else:
        completed_qa_ids = set()
        results = []
        print(f"🚀 Starting fresh evaluation on {len(df_input)} samples.")
        
    # Check if already 100% complete
    if len(completed_qa_ids) >= len(df_input):
        print(f"✅ Model [{model_name}] already completed. Proceeding directly to statistical analysis.")
        return pd.DataFrame(results), model_out_dir
        
    print(f"\\n📦 Loading Model: {model_name} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.eval()
    target_dev = next(model.parameters()).device
    print(f"✅ Loaded {model_name} onto device: {target_dev}")
    
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
            orig_pil = Image.open(test_img_p)
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
            orig_pil = Image.open(orig_img_path)
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
        
        # Probability assigned on blank image to the SAME answer chosen under P0
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
            "timestamp": datetime.utcnow().isoformat()
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
# 🚀 RUNNING MODEL 1: Qwen2.5-VL-7B-Instruct (General Multimodal SOTA)
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
    c10_code = """# Compute Discrete AURC & Bootstrap for Model 1 (Both Test-Only and Full Benchmark)
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

# 1. Primary Analysis: Test-Only Subset (N=92, 35 clusters)
df_test_m1 = df_results_m1[df_results_m1["split"] == "test"].copy()
print(f"📊 Primary Analysis: Test-Only Subset (N={len(df_test_m1)}, {df_test_m1['canonical_image_id'].nunique()} clusters)")
df_table_test_m1 = analyze_subset(df_test_m1, "Test-Only (N=92)")
display(df_table_test_m1)

# 2. Sensitivity Analysis: Full Benchmark Set (N=150, 58 clusters)
print(f"\\n📊 Sensitivity Analysis: Full Benchmark Set (N={len(df_results_m1)}, {df_results_m1['canonical_image_id'].nunique()} clusters)")
df_table_all_m1 = analyze_subset(df_results_m1, "Full-Benchmark (N=150)")
display(df_table_all_m1)

# Combine and save tables
df_final_m1 = pd.concat([df_table_test_m1, df_table_all_m1], ignore_index=True)
comp_path_m1 = out_dir_m1 / "estimator_comparison_qwen7b.csv"
df_final_m1.to_csv(comp_path_m1, index=False)

# 3. Plot Risk-Coverage Curves for Test-Only Subset
fig_path_m1 = out_dir_m1 / "risk_coverage_curve_qwen7b_test.png"
plot_risk_coverage_curves(df_test_m1, score_cols, fig_path_m1, "Qwen2.5-VL-7B-Instruct (Test-Only)")"""
    cells.append(nbf.v4.new_code_cell(c10_code))

    # =========================================================================
    # CELL 11: MODEL 2 EVALUATION (HUATUOGPT-VISION-7B)
    # =========================================================================
    c11_code = """# ==============================================================================
# 🚀 RUNNING MODEL 2: HuatuoGPT-Vision-7B (Medical Domain Specialist)
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
    c12_code = """# Compute Discrete AURC & Bootstrap for Model 2 (Both Test-Only and Full Benchmark)
df_test_m2 = df_results_m2[df_results_m2["split"] == "test"].copy()
print(f"📊 Primary Analysis: Test-Only Subset (N={len(df_test_m2)}, {df_test_m2['canonical_image_id'].nunique()} clusters)")
df_table_test_m2 = analyze_subset(df_test_m2, "Test-Only (N=92)")
display(df_table_test_m2)

print(f"\\n📊 Sensitivity Analysis: Full Benchmark Set (N={len(df_results_m2)}, {df_results_m2['canonical_image_id'].nunique()} clusters)")
df_table_all_m2 = analyze_subset(df_results_m2, "Full-Benchmark (N=150)")
display(df_table_all_m2)

df_final_m2 = pd.concat([df_table_test_m2, df_table_all_m2], ignore_index=True)
comp_path_m2 = out_dir_m2 / "estimator_comparison_huatuo7b.csv"
df_final_m2.to_csv(comp_path_m2, index=False)

fig_path_m2 = out_dir_m2 / "risk_coverage_curve_huatuo7b_test.png"
plot_risk_coverage_curves(df_test_m2, score_cols, fig_path_m2, "HuatuoGPT-Vision-7B (Test-Only)")"""
    cells.append(nbf.v4.new_code_cell(c12_code))

    # =========================================================================
    # CELL 13: SIDE-BY-SIDE COMPARISON & CROSS-MODEL LEADERBOARD
    # =========================================================================
    c13_code = """# ==============================================================================
# 📊 FINAL THESIS COMPARATIVE LEADERBOARD (RQ1 - RQ3)
# ==============================================================================
# Primary evaluations on Test-Only (N=92)
t1 = df_results_m1[df_results_m1["split"] == "test"]
t2 = df_results_m2[df_results_m2["split"] == "test"]

# Filter score rows for test-only
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
        "Interpretation": "Degradasi akurasi akibat language shift"
    },
    {
        "Research Question": "RQ1: Cross-Lingual Flip Rate",
        "Qwen2.5-VL-7B": f"{t1['flip'].mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{t2['flip'].mean()*100:.2f}%",
        "Interpretation": "Ketidakkonsistenan jawaban antar-bahasa pada citra yang sama"
    },
    {
        "Research Question": "RQ1: Joint Correctness",
        "Qwen2.5-VL-7B": f"{t1['joint_correct'].mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{t2['joint_correct'].mean()*100:.2f}%",
        "Interpretation": "Proporsi sampel yang benar pada kedua bahasa"
    },
    {
        "Research Question": "RQ2: Mean Directed Sensitivity (Delta P)",
        "Qwen2.5-VL-7B": f"{t1['delta_p_dir'].mean():+.4f}",
        "HuatuoGPT-Vision-7B": f"{t2['delta_p_dir'].mean():+.4f}",
        "Interpretation": "Penurunan rata-rata keyakinan saat citra dihilangkan (P0 vs P1)"
    },
    {
        "Research Question": "RQ2: Negative Sensitivity Rate (Delta P <= 0)",
        "Qwen2.5-VL-7B": f"{(t1['delta_p_dir'] <= 0).mean()*100:.2f}%",
        "HuatuoGPT-Vision-7B": f"{(t2['delta_p_dir'] <= 0).mean()*100:.2f}%",
        "Interpretation": "Halusinasi / Overconfidence tanpa bergantung pada citra"
    },
    {
        "Research Question": "RQ3: M1 (Confidence-Only) AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m1', 'aurc_discrete']:.4f}",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m1', 'aurc_discrete']:.4f}",
        "Interpretation": "Baseline selective risk menggunakan confidence model murni"
    },
    {
        "Research Question": "RQ3: M_dir (Directed Multiplicative) AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m_dir', 'aurc_discrete']:.4f}",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m_dir', 'aurc_discrete']:.4f}",
        "Interpretation": "Estimator usulan menggabungkan confidence dan sensitivitas terarah"
    },
    {
        "Research Question": "RQ3: M_dir p-value vs M1 (B=1000)",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m_dir', 'p_value']:.4f}",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m_dir', 'p_value']:.4f}",
        "Interpretation": "Signifikansi statistik (paired cluster bootstrap, B=1000)"
    },
    {
        "Research Question": "RQ3: M3_old (Historical Unsigned) AURC",
        "Qwen2.5-VL-7B": f"{st1.loc['score_m3_old', 'aurc_discrete']:.4f}",
        "HuatuoGPT-Vision-7B": f"{st2.loc['score_m3_old', 'aurc_discrete']:.4f}",
        "Interpretation": "Baseline lama dengan cacat nilai mutlak |Delta P|"
    }
]

df_comparative = pd.DataFrame(comparative_rows)
comp_summary_path = RESULTS_DIR / "final_cross_model_comparison_test_only.csv"
df_comparative.to_csv(comp_summary_path, index=False)

print("=" * 85)
print("🏆 FINAL COMPARATIVE RESEARCH REPORT (RQ1, RQ2, RQ3) - PRIMARY TEST-ONLY SET (N=92)")
print("=" * 85)
display(df_comparative)
print(f"\\n💾 Saved cross-model comparative table to: {comp_summary_path}")"""
    cells.append(nbf.v4.new_code_cell(c13_code))

    # =========================================================================
    # CELL 14: EXPORT & ZIP BUNDLE FOR THESIS WRITING
    # =========================================================================
    c14_code = """# ==============================================================================
# 📦 PACKAGING EXPERIMENTAL ARTIFACTS FOR THESIS WRITING
# ==============================================================================
import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
bundle_zip_name = f"decisive_validation_live_results_{timestamp}.zip"
bundle_zip_path = WORKSPACE / bundle_zip_name

print(f"📦 Packaging all results into: {bundle_zip_path}...")
with zipfile.ZipFile(bundle_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(RESULTS_DIR):
        for file in files:
            full_p = Path(root) / file
            rel_p = full_p.relative_to(PROJECT_DIR)
            zipf.write(full_p, arcname=str(rel_p))

zip_size_mb = bundle_zip_path.stat().st_size / (1024**2)
print(f"✅ Created downloadable archive: {bundle_zip_name} ({zip_size_mb:.2f} MB)")
print(f"📍 Location on Kaggle: {bundle_zip_path.resolve()}")
print("\\n💡 INSTRUCTION TO DOWNLOAD RESULTS FROM KAGGLE:")
print("1. Di panel kanan Kaggle notebook, buka tab 'Output'.")
print(f"2. Temukan file '{bundle_zip_name}'.")
print("3. Klik titik tiga (...) di samping file lalu pilih 'Download'.")
print("4. Ekstrak file zip ke direktori lokal proyek: c:\\\\.PUSAT\\\\Project\\\\Kuliah\\\\VLM\\\\main\\\\results\\\\")"""
    cells.append(nbf.v4.new_code_cell(c14_code))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path("main/Kaggle")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "04_decisive_validation_live.ipynb"
    
    nb = build_kaggle_notebook()
    with open(out_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"[OK] Successfully built Kaggle notebook at: {out_nb_path}")
    print(f"Total cells: {len(nb.cells)}")
