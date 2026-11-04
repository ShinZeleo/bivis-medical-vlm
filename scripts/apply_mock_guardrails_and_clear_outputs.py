import sys, json
from pathlib import Path

# 1. Update notebook 10 rev06
nb_path = Path("notebooks/kaggle/rev06/10_standalone_aggregation_evaluation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_3_src = """# ==============================================================================
# STAGE 3: AUTHORITATIVE FROZEN MANIFEST ENFORCEMENT
# ==============================================================================
import zlib
import base64
import hashlib
import io
import pandas as pd

def load_and_verify_manifest():
    search_paths = [
        Path("data/manifests/manifest_rev05.csv"),
        Path("artifacts/experiments/decisive_validation/rev05/manifest_rev05.csv"),
        Path("artifacts/experiments/decisive_validation/rev05/metrics/manifest_rev05.csv"),
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
        if p.exists() and p.is_file():
            raw_bytes = p.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() == AUTHORITATIVE_MANIFEST_SHA256:
                df_manifest = pd.read_csv(io.BytesIO(raw_bytes))
                loaded_source = str(p)
                break
                
    if df_manifest is None:
        if not EMBEDDED_MANIFEST_B64:
            raise RuntimeError(f"❌ Authoritative manifest with SHA-256 {AUTHORITATIVE_MANIFEST_SHA256} not found in search paths and embedded fallback unavailable!")
        print("ℹ️ Local manifest file not found. Decompressing embedded base64 verified manifest...")
        try:
            decompressed_bytes = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64))
            calc_sha = hashlib.sha256(decompressed_bytes).hexdigest()
            if calc_sha != AUTHORITATIVE_MANIFEST_SHA256:
                raise RuntimeError(f"❌ Embedded manifest hash mismatch! Expected {AUTHORITATIVE_MANIFEST_SHA256}, got {calc_sha}")
            df_manifest = pd.read_csv(io.BytesIO(decompressed_bytes))
            loaded_source = "EMBEDDED_VERIFIED_BASE64"
        except Exception as e:
            raise RuntimeError(f"❌ Failed to decompress/verify embedded manifest: {e}")
        
    if len(df_manifest) != 2244:
        raise RuntimeError(f"❌ Manifest row count mismatch! Expected 2244 total QA rows, got {len(df_manifest)}")
    print(f"✅ Authoritative Manifest 100% Cryptographically Verified ({loaded_source})")
    return df_manifest

df_manifest = load_and_verify_manifest()
"""

cell_4_src = """# ==============================================================================
# STAGE 3: DUAL-MODEL CHECKPOINT DISCOVERY & STRICT INTEGRITY AUDIT
# ==============================================================================
MODEL_1_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_2_NAME = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"

def discover_checkpoint(model_name):
    slug = model_name.replace("/", "_").replace("-", "_")
    target_name = f"live_validation_checkpoint_{slug}.csv"
    
    candidates = [
        Path("artifacts/experiments/decisive_validation/rev06/predictions") / ("qwen" if "qwen" in slug.lower() and "huatuo" not in slug.lower() else "huatuo") / target_name,
        RESULTS_ROOT / slug / target_name,
        RESULTS_ROOT / target_name,
        Path("artifacts/experiments/decisive_validation/rev05/predictions") / ("qwen" if "qwen" in slug.lower() and "huatuo" not in slug.lower() else "huatuo") / target_name,
        Path("results/decisive_validation/rev05") / slug / target_name,
        Path("/kaggle/working") / slug / target_name,
        Path("/kaggle/working") / target_name,
        Path("main/results/decisive_validation/rev05") / slug / target_name
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        short_kw = "qwen" if ("qwen" in slug.lower() and "huatuo" not in slug.lower()) else "huatuo"
        for p in kaggle_input.rglob("*.csv"):
            nl = p.name.lower()
            if (short_kw in nl and "checkpoint" in nl) or slug.lower() in nl:
                candidates.insert(0, p)
                
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 1000:
            # Strictly reject mock checkpoints
            if "mock" in p.name.lower() or "mock" in str(p).lower():
                raise RuntimeError(f"❌ Mock checkpoint file rejected: {p}")
            df = pd.read_csv(p)
            if "is_mock" in df.columns and (df["is_mock"] == True).any():
                raise RuntimeError(f"❌ Mock checkpoint detected via is_mock column in {p}! Refusing mock data.")
            if (len(df) == 1133 and 
                df["revision_id"].iloc[0] in [REVISION_ID, "rev05b", "rev06"] and
                df["manifest_sha256"].iloc[0] == AUTHORITATIVE_MANIFEST_SHA256 and
                df["model_id"].iloc[0] == model_name and
                np.isfinite(df["score_m_norm_seq"]).all() and
                df["qa_id"].duplicated().sum() == 0):
                print(f"✅ Valid 100% complete checkpoint found for {model_name.split('/')[-1]}: {p}")
                return df
                
    raise FileNotFoundError(f"❌ Complete 1,133-row checkpoint for {model_name} not found!")

df_qwen = discover_checkpoint(MODEL_1_NAME)
df_huatuo = discover_checkpoint(MODEL_2_NAME)
print(f"📊 Qwen Checkpoint: {len(df_qwen)} rows, {len(df_qwen.columns)} columns")
print(f"📊 Huatuo Checkpoint: {len(df_huatuo)} rows, {len(df_huatuo.columns)} columns")
"""

nb["cells"][3]["source"] = [l + "\n" for l in cell_3_src.splitlines()]
nb["cells"][4]["source"] = [l + "\n" for l in cell_4_src.splitlines()]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 3 & 4 in notebook 10 rev06.")

# 2. Clear all outputs in rev06 notebooks
rev06_dir = Path("notebooks/kaggle/rev06")
for nb_file in rev06_dir.glob("*.ipynb"):
    with open(nb_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleared_count = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                cell["outputs"] = []
                cell["execution_count"] = None
                cleared_count += 1
    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"Cleared outputs for {nb_file.name}: {cleared_count} cells stripped.")
