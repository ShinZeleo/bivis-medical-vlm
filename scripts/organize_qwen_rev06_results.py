import sys
import shutil
import hashlib
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

src_dir = Path("Qwen_Qwen2.5_VL_7B_Instruct_complete_validation_results")
base_rev06 = Path("artifacts/experiments/decisive_validation/rev06")

pred_dir = base_rev06 / "predictions" / "qwen"
metrics_dir = base_rev06 / "metrics" / "qwen"
figures_dir = base_rev06 / "figures" / "qwen"

pred_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

manifest_records = []

def record_file(src_path, dst_path, category):
    shutil.copy2(src_path, dst_path)
    file_bytes = dst_path.read_bytes()
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    manifest_records.append({
        "filename": dst_path.name,
        "category": category,
        "rel_path": str(dst_path.relative_to(base_rev06)),
        "size_bytes": len(file_bytes),
        "sha256": sha256
    })
    print(f"📁 Organized [{category}]: {dst_path.name} -> {dst_path}")

# 1. Prediction Checkpoint
ckpt_file = src_dir / "live_validation_checkpoint_Qwen_Qwen2.5_VL_7B_Instruct.csv"
if ckpt_file.exists():
    record_file(ckpt_file, pred_dir / ckpt_file.name, "Prediction Checkpoint")

# 2. Metrics CSVs
for m_name in [
    "rq1_bilingual_metrics_Qwen_Qwen2.5_VL_7B_Instruct.csv",
    "rq2_visual_intervention_Qwen_Qwen2.5_VL_7B_Instruct.csv",
    "rq3_selective_prediction_Qwen_Qwen2.5_VL_7B_Instruct.csv",
    "risk_ablation_Qwen_Qwen2.5_VL_7B_Instruct.csv"
]:
    m_file = src_dir / m_name
    if m_file.exists():
        record_file(m_file, metrics_dir / m_name, "Evaluation Metric")

# 3. Figures
fig_file = src_dir / "risk_coverage_primary_test_Qwen2.5_VL_7B.png"
if fig_file.exists():
    record_file(fig_file, figures_dir / fig_file.name, "Figure / Plot")

# Write manifest for organized rev06 artifacts
df_manifest = pd.DataFrame(manifest_records)
manifest_csv = base_rev06 / "qwen_artifacts_manifest.csv"
df_manifest.to_csv(manifest_csv, index=False)
print(f"\n✅ Artifact manifest saved: {manifest_csv}")

# Remove temporary zip in root if exists
temp_zip = Path("Qwen_Qwen2.5_VL_7B_Instruct_complete_validation_results.zip")
if temp_zip.exists():
    temp_zip.unlink()
    print("🧹 Cleaned up temporary zip in root directory.")

print("🎉 Successfully organized all Qwen rev06 experimental results!")
