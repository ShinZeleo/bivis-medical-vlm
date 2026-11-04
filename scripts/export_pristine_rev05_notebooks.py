import sys
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dst_dir = Path("notebooks/kaggle/rev05")
dst_dir.mkdir(parents=True, exist_ok=True)

mapping = [
    ("main/notebooks/kaggle/08_decisive_validation_rev05.ipynb", dst_dir / "08_qwen_decisive_validation_rev05.ipynb"),
    ("main/notebooks/kaggle/09_infer_huatuo_resume.ipynb", dst_dir / "09_huatuo_decisive_validation_rev05.ipynb"),
    ("main/notebooks/kaggle/10_standalone_aggregation_evaluation.ipynb", dst_dir / "10_standalone_aggregation_evaluation_rev05.ipynb")
]

for src_git, dst_file in mapping:
    print(f"Exporting {src_git} to {dst_file}...")
    res = subprocess.run(
        ["git", "show", f"v11.0-rev05-frozen:{src_git}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if res.returncode == 0:
        with open(dst_file, "wb") as f:
            f.write(res.stdout)
        print(f"✅ Successfully exported {dst_file.name} ({len(res.stdout)} bytes)")
    else:
        print(f"❌ Error exporting {src_git}: {res.stderr.decode('utf-8', errors='ignore')}")

print("Pristine rev05 notebooks export complete!")
