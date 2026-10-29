#!/usr/bin/env python3
"""
Executes 10_standalone_aggregation_evaluation.ipynb end-to-end
and saves executed outputs directly into the notebook.
"""
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import nbformat
from nbclient import NotebookClient

nb_path = Path("main/notebooks/kaggle/10_standalone_aggregation_evaluation.ipynb")
print(f"📖 Loading notebook: {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)

print(f"🚀 Starting NotebookClient execution (timeout=600s)...")
t0 = time.time()
client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()
elapsed = time.time() - t0
print(f"⏱️ Notebook executed successfully in {elapsed:.2f} seconds!")

print(f"💾 Saving executed outputs in-place to {nb_path}...")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("🎉 In-place execution and output saving complete!")
