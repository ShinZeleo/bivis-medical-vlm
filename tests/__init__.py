"""
Test suite for VLM Research package and selective prediction pipeline.
"""
import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH for all test modules
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

