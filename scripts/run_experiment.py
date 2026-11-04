#!/usr/bin/env python3
"""
CLI entry point to execute VLM decisive validation experiments and selective prediction evaluation.
"""

import sys
import argparse
from pathlib import Path

# Add src to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description="Run Medical VLM Decisive Validation Experiment")
    parser.add_argument("--model", type=str, default="all", choices=["qwen", "huatuo", "all"],
                        help="Model to evaluate: 'qwen' (Qwen2.5-VL-7B), 'huatuo' (HuatuoGPT-Vision-7B), or 'all'")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--dry-run", action="store_true", help="Execute dry-run without downloading model weights")
    args = parser.parse_args()

    print(f"Initializing VLM Experiment Runner for model selection: {args.model}")
    print(f"Project root: {PROJECT_ROOT}")
    
    if args.dry_run:
        print("Dry run requested. Executing preflight unit test suite instead...")
        import subprocess
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests" / "test_decisive_validation.py")])
        sys.exit(result.returncode)
    else:
        # Import and execute decisive validation
        from scripts.run_decisive_validation import main as run_engine
        run_engine()

if __name__ == "__main__":
    main()
