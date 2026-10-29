#!/usr/bin/env python3
"""
evaluate_candidate_model.py
===========================
Automated evaluation runner for Medical VLM candidates (e.g. Qwen2.5-VL-32B-Instruct,
HuatuoGPT-Vision-7B, Qwen2.5-VL-7B) on the frozen IndoRad-VQA bilingual manifest.

Supports:
- 4-bit NF4 quantization for high-capacity models (fits 32B onto Kaggle T4x2 / Colab A100/L4)
- Multi-GPU auto device mapping
- Frozen evaluation manifest (results/bilingual_pair_manifest.csv)
- Dual reporting: Primary Analysis (CLEAN pairs) + Sensitivity Analysis (ALL pairs)
- 95% Wilson Score confidence intervals
- Exact model checkpoint tagging in CSV and JSON outputs
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a candidate VLM on frozen IndoRad-VQA manifest.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct",
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-VL-32B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--manifest", type=str, default="results/bilingual_pair_manifest.csv",
                        help="Path to frozen evaluation manifest CSV")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save predictions and metrics")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of pairs to evaluate (for dry runs)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    return parser.parse_args()

def normalize_answer(pred_text: str) -> str:
    """Deterministic binary normalization conforming to protocol."""
    t = pred_text.strip().upper()
    if "YES" in t or "YA" in t:
        return "YES"
    elif "NO" in t or "TIDAK" in t:
        return "NO"
    return t[:10]

def wilson_ci(p: float, n: int, z: float = 1.96) -> list[float]:
    """Calculate 95% Wilson Score interval for a proportion."""
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    delta = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return [round(max(0.0, centre - delta) * 100, 2), round(min(1.0, centre + delta) * 100, 2)]

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / args.manifest
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🔬 CANDIDATE VLM EVALUATION PIPELINE")
    print(f"Model Under Test : {args.model_id}")
    print(f"Evaluation Manifest : {manifest_path}")
    print("=" * 75)

    if not manifest_path.exists():
        print(f"❌ Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    df_manifest = pd.read_csv(manifest_path)
    if args.limit is not None and args.limit > 0:
        df_manifest = df_manifest.head(args.limit)
    print(f"Loaded {len(df_manifest)} evaluation pairs.")

    # Lazy import of torch and transformers to keep script light if checking help
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    # Quantization setup: 4-bit NF4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print(f"\n📦 Loading model [{args.model_id}] with 4-bit NF4 quantization...")
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 512

    try:
        processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model.eval()
        print("✅ Model loaded successfully onto accelerator devices.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Token sets for binary confidence
    tokens_yes = [
        processor.tokenizer.encode("YES", add_special_tokens=False)[0],
        processor.tokenizer.encode(" YES", add_special_tokens=False)[0],
        processor.tokenizer.encode("Yes", add_special_tokens=False)[0],
        processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    ]
    tokens_no = [
        processor.tokenizer.encode("NO", add_special_tokens=False)[0],
        processor.tokenizer.encode(" NO", add_special_tokens=False)[0],
        processor.tokenizer.encode("No", add_special_tokens=False)[0],
        processor.tokenizer.encode(" No", add_special_tokens=False)[0]
    ]

    def infer_single(img_path_or_obj, q_text: str, lang: str):
        if lang == "en":
            prompt = f"Answer the radiology visual question based on the image. Provide only the answer.\nQuestion: {q_text}"
        else:
            prompt = f"Jawab pertanyaan radiologi berdasarkan gambar. Berikan hanya jawabannya.\nPertanyaan: {q_text}"

        messages = [{"role": "user", "content": [{"type": "image", "image": img_path_or_obj}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        target_dev = next(model.parameters()).device
        inputs = processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(target_dev)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False
            )

        gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        raw_ans = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        norm_ans = normalize_answer(raw_ans)

        # Normalized binary logit posterior
        if len(outputs.scores) > 0:
            first_logits = outputs.scores[0][0]
            probs = F.softmax(first_logits, dim=-1)
            prob_yes = sum(probs[t].item() for t in tokens_yes if t < len(probs))
            prob_no = sum(probs[t].item() for t in tokens_no if t < len(probs))
            total_mass = prob_yes + prob_no
            if total_mass > 1e-6:
                conf = max(prob_yes / total_mass, prob_no / total_mass)
            else:
                conf = 0.50
        else:
            conf = 0.50

        del inputs, outputs
        return raw_ans, norm_ans, float(conf)

    # Image lookup directory
    img_dir = project_root / "data" / "images"

    results = []
    print(f"\n🚀 Running bilingual inference on {len(df_manifest)} pairs...")
    for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc=f"Evaluating {args.model_id.split('/')[-1]}"):
        qa_id = row["qa_id"]
        img_id = row["image_id"]
        img_path = img_dir / f"{img_id}.png"

        # Check if local image exists
        if not img_path.exists():
            print(f"⚠️ Image {img_path} not found locally! Skipping qa_id={qa_id}")
            continue

        gt = str(row["reference_answer"]).strip().upper()
        raw_en, norm_en, conf_en = infer_single(str(img_path), row["question_en"], lang="en")
        raw_id, norm_id, conf_id = infer_single(str(img_path), row["question_id"], lang="id")

        corr_en = (norm_en == gt)
        corr_id = (norm_id == gt)
        flip = (norm_en != norm_id)
        joint_corr = (corr_en and corr_id)

        results.append({
            "qa_id": qa_id,
            "image_id": img_id,
            "pixel_sha256": row.get("pixel_sha256", ""),
            "translation_status": row.get("translation_status", "CLEAN"),
            "reference_answer": gt,
            "raw_answer_en": raw_en,
            "normalized_answer_en": norm_en,
            "confidence_en": conf_en,
            "correct_en": corr_en,
            "valid_format_en": norm_en in ["YES", "NO"],
            "raw_answer_id": raw_id,
            "normalized_answer_id": norm_id,
            "confidence_id": conf_id,
            "correct_id": corr_id,
            "valid_format_id": norm_id in ["YES", "NO"],
            "flip": flip,
            "joint_correct": joint_corr,
            "model_id": args.model_id
        })
        torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)

    # Save predictions
    safe_name = args.model_id.replace("/", "_")
    pred_path = output_dir / f"predictions_{safe_name}.csv"
    df_res.to_csv(pred_path, index=False)
    print(f"\n💾 Predictions successfully saved to: {pred_path}")

    # Compute metrics function
    def compute_summary(df, subset_name):
        n = len(df)
        if n == 0:
            return {}
        acc_en = float(df["correct_en"].mean())
        acc_id = float(df["correct_id"].mean())
        gap = float(abs(acc_en - acc_id))
        flip = float(df["flip"].mean())
        joint = float(df["joint_correct"].mean())
        conf_en = float(df["confidence_en"].mean())
        conf_id = float(df["confidence_id"].mean())
        v_en = int(df["valid_format_en"].sum())
        v_id = int(df["valid_format_id"].sum())

        return {
            "n": n,
            "accuracy_en": acc_en,
            "accuracy_en_ci95": wilson_ci(acc_en, n),
            "accuracy_id": acc_id,
            "accuracy_id_ci95": wilson_ci(acc_id, n),
            "language_gap": gap,
            "flip_rate": flip,
            "flip_rate_ci95": wilson_ci(flip, n),
            "joint_correctness": joint,
            "joint_correctness_ci95": wilson_ci(joint, n),
            "mean_confidence_en": conf_en,
            "mean_confidence_id": conf_id,
            "valid_format_en": v_en,
            "valid_format_id": v_id
        }

    clean_subset = df_res[df_res["translation_status"] == "CLEAN"]
    metrics_clean = compute_summary(clean_subset, "CLEAN (Primary)")
    metrics_all = compute_summary(df_res, "ALL (Sensitivity)")

    full_report = {
        "model_id": args.model_id,
        "checkpoint": f"{args.model_id} (4-bit NF4)",
        "evaluation_manifest": str(manifest_path.name),
        "primary_analysis_clean": metrics_clean,
        "sensitivity_analysis_all": metrics_all
    }

    metrics_path = output_dir / f"metrics_{safe_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"📊 Metrics successfully saved to: {metrics_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f"🏆 EVALUATION RESULTS SUMMARY: {args.model_id}")
    print("=" * 75)
    print(f"1. English Accuracy   (Acc_EN) : {metrics_clean.get('accuracy_en', 0)*100:.2f}%  [95% CI: {metrics_clean.get('accuracy_en_ci95')}]")
    print(f"2. Indonesian Accuracy(Acc_ID) : {metrics_clean.get('accuracy_id', 0)*100:.2f}%  [95% CI: {metrics_clean.get('accuracy_id_ci95')}]")
    print(f"3. Language Gap       (|EN-ID|): {metrics_clean.get('language_gap', 0)*100:.2f} pts")
    print(f"4. Cross-Lingual Flip (Flip)   : {metrics_clean.get('flip_rate', 0)*100:.2f}%  [95% CI: {metrics_clean.get('flip_rate_ci95')}]")
    print(f"5. Joint Correctness  (Joint)  : {metrics_clean.get('joint_correctness', 0)*100:.2f}%  [95% CI: {metrics_clean.get('joint_correctness_ci95')}]")
    print("=" * 75)

if __name__ == "__main__":
    main()
