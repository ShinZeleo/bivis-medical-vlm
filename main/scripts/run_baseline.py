import os
import sys
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.image_utils import get_image_path, ensure_images_extracted
from src.models.qwen_loader import load_qwen_vl
from qwen_vl_utils import process_vision_info

def run_baseline(
    sample_size: int = 20,
    model: Optional[object] = None,
    processor: Optional[object] = None,
    target_device: Optional[torch.device] = None,
    random_seed: int = config.RANDOM_SEED,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Run baseline zero-shot evaluation on paired English-Indonesian questions.
    Can be called directly from Jupyter Notebook or CLI.
    """
    ensure_images_extracted()
    
    # Locate manifest
    manifest_path = config.DATA_DIR / "indorad_vqa_final_manifest.csv"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "data" / "indorad_vqa_final_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    eligible_df = df[df["is_eligible_primary"] == True]
    if len(eligible_df) == 0:
        eligible_df = df
        
    sample_df = eligible_df.sample(min(sample_size, len(eligible_df)), random_state=random_seed)
    print(f"[run_baseline] Evaluating on {len(sample_df)} paired questions...")
    
    # Load model if not provided
    if model is None or processor is None:
        processor, model, target_device = load_qwen_vl()
    elif target_device is None:
        target_device = next(model.parameters()).device
        
    results = []
    
    for idx, row in sample_df.iterrows():
        img_id = row.get("image_id", "")
        try:
            img_path = get_image_path(img_id)
        except Exception as e:
            print(f"Skipping {img_id}: {e}")
            continue
            
        q_en = str(row.get('question_en', row.get('question', 'Is this normal?')))
        q_id = str(row.get('question_id', row.get('question_indonesian', 'Apakah ini normal?')))
        gt_en = str(row.get('answer_en', row.get('answer', ''))).strip().upper()
        
        prompts = {
            "en": f"Answer only YES or NO.\n{q_en}",
            "id": f"Jawab hanya dengan YES atau NO.\n{q_id}"
        }
        
        row_preds = {}
        for lang, prompt_text in prompts.items():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path.resolve())},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(target_device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=10)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            parsed = "YES" if "YES" in output_text.upper() else ("NO" if "NO" in output_text.upper() else output_text)
            row_preds[lang] = parsed
            
            results.append({
                "qid": row.get("qid", idx),
                "image_id": img_id,
                "language": lang,
                "prompt": prompt_text,
                "ground_truth": gt_en,
                "raw_output": output_text,
                "parsed_prediction": parsed,
                "is_correct": (parsed == gt_en) if gt_en in ["YES", "NO"] else None
            })
            print(f"[{lang.upper()}] Pred: {parsed} | Raw: '{output_text}' (GT: {gt_en})")
            
    res_df = pd.DataFrame(results)
    
    if save_output:
        out_dir = config.OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "pilot_predictions.csv"
        res_df.to_csv(out_file, index=False)
        print(f"[run_baseline] Saved results to {out_file}")
        
    return res_df

def main():
    run_baseline(sample_size=10, save_output=True)

if __name__ == "__main__":
    main()
