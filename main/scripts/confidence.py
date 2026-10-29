import os
import sys
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.image_utils import get_image_path, ensure_images_extracted
from src.models.qwen_loader import load_qwen_vl
from qwen_vl_utils import process_vision_info

def run_confidence_evaluation(
    sample_size: int = 10,
    model: Optional[object] = None,
    processor: Optional[object] = None,
    target_device: Optional[torch.device] = None,
    random_seed: int = config.RANDOM_SEED,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Extract token-level confidence (logprobs) for canonical YES / NO answers.
    Can be called directly from Jupyter Notebook or CLI.
    """
    ensure_images_extracted()
    
    manifest_path = config.DATA_DIR / "indorad_vqa_final_manifest.csv"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "data" / "indorad_vqa_final_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    full_df = pd.read_csv(manifest_path)
    eligible_df = full_df[full_df["is_eligible_primary"] == True]
    if len(eligible_df) == 0:
        eligible_df = full_df
        
    df = eligible_df.sample(min(sample_size, len(eligible_df)), random_state=random_seed)
    print(f"[run_confidence_evaluation] Extracting confidence on {len(df)} pairs...")
    
    if model is None or processor is None:
        processor, model, target_device = load_qwen_vl()
    elif target_device is None:
        target_device = next(model.parameters()).device
        
    # Tokenize berbagai variasi YES/NO (termasuk yang memiliki spasi di depan)
    token_yes_upper = processor.tokenizer.encode("YES", add_special_tokens=False)[0]
    token_no_upper = processor.tokenizer.encode("NO", add_special_tokens=False)[0]
    token_yes_title = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_no_title = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    
    token_yes_space_upper = processor.tokenizer.encode(" YES", add_special_tokens=False)[0]
    token_no_space_upper = processor.tokenizer.encode(" NO", add_special_tokens=False)[0]
    token_yes_space_title = processor.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    token_no_space_title = processor.tokenizer.encode(" No", add_special_tokens=False)[0]
    
    results = []
    
    for idx, row in df.iterrows():
        img_id = row.get("image_id", "")
        try:
            img_path = get_image_path(img_id)
        except Exception as e:
            print(f"Skipping {img_id}: {e}")
            continue
            
        q_en = str(row.get('question_en', row.get('question', 'Is this normal?')))
        q_id = str(row.get('question_id', row.get('question_indonesian', 'Apakah ini normal?')))
        gt_en = str(row.get('answer_en', row.get('answer', ''))).strip().upper()
        
        # PROMPT FIX: Memaksa instruksi menggunakan Bahasa Inggris untuk kedua bahasa!
        prompts = {
            "en": f"Answer only YES or NO.\n{q_en}",
            "id": f"Answer only YES or NO. Question:\n{q_id}"
        }
        
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
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            logits = outputs.scores[0][0]
            probs = torch.softmax(logits, dim=-1)
            
            # Menggabungkan probabilitas seluruh variasi kata YES dan NO
            p_yes = (probs[token_yes_upper].item() + probs[token_yes_title].item() +
                     probs[token_yes_space_upper].item() + probs[token_yes_space_title].item())
            p_no = (probs[token_no_upper].item() + probs[token_no_title].item() +
                    probs[token_no_space_upper].item() + probs[token_no_space_title].item())
            
            total_prob = p_yes + p_no
            
            # Jika model benar-benar menjawab dengan kata selain variasi YES/NO, baru berikan 0.5
            norm_p_yes = p_yes / total_prob if total_prob > 0 else 0.5
            norm_p_no = p_no / total_prob if total_prob > 0 else 0.5
            
            confidence = max(norm_p_yes, norm_p_no)
            pred = "YES" if norm_p_yes >= norm_p_no else "NO"
            
            top_token_id = torch.argmax(logits).item()
            top_token_str = processor.tokenizer.decode([top_token_id]).strip()
            
            results.append({
                "qid": row.get("qid", idx),
                "image_id": img_id,
                "language": lang,
                "question": q_en if lang == "en" else q_id,
                "ground_truth": gt_en,
                "top_generated_token": top_token_str,
                "pred": pred,
                "confidence": round(confidence, 4),
                "norm_p_yes": round(norm_p_yes, 4),
                "norm_p_no": round(norm_p_no, 4),
                "is_correct": (pred == gt_en) if gt_en in ["YES", "NO"] else None
            })
            print(f"[{lang.upper()}] Pred: {pred} (Conf: {confidence:.2f}, GT: {gt_en})")
            
    res_df = pd.DataFrame(results)
    
    if save_output:
        out_dir = config.OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "confidence_results.csv"
        res_df.to_csv(out_file, index=False)
        print(f"[run_confidence_evaluation] Saved confidence evaluations to {out_file}")
        
    return res_df

def main():
    run_confidence_evaluation(sample_size=10, save_output=True)

if __name__ == "__main__":
    main()
