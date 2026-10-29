import json
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List

def check_image_leakage(dfs: Dict[str, pd.DataFrame]) -> List:
    """Check for image identity overlap across dataset splits."""
    if len(dfs) < 2:
        return []
        
    train_df = dfs.get('train', pd.DataFrame())
    test_df = dfs.get('test', pd.DataFrame())
    
    # Check by image_name or pixel_sha256 if present
    col = 'image_name' if 'image_name' in train_df.columns else ('pixel_sha256' if 'pixel_sha256' in train_df.columns else None)
    if col and col in test_df.columns:
        train_imgs = set(train_df[col].dropna())
        test_imgs = set(test_df[col].dropna())
        overlap = list(train_imgs & test_imgs)
        return overlap
    return []

def check_question_leakage(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Check for exact question string overlap across dataset splits."""
    if len(dfs) < 2:
        return {}
        
    train_df = dfs.get('train', pd.DataFrame())
    test_df = dfs.get('test', pd.DataFrame())
    
    results = {}
    for col in ['question', 'question_indonesian']:
        if col in train_df.columns and col in test_df.columns:
            train_q = set(train_df[col].dropna().str.lower().str.strip())
            test_q = set(test_df[col].dropna().str.lower().str.strip())
            overlap = list(train_q & test_q)
            results[col] = {
                'count': len(overlap),
                'examples': overlap[:5]
            }
    return results

def check_answer_leakage(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Check for exact question+answer pair overlap across dataset splits."""
    if len(dfs) < 2:
        return {}
        
    train_df = dfs.get('train', pd.DataFrame())
    test_df = dfs.get('test', pd.DataFrame())
    
    results = {}
    if 'question' in train_df.columns and 'answer' in train_df.columns:
        train_qa = set(zip(train_df['question'].str.lower().str.strip(), train_df['answer'].str.lower().str.strip()))
        test_qa = set(zip(test_df['question'].str.lower().str.strip(), test_df['answer'].str.lower().str.strip()))
        overlap = [list(x) for x in list(train_qa & test_qa)]
        results['question_answer'] = {
            'count': len(overlap),
            'examples': overlap[:5]
        }
    return results

def check_bilingual_pair_split_integrity(dfs: Dict[str, pd.DataFrame]) -> bool:
    """Verify that paired EN and ID Q&A remain together in the same split."""
    for split_name, df in dfs.items():
        if 'question' in df.columns and 'question_indonesian' in df.columns:
            mismatch = df['question'].notnull() != df['question_indonesian'].notnull()
            if mismatch.any():
                return False
    return True

def compute_leakage_summary(img_leakage: List, q_leakage: Dict, a_leakage: Dict) -> Dict:
    """Combine leakage checks into a single summary dictionary."""
    return {
        'image_leakage_count': len(img_leakage),
        'question_leakage': q_leakage,
        'answer_leakage': a_leakage
    }

def recommend_image_level_split(df_combined: pd.DataFrame) -> pd.DataFrame:
    """Recommend an image-level split grouped by image identity to eliminate image leakage."""
    df_out = df_combined.copy()
    img_col = 'image_name' if 'image_name' in df_out.columns else ('pixel_sha256' if 'pixel_sha256' in df_out.columns else 'qid')
    
    unique_images = df_out[img_col].unique()
    import numpy as np
    np.random.seed(42)
    shuffled_imgs = np.random.permutation(unique_images)
    
    n_train = int(len(shuffled_imgs) * 0.8)
    train_imgs = set(shuffled_imgs[:n_train])
    
    df_out['recommended_split'] = df_out[img_col].apply(lambda x: 'train' if x in train_imgs else 'test')
    return df_out

def save_leakage_audit(summary: Dict, rec_split: pd.DataFrame, output_dir: Union[str, Path]):
    """Save leakage audit summary JSON and recommended split manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "leakage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    if not rec_split.empty:
        rec_split.to_csv(output_dir / "recommended_image_split.csv", index=False)
