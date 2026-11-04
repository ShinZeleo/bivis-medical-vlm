import pandas as pd
from pathlib import Path
from typing import Dict, Union

def analyze_pairing_structure(df: pd.DataFrame) -> Dict[str, int]:
    """Analyze the bilingual pair coverage and structure."""
    total = len(df)
    has_both_q = (df['question'].notnull() & df['question_indonesian'].notnull()).sum() if 'question_indonesian' in df.columns else 0
    has_both_a = (df['answer'].notnull() & df['answer_indonesian'].notnull()).sum() if 'answer_indonesian' in df.columns else 0
    valid_pairs = (has_both_q == total and has_both_a == total)
    
    return {
        'total_rows': total,
        'has_both_questions': int(has_both_q),
        'has_both_answers': int(has_both_a),
        'valid_pairs_count': int(has_both_q) if valid_pairs else 0
    }

def validate_bilingual_pairs(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Validate paired Q&A items and tag match quality."""
    val_rows = []
    
    for idx, row in df.iterrows():
        q_en = str(row.get('question', ''))
        q_id = str(row.get('question_indonesian', ''))
        a_en = str(row.get('answer', '')).lower().strip()
        a_id = str(row.get('answer_indonesian', '')).lower().strip()
        
        # Categorize answer match type
        if a_en == a_id:
            match_type = 'exact_match'
        elif a_en in ['yes', 'no'] and a_id in ['ya', 'tidak', 'tidak ada']:
            match_type = 'semantic_likely_equivalent'
        else:
            match_type = 'different'
            
        pair_validity = 'valid' if (q_en and q_id and a_en and a_id) else 'invalid'
        
        val_rows.append({
            'split': split_name,
            'question_en': q_en,
            'question_id': q_id,
            'answer_en': a_en,
            'answer_id': a_id,
            'answer_match_type': match_type,
            'pair_validity': pair_validity
        })
        
    return pd.DataFrame(val_rows)

def summarize_pairing(val_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Summarize bilingual pairing validation results."""
    total = len(val_df)
    valid_cnt = (val_df['pair_validity'] == 'valid').sum()
    partial_cnt = (val_df['pair_validity'] == 'partial').sum()
    invalid_cnt = (val_df['pair_validity'] == 'invalid').sum()
    
    match_counts = val_df['answer_match_type'].value_counts().to_dict()
    
    return {
        'total_pairs': total,
        'valid': int(valid_cnt),
        'valid_pct': round(valid_cnt / total * 100, 2) if total > 0 else 0,
        'partial': int(partial_cnt),
        'partial_pct': round(partial_cnt / total * 100, 2) if total > 0 else 0,
        'invalid': int(invalid_cnt),
        'invalid_pct': round(invalid_cnt / total * 100, 2) if total > 0 else 0,
        'answer_match_breakdown': match_counts
    }

def sample_pairs_for_review(val_df: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
    """Sample bilingual pairs for manual inspection."""
    return val_df.sample(min(sample_size, len(val_df)), random_state=42)

def save_pairing_audit(val_df: pd.DataFrame, summary: Dict, output_dir: Union[str, Path]):
    """Save pairing validation DataFrame and summary to manifests directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_df.to_csv(output_dir / "bilingual_pairing_audit.csv", index=False)
    
    # Save flat summary
    flat_summary = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    if 'answer_match_breakdown' in summary:
        for k, v in summary['answer_match_breakdown'].items():
            flat_summary[f"answer_match_breakdown_{k}"] = v
    pd.DataFrame([flat_summary]).to_csv(output_dir / "bilingual_pairing_summary.csv", index=False)
