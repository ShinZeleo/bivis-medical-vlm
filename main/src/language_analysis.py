import pandas as pd
from typing import Dict

def compare_language_distributions(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare language text availability across splits."""
    records = []
    for split_name, df in dfs.items():
        total = len(df)
        has_en = (df['question'].notnull() & df['answer'].notnull()).sum()
        has_id = (df['question_indonesian'].notnull() & df['answer_indonesian'].notnull()).sum() if 'question_indonesian' in df.columns else 0
        records.append({
            'Split': split_name,
            'Total Rows': total,
            'EN Available': has_en,
            'EN %': round(has_en / total * 100, 2),
            'ID Available': has_id,
            'ID %': round(has_id / total * 100, 2) if total > 0 else 0
        })
    return pd.DataFrame(records)

def analyze_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Compute word and character lengths for questions and answers in EN and ID."""
    df_out = df.copy()
    
    if 'question' in df_out.columns:
        df_out['len_q_en'] = df_out['question'].astype(str).apply(lambda x: len(x.split()))
    if 'question_indonesian' in df_out.columns:
        df_out['len_q_id'] = df_out['question_indonesian'].astype(str).apply(lambda x: len(x.split()))
    if 'answer' in df_out.columns:
        df_out['len_a_en'] = df_out['answer'].astype(str).apply(lambda x: len(x.split()))
    if 'answer_indonesian' in df_out.columns:
        df_out['len_a_id'] = df_out['answer_indonesian'].astype(str).apply(lambda x: len(x.split()))
        
    return df_out

def detect_language_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect translation anomalies such as empty fields or identical non-translated strings."""
    anomalies = []
    
    for idx, row in df.iterrows():
        reasons = []
        q_en = str(row.get('question', '')).strip().lower()
        q_id = str(row.get('question_indonesian', '')).strip().lower()
        a_en = str(row.get('answer', '')).strip().lower()
        a_id = str(row.get('answer_indonesian', '')).strip().lower()
        
        if not q_id or q_id == 'nan':
            reasons.append("Missing Indonesian Question")
        if not a_id or a_id == 'nan':
            reasons.append("Missing Indonesian Answer")
            
        # Check if question is untranslated (identical long string)
        if len(q_en) > 15 and q_en == q_id:
            reasons.append("Identical Question (Untranslated)")
            
        if reasons:
            row_dict = row.to_dict()
            row_dict['anomaly_reason'] = "; ".join(reasons)
            anomalies.append(row_dict)
            
    return pd.DataFrame(anomalies)
