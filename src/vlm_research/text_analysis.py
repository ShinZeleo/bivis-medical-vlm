import pandas as pd
from pathlib import Path
from typing import Dict, Union

def analyze_question_types(df: pd.DataFrame) -> pd.Series:
    """Analyze distribution of question types."""
    if 'question_type' in df.columns:
        return df['question_type'].value_counts(dropna=False)
    return pd.Series()

def analyze_answer_types(df: pd.DataFrame) -> pd.Series:
    """Analyze distribution of answer types (CLOSED / OPEN)."""
    if 'answer_type' in df.columns:
        return df['answer_type'].value_counts(dropna=False)
    return pd.Series()

def check_yes_no_questions(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Check how many questions are Closed (Yes/No) questions."""
    total = len(df)
    closed_cnt = (df['answer_type'] == 'CLOSED').sum() if 'answer_type' in df.columns else 0
    open_cnt = total - closed_cnt
    
    return {
        'total_questions': total,
        'yes_no_count': int(closed_cnt),
        'yes_no_pct': round(closed_cnt / total * 100, 2) if total > 0 else 0,
        'open_ended_count': int(open_cnt),
        'open_ended_pct': round(open_cnt / total * 100, 2) if total > 0 else 0
    }

def analyze_answer_distribution(df: pd.DataFrame) -> pd.Series:
    """Analyze top English and Indonesian answers."""
    if 'answer' in df.columns:
        return df['answer'].value_counts().head(20)
    return pd.Series()

def save_text_analysis(data: Dict, output_dir: Union[str, Path]):
    """Save text analysis summaries to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'yes_no' in data:
        pd.DataFrame([data['yes_no']]).to_csv(output_dir / "yes_no_stats.csv", index=False)
