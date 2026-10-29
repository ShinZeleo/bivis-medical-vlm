import pandas as pd
from pathlib import Path
from typing import Dict, Union

def generate_quality_report(df: pd.DataFrame, split_name: str) -> Dict[str, Union[pd.DataFrame, int]]:
    """
    Generate missing values report and duplicate analysis.
    Safely handles unhashable columns (e.g. image dicts/objects).
    """
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing_Count']
    missing_df['Missing_Pct'] = (missing_df['Missing_Count'] / len(df) * 100).round(2)
    
    # Duplicates check
    dup_q_en = df.duplicated(subset=['question']).sum() if 'question' in df.columns else 0
    dup_q_id = df.duplicated(subset=['question_indonesian']).sum() if 'question_indonesian' in df.columns else 0
    
    try:
        dup_full = df.duplicated().sum()
    except TypeError:
        # Exclude unhashable columns (like 'image' dicts)
        cols_to_check = [c for c in df.columns if c != 'image' and not isinstance(df[c].iloc[0] if len(df) > 0 else None, (dict, list))]
        dup_full = df.duplicated(subset=cols_to_check).sum()
    
    return {
        'missing_values': missing_df,
        'duplicates': {
            'duplicate_english_questions': int(dup_q_en),
            'duplicate_indonesian_questions': int(dup_q_id),
            'exact_duplicate_rows': int(dup_full)
        }
    }

def save_quality_report(report: Dict, output_path: Union[str, Path]):
    """Save quality report missing values to CSV."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if 'missing_values' in report:
        report['missing_values'].to_csv(output_path / "missing_values_report.csv", index=False)
