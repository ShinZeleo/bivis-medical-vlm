import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Union

def setup_style():
    """Set default plotting style for figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

def plot_question_length_distribution(df: pd.DataFrame, output_path: Union[str, Path] = None):
    """Plot word length distribution of English vs Indonesian questions."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    
    if 'len_q_en' in df.columns and 'len_q_id' in df.columns:
        sns.kdeplot(df['len_q_en'], label="English Questions", fill=True, ax=ax, alpha=0.4, color='#1f77b4')
        sns.kdeplot(df['len_q_id'], label="Indonesian Questions", fill=True, ax=ax, alpha=0.4, color='#ff7f0e')
    
    ax.set_title("Question Length Distribution (Word Count)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()

def plot_answer_length_distribution(df: pd.DataFrame, output_path: Union[str, Path] = None):
    """Plot word length distribution of English vs Indonesian answers."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    
    if 'len_a_en' in df.columns and 'len_a_id' in df.columns:
        sns.kdeplot(df['len_a_en'], label="English Answers", fill=True, ax=ax, alpha=0.4, color='#2ca02c')
        sns.kdeplot(df['len_a_id'], label="Indonesian Answers", fill=True, ax=ax, alpha=0.4, color='#d62728')
        
    ax.set_title("Answer Length Distribution (Word Count)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()

def plot_bilingual_pair_validity(summary: Dict[str, Union[int, float]], output_path: Union[str, Path] = None):
    """Plot summary pie chart of bilingual pair validity."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    
    labels = list(summary.keys())
    values = list(summary.values())
    colors = ['#2ca02c', '#ff7f0e', '#d62728'][:len(labels)]
    
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax.set_title("Bilingual Pair Validity Breakdown")
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()

def plot_answer_type_distribution(df: pd.DataFrame, output_path: Union[str, Path] = None):
    """Plot answer type distribution (CLOSED vs OPEN)."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    
    if 'answer_type' in df.columns:
        counts = df['answer_type'].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax, palette="Blues_d")
        ax.set_title("Answer Type Distribution (CLOSED vs OPEN)")
        ax.set_ylabel("Count")
        
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()
