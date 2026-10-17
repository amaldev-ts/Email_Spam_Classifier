"""
evaluate.py
-----------
Evaluation metrics computation, plotting, error analysis, and report formatting
for the UCI Spambase email classification system.

Conventions:
    Class 0: Not Spam (Legitimate Email) - Negative Class
    Class 1: Spam - Positive Class for Precision, Recall, and F1-score.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import FEATURE_NAMES

CLASS_NAMES = ["Not Spam", "Spam"]


def compute_metrics(y_true, y_pred) -> Dict:
    """Calculate and return a dictionary of standard classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fpr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm,
        "report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
        ),
    }


def extract_errors(
    y_true,
    y_pred,
    X_features: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """
    Extract all misclassified test cases (False Positives and False Negatives)
    with the top non-zero indicative features for qualitative diagnostic analysis.
    """
    if isinstance(X_features, np.ndarray):
        X_df = pd.DataFrame(X_features, columns=feature_names)
    else:
        X_df = X_features.reset_index(drop=True)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    rows = []
    for i in range(len(y_true_arr)):
        actual = y_true_arr[i]
        pred = y_pred_arr[i]

        if actual != pred:
            err_type = "False Positive" if (actual == 0 and pred == 1) else "False Negative"
            
            # Find the top 3 highest feature values in this sample
            sample_feats = X_df.iloc[i]
            top_feats = sample_feats[sample_feats > 0].sort_values(ascending=False).head(4)
            top_feat_str = "; ".join([f"{k}={v:.2f}" for k, v in top_feats.items()]) if not top_feats.empty else "All features ~0"

            rows.append({
                "Sample_Index": i,
                "Error_Type": err_type,
                "Actual": "Not Spam" if actual == 0 else "Spam",
                "Predicted": "Spam" if pred == 1 else "Not Spam",
                "Top_Indicative_Features": top_feat_str,
                "capital_run_total": float(sample_feats.get("capital_run_length_total", 0.0)),
                "char_freq_dollar": float(sample_feats.get("char_freq_$", 0.0)),
                "char_freq_exclamation": float(sample_feats.get("char_freq_!", 0.0)),
                "word_freq_free": float(sample_feats.get("word_freq_free", 0.0)),
            })

    return pd.DataFrame(rows)


def plot_class_distribution(labels, path: Path) -> None:
    """Plot bar chart of class counts with percentage annotations."""
    counts = [int((labels == 0).sum()), int((labels == 1).sum())]
    total = sum(counts)

    plt.figure(figsize=(6, 4.5))
    ax = sns.barplot(
        x=CLASS_NAMES,
        y=counts,
        hue=CLASS_NAMES,
        palette=["#3b82f6", "#ef4444"],
        legend=False,
    )

    for i, value in enumerate(counts):
        pct = (value / total) * 100
        ax.text(
            i,
            value + 35,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Class Distribution (UCI Spambase)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Number of Email Instances", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, path: Path, title: str = "Confusion Matrix (Spambase GaussianNB)") -> None:
    """Plot annotated heatmap of the confusion matrix."""
    plt.figure(figsize=(6, 5))
    total = cm.sum()
    labels = []
    names = [["TN", "FP"], ["FN", "TP"]]

    for i in range(2):
        row = []
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100
            row.append(f"{names[i][j]}\n{count:,}\n({pct:.2f}%)")
        labels.append(row)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 11, "weight": "bold"},
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def format_metrics_text(metrics: Dict, run_info: Dict, baseline_metrics: Optional[Dict] = None) -> str:
    """Format comprehensive evaluation results and trade-off commentary."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    fpr = metrics["false_positive_rate"]

    lines = [
        "EMAIL SPAM CLASSIFICATION - UCI SPAMBASE (GAUSSIAN NAIVE BAYES)",
        "=" * 64,
        f"Dataset File       : {run_info['file']} (UCI Spambase Email Dataset)",
        f"Total Instances    : {run_info['total']:,}  (Not Spam: {run_info['not_spam']:,}, Spam: {run_info['spam']:,})",
        f"Training Samples   : {run_info['train_size']:,}",
        f"Testing Samples    : {run_info['test_size']:,}",
        f"Input Features     : {run_info['num_features']} continuous features",
        f"Feature Transform  : {run_info.get('transform', 'Log1p Transformation (log(1 + x))')}",
        f"Variance Smoothing : {run_info.get('var_smoothing', 0.05)}",
        f"Selected Recipe    : {run_info.get('selected_model', 'N/A')}",
        f"Random Seed        : {run_info['seed']}",
        "",
        "HELD-OUT TEST SET EVALUATION (Spam = Positive Class)",
        "-" * 64,
        f"Accuracy            : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"Spam Precision      : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)",
        f"Spam Recall         : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)",
        f"Spam F1-score       : {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)",
        f"False Positive Rate : {fpr:.4f} ({fpr*100:.2f}%)",
        "",
    ]

    if baseline_metrics is not None:
        b_acc = baseline_metrics["accuracy"]
        b_prec = baseline_metrics["precision"]
        b_rec = baseline_metrics["recall"]
        b_f1 = baseline_metrics["f1"]
        b_fpr = baseline_metrics["false_positive_rate"]
        b_fp = baseline_metrics["fp"]

        d_acc = (metrics["accuracy"] - b_acc) * 100
        d_prec = (metrics["precision"] - b_prec) * 100
        d_rec = (metrics["recall"] - b_rec) * 100
        d_f1 = (metrics["f1"] - b_f1) * 100
        d_fpr = (fpr - b_fpr) * 100

        lines.extend([
            "COMPARISON AGAINST UNREGULARIZED RAW BASELINE (Raw Features, var_smoothing=1e-9):",
            f"  Accuracy    : {b_acc*100:.2f}%  ->  {metrics['accuracy']*100:.2f}%  ({d_acc:+.2f}%)",
            f"  Precision   : {b_prec*100:.2f}%  ->  {metrics['precision']*100:.2f}%  ({d_prec:+.2f}%)",
            f"  Recall      : {b_rec*100:.2f}%  ->  {metrics['recall']*100:.2f}%  ({d_rec:+.2f}%)",
            f"  F1-Score    : {b_f1*100:.2f}%  ->  {metrics['f1']*100:.2f}%  ({d_f1:+.2f}%)",
            f"  FP Rate     : {b_fpr*100:.2f}%  ->  {fpr*100:.2f}%  ({d_fpr:+.2f}%)",
            f"  False Positives: {b_fp} in baseline vs {fp} in improved model.",
            "",
        ])

    lines.extend([
        "CONFUSION MATRIX BREAKDOWN",
        "----------------------------------------------------------------",
        "                 Predicted Not Spam   Predicted Spam",
        f"Actual Not Spam  {tn:<20d} {fp:<15d}",
        f"Actual Spam      {fn:<20d} {tp:<15d}",
        "",
        f"True Negatives  (TN) = {tn:<5d} : Not spam emails correctly delivered to Inbox.",
        f"False Positives (FP) = {fp:<5d} : Not spam emails incorrectly flagged as Spam.",
        f"False Negatives (FN) = {fn:<5d} : Spam emails that slipped into Inbox.",
        f"True Positives  (TP) = {tp:<5d} : Spam emails successfully intercepted.",
        "",
        "CLASSIFICATION REPORT",
        "-" * 64,
        metrics["report"],
        "METHODOLOGICAL ANALYSIS & GAUSSIAN DISTRIBUTION FIT",
        "-" * 64,
        "The UCI Spambase dataset comprises 57 continuous attributes. Many features",
        "(word percentages and capital letter run lengths) exhibit severe right-skewness",
        "with heavy zero-inflation. Applying a Log1p transformation maps these skewed",
        "distributions closer to the bell-shaped Gaussian density required by GaussianNB,",
        "while tuning var_smoothing provides variance regularization against outliers.",
    ])
    return "\n".join(lines)


def save_metrics_text(text: str, path: Path) -> None:
    """Save text content to file with UTF-8 encoding."""
    Path(path).write_text(text + "\n", encoding="utf-8")
