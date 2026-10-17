"""
data_loader.py
--------------
Find, load, audit, and validate the official UCI Spambase dataset.

Dataset Reference:
    Creators: Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
    Organization: Hewlett-Packard Laboratories (Palo Alto, CA), 1999
    Source: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/94/spambase)

Dataset Characteristics:
    - 4,601 email instances (1,813 Spam [39.4%], 2,788 Not Spam [60.6%])
    - 57 continuous numeric attributes + 1 binary nominal target:
        * 48 word-frequency percentages (word_freq_WORD, 0-100%)
        * 6 character-frequency percentages (char_freq_CHAR, 0-100%)
        * 3 capital letter run-length statistics:
            - capital_run_length_average (continuous float)
            - capital_run_length_longest (continuous integer)
            - capital_run_length_total (continuous integer)
        * 1 class label (1 = spam, 0 = not spam)

Key Data Validation Principles:
    - Track and record any malformed/bad rows explicitly.
    - Check for identical feature vectors associated with conflicting labels.
    - Deduplicate exact duplicate instances while auditing counts.
    - Verify class distribution for stratified partitioning.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Default data directory relative to this file
DATA_DIR = Path(__file__).resolve().parent / "data"

# Standard 57 Spambase feature names + label
SPAMBASE_WORDS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses",
    "free", "business", "email", "you", "credit", "your", "font", "000",
    "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet",
    "857", "data", "415", "85", "technology", "1999", "parts", "pm",
    "direct", "cs", "meeting", "original", "project", "re", "edu",
    "table", "conference",
]

SPAMBASE_CHARS = [";", "(", "[", "!", "$", "#"]

FEATURE_NAMES: List[str] = (
    [f"word_freq_{w}" for w in SPAMBASE_WORDS]
    + [f"char_freq_{c}" for c in SPAMBASE_CHARS]
    + [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
)

COLUMN_NAMES: List[str] = FEATURE_NAMES + ["label"]


def find_dataset_file(data_dir: Path = DATA_DIR) -> Path:
    """Locate the Spambase data file inside data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    preferred = ["spambase.data", "spambase.csv", "spambase.data.txt"]
    for name in preferred:
        path = data_dir / name
        if path.is_file():
            return path

    for path in sorted(data_dir.iterdir()):
        if path.is_file() and "spambase" in path.name.lower() and path.suffix.lower() in {".data", ".csv", ".txt"}:
            return path

    raise FileNotFoundError(
        f"No Spambase data file found in '{data_dir}'.\n"
        "Ensure 'spambase.data' exists in the data/ folder."
    )


def load_dataset(path=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load, audit, and clean the UCI Spambase dataset.
    
    Returns:
        df   : pd.DataFrame with 57 feature columns and 1 'label' column
        info : dict containing detailed auditing statistics
    """
    path = Path(path) if path else find_dataset_file()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Read physical line count
    total_physical_lines = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                total_physical_lines += 1

    bad_lines = []

    def handle_bad_line(line_content):
        bad_lines.append(line_content)
        return None

    # Load CSV
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        on_bad_lines=handle_bad_line,
        engine="python",
        encoding="utf-8",
    )

    info = {
        "file": path.name,
        "physical_lines": total_physical_lines,
        "rows_read": len(df),
        "bad_lines_skipped": len(bad_lines),
    }

    if df.empty:
        raise ValueError(f"Dataset file '{path.name}' is empty.")

    # 1. Missing / null values
    before = len(df)
    df = df.dropna()
    info["missing_removed"] = before - len(df)

    # 2. Verify target label encoding
    df["label"] = df["label"].astype(int)
    invalid_labels = ~df["label"].isin([0, 1])
    info["invalid_labels_removed"] = int(invalid_labels.sum())
    df = df[~invalid_labels]

    # 3. Check for identical feature vectors with conflicting labels
    feature_cols = FEATURE_NAMES
    grouped_labels = df.groupby(feature_cols)["label"].nunique()
    conflicting_keys = grouped_labels[grouped_labels > 1].index
    info["conflicting_groups_detected"] = len(conflicting_keys)

    if len(conflicting_keys) > 0:
        conflict_set = set(conflicting_keys)
        mask = df[feature_cols].apply(tuple, axis=1).isin(conflict_set)
        info["conflicting_rows_removed"] = int(mask.sum())
        df = df[~mask]
    else:
        info["conflicting_rows_removed"] = 0

    # 4. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    info["duplicates_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    info["rows_final"] = len(df)

    # 5. Class distribution statistics
    class_counts = df["label"].value_counts().to_dict()
    n_not_spam = class_counts.get(0, 0)
    n_spam = class_counts.get(1, 0)
    info["not_spam_count"] = n_not_spam
    info["spam_count"] = n_spam
    info["not_spam_pct"] = (n_not_spam / len(df)) * 100 if len(df) > 0 else 0.0
    info["spam_pct"] = (n_spam / len(df)) * 100 if len(df) > 0 else 0.0
    info["num_features"] = len(feature_cols)

    return df, info
