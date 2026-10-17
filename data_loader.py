"""
data_loader.py
--------------
Find, load, validate, and audit the labelled spam dataset.

Supported formats (auto-detected):
  1. UCI SMS Spam Collection raw file  -> 'SMSSpamCollection'
       tab-separated, NO header, each line: <label><TAB><message>
  2. The same UCI zip file             -> 'smsspamcollection.zip' (auto-extracted)
  3. CSV files with a header row, e.g. Kaggle 'spam.csv' (columns v1, v2) or any
     CSV that has a label column and a text column (see LABEL_COLUMNS / TEXT_COLUMNS).

Key Data Validation Principles:
  - Track and record any malformed/bad lines explicitly (no silent skips).
  - Normalise label spellings to binary integers: 0 = ham (legitimate), 1 = spam.
  - Verify and resolve identical texts with conflicting labels.
  - Remove exact duplicate messages while maintaining an audited count.
  - Validate minimum class counts for stratified splitting.

Output of load_dataset():
    df   : pandas DataFrame with exactly two columns ['label', 'text']
    info : dict with complete cleaning and dataset audit statistics
"""

import csv
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from preprocessing import clean_text

# Default data folder, always relative to this file
DATA_DIR = Path(__file__).resolve().parent / "data"

# Accepted label spellings -> numeric class (0 = ham, 1 = spam)
LABEL_MAP = {
    "ham": 0, "not spam": 0, "notspam": 0, "legitimate": 0, "legit": 0, "0": 0,
    "spam": 1, "junk": 1, "1": 1,
}

# Possible column names in CSV files (compared in lower case)
LABEL_COLUMNS = ["label", "v1", "category", "class", "target", "spam"]
TEXT_COLUMNS = ["text", "v2", "message", "email", "body", "content"]


def _extract_zip_if_present(data_dir: Path) -> None:
    """If the UCI zip is in data/, extract its files (flat, no sub-folders)."""
    for zip_path in data_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.namelist():
                    name = Path(member).name
                    if not name or (data_dir / name).exists():
                        continue
                    (data_dir / name).write_bytes(archive.read(member))
        except zipfile.BadZipFile:
            raise ValueError(f"'{zip_path.name}' is not a valid zip file. Download it again.")


def find_dataset_file(data_dir: Path = DATA_DIR) -> Path:
    """Return the path of the dataset file inside data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    _extract_zip_if_present(data_dir)

    preferred = ["SMSSpamCollection", "SMSSpamCollection.txt", "spam.csv"]
    for name in preferred:
        if (data_dir / name).is_file():
            return data_dir / name

    for path in sorted(data_dir.iterdir()):
        is_data = path.suffix.lower() in {".csv", ".tsv", ".txt"}
        is_readme = path.name.lower().startswith("readme")
        if path.is_file() and is_data and not is_readme:
            return path

    raise FileNotFoundError(
        f"No dataset file found in '{data_dir}'.\n"
        "Download the UCI SMS Spam Collection and place the file named "
        "'SMSSpamCollection' (or the zip) in the data/ folder. See README.md."
    )


def _read_raw_table(path: Path) -> Tuple[pd.DataFrame, int, int]:
    """
    Read the file into a DataFrame with columns 'label' and 'text'.
    Returns (df, total_physical_lines, bad_lines_count).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Count total physical non-empty lines in the source file
    total_physical_lines = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                total_physical_lines += 1

    bad_lines = []

    def handle_bad_line(line_content):
        bad_lines.append(line_content)
        return None

    # Try UTF-8 first, fallback to latin-1
    encodings = ["utf-8", "latin-1"]
    last_err = None

    for enc in encodings:
        try:
            if suffix in {"", ".txt", ".tsv"}:
                df = pd.read_csv(
                    path,
                    sep="\t",
                    header=None,
                    names=["label", "text"],
                    quoting=csv.QUOTE_NONE,
                    on_bad_lines=handle_bad_line,
                    engine="python",
                    encoding=enc,
                )
                return df, total_physical_lines, len(bad_lines)

            if suffix == ".csv":
                df = pd.read_csv(
                    path,
                    on_bad_lines=handle_bad_line,
                    engine="python",
                    encoding=enc,
                )
                lower_to_orig = {str(c).strip().lower(): c for c in df.columns}
                label_col = next((lower_to_orig[c] for c in LABEL_COLUMNS if c in lower_to_orig), None)
                text_col = next((lower_to_orig[c] for c in TEXT_COLUMNS if c in lower_to_orig), None)
                if label_col is None or text_col is None:
                    raise ValueError(
                        f"Could not find label/text columns in '{path.name}'. "
                        f"Found: {list(df.columns)}. Expected label in {LABEL_COLUMNS} and text in {TEXT_COLUMNS}."
                    )
                df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
                return df, total_physical_lines, len(bad_lines)
        except UnicodeDecodeError as err:
            last_err = err
            continue

    if last_err:
        raise last_err
    raise ValueError(f"Unsupported file format '{suffix}'. Use the UCI file or a .csv file.")


def load_dataset(path=None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load, clean, validate, and audit the spam dataset.
    
    Returns:
        df   : pd.DataFrame with columns ['label', 'text']
        info : dict with comprehensive cleaning and validation audit counts
    """
    path = Path(path) if path else find_dataset_file()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        df, total_physical_lines, bad_lines_count = _read_raw_table(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file '{path.name}' is empty.")

    info = {
        "file": path.name,
        "physical_lines": total_physical_lines,
        "rows_read": len(df),
        "bad_lines_skipped": bad_lines_count,
    }

    if df.empty:
        raise ValueError(f"Dataset file '{path.name}' has no rows.")

    # 1. Missing values
    before = len(df)
    df = df.dropna(subset=["label", "text"])
    info["missing_removed"] = before - len(df)

    # 2. Labels normalisation
    normalised = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    info["invalid_labels_removed"] = int(normalised.isna().sum())
    df = df.assign(label=normalised).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 3. Text cleaning: whitespace normalisation, collapse multiple spaces
    df["text"] = df["text"].apply(clean_text)
    before = len(df)
    df = df[df["text"] != ""]
    info["empty_text_removed"] = before - len(df)

    # 4. Check for conflicting labels on identical text
    label_counts_per_text = df.groupby("text")["label"].nunique()
    conflicting_texts = label_counts_per_text[label_counts_per_text > 1].index.tolist()
    info["conflicting_labels_detected"] = len(conflicting_texts)

    if conflicting_texts:
        # Exclude conflicting messages to prevent ambiguous ground truth
        before = len(df)
        df = df[~df["text"].isin(conflicting_texts)]
        info["conflicting_rows_removed"] = before - len(df)
    else:
        info["conflicting_rows_removed"] = 0

    # 5. Exact duplicate messages removal (retain first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    info["duplicates_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    info["rows_final"] = len(df)

    # 6. Class distribution validation
    class_counts = df["label"].value_counts().to_dict()
    n_ham = class_counts.get(0, 0)
    n_spam = class_counts.get(1, 0)
    info["ham_count"] = n_ham
    info["spam_count"] = n_spam
    info["ham_pct"] = (n_ham / len(df)) * 100 if len(df) > 0 else 0.0
    info["spam_pct"] = (n_spam / len(df)) * 100 if len(df) > 0 else 0.0

    if len(class_counts) < 2:
        raise ValueError("Dataset must contain BOTH spam and ham messages after cleaning.")
    if min(class_counts.values()) < 2:
        raise ValueError("Each class needs at least 2 samples for stratified train/test split.")

    return df[["label", "text"]], info
