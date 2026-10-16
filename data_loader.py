"""
data_loader.py
--------------
Find, load and validate the labelled spam dataset stored in the data/ folder.

Supported formats (auto-detected):
  1. UCI SMS Spam Collection raw file  -> 'SMSSpamCollection'
       tab-separated, NO header, each line:  <label><TAB><message>
  2. The same UCI zip file             -> 'smsspamcollection.zip' (auto-extracted)
  3. CSV files with a header row, e.g. Kaggle 'spam.csv' (columns v1,v2) or any
     CSV that has a label column and a text column (see the name lists below).

Output of load_dataset(): a pandas DataFrame with exactly two columns:
    label : 0 = ham (not spam), 1 = spam
    text  : cleaned message text
"""

import csv
import zipfile
from pathlib import Path

import pandas as pd

from preprocessing import clean_text

# Default data folder, always relative to this file (works from any directory)
DATA_DIR = Path(__file__).resolve().parent / "data"

# Accepted label spellings -> numeric class (0 = ham, 1 = spam)
LABEL_MAP = {
    "ham": 0, "not spam": 0, "notspam": 0, "legitimate": 0, "legit": 0, "0": 0,
    "spam": 1, "junk": 1, "1": 1,
}

# Possible column names in CSV files (compared in lower case)
LABEL_COLUMNS = ["label", "v1", "category", "class", "target", "spam"]
TEXT_COLUMNS = ["text", "v2", "message", "email", "body", "content"]


def _extract_zip_if_present(data_dir):
    """If the UCI zip is in data/, extract its files (flat, no sub-folders)."""
    for zip_path in data_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.namelist():
                    name = Path(member).name          # drop any folder parts (safe)
                    if not name or (data_dir / name).exists():
                        continue
                    (data_dir / name).write_bytes(archive.read(member))
        except zipfile.BadZipFile:
            raise ValueError(f"'{zip_path.name}' is not a valid zip file. Download it again.")


def find_dataset_file(data_dir=DATA_DIR):
    """Return the path of the dataset file inside data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    _extract_zip_if_present(data_dir)

    # Preferred file names first, then any other csv/tsv/txt file
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


def _read_csv_flexible(path, **kwargs):
    """Read a file as UTF-8, falling back to latin-1 (common for spam datasets)."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def _read_raw_table(path):
    """Read the file into a DataFrame with columns 'label' and 'text'."""
    path = Path(path)
    suffix = path.suffix.lower()

    # UCI raw format: no extension (or .txt/.tsv), tab separated, no header
    if suffix in {"", ".txt", ".tsv"}:
        df = _read_csv_flexible(
            path, sep="\t", header=None, names=["label", "text"],
            quoting=csv.QUOTE_NONE, on_bad_lines="skip",
        )
        return df

    if suffix == ".csv":
        df = _read_csv_flexible(path)
        lower_to_original = {str(c).strip().lower(): c for c in df.columns}
        label_col = next((lower_to_original[c] for c in LABEL_COLUMNS if c in lower_to_original), None)
        text_col = next((lower_to_original[c] for c in TEXT_COLUMNS if c in lower_to_original), None)
        if label_col is None or text_col is None:
            raise ValueError(
                f"Could not find label/text columns in '{path.name}'. "
                f"Columns found: {list(df.columns)}. "
                f"Expected a label column {LABEL_COLUMNS} and a text column {TEXT_COLUMNS}."
            )
        return df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})

    raise ValueError(f"Unsupported file format '{suffix}'. Use the UCI file or a .csv file.")


def load_dataset(path=None):
    """
    Load and validate the dataset.
    Returns (DataFrame[label, text], info_dict_with_cleaning_counts).
    """
    path = Path(path) if path else find_dataset_file()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        df = _read_raw_table(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file '{path.name}' is empty.")

    info = {"file": path.name, "rows_read": len(df)}
    if df.empty:
        raise ValueError(f"Dataset file '{path.name}' has no rows.")

    # 1) Missing values
    before = len(df)
    df = df.dropna(subset=["label", "text"])
    info["missing_removed"] = before - len(df)

    # 2) Labels: normalise spelling, drop anything unrecognised
    normalised = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    info["invalid_labels_removed"] = int(normalised.isna().sum())
    df = df.assign(label=normalised).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 3) Text: clean whitespace, drop empty messages
    df["text"] = df["text"].apply(clean_text)
    before = len(df)
    df = df[df["text"] != ""]
    info["empty_text_removed"] = before - len(df)

    # 4) Exact duplicate messages (keep the first one)
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    info["duplicates_removed"] = before - len(df)

    df = df.reset_index(drop=True)
    info["rows_final"] = len(df)

    # 5) Final validation: both classes needed, at least 2 each for stratified split
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Dataset must contain BOTH spam and ham messages after cleaning.")
    if counts.min() < 2:
        raise ValueError("Each class needs at least 2 samples for a stratified train/test split.")

    return df[["label", "text"]], info
