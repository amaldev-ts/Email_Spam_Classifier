"""
tests/test_data_loader.py
-------------------------
Unit tests for UCI Spambase dataset finding, loading, and auditing.
"""

from pathlib import Path
import pandas as pd
import pytest

from data_loader import (
    COLUMN_NAMES,
    DATA_DIR,
    FEATURE_NAMES,
    find_dataset_file,
    load_dataset,
)


def test_find_dataset_file():
    """Verify that the dataset finder locates spambase.data in the data folder."""
    data_path = find_dataset_file()
    assert data_path.is_file(), f"File does not exist: {data_path}"
    assert data_path.name in {"spambase.data", "spambase.csv", "spambase.data.txt"}


def test_canonical_dataset_loading_default():
    """Verify default loading preserves canonical 4,601 instances with full schema."""
    df, info = load_dataset(clean_duplicates=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4601, f"Expected 4,601 rows, got {len(df)}"
    assert list(df.columns) == COLUMN_NAMES
    assert len(FEATURE_NAMES) == 57

    # Target label integrity
    assert set(df["label"].unique()) == {0, 1}
    assert df["label"].isna().sum() == 0

    # Class distribution
    assert info["not_spam_count"] == 2788
    assert info["spam_count"] == 1813
    assert info["not_spam_count"] + info["spam_count"] == 4601

    # Auditing info
    assert info["exact_duplicates_count"] == 391
    assert info["conflicting_groups_detected"] == 3
    assert info["conflicting_rows_detected"] == 6
    assert info["clean_duplicates_applied"] is False


def test_deduplicated_dataset_loading():
    """Verify clean_duplicates=True removes exact duplicates and conflicting rows."""
    df, info = load_dataset(clean_duplicates=True)

    assert len(df) == 4204, f"Expected 4,204 rows, got {len(df)}"
    assert info["clean_duplicates_applied"] is True
    assert info["duplicates_removed"] == 391
    assert info["conflicting_rows_removed"] == 6

    assert info["not_spam_count"] == 2528
    assert info["spam_count"] == 1676
    assert info["not_spam_count"] + info["spam_count"] == 4204


def test_dataset_no_nulls():
    """Verify that there are zero missing or null values in features or labels."""
    df, _ = load_dataset()
    assert df[FEATURE_NAMES].isna().sum().sum() == 0
    assert df["label"].isna().sum() == 0


def test_invalid_path_raises_error():
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("data/non_existent_file.xyz"))
