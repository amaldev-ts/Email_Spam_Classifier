"""
tests/test_pipeline.py
----------------------
Unit tests for metric computation, error extraction, cross-validation search,
and programmatic model selection.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from data_loader import FEATURE_NAMES, load_dataset
from evaluate import compute_metrics, extract_errors
from main import (
    compare_dataset_cleaning_policies,
    run_cross_validation_experiments,
)


@pytest.fixture(scope="module")
def sample_dataset():
    """Load a modest subset for fast integration tests."""
    df, _ = load_dataset()
    sample = df.sample(n=300, random_state=42)
    X = sample[FEATURE_NAMES]
    y = sample["label"]
    return X, y


def test_compute_metrics():
    """Verify compute_metrics calculates correct classification statistics."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.50
    assert metrics["precision"] == 0.50
    assert metrics["recall"] == 0.50
    assert metrics["f1"] == 0.50
    assert metrics["false_positive_rate"] == 0.50
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1


def test_extract_errors():
    """Verify error extraction identifies and characterizes misclassifications."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([1, 0, 0])
    X = pd.DataFrame(
        np.zeros((3, len(FEATURE_NAMES))),
        columns=FEATURE_NAMES,
    )
    X.loc[0, "word_freq_free"] = 5.0
    X.loc[0, "capital_run_length_total"] = 100.0

    errors_df = extract_errors(y_true, y_pred, X, FEATURE_NAMES)
    assert len(errors_df) == 2

    # First error: Actual 0 (Not Spam), Predicted 1 (Spam) -> False Positive
    fp_row = errors_df[errors_df["Sample_Index"] == 0].iloc[0]
    assert fp_row["Error_Type"] == "False Positive"
    assert fp_row["Actual"] == "Not Spam"
    assert fp_row["Predicted"] == "Spam"
    assert "word_freq_free=5.00" in fp_row["Top_Indicative_Features"]

    # Second error: Actual 1 (Spam), Predicted 0 (Not Spam) -> False Negative
    fn_row = errors_df[errors_df["Sample_Index"] == 1].iloc[0]
    assert fn_row["Error_Type"] == "False Negative"
    assert fn_row["Actual"] == "Spam"
    assert fn_row["Predicted"] == "Not Spam"


def test_cross_validation_and_programmatic_selection(sample_dataset):
    """Verify that run_cross_validation_experiments programmatically selects highest CV F1."""
    X, y = sample_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        df_cv, best_config = run_cross_validation_experiments(X, y, output_dir=tmp_path)

        assert isinstance(df_cv, pd.DataFrame)
        assert len(df_cv) == 12
        assert (tmp_path / "cv_experiments.csv").is_file()

        # Check required columns
        for col in ["Experiment", "CV_Accuracy_Mean", "CV_Precision_Mean", "CV_Recall_Mean", "CV_F1_Mean"]:
            assert col in df_cv.columns

        # Verify programmatic selection picks the row with highest CV_F1_Mean
        expected_best_f1 = df_cv["CV_F1_Mean"].max()
        assert best_config["cv_f1_mean"] == expected_best_f1
        assert "pipeline" in best_config
        assert "name" in best_config
        assert "id" in best_config


def test_compare_dataset_cleaning_policies():
    """Verify dataset cleaning comparison produces valid comparative metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        df_comp = compare_dataset_cleaning_policies(output_dir=tmp_path)

        assert isinstance(df_comp, pd.DataFrame)
        assert len(df_comp) == 2
        assert (tmp_path / "dataset_cleaning_comparison.csv").is_file()

        # Canonical vs Deduplicated rows
        canonical_row = df_comp[df_comp["Clean_Duplicates"] == False].iloc[0]
        dedup_row = df_comp[df_comp["Clean_Duplicates"] == True].iloc[0]

        assert canonical_row["Total_Rows"] == 4601
        assert dedup_row["Total_Rows"] == 4204
        assert canonical_row["CV_F1_Mean"] > 0.80
        assert dedup_row["CV_F1_Mean"] > 0.80
