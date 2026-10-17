"""
tests/test_model.py
-------------------
Unit tests for pipeline construction, training, prediction, and artifact serialization.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from data_loader import FEATURE_NAMES, load_dataset
from model import (
    DEFAULT_VAR_SMOOTHING,
    build_spambase_pipeline,
    load_artifacts,
    predict_labels,
    predict_probabilities,
    save_artifacts,
    train_model,
)


@pytest.fixture(scope="module")
def sample_data():
    """Load a small sample of the Spambase dataset for fast testing."""
    df, _ = load_dataset()
    sample = df.sample(n=100, random_state=42)
    X = sample[FEATURE_NAMES]
    y = sample["label"]
    return X, y


def test_build_spambase_pipeline():
    """Verify that build_spambase_pipeline constructs valid scikit-learn Pipelines."""
    pipe_log = build_spambase_pipeline(use_log=True, var_smoothing=0.05)
    assert isinstance(pipe_log, Pipeline)
    assert "log1p" in pipe_log.named_steps
    assert "classifier" in pipe_log.named_steps
    assert pipe_log.named_steps["classifier"].var_smoothing == 0.05

    pipe_raw = build_spambase_pipeline(use_log=False, var_smoothing=1e-9)
    assert isinstance(pipe_raw, Pipeline)
    assert "log1p" not in pipe_raw.named_steps
    assert pipe_raw.named_steps["classifier"].var_smoothing == 1e-9


def test_train_model(sample_data):
    """Verify that train_model fits successfully and produces callable pipeline."""
    X, y = sample_data
    pipeline = train_model(X, y, use_log=True, var_smoothing=0.05)

    assert hasattr(pipeline.named_steps["classifier"], "classes_")
    assert set(pipeline.named_steps["classifier"].classes_) == {0, 1}


def test_predict_labels_and_probabilities(sample_data):
    """Verify shape, valid values, and probability calibration summing to 1.0."""
    X, y = sample_data
    pipeline = train_model(X, y, use_log=True, var_smoothing=0.05)

    labels = predict_labels(pipeline, X)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(X)
    assert set(np.unique(labels)).issubset({0, 1})

    probs = predict_probabilities(pipeline, X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Probabilities must sum to 1.0 for each sample
    prob_sums = probs.sum(axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)


def test_save_and_load_artifacts(sample_data):
    """Verify serialization and deserialization preserves model predictions."""
    X, y = sample_data
    pipeline = train_model(X, y, use_log=True, var_smoothing=0.05)
    orig_preds = predict_probabilities(pipeline, X)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_artifacts(pipeline, output_dir=tmp_path)

        assert (tmp_path / "spambase_gaussian_nb_pipeline.joblib").is_file()
        assert (tmp_path / "gaussian_nb_model.joblib").is_file()

        loaded_pipe = load_artifacts(output_dir=tmp_path)
        loaded_preds = predict_probabilities(loaded_pipe, X)

        np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-6)
