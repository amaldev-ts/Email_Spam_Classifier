"""
tests/test_predict.py
---------------------
Unit tests for CLI prediction functions, input validation, and sample vectors.
"""

import numpy as np
import pytest

from data_loader import FEATURE_NAMES
from model import load_artifacts
from predict import (
    SAMPLE_NOT_SPAM_VECTOR,
    SAMPLE_SPAM_VECTOR,
    format_probability,
    predict_from_features,
    predict_from_text,
)


@pytest.fixture(scope="module")
def pipeline():
    """Load fitted pipeline artifact for inference tests."""
    return load_artifacts()


def test_format_probability():
    """Verify probability formatting prevents misleading absolute certainty."""
    assert format_probability(0.99999) == "> 99.99%"
    assert format_probability(0.00001) == "< 0.01%"
    assert format_probability(0.7523) == "75.23%"
    assert format_probability(0.50) == "50.00%"


def test_predict_from_features_valid(pipeline):
    """Verify inference on valid 57-feature numeric vectors."""
    features = np.zeros(57, dtype=np.float64)
    result = predict_from_features(features, pipeline)

    assert "label" in result
    assert result["label"] in {"SPAM", "NOT SPAM"}
    assert "p_not_spam" in result
    assert "p_spam" in result
    assert 0.0 <= result["p_not_spam"] <= 1.0
    assert 0.0 <= result["p_spam"] <= 1.0
    assert abs((result["p_not_spam"] + result["p_spam"]) - 1.0) < 1e-5


def test_predict_from_features_dimension_mismatch(pipeline):
    """Verify ValueError is raised if feature count is not 57."""
    short_features = np.zeros(10, dtype=np.float64)
    with pytest.raises(ValueError, match="Expected 57 features"):
        predict_from_features(short_features, pipeline)

    long_features = np.zeros(60, dtype=np.float64)
    with pytest.raises(ValueError, match="Expected 57 features"):
        predict_from_features(long_features, pipeline)


def test_predict_from_text(pipeline):
    """Verify text-based heuristic prediction pipeline."""
    spam_text = "Claim your free money prize from our internet business now!"
    result = predict_from_text(spam_text, pipeline)
    assert result["label"] == "SPAM"
    assert result["is_approximate_text"] is True

    legit_text = "Hi George, attached are the lab notes for our hp project meeting."
    result_legit = predict_from_text(legit_text, pipeline)
    assert result_legit["label"] == "NOT SPAM"
    assert result_legit["is_approximate_text"] is True


def test_predict_from_text_empty_raises(pipeline):
    """Verify empty or whitespace strings raise ValueError."""
    with pytest.raises(ValueError, match="Email text is empty"):
        predict_from_text("", pipeline)

    with pytest.raises(ValueError, match="Email text is empty"):
        predict_from_text("   \n\t   ", pipeline)


def test_sample_vectors(pipeline):
    """Verify built-in test sample vectors have valid length and expected labels."""
    assert len(SAMPLE_SPAM_VECTOR) == 57
    assert len(SAMPLE_NOT_SPAM_VECTOR) == 57

    spam_res = predict_from_features(np.array(SAMPLE_SPAM_VECTOR), pipeline)
    assert spam_res["label"] == "SPAM"
    assert spam_res["p_spam"] > 0.90

    not_spam_res = predict_from_features(np.array(SAMPLE_NOT_SPAM_VECTOR), pipeline)
    assert not_spam_res["label"] == "NOT SPAM"
    assert not_spam_res["p_not_spam"] > 0.90
