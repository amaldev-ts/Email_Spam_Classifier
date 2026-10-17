"""
tests/test_preprocessing.py
----------------------------
Unit tests for text cleaning, approximate feature extraction, and Log1p transform.
"""

import numpy as np
import pytest

from preprocessing import clean_text, extract_spambase_features, make_log1p_transformer
from data_loader import FEATURE_NAMES, SPAMBASE_WORDS, SPAMBASE_CHARS


def test_clean_text():
    """Verify that clean_text removes extra whitespace and handles edge cases."""
    raw = "  Hello \t world \n\n this is an   email!  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world this is an email!"
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_extract_spambase_features_shape_and_type():
    """Verify feature extractor output shape, dtype, and feature count."""
    text = "Claim your free money prize from our internet business now!"
    feats = extract_spambase_features(text)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (57,)
    assert feats.dtype == np.float64
    assert np.all(feats >= 0), "All extracted features must be non-negative"


def test_extract_spambase_features_empty_text():
    """Verify safe fallback for empty or whitespace text."""
    feats_empty = extract_spambase_features("")
    assert feats_empty.shape == (57,)
    assert np.all(feats_empty >= 0)

    feats_space = extract_spambase_features("    ")
    assert feats_space.shape == (57,)


def test_extract_spambase_features_signals():
    """Verify that known words and characters trigger positive frequencies."""
    text = "FREE money $$$ from our business!"
    feats = extract_spambase_features(text)

    # Check that word_freq_free, word_freq_money, word_freq_our, char_freq_$ are > 0
    free_idx = FEATURE_NAMES.index("word_freq_free")
    money_idx = FEATURE_NAMES.index("word_freq_money")
    dollar_idx = FEATURE_NAMES.index("char_freq_$")

    assert feats[free_idx] > 0.0
    assert feats[money_idx] > 0.0
    assert feats[dollar_idx] > 0.0

    # Capital run length total should be positive (FREE is 4 capital letters)
    cap_tot_idx = FEATURE_NAMES.index("capital_run_length_total")
    assert feats[cap_tot_idx] >= 4.0


def test_make_log1p_transformer():
    """Verify that make_log1p_transformer applies log(1 + x) correctly."""
    transformer = make_log1p_transformer()
    data = np.array([[0.0, 1.0, 9.0], [10.0, 99.0, 0.0]])
    transformed = transformer.transform(data)

    expected = np.log1p(data)
    np.testing.assert_allclose(transformed, expected, rtol=1e-5)
