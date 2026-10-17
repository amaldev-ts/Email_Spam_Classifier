"""
preprocessing.py
----------------
Feature definition, distributional transformation, and raw-email feature extraction
for the UCI Spambase email classification system.

This module provides:
  1. The official 57-feature schema definitions from UCI Spambase.
  2. An interactive `extract_spambase_features()` engine that converts any raw email
     text into the exact 57 Spambase features (satisfies Requirement #2 of CST413).
  3. Distributional transforms (Log1p transformation) to mitigate severe right-skewness
     and align continuous features closer to Gaussian normality.
"""

import re
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from data_loader import FEATURE_NAMES, SPAMBASE_CHARS, SPAMBASE_WORDS


def clean_text(text: str) -> str:
    """Normalize whitespace and strip null/non-string content."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def extract_spambase_features(text: str) -> np.ndarray:
    """
    Extract the 57 UCI Spambase continuous features from raw email text.

    Feature Definitions (UCI Spambase Specification):
      - 48 word frequencies: 100 * (count of WORD / total words in email).
        Words are defined as strings of alphanumeric characters.
      - 6 character frequencies: 100 * (count of CHAR / total characters in email).
        Target characters: ';', '(', '[', '!', '$', '#'
      - 3 capital letter run-length statistics:
          * capital_run_length_average : mean length of contiguous uppercase runs
          * capital_run_length_longest : maximum length of contiguous uppercase runs
          * capital_run_length_total   : sum of all contiguous uppercase runs
    
    Returns:
        np.ndarray of shape (57,) with dtype float64.
    """
    if not isinstance(text, str) or not text.strip():
        # Fallback for empty text
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    # 1. Word Frequencies (48 attributes)
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    total_words = len(words)
    word_freqs = []
    for target_word in SPAMBASE_WORDS:
        count = sum(1 for w in words if w == target_word)
        pct = (100.0 * count / total_words) if total_words > 0 else 0.0
        word_freqs.append(pct)

    # 2. Character Frequencies (6 attributes)
    total_chars = len(text)
    char_freqs = []
    for target_char in SPAMBASE_CHARS:
        count = text.count(target_char)
        pct = (100.0 * count / total_chars) if total_chars > 0 else 0.0
        char_freqs.append(pct)

    # 3. Capital Run-Length Statistics (3 attributes)
    capital_runs = [len(match) for match in re.findall(r"[A-Z]+", text)]
    if capital_runs:
        cap_avg = float(np.mean(capital_runs))
        cap_longest = int(max(capital_runs))
        cap_total = int(sum(capital_runs))
    else:
        cap_avg = 1.0
        cap_longest = 1
        cap_total = 0

    features = word_freqs + char_freqs + [cap_avg, cap_longest, cap_total]
    return np.array(features, dtype=np.float64)


def make_log1p_transformer() -> FunctionTransformer:
    """
    Return a scikit-learn transformer that applies log(1 + x) to features.
    
    Why Log1p is Essential for GaussianNB on Spambase:
      Percentages and run-length counts in Spambase are heavily zero-inflated
      and right-skewed (capital_run_length_total spans 1 to 15,841).
      Raw features violate the Gaussian normality assumption and cause feature
      variances to span orders of magnitude (from 0.01 to 3.7e5).
      Log1p stabilizes feature variances and maps skewed distributions into
      bell-shaped, near-normal profiles compatible with GaussianNB.
    """
    return FunctionTransformer(
        func=np.log1p,
        validate=False,
    )
