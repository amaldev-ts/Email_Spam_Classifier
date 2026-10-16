"""
preprocessing.py
----------------
Text cleaning and numerical feature extraction.

Feature type: word frequency (bag-of-words) using CountVectorizer.

Why dense arrays?
    CountVectorizer returns a SPARSE matrix (it stores only non-zero counts).
    GaussianNB does not accept sparse input, so we convert the matrix to a
    DENSE NumPy array with .toarray() before training / predicting.

Memory note:
    A dense matrix stores every cell, including all the zeros:
        memory = rows x columns x bytes_per_number
    To keep this small we (1) limit the vocabulary to MAX_FEATURES words and
    (2) use float32 (4 bytes) instead of float64 (8 bytes).
    Example: 4,500 messages x 2,000 words x 4 bytes ~= 36 MB.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Vocabulary settings (optimized via stratified cross-validation)
MAX_FEATURES = 2000   # 2000 most informative word features
MIN_DOC_FREQ = 2      # ignore words appearing in fewer than 2 messages


def clean_text(text):
    """Return text with missing values handled and extra whitespace removed."""
    if not isinstance(text, str):
        return ""                      # None / NaN / numbers -> empty string
    return " ".join(text.split())      # collapse tabs, newlines, repeated spaces


def build_vectorizer(max_features=MAX_FEATURES, min_df=MIN_DOC_FREQ, stop_words="english", lowercase=True):
    """Create a NEW, unfitted CountVectorizer (word-frequency features)."""
    return CountVectorizer(
        lowercase=lowercase,
        stop_words=stop_words,
        max_features=max_features,
        min_df=min_df,
    )


def to_dense(sparse_matrix):
    """Convert a sparse matrix to a dense float32 NumPy array for GaussianNB."""
    return sparse_matrix.toarray().astype(np.float32)


def fit_vectorizer(train_texts, max_features=MAX_FEATURES, min_df=MIN_DOC_FREQ):
    """
    Fit the vectorizer on TRAINING text only (prevents data leakage).
    Returns (fitted_vectorizer, dense_training_features).
    """
    vectorizer = build_vectorizer(max_features=max_features, min_df=min_df)
    sparse_features = vectorizer.fit_transform(train_texts)
    return vectorizer, to_dense(sparse_features)


def transform_texts(vectorizer, texts):
    """
    Transform text with an ALREADY FITTED vectorizer (never re-fits).
    Used for the test set and for new messages typed by the user.
    """
    sparse_features = vectorizer.transform(texts)
    return to_dense(sparse_features)
