"""
preprocessing.py
----------------
Text cleaning, numerical feature extraction, and dense array transformation.

Feature type: word frequency (bag-of-words) via CountVectorizer.

Why dense arrays?
    CountVectorizer generates a SPARSE matrix (storing only non-zero word counts).
    scikit-learn's GaussianNB requires continuous feature values and does not accept
    sparse matrices. Therefore, we convert the sparse matrix to a DENSE NumPy array
    using .toarray() before training or inference.

Memory Management:
    A dense matrix stores every cell, including zeros:
        Memory (bytes) = rows x columns x bytes_per_element
    To maintain a light footprint:
      1. We constrain the vocabulary (default max_features=2,000 words).
      2. We cast to float32 (4 bytes per cell) instead of float64 (8 bytes).
      Example: 4,128 training messages x 2,000 features x 4 bytes ~= 33 MB.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer

# Default vocabulary settings
MAX_FEATURES = 2000   # Most informative word features
MIN_DOC_FREQ = 2      # Ignore words appearing in fewer than 2 messages


def clean_text(text) -> str:
    """Return text with nulls handled and redundant whitespace collapsed."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def build_vectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DOC_FREQ,
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 1),
) -> CountVectorizer:
    """Create a new, unfitted CountVectorizer for word-frequency extraction."""
    return CountVectorizer(
        lowercase=lowercase,
        stop_words=stop_words,
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )


def to_dense(sparse_matrix) -> np.ndarray:
    """Convert a scipy sparse matrix to a dense float32 NumPy array for GaussianNB."""
    if hasattr(sparse_matrix, "toarray"):
        return sparse_matrix.toarray().astype(np.float32)
    return np.asarray(sparse_matrix, dtype=np.float32)


def make_dense_transformer() -> FunctionTransformer:
    """Return a scikit-learn compatible transformer for seamless Pipeline usage."""
    return FunctionTransformer(
        func=to_dense,
        accept_sparse=True,
        validate=False,
    )


def fit_vectorizer(train_texts, max_features=MAX_FEATURES, min_df=MIN_DOC_FREQ, stop_words="english"):
    """
    Fit the vectorizer on TRAINING text only (prevents test data leakage).
    Returns (fitted_vectorizer, dense_training_features).
    """
    vectorizer = build_vectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
    )
    sparse_features = vectorizer.fit_transform(train_texts)
    return vectorizer, to_dense(sparse_features)


def transform_texts(vectorizer, texts):
    """
    Transform text using an ALREADY FITTED vectorizer (never re-fits).
    Used for test partitions and new interactive prediction inputs.
    """
    sparse_features = vectorizer.transform(texts)
    return to_dense(sparse_features)
