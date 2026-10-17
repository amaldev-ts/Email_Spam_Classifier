"""
model.py
--------
Gaussian Naïve Bayes model definition, training, pipeline construction, and persistence.

Probabilistic Foundation:
    GaussianNB assumes each feature x_i (word frequency/count) follows a univariate
    normal distribution conditioned on class C:
        P(x_i | C) = 1 / sqrt(2 * pi * sigma_{C,i}^2) * exp( - (x_i - mu_{C,i})^2 / (2 * sigma_{C,i}^2) )
    
    Using Bayes' theorem and the class-conditional feature independence assumption:
        P(C | x) proportional to P(C) * prod_{i=1}^D P(x_i | C)

Why Variance Smoothing (`var_smoothing`) is Essential:
    Word counts are sparse; most words are zero for most messages. Without smoothing
    (default var_smoothing=1e-9), feature variance sigma_{C,i}^2 collapses near zero.
    When a test legitimate (ham) message contains a rare word never seen in training ham,
    the denominator in the exponent causes the Gaussian likelihood to approach 0,
    resulting in astronomical false positives.
    
    Regularizing feature variances with `var_smoothing = 0.05` adds an additive buffer:
        sigma_{C,i,regularized}^2 = sigma_{C,i}^2 + epsilon * max_j(sigma_j^2)
    This stabilizes probabilities, dramatically increasing Spam Precision while maintaining
    high Recall.
"""

from pathlib import Path
from typing import Tuple

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from preprocessing import build_vectorizer, make_dense_transformer

# Output folder, always relative to this file
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_FILE = "gaussian_nb_model.joblib"
VECTORIZER_FILE = "count_vectorizer.joblib"
PIPELINE_FILE = "spam_classifier_pipeline.joblib"

# Tuned variance smoothing factor (selected via 5-fold Stratified Cross-Validation on training data)
DEFAULT_VAR_SMOOTHING = 0.05


def build_gaussian_nb_pipeline(vectorizer=None, var_smoothing=DEFAULT_VAR_SMOOTHING) -> Pipeline:
    """
    Construct a complete, leak-free scikit-learn Pipeline:
    Text -> Vectorizer -> DenseTransformer -> GaussianNB.
    """
    if vectorizer is None:
        vectorizer = build_vectorizer()
    return Pipeline([
        ("vectorizer", vectorizer),
        ("to_dense", make_dense_transformer()),
        ("classifier", GaussianNB(var_smoothing=var_smoothing)),
    ])


def train_model(X_train, y_train, var_smoothing=DEFAULT_VAR_SMOOTHING) -> GaussianNB:
    """Train and return a GaussianNB classifier on dense feature arrays."""
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    return model


def predict_labels(model, X):
    """Return predicted class labels (0 = ham, 1 = spam)."""
    return model.predict(X)


def predict_probabilities(model, X):
    """Return predicted class probabilities [P(ham), P(spam)]."""
    return model.predict_proba(X)


def save_artifacts(model, vectorizer, pipeline=None, output_dir=OUTPUT_DIR) -> None:
    """Save trained model, vectorizer, and full pipeline to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / MODEL_FILE)
    joblib.dump(vectorizer, output_dir / VECTORIZER_FILE)
    if pipeline is not None:
        joblib.dump(pipeline, output_dir / PIPELINE_FILE)


def load_artifacts(output_dir=OUTPUT_DIR) -> Tuple[GaussianNB, object]:
    """Load (model, vectorizer). Raises FileNotFoundError if not trained yet."""
    output_dir = Path(output_dir)
    model_path = output_dir / MODEL_FILE
    vectorizer_path = output_dir / VECTORIZER_FILE
    if not model_path.is_file() or not vectorizer_path.is_file():
        raise FileNotFoundError(
            "Saved model not found in 'outputs/'. Train first with:  python main.py"
        )
    return joblib.load(model_path), joblib.load(vectorizer_path)


def load_pipeline(output_dir=OUTPUT_DIR) -> Pipeline:
    """Load the full end-to-end Pipeline artifact."""
    output_dir = Path(output_dir)
    pipe_path = output_dir / PIPELINE_FILE
    if not pipe_path.is_file():
        raise FileNotFoundError(
            "Saved pipeline not found in 'outputs/'. Train first with:  python main.py"
        )
    return joblib.load(pipe_path)
