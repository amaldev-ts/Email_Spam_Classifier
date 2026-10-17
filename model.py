"""
model.py
--------
Gaussian Naïve Bayes modeling, pipeline construction, and persistence
for the UCI Spambase email classifier.

Probabilistic Basis of GaussianNB on Spambase:
    Under Bayes' theorem, the posterior probability of an email being Spam (class 1)
    or Not Spam (class 0) given its 57 continuous features x = [x_1, ..., x_57] is:
        P(C | x) proportional to P(C) * prod_{i=1}^{57} P(x_i | C)
    
    Each continuous feature x_i is modeled as a univariate Gaussian distribution:
        P(x_i | C) = 1 / sqrt(2 * pi * sigma_{C,i}^2) * exp( - (x_i - mu_{C,i})^2 / (2 * sigma_{C,i}^2) )

Variance Smoothing Regularization:
    When empirical feature variances are small, GaussianNB can become overly sensitive
    to single-feature outliers. Regularization via `var_smoothing` adds a fraction epsilon
    of the maximum feature variance across all features:
        sigma_{C,i,smoothed}^2 = sigma_{C,i}^2 + epsilon * max_j(sigma_j^2)
"""

from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from preprocessing import make_log1p_transformer

# Output folder, always relative to this file
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PIPELINE_FILE = "spambase_gaussian_nb_pipeline.joblib"
MODEL_FILE = "gaussian_nb_model.joblib"

# Tuned default variance smoothing factor
DEFAULT_VAR_SMOOTHING = 0.05


def build_spambase_pipeline(use_log: bool = True, var_smoothing: float = DEFAULT_VAR_SMOOTHING) -> Pipeline:
    """
    Construct a complete scikit-learn Pipeline for Spambase classification.
    Optionally applies log1p transformation before GaussianNB.
    """
    steps = []
    if use_log:
        steps.append(("log1p", make_log1p_transformer()))
    steps.append(("classifier", GaussianNB(var_smoothing=var_smoothing)))
    return Pipeline(steps)


def train_model(X_train, y_train, use_log: bool = True, var_smoothing: float = DEFAULT_VAR_SMOOTHING) -> Pipeline:
    """Train and return a fitted Spambase Pipeline."""
    pipeline = build_spambase_pipeline(use_log=use_log, var_smoothing=var_smoothing)
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_labels(pipeline: Pipeline, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Return predicted class labels (0 = Not Spam, 1 = Spam)."""
    return pipeline.predict(X)


def predict_probabilities(pipeline: Pipeline, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Return predicted class probabilities [P(Not Spam), P(Spam)]."""
    return pipeline.predict_proba(X)


def save_artifacts(pipeline: Pipeline, output_dir: Path = OUTPUT_DIR) -> None:
    """Serialize the trained pipeline artifact to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / PIPELINE_FILE)
    # Also save the classifier separately for convenience
    if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
        joblib.dump(pipeline.named_steps["classifier"], output_dir / MODEL_FILE)


def load_artifacts(output_dir: Path = OUTPUT_DIR) -> Pipeline:
    """Load the trained Spambase pipeline. Raises FileNotFoundError if not trained yet."""
    output_dir = Path(output_dir)
    pipe_path = output_dir / PIPELINE_FILE
    if not pipe_path.is_file():
        raise FileNotFoundError(
            "Saved model pipeline not found in 'outputs/'. Train first with:  python main.py"
        )
    return joblib.load(pipe_path)
