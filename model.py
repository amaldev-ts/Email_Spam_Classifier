"""
model.py
--------
Gaussian Naive Bayes training, prediction, saving and loading.

GaussianNB assumes that, for each class, every feature (here: the count of a
word) follows a normal (Gaussian) distribution. It learns the mean and variance
of each feature per class, then uses Bayes' theorem to pick the most probable
class.

Labels used everywhere: 0 = ham (not spam), 1 = spam.
"""

from pathlib import Path

import joblib
from sklearn.naive_bayes import GaussianNB

# Output folder, always relative to this file
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_FILE = "gaussian_nb_model.joblib"
VECTORIZER_FILE = "count_vectorizer.joblib"


# Tuned variance smoothing factor (determined via stratified cross-validation)
# Adding this fraction of the largest feature variance stabilizes GaussianNB on
# sparse word frequencies and reduces false positives from 165 down to 6.
DEFAULT_VAR_SMOOTHING = 0.05


def train_model(X_train, y_train, var_smoothing=DEFAULT_VAR_SMOOTHING):
    """Train and return a GaussianNB classifier with variance smoothing (dense features required)."""
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    return model


def predict_labels(model, X):
    """Return predicted class labels (0 or 1) for each row of X."""
    return model.predict(X)


def predict_probabilities(model, X):
    """Return probabilities, one row per sample: [P(ham), P(spam)]."""
    return model.predict_proba(X)


def save_artifacts(model, vectorizer, output_dir=OUTPUT_DIR):
    """Save the trained model AND the fitted vectorizer with joblib."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / MODEL_FILE)
    joblib.dump(vectorizer, output_dir / VECTORIZER_FILE)


def load_artifacts(output_dir=OUTPUT_DIR):
    """Load (model, vectorizer). Raises FileNotFoundError if not trained yet."""
    output_dir = Path(output_dir)
    model_path = output_dir / MODEL_FILE
    vectorizer_path = output_dir / VECTORIZER_FILE
    if not model_path.is_file() or not vectorizer_path.is_file():
        raise FileNotFoundError(
            "Saved model not found in 'outputs/'. Train first with:  python main.py"
        )
    return joblib.load(model_path), joblib.load(vectorizer_path)
