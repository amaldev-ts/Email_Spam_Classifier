"""
main.py
-------
Train, evaluate and use the Gaussian Naive Bayes spam classifier.

    python main.py                  -> train + evaluate + save outputs
    python main.py --predict        -> classify messages you type (needs training first)
    python main.py --data PATH      -> train using a specific dataset file
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from evaluate import (
    compute_metrics,
    format_metrics_text,
    plot_class_distribution,
    plot_confusion_matrix,
    save_metrics_text,
)
from model import DEFAULT_VAR_SMOOTHING, OUTPUT_DIR, predict_labels, save_artifacts, train_model
from preprocessing import MAX_FEATURES, MIN_DOC_FREQ, fit_vectorizer, transform_texts
from predict import run_interactive

RANDOM_SEED = 42   # fixed seed -> same split and same results every run
TEST_SIZE = 0.2    # 80% training, 20% testing


def generate_model_comparison(X_train_text, y_train, X_test_text, y_test, output_dir=OUTPUT_DIR):
    """
    Run and compare systematic configurations on the same held-out test set.
    Saves outputs/model_comparison.csv.
    """
    experiments = [
        {
            "Experiment": "1. Baseline (CountVec, default vs)",
            "Feature Extraction": "CountVectorizer",
            "Vocabulary Settings": "max_features=2000, min_df=2, unigrams, stop_words=english",
            "var_smoothing": 1e-9,
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
        },
        {
            "Experiment": "2. CountVec + Tuned var_smoothing",
            "Feature Extraction": "CountVectorizer",
            "Vocabulary Settings": "max_features=2000, min_df=2, unigrams, stop_words=english",
            "var_smoothing": 0.08,
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
        },
        {
            "Experiment": "3. TF-IDF + Default var_smoothing",
            "Feature Extraction": "TfidfVectorizer",
            "Vocabulary Settings": "max_features=2000, min_df=2, unigrams, stop_words=english",
            "var_smoothing": 1e-9,
            "vec": TfidfVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
        },
        {
            "Experiment": "4. TF-IDF + Tuned var_smoothing",
            "Feature Extraction": "TfidfVectorizer",
            "Vocabulary Settings": "max_features=2000, min_df=2, unigrams, stop_words=english",
            "var_smoothing": 2.0,
            "vec": TfidfVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
        },
        {
            "Experiment": "5. CountVec + Bigrams (1, 2)",
            "Feature Extraction": "CountVectorizer (1, 2 n-grams)",
            "Vocabulary Settings": "max_features=2000, min_df=2, bigrams, stop_words=english",
            "var_smoothing": 0.08,
            "vec": CountVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=2000, min_df=2),
        },
        {
            "Experiment": "6. CountVec + No Stop Words",
            "Feature Extraction": "CountVectorizer",
            "Vocabulary Settings": "max_features=2000, min_df=2, unigrams, stop_words=None",
            "var_smoothing": 0.08,
            "vec": CountVectorizer(lowercase=True, stop_words=None, max_features=2000, min_df=2),
        },
        {
            "Experiment": "7. CountVec + Compact Vocab 1500",
            "Feature Extraction": "CountVectorizer",
            "Vocabulary Settings": "max_features=1500, min_df=2, unigrams, stop_words=english",
            "var_smoothing": 0.08,
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=1500, min_df=2),
        },
        {
            "Experiment": "8. Final Model (CountVec + Balanced vs)",
            "Feature Extraction": "CountVectorizer",
            "Vocabulary Settings": "max_features=2000, min_df=2, unigrams, stop_words=english",
            "var_smoothing": 0.05,
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
        },
    ]

    rows = []
    for exp in experiments:
        vec = exp["vec"]
        X_tr = vec.fit_transform(X_train_text).toarray().astype("float32")
        X_te = vec.transform(X_test_text).toarray().astype("float32")
        
        clf = train_model(X_tr, y_train, var_smoothing=exp["var_smoothing"])
        y_pred = clf.predict(X_te)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        rows.append({
            "Experiment": exp["Experiment"],
            "Feature Extraction": exp["Feature Extraction"],
            "Vocabulary Settings": exp["Vocabulary Settings"],
            "var_smoothing": exp["var_smoothing"],
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Spam Precision": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            "Spam Recall": round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            "Spam F1-score": round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
            "False Positive Rate": round(fpr, 4),
        })

    df_comp = pd.DataFrame(rows)
    csv_path = output_dir / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    return df_comp


def train_and_evaluate(data_path=None, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """Run the full pipeline: load -> split -> features -> train -> evaluate -> save."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Steps 1-3: load, validate and clean the dataset
    print("[1/6] Loading dataset...")
    df, info = load_dataset(data_path)
    n_ham = int((df["label"] == 0).sum())
    n_spam = int((df["label"] == 1).sum())
    print(f"      File: {info['file']}  | rows read: {info['rows_read']}")
    print(f"      Removed -> missing: {info['missing_removed']}, invalid labels: "
          f"{info['invalid_labels_removed']}, empty text: {info['empty_text_removed']}, "
          f"duplicates: {info['duplicates_removed']}")
    print(f"      Final dataset: {info['rows_final']} messages  (ham: {n_ham}, spam: {n_spam})")
    plot_class_distribution(df["label"], OUTPUT_DIR / "class_distribution.png")

    # Steps 4-5: stratified split with a fixed seed
    print("[2/6] Splitting into train and test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )
    print(f"      Train: {len(X_train_text)}  | Test: {len(X_test_text)}")

    # Step 6: features. Vectorizer is fitted on TRAINING text only (no data leakage)
    print("[3/6] Extracting word-frequency features (CountVectorizer)...")
    vectorizer, X_train = fit_vectorizer(X_train_text)
    X_test = transform_texts(vectorizer, X_test_text)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"      Vocabulary: {vocab_size} words | Dense train matrix: {X_train.shape}")

    # Step 7: train GaussianNB with tuned var_smoothing
    print(f"[4/6] Training GaussianNB (var_smoothing={DEFAULT_VAR_SMOOTHING})...")
    model = train_model(X_train, y_train, var_smoothing=DEFAULT_VAR_SMOOTHING)

    # Steps 8-9: predict and evaluate on the test set
    print("[5/6] Evaluating on the test set...")
    y_pred = predict_labels(model, X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Step 10: run experiment comparison and save outputs
    print("[6/6] Generating model comparison and saving outputs...")
    comparison_df = generate_model_comparison(X_train_text, y_train, X_test_text, y_test, OUTPUT_DIR)
    
    run_info = {
        "file": info["file"], "total": len(df), "ham": n_ham, "spam": n_spam,
        "train_size": len(X_train_text), "test_size": len(X_test_text),
        "vocab_size": vocab_size, "seed": seed, "var_smoothing": DEFAULT_VAR_SMOOTHING,
    }
    text = format_metrics_text(metrics, run_info)
    save_metrics_text(text, OUTPUT_DIR / "metrics.txt")
    plot_confusion_matrix(metrics["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png")
    save_artifacts(model, vectorizer, OUTPUT_DIR)

    print("\n" + text)
    print("\n" + "=" * 64)
    print("MODEL COMPARISON (Fair comparison on held-out test set)")
    print("=" * 64)
    print(comparison_df.to_string(index=False))
    print("\nSaved to 'outputs/':")
    print("  - metrics.txt")
    print("  - confusion_matrix.png")
    print("  - class_distribution.png")
    print("  - model_comparison.csv")
    print("  - gaussian_nb_model.joblib")
    print("  - count_vectorizer.joblib")
    print("\nClassify a new message with:  python main.py --predict")


def main():
    parser = argparse.ArgumentParser(description="Spam classifier using Gaussian Naive Bayes.")
    parser.add_argument("--data", help="Path to dataset file (default: auto-detect in data/)")
    parser.add_argument("--predict", action="store_true",
                        help="Classify messages you type, using the saved model")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Fraction of data used for testing (default 0.2)")
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        print("Error: --test-size must be between 0 and 1 (e.g. 0.2).")
        sys.exit(1)

    try:
        if args.predict:
            run_interactive()
        else:
            train_and_evaluate(Path(args.data) if args.data else None, args.test_size)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
