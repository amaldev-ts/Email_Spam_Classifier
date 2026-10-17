"""
main.py
-------
End-to-end training, cross-validation, evaluation, and artifact generation
for Gaussian Naïve Bayes spam classification.

Strict Machine Learning Workflow:
  1. Load, validate, and clean dataset (check for bad lines, conflicting labels, duplicates).
  2. Perform stratified train/test split (80/20, seed=42). Test set is completely isolated.
  3. Run 5-fold Stratified Cross-Validation ON TRAINING DATA ONLY across systematic
     GaussianNB configurations (smoothing, vocabulary size, min_df, n-grams, feature types).
  4. Refit the selected optimal GaussianNB model on the full training set.
  5. Evaluate the final model ONCE on the held-out test set. Also evaluate baseline for direct contrast.
  6. Extract False Positives and False Negatives for qualitative error analysis.
  7. Persist model artifacts, metric reports, CSV tables, and visualization plots.

CLI Usage:
    python main.py                  -> train, CV benchmark, evaluate, save
    python main.py --predict        -> interactive prediction CLI
    python main.py --data PATH      -> train with a custom dataset path
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from data_loader import load_dataset
from evaluate import (
    compute_metrics,
    extract_errors,
    format_metrics_text,
    plot_class_distribution,
    plot_confusion_matrix,
    save_metrics_text,
)
from model import (
    DEFAULT_VAR_SMOOTHING,
    OUTPUT_DIR,
    build_gaussian_nb_pipeline,
    save_artifacts,
    train_model,
)
from predict import run_interactive
from preprocessing import (
    MAX_FEATURES,
    MIN_DOC_FREQ,
    build_vectorizer,
    fit_vectorizer,
    make_dense_transformer,
    transform_texts,
)

RANDOM_SEED = 42
TEST_SIZE = 0.20


def run_cross_validation_experiments(
    X_train_text: pd.Series, y_train: pd.Series, output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """
    Run 5-Fold Stratified Cross-Validation on the training partition only.
    Tests smoothing, vocabulary size, min_df, n-grams, and feature representation.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "precision", "recall", "f1"]

    experiment_definitions = [
        {
            "id": 1,
            "name": "1. Baseline (vs=1e-9)",
            "category": "Baseline",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
            "var_smoothing": 1e-9,
            "description": "Unregularized default GaussianNB",
        },
        {
            "id": 2,
            "name": "2. Smoothing vs=1e-5",
            "category": "Variance Smoothing",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
            "var_smoothing": 1e-5,
            "description": "Minimal smoothing",
        },
        {
            "id": 3,
            "name": "3. Smoothing vs=1e-2",
            "category": "Variance Smoothing",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
            "var_smoothing": 1e-2,
            "description": "Moderate smoothing",
        },
        {
            "id": 4,
            "name": "4. Smoothing vs=0.05",
            "category": "Variance Smoothing",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
            "var_smoothing": 0.05,
            "description": "Balanced regularization (optimal)",
        },
        {
            "id": 5,
            "name": "5. Smoothing vs=0.08",
            "category": "Variance Smoothing",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
            "var_smoothing": 0.08,
            "description": "High precision smoothing",
        },
        {
            "id": 6,
            "name": "6. Vocab max_features=1000",
            "category": "Vocabulary Capacity",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=1000, min_df=2),
            "var_smoothing": 0.05,
            "description": "Compact 1000-word vocabulary",
        },
        {
            "id": 7,
            "name": "7. Vocab max_features=3000",
            "category": "Vocabulary Capacity",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=3000, min_df=2),
            "var_smoothing": 0.05,
            "description": "Expanded 3000-word vocabulary",
        },
        {
            "id": 8,
            "name": "8. Min_df=1 (no pruning)",
            "category": "Document Frequency",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=1),
            "var_smoothing": 0.05,
            "description": "Keep hapax legomena",
        },
        {
            "id": 9,
            "name": "9. Min_df=3 (rare pruned)",
            "category": "Document Frequency",
            "vec": CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=3),
            "var_smoothing": 0.05,
            "description": "Prune words with frequency < 3",
        },
        {
            "id": 10,
            "name": "10. N-grams (1, 2)",
            "category": "N-gram Range",
            "vec": CountVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=2000, min_df=2),
            "var_smoothing": 0.05,
            "description": "Unigrams + Bigrams",
        },
        {
            "id": 11,
            "name": "11. No Stop Words",
            "category": "Stop-Word Policy",
            "vec": CountVectorizer(lowercase=True, stop_words=None, max_features=2000, min_df=2),
            "var_smoothing": 0.05,
            "description": "Retain common English stop words",
        },
        {
            "id": 12,
            "name": "12. TF-IDF Representation",
            "category": "Feature Type",
            "vec": TfidfVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2),
            "var_smoothing": 2.0,
            "description": "TF-IDF features with scaled vs",
        },
    ]

    results = []
    for exp in experiment_definitions:
        pipe = Pipeline([
            ("vectorizer", exp["vec"]),
            ("to_dense", make_dense_transformer()),
            ("classifier", GaussianNB(var_smoothing=exp["var_smoothing"])),
        ])

        cv_scores = cross_validate(
            pipe,
            X_train_text,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        acc_mean = float(cv_scores["test_accuracy"].mean())
        acc_std = float(cv_scores["test_accuracy"].std())
        prec_mean = float(cv_scores["test_precision"].mean())
        prec_std = float(cv_scores["test_precision"].std())
        rec_mean = float(cv_scores["test_recall"].mean())
        rec_std = float(cv_scores["test_recall"].std())
        f1_mean = float(cv_scores["test_f1"].mean())
        f1_std = float(cv_scores["test_f1"].std())

        results.append({
            "Experiment": exp["name"],
            "Category": exp["category"],
            "var_smoothing": exp["var_smoothing"],
            "CV_Accuracy_Mean": round(acc_mean, 4),
            "CV_Accuracy_Std": round(acc_std, 4),
            "CV_Precision_Mean": round(prec_mean, 4),
            "CV_Precision_Std": round(prec_std, 4),
            "CV_Recall_Mean": round(rec_mean, 4),
            "CV_Recall_Std": round(rec_std, 4),
            "CV_F1_Mean": round(f1_mean, 4),
            "CV_F1_Std": round(f1_std, 4),
            "Description": exp["description"],
        })

    df_cv = pd.DataFrame(results)
    csv_path = output_dir / "cv_experiments.csv"
    df_cv.to_csv(csv_path, index=False)
    return df_cv


def train_and_evaluate(data_path=None, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED) -> None:
    """Full leak-free execution: validate -> split -> CV search -> fit -> test -> save."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Acquisition, Validation, and Cleaning
    print("[1/7] Loading and validating dataset...")
    df, info = load_dataset(data_path)
    print(f"      Source file          : {info['file']}")
    print(f"      Physical file lines  : {info['physical_lines']:,}")
    print(f"      Rows parsed          : {info['rows_read']:,}")
    print(f"      Malformed lines      : {info['bad_lines_skipped']}")
    print(f"      Missing rows removed : {info['missing_removed']}")
    print(f"      Invalid label rows   : {info['invalid_labels_removed']}")
    print(f"      Empty text rows      : {info['empty_text_removed']}")
    print(f"      Conflicting labels   : {info['conflicting_labels_detected']} detected ({info['conflicting_rows_removed']} rows removed)")
    print(f"      Duplicates removed   : {info['duplicates_removed']:,}")
    print(f"      Cleaned dataset      : {info['rows_final']:,} messages")
    print(f"      Class distribution   : Ham = {info['ham_count']:,} ({info['ham_pct']:.2f}%), Spam = {info['spam_count']:,} ({info['spam_pct']:.2f}%)")

    plot_class_distribution(df["label"], OUTPUT_DIR / "class_distribution.png")

    # 2. Stratified Train/Test Split (80/20)
    print("[2/7] Splitting dataset into stratified train and test partitions...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    print(f"      Training partition   : {len(X_train_text):,} samples (Ham: {(y_train == 0).sum():,}, Spam: {(y_train == 1).sum():,})")
    print(f"      Held-out test set    : {len(X_test_text):,} samples (Ham: {(y_test == 0).sum():,}, Spam: {(y_test == 1).sum():,})")
    print("      Note: Held-out test set is STRICTLY ISOLATED during model tuning.")

    # 3. Stratified 5-Fold Cross-Validation on Training Data Only
    print("[3/7] Running 5-Fold Stratified Cross-Validation on training data across GaussianNB configurations...")
    df_cv = run_cross_validation_experiments(X_train_text, y_train, OUTPUT_DIR)
    print("      Saved CV results to: outputs/cv_experiments.csv")

    # 4. Fit Final Selected GaussianNB Model on Full Training Set
    print(f"[4/7] Fitting selected GaussianNB model on full training set (var_smoothing={DEFAULT_VAR_SMOOTHING})...")
    vectorizer, X_train_dense = fit_vectorizer(
        X_train_text, max_features=MAX_FEATURES, min_df=MIN_DOC_FREQ, stop_words="english"
    )
    final_model = train_model(X_train_dense, y_train, var_smoothing=DEFAULT_VAR_SMOOTHING)

    # Also build full end-to-end Pipeline
    final_pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("to_dense", make_dense_transformer()),
        ("classifier", final_model),
    ])

    # Also fit unregularized baseline model for rigorous before/after comparison
    baseline_model = train_model(X_train_dense, y_train, var_smoothing=1e-9)

    # 5. Held-Out Test Set Evaluation
    print("[5/7] Evaluating final model and baseline on held-out test set...")
    X_test_dense = transform_texts(vectorizer, X_test_text)

    y_pred_final = final_model.predict(X_test_dense)
    y_pred_baseline = baseline_model.predict(X_test_dense)

    final_metrics = compute_metrics(y_test, y_pred_final)
    baseline_metrics = compute_metrics(y_test, y_pred_baseline)

    # Save test comparison table
    test_comp_df = pd.DataFrame([
        {
            "Model": "Baseline GaussianNB (Unregularized)",
            "var_smoothing": 1e-9,
            "Accuracy": round(baseline_metrics["accuracy"], 4),
            "Precision": round(baseline_metrics["precision"], 4),
            "Recall": round(baseline_metrics["recall"], 4),
            "F1-Score": round(baseline_metrics["f1"], 4),
            "FP_Rate": round(baseline_metrics["false_positive_rate"], 4),
            "FP_Count": baseline_metrics["fp"],
            "FN_Count": baseline_metrics["fn"],
        },
        {
            "Model": "Final Regularized GaussianNB",
            "var_smoothing": DEFAULT_VAR_SMOOTHING,
            "Accuracy": round(final_metrics["accuracy"], 4),
            "Precision": round(final_metrics["precision"], 4),
            "Recall": round(final_metrics["recall"], 4),
            "F1-Score": round(final_metrics["f1"], 4),
            "FP_Rate": round(final_metrics["false_positive_rate"], 4),
            "FP_Count": final_metrics["fp"],
            "FN_Count": final_metrics["fn"],
        },
    ])
    test_comp_df.to_csv(OUTPUT_DIR / "test_comparison.csv", index=False)

    # 6. Qualitative and Quantitative Error Analysis
    print("[6/7] Extracting False Positives and False Negatives for error analysis...")
    df_errors = extract_errors(y_test, y_pred_final, X_test_text)
    df_errors.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
    n_fp = (df_errors["Error_Type"] == "False Positive").sum()
    n_fn = (df_errors["Error_Type"] == "False Negative").sum()
    print(f"      Total test errors: {len(df_errors)}  (False Positives: {n_fp}, False Negatives: {n_fn})")

    # 7. Persistence and Formatting
    print("[7/7] Persisting artifacts and reports...")
    run_info = {
        "file": info["file"],
        "total": len(df),
        "ham": info["ham_count"],
        "spam": info["spam_count"],
        "train_size": len(X_train_text),
        "test_size": len(X_test_text),
        "vocab_size": len(vectorizer.vocabulary_),
        "seed": seed,
        "var_smoothing": DEFAULT_VAR_SMOOTHING,
    }
    metrics_text = format_metrics_text(final_metrics, run_info, baseline_metrics=baseline_metrics)
    save_metrics_text(metrics_text, OUTPUT_DIR / "metrics.txt")
    plot_confusion_matrix(final_metrics["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png")
    save_artifacts(final_model, vectorizer, pipeline=final_pipeline, output_dir=OUTPUT_DIR)

    # Console Summary
    print("\n" + metrics_text)
    print("\n" + "=" * 80)
    print("5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY (TRAINING DATA ONLY)")
    print("=" * 80)
    summary_cols = ["Experiment", "CV_Accuracy_Mean", "CV_Precision_Mean", "CV_Recall_Mean", "CV_F1_Mean"]
    print(df_cv[summary_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET BEFORE/AFTER COMPARISON")
    print("=" * 80)
    print(test_comp_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("QUALITATIVE ERROR SAMPLES (From outputs/error_analysis.csv)")
    print("=" * 80)
    if not df_errors.empty:
        for err_type in ["False Positive", "False Negative"]:
            subset = df_errors[df_errors["Error_Type"] == err_type]
            print(f"\n--- Sample {err_type}s ({len(subset)} total) ---")
            for idx, row in subset.head(3).iterrows():
                preview = row["Message"][:90] + ("..." if len(row["Message"]) > 90 else "")
                print(f"  * [{row['Actual']} -> {row['Predicted']}] {preview}")

    print("\nAll artifacts successfully saved to 'outputs/':")
    print("  - metrics.txt                : Complete metrics report and discussion")
    print("  - cv_experiments.csv         : 5-Fold CV results on training partition")
    print("  - test_comparison.csv        : Test set before/after comparison table")
    print("  - error_analysis.csv         : Full list of FP and FN misclassifications")
    print("  - confusion_matrix.png       : Annotated confusion matrix heatmap")
    print("  - class_distribution.png     : Class distribution bar chart")
    print("  - gaussian_nb_model.joblib   : Trained GaussianNB classifier")
    print("  - count_vectorizer.joblib    : Fitted CountVectorizer")
    print("  - spam_classifier_pipeline.joblib : End-to-end Pipeline")
    print("\nClassify interactive messages with:  python main.py --predict")


def main():
    parser = argparse.ArgumentParser(description="Spam classifier using Gaussian Naive Bayes.")
    parser.add_argument("--data", help="Path to dataset file (default: auto-detect in data/)")
    parser.add_argument("--predict", action="store_true", help="Classify messages interactively using saved model")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Held-out test fraction (default: 0.20)")
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
        print(f"\nExecution Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
