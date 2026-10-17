"""
main.py
-------
End-to-end training, 5-fold cross-validation, test evaluation, and artifact generation
for Gaussian Naïve Bayes email classification on the UCI Spambase dataset.

Strict Machine Learning Workflow:
  1. Load, audit, and validate the UCI Spambase dataset (check conflicts, duplicates, types).
  2. Stratified 80/20 train/test split (seed=42). Held-out test set is strictly isolated.
  3. Run 5-fold Stratified Cross-Validation ON TRAINING DATA ONLY across systematic
     GaussianNB configurations (smoothing values, log1p transformation, scaling, feature subsets).
  4. Refit the selected optimal GaussianNB pipeline on the full training set.
  5. Evaluate the final model ONCE on the held-out test set; compare against the raw baseline.
  6. Extract False Positives and False Negatives for qualitative diagnostic error analysis.
  7. Persist model artifacts, metric summaries, CSV tables, and visualization plots.

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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data_loader import FEATURE_NAMES, load_dataset
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
    build_spambase_pipeline,
    save_artifacts,
    train_model,
)
from predict import run_interactive
from preprocessing import make_log1p_transformer

RANDOM_SEED = 42
TEST_SIZE = 0.20


def run_cross_validation_experiments(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """
    Execute 5-Fold Stratified Cross-Validation strictly on the training partition.
    Evaluates variance smoothing values, log1p transformation, scaling, and feature subsets.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["accuracy", "precision", "recall", "f1"]

    # First 48 features are word frequencies
    word_feature_names = FEATURE_NAMES[:48]

    experiment_definitions = [
        {
            "id": 1,
            "name": "1. Baseline (Raw, vs=1e-9)",
            "category": "Baseline",
            "pipeline": Pipeline([("classifier", GaussianNB(var_smoothing=1e-9))]),
            "features": "all",
            "description": "Default unregularized GaussianNB on raw features",
        },
        {
            "id": 2,
            "name": "2. Raw vs=1e-5",
            "category": "Raw Variance Sweep",
            "pipeline": Pipeline([("classifier", GaussianNB(var_smoothing=1e-5))]),
            "features": "all",
            "description": "Raw features with minimal smoothing",
        },
        {
            "id": 3,
            "name": "3. Raw vs=1e-2",
            "category": "Raw Variance Sweep",
            "pipeline": Pipeline([("classifier", GaussianNB(var_smoothing=1e-2))]),
            "features": "all",
            "description": "Raw features with moderate smoothing",
        },
        {
            "id": 4,
            "name": "4. Raw vs=0.05",
            "category": "Raw Variance Sweep",
            "pipeline": Pipeline([("classifier", GaussianNB(var_smoothing=0.05))]),
            "features": "all",
            "description": "Raw features with vs=0.05",
        },
        {
            "id": 5,
            "name": "5. Log1p vs=1e-9",
            "category": "Log1p Transformation",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=1e-9))]),
            "features": "all",
            "description": "Log1p transform with default smoothing",
        },
        {
            "id": 6,
            "name": "6. Log1p vs=1e-4",
            "category": "Log1p Transformation",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=1e-4))]),
            "features": "all",
            "description": "Log1p transform with vs=1e-4",
        },
        {
            "id": 7,
            "name": "7. Log1p vs=1e-3",
            "category": "Log1p Transformation",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=1e-3))]),
            "features": "all",
            "description": "Log1p transform with vs=1e-3",
        },
        {
            "id": 8,
            "name": "8. Log1p vs=1e-2",
            "category": "Log1p Transformation",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=1e-2))]),
            "features": "all",
            "description": "Log1p transform with vs=1e-2",
        },
        {
            "id": 9,
            "name": "9. Log1p vs=0.05 (Selected)",
            "category": "Log1p Transformation",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=0.05))]),
            "features": "all",
            "description": "Optimal Log1p + balanced variance regularization",
        },
        {
            "id": 10,
            "name": "10. Log1p vs=0.10",
            "category": "Log1p Transformation",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=0.10))]),
            "features": "all",
            "description": "Log1p with higher smoothing",
        },
        {
            "id": 11,
            "name": "11. Log1p + StandardScaler",
            "category": "Scaling & Standardization",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("scaler", StandardScaler()), ("classifier", GaussianNB(var_smoothing=0.01))]),
            "features": "all",
            "description": "Log1p followed by unit variance scaling",
        },
        {
            "id": 12,
            "name": "12. Word Features Only (48)",
            "category": "Feature Subset",
            "pipeline": Pipeline([("log", make_log1p_transformer()), ("classifier", GaussianNB(var_smoothing=0.05))]),
            "features": "words_only",
            "description": "Exclude punctuation and capital run-lengths",
        },
    ]

    results = []
    for exp in experiment_definitions:
        X_curr = X_train[word_feature_names] if exp["features"] == "words_only" else X_train
        pipe = exp["pipeline"]

        cv_scores = cross_validate(
            pipe,
            X_curr,
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

    # 1. Dataset Loading, Auditing, and Validation
    print("[1/7] Loading and auditing UCI Spambase dataset...")
    df, info = load_dataset(data_path)
    print(f"      Source file          : {info['file']}")
    print(f"      Physical file lines  : {info['physical_lines']:,}")
    print(f"      Rows parsed          : {info['rows_read']:,}")
    print(f"      Malformed lines      : {info['bad_lines_skipped']}")
    print(f"      Missing rows removed : {info['missing_removed']}")
    print(f"      Invalid label rows   : {info['invalid_labels_removed']}")
    print(f"      Conflicting groups   : {info['conflicting_groups_detected']} detected ({info['conflicting_rows_removed']} rows removed)")
    print(f"      Duplicates removed   : {info['duplicates_removed']:,}")
    print(f"      Cleaned dataset      : {info['rows_final']:,} email instances")
    print(f"      Class distribution   : Not Spam = {info['not_spam_count']:,} ({info['not_spam_pct']:.2f}%), Spam = {info['spam_count']:,} ({info['spam_pct']:.2f}%)")

    plot_class_distribution(df["label"], OUTPUT_DIR / "class_distribution.png")

    # 2. Stratified Train/Test Split (80/20)
    print("[2/7] Splitting dataset into stratified train and test partitions...")
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    print(f"      Training partition   : {len(X_train):,} samples (Not Spam: {(y_train == 0).sum():,}, Spam: {(y_train == 1).sum():,})")
    print(f"      Held-out test set    : {len(X_test):,} samples (Not Spam: {(y_test == 0).sum():,}, Spam: {(y_test == 1).sum():,})")
    print("      Note: Held-out test set is STRICTLY ISOLATED during model tuning.")

    # 3. Stratified 5-Fold Cross-Validation on Training Data Only
    print("[3/7] Running 5-Fold Stratified Cross-Validation on training data across GaussianNB configurations...")
    df_cv = run_cross_validation_experiments(X_train, y_train, OUTPUT_DIR)
    print("      Saved CV results to: outputs/cv_experiments.csv")

    # 4. Fit Final Selected GaussianNB Model on Full Training Set
    print(f"[4/7] Fitting selected GaussianNB model on full training set (Log1p + var_smoothing={DEFAULT_VAR_SMOOTHING})...")
    final_pipeline = train_model(X_train, y_train, use_log=True, var_smoothing=DEFAULT_VAR_SMOOTHING)

    # Also fit raw baseline model for direct before/after contrast
    baseline_pipeline = train_model(X_train, y_train, use_log=False, var_smoothing=1e-9)

    # 5. Held-Out Test Set Evaluation
    print("[5/7] Evaluating final model and baseline on held-out test set...")
    y_pred_final = final_pipeline.predict(X_test)
    y_pred_baseline = baseline_pipeline.predict(X_test)

    final_metrics = compute_metrics(y_test, y_pred_final)
    baseline_metrics = compute_metrics(y_test, y_pred_baseline)

    test_comp_df = pd.DataFrame([
        {
            "Model": "Baseline GaussianNB (Raw, vs=1e-9)",
            "Transform": "None (Raw)",
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
            "Model": "Final Improved GaussianNB (Log1p + vs=0.05)",
            "Transform": "Log1p",
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
    df_errors = extract_errors(y_test, y_pred_final, X_test, feature_names=FEATURE_NAMES)
    df_errors.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
    n_fp = (df_errors["Error_Type"] == "False Positive").sum()
    n_fn = (df_errors["Error_Type"] == "False Negative").sum()
    print(f"      Total test errors: {len(df_errors)}  (False Positives: {n_fp}, False Negatives: {n_fn})")

    # 7. Persistence and Reports
    print("[7/7] Persisting artifacts and reports...")
    run_info = {
        "file": info["file"],
        "total": len(df),
        "not_spam": info["not_spam_count"],
        "spam": info["spam_count"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(FEATURE_NAMES),
        "seed": seed,
        "var_smoothing": DEFAULT_VAR_SMOOTHING,
    }
    metrics_text = format_metrics_text(final_metrics, run_info, baseline_metrics=baseline_metrics)
    save_metrics_text(metrics_text, OUTPUT_DIR / "metrics.txt")
    plot_confusion_matrix(final_metrics["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png")
    save_artifacts(final_pipeline, output_dir=OUTPUT_DIR)

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
    print("SAMPLE TEST ERRORS (From outputs/error_analysis.csv)")
    print("=" * 80)
    if not df_errors.empty:
        for err_type in ["False Positive", "False Negative"]:
            subset = df_errors[df_errors["Error_Type"] == err_type]
            print(f"\n--- Sample {err_type}s ({len(subset)} total) ---")
            for idx, row in subset.head(3).iterrows():
                print(f"  * [Index {row['Sample_Index']}] Top features: {row['Top_Indicative_Features']}")

    print("\nAll artifacts successfully saved to 'outputs/':")
    print("  - metrics.txt                : Complete metrics report and discussion")
    print("  - cv_experiments.csv         : 5-Fold CV results on training partition")
    print("  - test_comparison.csv        : Test set before/after comparison table")
    print("  - error_analysis.csv         : Full list of FP and FN misclassifications")
    print("  - confusion_matrix.png       : Annotated confusion matrix heatmap")
    print("  - class_distribution.png     : Class distribution bar chart")
    print("  - spambase_gaussian_nb_pipeline.joblib : Full fitted Pipeline")
    print("  - gaussian_nb_model.joblib   : Trained GaussianNB classifier")
    print("\nClassify emails with:  python predict.py --text \"...\"")


def main():
    parser = argparse.ArgumentParser(description="UCI Spambase classifier using Gaussian Naive Bayes.")
    parser.add_argument("--data", help="Path to spambase.data file (default: auto-detect in data/)")
    parser.add_argument("--predict", action="store_true", help="Classify messages interactively")
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
