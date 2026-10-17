"""
predict.py
----------
Classify an email message or feature vector as SPAM or NOT SPAM
using the trained Gaussian Naïve Bayes pipeline on UCI Spambase.

Usage:
    1. Classify raw email text:
       python predict.py --text "Claim your free money prize from our internet business now!"

    2. Classify a row of 57 numeric features:
       python predict.py --features "0,0.64,0.64,0,...,61,278"

    3. Interactive mode:
       python predict.py
"""

import argparse
import sys
from typing import Dict, Union

import numpy as np
import pandas as pd

from data_loader import FEATURE_NAMES
from model import load_artifacts, predict_labels, predict_probabilities
from preprocessing import clean_text, extract_spambase_features


def format_probability(p: float) -> str:
    """Format probability without misleading extreme certainty."""
    if p >= 0.99995:
        return "> 99.99%"
    elif p <= 0.00005:
        return "< 0.01%"
    else:
        return f"{p * 100:.2f}%"


def predict_from_features(features: np.ndarray, pipeline) -> Dict:
    """Run prediction on a 57-dimensional feature vector."""
    if features.ndim == 1:
        features = features.reshape(1, -1)

    df_feats = pd.DataFrame(features, columns=FEATURE_NAMES)
    label_idx = int(predict_labels(pipeline, df_feats)[0])
    p_not_spam, p_spam = predict_probabilities(pipeline, df_feats)[0]

    return {
        "label": "SPAM" if label_idx == 1 else "NOT SPAM",
        "p_not_spam": float(p_not_spam),
        "p_spam": float(p_spam),
        "features": features[0],
    }


def predict_from_text(raw_text: str, pipeline) -> Dict:
    """Extract Spambase features from raw email text and predict."""
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise ValueError("Email text is empty. Please provide some text to classify.")

    features = extract_spambase_features(cleaned)
    result = predict_from_features(features, pipeline)
    result["raw_text"] = cleaned
    return result


def print_result(result: Dict) -> None:
    """Display prediction outcome and probabilistic confidence."""
    label = result["label"]
    p_not_spam = result["p_not_spam"]
    p_spam = result["p_spam"]

    max_p = max(p_not_spam, p_spam)
    if max_p >= 0.90:
        confidence = "High Confidence"
    elif max_p >= 0.70:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence (Borderline)"

    print("\n" + "=" * 52)
    print(f"  PREDICTED CLASS  : {label}")
    print(f"  Confidence Level : {confidence}")
    print("-" * 52)
    print(f"  Not Spam Probability : {format_probability(p_not_spam):<10} (p = {p_not_spam:.4f})")
    print(f"  Spam Probability     : {format_probability(p_spam):<10} (p = {p_spam:.4f})")
    print("=" * 52)

    # Show indicative features if non-zero
    feats = result.get("features")
    if feats is not None:
        s = pd.Series(feats, index=FEATURE_NAMES)
        top = s[s > 0].sort_values(ascending=False).head(3)
        if not top.empty:
            print("  Top Detected Spambase Signals:")
            for k, v in top.items():
                print(f"    * {k:<28}: {v:.2f}")
            print("-" * 52)

    print("  * Note: Probabilities represent Gaussian Naive Bayes")
    print("    posterior likelihoods under feature independence.")
    print("=" * 52 + "\n")


def run_interactive():
    """Interactive loop for email classification."""
    pipeline = load_artifacts()
    print("Spambase email classifier ready.")
    print("Enter raw email text to classify (type 'quit' to exit):")
    while True:
        try:
            text = input("Email> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit", "q"}:
            break
        try:
            res = predict_from_text(text, pipeline)
            print_result(res)
        except ValueError as err:
            print(f"  {err}")


def main():
    parser = argparse.ArgumentParser(description="Predict spam/not spam using UCI Spambase GaussianNB.")
    parser.add_argument("--text", help="Raw email text to classify")
    parser.add_argument("--features", help="Comma-separated 57 numeric values")
    args = parser.parse_args()

    try:
        pipeline = load_artifacts()

        if args.text is not None:
            res = predict_from_text(args.text, pipeline)
            print_result(res)
        elif args.features is not None:
            vals = [float(x.strip()) for x in args.features.split(",") if x.strip()]
            if len(vals) != 57:
                print(f"Error: Expected 57 comma-separated numbers, received {len(vals)}.")
                sys.exit(1)
            res = predict_from_features(np.array(vals, dtype=np.float64), pipeline)
            print_result(res)
        else:
            run_interactive()
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
