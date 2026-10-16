"""
predict.py
----------
Classify a message as SPAM or HAM using the SAVED model and SAVED vectorizer.
The vectorizer is only loaded and used to transform - it is never re-fitted.

Usage (from the project folder):
    python predict.py                       -> interactive prompt
    python predict.py --text "Win a free prize now"
(You can also run:  python main.py --predict)
"""

import argparse
import sys

from model import OUTPUT_DIR, load_artifacts, predict_labels, predict_probabilities
from preprocessing import clean_text, transform_texts


def format_probability(p):
    """
    Format a probability clearly without misleading certainty.
    Avoids printing raw 1.0000 or 0.0000 as proof of absolute certainty.
    """
    if p >= 0.99995:
        return "> 99.99%"
    elif p <= 0.00005:
        return "< 0.01%"
    else:
        return f"{p * 100:.2f}%"


def predict_message(text, model, vectorizer):
    """
    Classify one message.
    Returns a dict: {'label': 'SPAM' or 'HAM', 'p_ham': float, 'p_spam': float}
    Raises ValueError if the message is empty.
    """
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Message is empty. Please type some text.")

    features = transform_texts(vectorizer, [cleaned])        # same vectorizer as training
    label = int(predict_labels(model, features)[0])
    p_ham, p_spam = predict_probabilities(model, features)[0]
    return {
        "label": "SPAM" if label == 1 else "HAM",
        "p_ham": float(p_ham),
        "p_spam": float(p_spam),
    }


def print_result(result):
    """Display a prediction clearly with calibrated probability presentation."""
    label = result["label"]
    p_ham = result["p_ham"]
    p_spam = result["p_spam"]

    # Determine confidence level based on margin
    max_p = max(p_ham, p_spam)
    if max_p >= 0.90:
        confidence = "High Confidence"
    elif max_p >= 0.70:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence (Borderline)"

    print("\n" + "=" * 50)
    print(f"  PREDICTED CLASS  : {label}")
    print(f"  Confidence Level : {confidence}")
    print("-" * 50)
    print(f"  Ham Probability  : {format_probability(p_ham):<10} (p = {p_ham:.4f})")
    print(f"  Spam Probability : {format_probability(p_spam):<10} (p = {p_spam:.4f})")
    print("=" * 50)
    print("  * Note: Probabilities represent Gaussian Naive Bayes")
    print("    posterior likelihoods under feature independence,")
    print("    not empirical ground-truth certainty.")
    print("=" * 50 + "\n")


def run_interactive(output_dir=OUTPUT_DIR):
    """Keep asking for messages until the user types 'quit'."""
    model, vectorizer = load_artifacts(output_dir)
    print("Spam classifier ready. Type a message and press Enter (type 'quit' to exit).")
    while True:
        try:
            text = input("Message> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit", "q"}:
            break
        try:
            print_result(predict_message(text, model, vectorizer))
        except ValueError as error:
            print(f"  {error}")


def main():
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("--text", help="Message to classify (skips the interactive prompt)")
    args = parser.parse_args()
    try:
        if args.text is not None:
            model, vectorizer = load_artifacts()
            print_result(predict_message(args.text, model, vectorizer))
        else:
            run_interactive()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
