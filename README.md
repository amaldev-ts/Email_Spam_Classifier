# Email/SMS Spam Classification Using Gaussian Naïve Bayes

A production-grade, reproducible machine learning project that classifies messages as **Spam** or **Not Spam (Ham)** using scikit-learn's **Gaussian Naïve Bayes (`GaussianNB`)** algorithm with variance smoothing regularization.

> **Dataset Disclosure:** The dataset used is the benchmark **UCI SMS Spam Collection** (Almeida et al.), containing **SMS text messages, not emails**. While the vectorization and classification pipeline is structurally identical for emails, the text features reflect SMS messaging characteristics (shorter text, phone numbers, and urgent promotional language).

---

## 1. Project Overview & Performance Highlights

This project addresses the classic "zero-variance collapse" problem when using `GaussianNB` on discrete bag-of-words text. By calibrating variance smoothing (`var_smoothing = 0.05`) and applying leak-free text preprocessing, **all four target metrics improved substantially** on the fixed held-out test set ($N = 1,032$):

| Metric | Original Baseline (`var_smoothing=1e-9`) | Improved Model (`var_smoothing=0.05`) | Improvement ($\Delta$) |
|---|:---:|:---:|:---:|
| **Accuracy** | 82.95% | **98.45%** | **+15.50%** |
| **Spam Precision** | 41.49% | **95.16%** | **+53.67%** |
| **Spam Recall** | 91.41% | **92.19%** | **+0.78%** |
| **Spam F1-score** | 57.07% | **93.65%** | **+36.58%** |
| **False Positive Rate** | 18.25% (165 FP) | **0.66% (6 FP)** | **-17.59% (-96.4% FP)** |

### Confusion Matrix Comparison (Held-Out Test Set: 904 Ham, 128 Spam)

```
BASELINE MODEL (var_smoothing = 1e-9):
                     Predicted Ham    Predicted Spam
Actual Ham (904) :        739              165       (165 False Positives! Precision = 41.49%)
Actual Spam (128):         11              117       (11 False Negatives, Recall = 91.41%)

IMPROVED MODEL (var_smoothing = 0.05):
                     Predicted Ham    Predicted Spam
Actual Ham (904) :        898                6       (Only 6 False Positives! Precision = 95.16%)
Actual Spam (128):         10              118       (10 False Negatives, Recall = 92.19%)
```

---

## 2. Why the Baseline Failed and How It Was Fixed

### The Problem
In standard `CountVectorizer`, matrices are sparse with mostly zeros. For rare words occurring 0 or 1 time in legitimate ham messages, the sample variance $\sigma^2$ collapses near zero. Under default `var_smoothing = 1e-9`, when a new test ham message contains a single word not heavily seen in ham, GaussianNB computes a likelihood under $10^{-500}$ ($z > 50$ standard deviations), incorrectly flagging legitimate messages as spam with astronomical overconfidence.

### The Solution: Variance Smoothing Regularization
In `GaussianNB`, `var_smoothing` adds a fraction of the maximum variance to all feature variances:
$$\sigma_{\text{regularized}}^2 = \sigma^2 + \text{var\_smoothing} \times \max(\sigma^2)$$
Setting `var_smoothing = 0.05` ensures feature variances have a stable lower bound, preventing any single rare word from dominating the joint likelihood. This drove precision up by **+53.67%** while maintaining over **92% recall**.

---

## 3. Systematic Model Comparison

Evaluated on the exact same held-out test set ($N = 1,032$, seed 42, 80/20 stratified split). Saved automatically as `outputs/model_comparison.csv`:

| Experiment | Feature Extraction | Vocabulary Settings | `var_smoothing` | Accuracy | Precision | Recall | F1-Score | False Positive Rate |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| 1. Baseline (CountVec, default vs) | CountVectorizer | 2000 words, min_df=2, unigrams, stop_words | $1.0 \times 10^{-9}$ | 82.95% | 41.49% | 91.41% | 57.07% | 18.25% (165 FP) |
| 2. CountVec + Tuned vs | CountVectorizer | 2000 words, min_df=2, unigrams, stop_words | $8.0 \times 10^{-2}$ | 98.35% | 96.64% | 89.84% | 93.12% | 0.44% (4 FP) |
| 3. TF-IDF + Default vs | TfidfVectorizer | 2000 words, min_df=2, unigrams, stop_words | $1.0 \times 10^{-9}$ | 82.56% | 40.44% | 85.94% | 55.00% | 17.92% (162 FP) |
| 4. TF-IDF + Tuned vs | TfidfVectorizer | 2000 words, min_df=2, unigrams, stop_words | $2.0 \times 10^{0}$ | 96.71% | 84.56% | 89.84% | 87.12% | 2.32% (21 FP) |
| 5. CountVec + Bigrams (1, 2) | CountVec (1, 2) | 2000 words, min_df=2, bigrams, stop_words | $8.0 \times 10^{-2}$ | 97.77% | 99.07% | 82.81% | 90.21% | 0.11% (1 FP) |
| 6. CountVec + No Stop Words | CountVectorizer | 2000 words, min_df=2, unigrams, no stop_words | $8.0 \times 10^{-2}$ | 95.54% | 96.59% | 66.41% | 78.70% | 0.33% (3 FP) |
| 7. CountVec + Compact Vocab 1500 | CountVectorizer | 1500 words, min_df=2, unigrams, stop_words | $8.0 \times 10^{-2}$ | 98.35% | 96.64% | 89.84% | 93.12% | 0.44% (4 FP) |
| **8. Final Model (CountVec + Balanced vs)** | **CountVectorizer** | **2000 words, min_df=2, unigrams, stop_words** | **$5.0 \times 10^{-2}$** | **98.45%** | **95.16%** | **92.19%** | **93.65%** | **0.66% (6 FP)** |

---

## 4. Leak-Free Architecture & Design

- **Strict Data Isolation:** Stratified 80/20 train/test split (`seed = 42`) is executed before vectorizer fitting.
- **No Test Leakage:** The `CountVectorizer` is fitted exclusively on training text (`X_train_text`). The test set is only transformed with the fitted vocabulary.
- **Cross-Validation on Train Only:** Hyperparameter tuning (`var_smoothing`, vocabulary size, n-grams) was selected using 5-fold Stratified Cross-Validation on the training partition.
- **Dense Array Optimization:** Dense float32 matrices (`np.float32`) keep memory usage under ~33 MB for 4,128 messages across 2,000 features.

---

## 5. Folder Structure

```
Email_Spam_Classifier/
├── main.py              # End-to-end training, evaluation, comparison & predict CLI
├── data_loader.py       # Dataset discovery, loading, duplicate removal, validation
├── preprocessing.py     # Text cleaning, CountVectorizer fitting & dense float32 conversion
├── model.py             # GaussianNB model training, var_smoothing, serialization
├── evaluate.py          # Metric computation, classification report, plots, formatting
├── predict.py           # Single-message and interactive prediction CLI
├── requirements.txt     # Python dependencies
├── README.md            # Complete project documentation
├── report.md            # Detailed academic and engineering project report
├── data/
│   └── SMSSpamCollection # UCI SMS Spam Collection dataset
└── outputs/
    ├── metrics.txt              # Detailed test set metrics & baseline deltas
    ├── model_comparison.csv     # 8-experiment benchmark comparison table
    ├── confusion_matrix.png     # Heatmap with counts and percentages
    ├── class_distribution.png   # Class distribution bar chart
    ├── gaussian_nb_model.joblib # Trained GaussianNB model artifact
    └── count_vectorizer.joblib  # Fitted CountVectorizer artifact
```

---

## 6. Setup & Execution

### Installation
Ensure Python 3.9+ is installed:
```powershell
cd Email_Spam_Classifier
pip install -r requirements.txt
```

### 1. Train and Evaluate Pipeline
```powershell
python main.py
```
This loads the data, cleans duplicates, splits 80/20, fits vectorizer on training data only, trains `GaussianNB(var_smoothing=0.05)`, evaluates on the test set, runs the comparison benchmark, and regenerates all artifacts in `outputs/`.

### 2. Classify Custom Messages
Single message prediction:
```powershell
python predict.py --text "WINNER! You have won a £1000 prize. Call 09061701461 to claim now."
```

Interactive terminal prediction:
```powershell
python main.py --predict
```

Example CLI Output:
```
==================================================
  PREDICTED CLASS  : SPAM
  Confidence Level : High Confidence
--------------------------------------------------
  Ham Probability  : < 0.01%    (p = 0.0000)
  Spam Probability : > 99.99%   (p = 1.0000)
==================================================
  * Note: Probabilities represent Gaussian Naive Bayes
    posterior likelihoods under feature independence,
    not empirical ground-truth certainty.
==================================================
```

---

## 7. Limitations of GaussianNB for Text

1. **Normality Assumption Violation:** Word counts are non-negative, discrete, and zero-inflated, violating the normal bell-curve assumption.
2. **Dense Memory Footprint:** GaussianNB does not accept sparse matrices, necessitating `.toarray()`. For corpora with tens of thousands of features, memory requirements scale quickly.
3. **Bag-of-Words Independence:** Bag-of-words ignores word order, context, negation ("not bad"), and syntax.
