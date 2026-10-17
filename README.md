# Email/SMS Spam Classification Using Gaussian Naïve Bayes

A production-grade, reproducible machine learning system that classifies messages as **Spam** or **Not Spam (Ham)** strictly implementing scikit-learn's **Gaussian Naïve Bayes (`GaussianNB`)** algorithm with variance smoothing regularization and leak-free 5-fold cross-validation.

> **Dataset Scope Disclosure:** The dataset used is the benchmark **UCI SMS Spam Collection** (Almeida et al.), consisting of **SMS text messages, not emails**. While the vectorization and classification pipeline is structurally identical for email body text, SMS messages have distinct characteristics (shorter lengths, mobile colloquialisms, phone numbers, and urgent promotional language) and lack email transport headers, SPF/DKIM authentication, MIME attachment types, and domain reputations.

---

## 1. Project Overview & Performance Highlights

This project addresses the classic "zero-variance collapse" problem that occurs when applying continuous `GaussianNB` models to discrete, sparse bag-of-words text. By calibrating variance smoothing (`var_smoothing = 0.05`), selecting parameters via **5-Fold Stratified Cross-Validation on the training partition only**, and evaluating on a strictly isolated held-out test partition ($N = 1,032$), **all target metrics improved substantially**:

| Metric | Original Baseline (`var_smoothing=1e-9`) | Regularized Model (`var_smoothing=0.05`) | Absolute Improvement ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 82.95% | **98.45%** | **+15.50%** | +18.69% |
| **Spam Precision** | 41.49% | **95.16%** | **+53.67%** | +129.36% |
| **Spam Recall** | 91.41% | **92.19%** | **+0.78%** | +0.85% |
| **Spam F1-Score** | 57.07% | **93.65%** | **+36.58%** | +64.10% |
| **False Positive Rate** | 18.25% (165 FP) | **0.66% (6 FP)** | **-17.59%** | **-96.36% (FP dropped)** |

### Confusion Matrix Comparison (Held-Out Test Set: 904 Ham, 128 Spam)

```
BASELINE MODEL (var_smoothing = 1e-9):
                     Predicted Ham    Predicted Spam
Actual Ham (904) :        739              165       (165 False Positives! Precision = 41.49%)
Actual Spam (128):         11              117       (11 False Negatives, Recall = 91.41%)

FINAL REGULARIZED MODEL (var_smoothing = 0.05):
                     Predicted Ham    Predicted Spam
Actual Ham (904) :        898                6       (Only 6 False Positives! Precision = 95.16%)
Actual Spam (128):         10              118       (10 False Negatives, Recall = 92.19%)
```

---

## 2. Theoretical Root Cause & Mathematical Solution

### The Problem: Zero-Variance Collapse
In bag-of-words text matrices, most words appear in only a tiny fraction of documents. For a word that occurs rarely or never in legitimate ham messages, its sample variance $\sigma_{\text{ham}, i}^2$ collapses near zero. Under default `var_smoothing = 1e-9`, when a new test ham message contains that word ($x_i = 1$), the Gaussian standardized z-score:
$$z = \frac{1 - \mu_{\text{ham}, i}}{\sigma_{\text{ham}, i}}$$
explodes ($z > 50$), driving the class likelihood $P(x_i \mid \text{ham})$ to under $10^{-500}$. Even if all other words are standard legitimate terms, this single feature's extreme likelihood penalty forces GaussianNB to falsely classify the message as spam (producing 165 False Positives and 41.49% precision in the baseline).

### The Solution: Variance Smoothing Regularization
In `GaussianNB`, `var_smoothing` regularizes feature variances by adding a fraction $\epsilon$ of the maximum feature variance across all features:
$$\sigma_{C,i,\text{smoothed}}^2 = \sigma_{C,i}^2 + \epsilon \cdot \max_{j}(\sigma_j^2)$$
Setting `var_smoothing = 0.05` establishes an additive variance floor that prevents single-word variance collapse, driving precision up by **+53.67%** while maintaining over **92% recall**.

---

## 3. Strict, Leak-Free Evaluation Methodology

```
Cleaned Dataset (5,160 messages: 4,518 Ham, 642 Spam)
                      │
                      ▼
        Stratified Split (80/20, seed=42)
         ┌────────────┴────────────┐
         ▼                         ▼
Training Partition (4,128)      Held-Out Test Set (1,032)
(3,614 Ham, 514 Spam)           (904 Ham, 128 Spam)
         │                                 │
         ▼                                 │ (Strictly isolated;
5-Fold Stratified CV                       │  never seen during tuning)
(Tune smoothing, vocab, min_df, n-grams)   │
         │                                 │
         ▼                                 ▼
Fit Final Model on Full Train ────► Single Final Test Evaluation
                                   (Produces test metrics & error analysis)
```

1. **Dataset Validation:** Raw file is parsed with malformed line tracking, conflict verification (0 conflicting labels detected), and deduplication (414 duplicates removed).
2. **Data Isolation:** Stratified 80/20 train/test split (`seed=42`) isolates 1,032 test samples.
3. **Training CV Only:** All model parameter exploration is conducted using **5-Fold Stratified Cross-Validation exclusively on the 4,128 training samples**.
4. **Single Test Evaluation:** The final model is evaluated once on the test partition, preventing test set contamination.

---

## 4. 5-Fold Stratified Cross-Validation Benchmark (Training Partition Only)

Saved automatically as `outputs/cv_experiments.csv`:

| Experiment | Category | `var_smoothing` | CV Accuracy | CV Precision | CV Recall | CV F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline (vs=1e-9) | Baseline | $1.0 \times 10^{-9}$ | $84.40\% \pm 0.64\%$ | $43.92\% \pm 1.13\%$ | $91.25\% \pm 1.62\%$ | $59.30\% \pm 1.36\%$ |
| 2. Smoothing vs=1e-5 | Variance Sweep | $1.0 \times 10^{-5}$ | $84.40\% \pm 0.64\%$ | $43.92\% \pm 1.13\%$ | $91.25\% \pm 1.62\%$ | $59.30\% \pm 1.36\%$ |
| 3. Smoothing vs=1e-2 | Variance Sweep | $1.0 \times 10^{-2}$ | $93.10\% \pm 2.60\%$ | $66.52\% \pm 8.93\%$ | $94.55\% \pm 1.90\%$ | $77.82\% \pm 6.76\%$ |
| **4. Smoothing vs=0.05** | **Variance Sweep** | **$5.0 \times 10^{-2}$** | **$98.35\% \pm 0.56\%$** | **$95.52\% \pm 2.21\%$** | **$91.05\% \pm 2.91\%$** | **$93.22\% \pm 2.31\%$** |
| 5. Smoothing vs=0.08 | Variance Sweep | $8.0 \times 10^{-2}$ | $98.30\% \pm 0.47\%$ | $98.04\% \pm 1.28\%$ | $88.14\% \pm 2.89\%$ | $92.81\% \pm 2.05\%$ |
| 6. Vocab max_features=1000 | Vocab Capacity | $5.0 \times 10^{-2}$ | $97.65\% \pm 0.60\%$ | $88.63\% \pm 3.17\%$ | $93.19\% \pm 1.62\%$ | $90.83\% \pm 2.23\%$ |
| 7. Vocab max_features=3000 | Vocab Capacity | $5.0 \times 10^{-2}$ | $98.38\% \pm 0.55\%$ | $96.66\% \pm 2.05\%$ | $90.08\% \pm 2.69\%$ | $93.25\% \pm 2.30\%$ |
| 8. Min_df=1 (no pruning) | Document Freq | $5.0 \times 10^{-2}$ | $98.38\% \pm 0.54\%$ | $95.57\% \pm 2.67\%$ | $91.25\% \pm 2.75\%$ | $93.33\% \pm 2.22\%$ |
| 9. Min_df=3 (rare pruned) | Document Freq | $5.0 \times 10^{-2}$ | $98.35\% \pm 0.50\%$ | $94.98\% \pm 2.18\%$ | $91.64\% \pm 2.18\%$ | $93.27\% \pm 2.03\%$ |
| 10. N-grams (1, 2) | N-gram Range | $5.0 \times 10^{-2}$ | $98.09\% \pm 0.33\%$ | $98.88\% \pm 1.01\%$ | $85.60\% \pm 2.06\%$ | $91.75\% \pm 1.48\%$ |
| 11. No Stop Words | Stop-Words | $5.0 \times 10^{-2}$ | $96.85\% \pm 0.56\%$ | $99.21\% \pm 0.65\%$ | $75.30\% \pm 4.11\%$ | $85.56\% \pm 2.87\%$ |
| 12. TF-IDF Representation | Feature Type | $2.0 \times 10^{0}$ | $97.14\% \pm 0.64\%$ | $86.92\% \pm 3.43\%$ | $90.86\% \pm 2.26\%$ | $88.81\% \pm 2.40\%$ |

---

## 5. Folder Structure & Output Artifacts

```
Email_Spam_Classifier/
├── main.py                          # Full pipeline: validation, CV, train, test, save
├── data_loader.py                   # Data reading, malformed line tracking, conflict check
├── preprocessing.py                 # Text cleaning, CountVectorizer, dense float32 conversion
├── model.py                         # GaussianNB definition, var_smoothing, pipeline export
├── evaluate.py                      # Metrics, confusion matrix, error analysis extraction
├── predict.py                       # CLI inference with calibrated probability output
├── requirements.txt                 # Exact dependency specifications
├── README.md                        # Project documentation
├── report.md                        # Formal CST413 academic report (10-section structure)
├── data/
│   └── SMSSpamCollection            # UCI SMS Spam Collection benchmark dataset
└── outputs/
    ├── metrics.txt                  # Full test set evaluation metrics & baseline comparison
    ├── cv_experiments.csv           # 12-configuration 5-fold cross-validation results table
    ├── test_comparison.csv          # Held-out test set before/after comparison table
    ├── error_analysis.csv           # Full text and labels of all test FP and FN errors
    ├── confusion_matrix.png         # Confusion matrix heatmap with counts & percentages
    ├── class_distribution.png       # Cleaned dataset class distribution plot
    ├── gaussian_nb_model.joblib     # Serialized GaussianNB classifier
    ├── count_vectorizer.joblib      # Serialized CountVectorizer
    └── spam_classifier_pipeline.joblib # Full scikit-learn Pipeline artifact
```

---

## 6. Setup and Execution

### Prerequisites
Python 3.10+ (tested on Python 3.13):
```powershell
pip install -r requirements.txt
```

### 1. Run Complete Training and Evaluation Pipeline
```powershell
python main.py
```
This script runs the entire machine learning workflow:
- Audits and cleans the dataset.
- Executes 5-fold Stratified Cross-Validation on the training data.
- Refits the selected optimal model on the training partition.
- Evaluates the model on the held-out test partition.
- Extracts misclassified samples to `outputs/error_analysis.csv`.
- Generates all visualization plots and metrics files in `outputs/`.

### 2. Classify Single Messages via CLI
```powershell
python predict.py --text "Congratulations! You have won a $1,000 Walmart gift card. Call 0800 now to claim."
```

Output:
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

### 3. Interactive Shell Prediction
```powershell
python main.py --predict
```

---

## 7. Submission Checklist for CST413

- [x] **Target Algorithm:** Gaussian Naïve Bayes (`GaussianNB`) retained as the core algorithm (not replaced by SVM or Logistic Regression).
- [x] **Evaluation Integrity:** Cross-validation performed strictly on training data; held-out test set evaluated once.
- [x] **Dataset Auditing:** Conflicting labels checked (0 found), bad lines tracked, duplicates audited (414 removed).
- [x] **Systematic Tuning:** Investigated smoothing ($10^{-9}$ to $0.08$), vocabulary capacity ($1000$ to $3000$), $min\_df$, n-grams, and stop words.
- [x] **Error Analysis:** Documented and analyzed all test False Positives and False Negatives.
- [x] **Academic Report:** Formatted according to the 10-section structure with Bayes' theorem worked example and SMS vs. email analysis.
- [x] **Clean Submission:** Exclude `.venv/`, `__pycache__/`, and `.git/` when creating the final ZIP file for submission.
