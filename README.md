# Email Spam Classification Using Gaussian Naïve Bayes (UCI Spambase)

A production-grade, reproducible machine learning system that classifies emails as **Spam** or **Not Spam (Ham)** strictly implementing scikit-learn's **Gaussian Naïve Bayes (`GaussianNB`)** algorithm on the official **UCI Spambase** dataset with Log-Transform distribution alignment and leak-free 5-fold cross-validation.

> **Dataset Scope:** The dataset used is the benchmark **UCI Spambase** (Hopkins et al., HP Labs, 1999), consisting of **4,601 genuine email instances** characterized by **57 continuous attributes** (48 word-frequency percentages, 6 character-frequency percentages, and 3 capital letter run-length statistics).

---

## 1. Project Overview & Performance Highlights

This project addresses the distributional challenges of applying continuous `GaussianNB` to skewed percentage and run-length attributes. By applying a **Log1p transformation ($\ln(1 + x)$)** and calibrating variance smoothing (`var_smoothing = 0.05`) strictly via **5-Fold Stratified Cross-Validation on the training partition only**, the classifier achieves superior performance on the held-out test partition ($N = 841$):

| Metric | Raw Baseline (`var_smoothing=1e-9`) | Improved Model (Log1p + `vs=0.05`) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 82.52% | **87.40%** | **+4.88%** | +5.91% |
| **Spam Precision** | 70.35% | **78.13%** | **+7.79%** | +11.07% |
| **Spam Recall** | 97.01% | **94.93%** | **-2.09%** | -2.15% |
| **Spam F1-Score** | 81.56% | **85.71%** | **+4.16%** | +5.10% |
| **False Positive Rate** | 27.08% (137 FP) | **17.59% (89 FP)** | **-9.49%** | **-35.04% (FP dropped)** |

### Confusion Matrix (Held-Out Test Set: 506 Ham, 335 Spam)

```
RAW BASELINE MODEL:
                     Predicted Ham    Predicted Spam
Actual Ham (506) :        369              137       (137 False Positives, Precision = 70.35%)
Actual Spam (335):         10              325       (10 False Negatives, Recall = 97.01%)

IMPROVED MODEL (Log1p + vs=0.05):
                     Predicted Ham    Predicted Spam
Actual Ham (506) :        417               89       (False Positives cut by 35.0% to 89!)
Actual Spam (335):         17              318       (17 False Negatives, Recall = 94.93%)
```

---

## 2. Theoretical Foundations: Why Log1p & Variance Smoothing Work

### The Problem: Variance Scale Disparity & Right-Skew
In UCI Spambase, feature variances span orders of magnitude:
- Word percentages have small variances ($\sigma^2 \sim 0.01$).
- Capital letter run lengths span up to $15,841$ with variances exceeding $3.7 \times 10^5$.

Because scikit-learn computes variance smoothing globally ($\sigma_{C,i,\text{regularized}}^2 = \sigma_{C,i}^2 + \text{var\_smoothing} \times \max(\sigma^2)$), setting smoothing on raw features adds massive variance to word frequencies, destroying their signal.

### The Solution: Log1p Transformation
Applying `Log1p` ($x' = \ln(1 + x)$):
1. Contracts long-tailed, right-skewed distributions toward a normal Gaussian bell curve.
2. Compresses the variance range, allowing `var_smoothing = 0.05` to provide balanced regularization across all 57 features simultaneously.

---

## 3. Strict, Leak-Free Evaluation Methodology

```
Cleaned Spambase Dataset (4,204 instances: 2,528 Ham, 1,676 Spam)
                            │
                            ▼
           Stratified Split (80/20, seed=42)
            ┌───────────────┴───────────────┐
            ▼                               ▼
Training Partition (3,363)              Held-Out Test Set (841)
(2,022 Ham, 1,341 Spam)                 (506 Ham, 335 Spam)
            │                                   │
            ▼                                   │ (Strictly isolated;
5-Fold Stratified CV                            │  never seen during tuning)
(Tune smoothing, transforms, subsets)           │
            │                                   │
            ▼                                   ▼
Fit Final Model on Full Train ──────────► Single Final Test Evaluation
                                         (Produces test metrics & error analysis)
```

1. **Dataset Audit:** 4,601 raw rows parsed; 3 conflicting feature groups (6 rows) filtered; 391 duplicates removed.
2. **Data Isolation:** Stratified 80/20 split (`seed=42`) isolates 841 test samples.
3. **Training CV Only:** All model exploration is conducted using **5-Fold Stratified Cross-Validation exclusively on the 3,363 training instances**.
4. **Single Test Evaluation:** Final model evaluated once on the test partition, preventing contamination.

---

## 4. 5-Fold Stratified Cross-Validation Benchmark (Training Partition Only)

Saved automatically as `outputs/cv_experiments.csv`:

| # | Experiment Configuration | Category | `var_smoothing` | CV Accuracy | CV Precision | CV Recall | CV F1-Score |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Baseline (Raw, vs=1e-9) | Baseline | $1.0 \times 10^{-9}$ | $82.31\% \pm 1.16\%$ | $70.67\% \pm 1.29\%$ | $95.15\% \pm 1.27\%$ | $81.10\% \pm 1.16\%$ |
| 2 | Raw vs=1e-5 | Raw Variance | $1.0 \times 10^{-5}$ | $68.30\% \pm 1.08\%$ | $83.11\% \pm 3.18\%$ | $25.73\% \pm 2.51\%$ | $39.23\% \pm 3.11\%$ |
| 3 | Raw vs=1e-2 | Raw Variance | $1.0 \times 10^{-2}$ | $64.97\% \pm 1.46\%$ | $79.85\% \pm 7.60\%$ | $16.18\% \pm 2.54\%$ | $26.87\% \pm 3.86\%$ |
| 4 | Raw vs=0.05 | Raw Variance | $5.0 \times 10^{-2}$ | $64.47\% \pm 1.11\%$ | $78.58\% \pm 6.97\%$ | $14.99\% \pm 1.71\%$ | $25.15\% \pm 2.66\%$ |
| 5 | Log1p vs=1e-9 | Log1p Transform | $1.0 \times 10^{-9}$ | $82.81\% \pm 0.91\%$ | $71.45\% \pm 0.98\%$ | $94.78\% \pm 1.69\%$ | $81.47\% \pm 1.01\%$ |
| 6 | Log1p vs=1e-4 | Log1p Transform | $1.0 \times 10^{-4}$ | $84.15\% \pm 1.05\%$ | $73.23\% \pm 1.28\%$ | $95.00\% \pm 1.10\%$ | $82.70\% \pm 1.06\%$ |
| 7 | Log1p vs=1e-3 | Log1p Transform | $1.0 \times 10^{-3}$ | $84.84\% \pm 1.21\%$ | $73.91\% \pm 1.48\%$ | $95.82\% \pm 0.76\%$ | $83.45\% \pm 1.20\%$ |
| 8 | Log1p vs=1e-2 | Log1p Transform | $1.0 \times 10^{-2}$ | $86.41\% \pm 1.71\%$ | $76.25\% \pm 2.38\%$ | $95.90\% \pm 0.58\%$ | $84.94\% \pm 1.66\%$ |
| **9** | **Log1p vs=0.05 (Selected)** | **Log1p Transform** | **$5.0 \times 10^{-2}$** | **$88.43\% \pm 1.27\%$** | **$80.66\% \pm 1.89\%$** | **$93.44\% \pm 1.39\%$** | **$86.57\% \pm 1.41\%$** |
| 10 | Log1p vs=0.10 | Log1p Transform | $1.0 \times 10^{-1}$ | $87.90\% \pm 1.73\%$ | $81.00\% \pm 2.42\%$ | $91.05\% \pm 1.40\%$ | $85.73\% \pm 1.95\%$ |
| 11 | Log1p + StandardScaler | Scaling | $1.0 \times 10^{-2}$ | $83.91\% \pm 1.01\%$ | $73.00\% \pm 1.17\%$ | $94.70\% \pm 1.32\%$ | $82.44\% \pm 1.07\%$ |
| 12 | Word Features Only (48) | Feature Subset | $5.0 \times 10^{-2}$ | $81.62\% \pm 1.18\%$ | $69.40\% \pm 1.34\%$ | $96.49\% \pm 0.69\%$ | $80.73\% \pm 1.10\%$ |

---

## 5. Folder Structure & Output Artifacts

```
Email_Spam_Classifier/
├── main.py                          # Full pipeline: validation, CV, train, test, save
├── data_loader.py                   # Data reading, 58-column schema, conflict check, auditing
├── preprocessing.py                 # Feature definitions, Log1p transform, raw-email feature engine
├── model.py                         # GaussianNB definition, var_smoothing, pipeline export
├── evaluate.py                      # Metrics, confusion matrix, error analysis extraction
├── predict.py                       # CLI inference supporting raw email text & feature rows
├── requirements.txt                 # Exact dependency specifications
├── README.md                        # Project documentation
├── report.md                        # Formal CST413 academic report (10-section structure)
├── data/
│   ├── spambase.data                # Official UCI Spambase dataset (4,601 instances)
│   ├── spambase.names               # Official attribute definitions
│   └── spambase.DOCUMENTATION       # HP Labs dataset documentation
└── outputs/
    ├── metrics.txt                  # Full test set evaluation metrics & baseline comparison
    ├── cv_experiments.csv           # 12-configuration 5-fold cross-validation results table
    ├── test_comparison.csv          # Held-out test set before/after comparison table
    ├── error_analysis.csv           # Full diagnostic list of test FP and FN errors
    ├── confusion_matrix.png         # Confusion matrix heatmap with counts & percentages
    ├── class_distribution.png       # Cleaned dataset class distribution plot
    ├── spambase_gaussian_nb_pipeline.joblib # Full fitted Pipeline
    └── gaussian_nb_model.joblib     # Trained GaussianNB classifier
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
This script executes the entire machine learning workflow:
- Audits and cleans the UCI Spambase dataset.
- Executes 5-fold Stratified Cross-Validation on the training data.
- Refits the optimal pipeline on the full training partition.
- Evaluates the model on the held-out test partition.
- Extracts misclassified samples to `outputs/error_analysis.csv`.
- Generates all visualization plots and metrics files in `outputs/`.

### 2. Classify Raw Email Messages via CLI
```powershell
python predict.py --text "Claim your free money prize from our internet business now!"
```

Output:
```
====================================================
  PREDICTED CLASS  : SPAM
  Confidence Level : High Confidence
----------------------------------------------------
  Ham Probability  : < 0.01%    (p = 0.0000)
  Spam Probability : > 99.99%   (p = 1.0000)
====================================================
  Top Detected Spambase Signals:
    * word_freq_our               : 10.00
    * word_freq_internet          : 10.00
    * word_freq_free              : 10.00
----------------------------------------------------
  * Note: Probabilities represent Gaussian Naive Bayes
    posterior likelihoods under feature independence.
====================================================
```

### 3. Classify Legitimate Emails
```powershell
python predict.py --text "Hi George, please find attached the meeting notes for our lab project conference."
```

Output:
```
====================================================
  PREDICTED CLASS  : HAM (Non-Spam)
  Confidence Level : High Confidence
----------------------------------------------------
  Ham Probability  : > 99.99%   (p = 1.0000)
  Spam Probability : < 0.01%    (p = 0.0000)
====================================================
  Top Detected Spambase Signals:
    * word_freq_our               : 7.69
    * word_freq_george            : 7.69
    * word_freq_lab               : 7.69
----------------------------------------------------
```

### 4. Interactive Shell Prediction
```powershell
python main.py --predict
```


