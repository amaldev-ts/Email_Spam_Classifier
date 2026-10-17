# Email Spam Classification Using Gaussian Naïve Bayes (UCI Spambase)

A production-grade, reproducible machine learning system that classifies emails as **Spam** or **Not Spam** strictly implementing scikit-learn's **Gaussian Naïve Bayes (`GaussianNB`)** algorithm on the official **UCI Spambase** dataset with Log-Transform distribution alignment, programmatic cross-validation model selection, and an automated test suite.

> **Dataset Scope:** The dataset used is the benchmark **UCI Spambase** (Hopkins et al., HP Labs, 1999), consisting of **4,601 genuine email instances** characterized by **57 continuous attributes** (48 word-frequency percentages, 6 character-frequency percentages, and 3 capital letter run-length statistics).

---

## 1. Project Overview & Performance Highlights

This project addresses the distributional challenges of applying continuous `GaussianNB` to skewed percentage and run-length attributes. By applying a **Log1p transformation ($\ln(1 + x)$)** and programmatically selecting variance smoothing (`var_smoothing = 0.05`) via **5-Fold Stratified Cross-Validation on the training partition only**, the classifier achieves superior performance on the isolated held-out test partition ($N = 921$):

| Metric | Raw Baseline (`var_smoothing=1e-9`) | Selected Model (Log1p + `vs=0.05`) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 83.39% | **87.62%** | **+4.23%** | +5.07% |
| **Spam Precision** | 71.78% | **78.75%** | **+6.97%** | +9.71% |
| **Spam Recall** | 95.32% | **93.94%** | **-1.38%** | -1.45% |
| **Spam F1-Score** | 81.89% | **85.68%** | **+3.78%** | +4.62% |
| **False Positive Rate** | 24.37% (136 FP) | **16.49% (92 FP)** | **-7.89%** | **-32.35% (FP cut)** |

### Confusion Matrix (Held-Out Test Set: 558 Not Spam, 363 Spam)

```
RAW BASELINE MODEL:
                     Predicted Not Spam   Predicted Spam
Actual Not Spam (558):      422                136       (136 False Positives, Precision = 71.78%)
Actual Spam (363):           17                346       (17 False Negatives, Recall = 95.32%)

SELECTED MODEL (Log1p + vs=0.05):
                     Predicted Not Spam   Predicted Spam
Actual Not Spam (558):      466                 92       (False Positives cut by 32.35% to 92!)
Actual Spam (363):           22                341       (22 False Negatives, Recall = 93.94%)
```

---

## 2. Dataset Policy & Comparative Cleaning Analysis

The system supports both the standard canonical benchmark dataset and a deduplicated subset via `data_loader.load_dataset(clean_duplicates=False/True)`:

1. **Canonical Benchmark ($N=4,601$, Default):**
   - Retains all 4,601 instances as published in the official UCI repository and academic literature (Hopkins et al., 1999).
   - **Why identical feature vectors occur naturally:** Spambase contains precomputed continuous feature vectors, not raw email bodies. Short or boilerplate emails where target vocabulary words are absent naturally yield identical zero-heavy vectors. Discarding them distorts the true distribution of real-world mail traffic.
   - **Why conflicting labels occur:** 3 pairs of zero/near-zero feature vectors share opposite ground-truth labels. This reflects the inherent information loss of projecting complex emails into 57 features (Bayes error), not corrupted data.
2. **Deduplicated Subset ($N=4,204$):**
   - Filters 391 exact duplicate rows and 3 conflicting groups (6 rows).
3. **Empirical Policy Comparison (`outputs/dataset_cleaning_comparison.csv`):**
   As demonstrated below, model performance is remarkably stable across both policies, confirming that the canonical 4,601 benchmark is both rigorous and representative:

| Dataset Policy | Total Rows | Train Rows | Test Rows | CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Benchmark (Default)** | **4,601** | **3,680** | **921** | **85.97%** | **87.62%** | **78.75%** | **93.94%** | **85.68%** | **92** |
| **Deduplicated Subset** | **4,204** | **3,363** | **841** | **86.57%** | **87.40%** | **78.13%** | **94.93%** | **85.71%** | **89** |

---

## 3. Strict, Leak-Free Evaluation Methodology

```
Canonical Spambase Dataset (4,601 instances: 2,788 Not Spam, 1,813 Spam)
                            │
                            ▼
           Stratified Split (80/20, seed=42)
            ┌───────────────┴───────────────┐
            ▼                               ▼
Training Partition (3,680)              Held-Out Test Set (921)
(2,230 Not Spam, 1,450 Spam)             (558 Not Spam, 363 Spam)
            │                                   │
            ▼                                   │ (Strictly isolated;
5-Fold Stratified CV                            │  never seen during tuning)
(12 candidate configurations evaluated)         │
            │                                   │
            ▼                                   │
Programmatic Best-Model Selection ──────────────┘
(Experiment #9: Log1p + vs=0.05, CV F1 = 85.97%)
            │
            ▼
Fit Selected Pipeline on Full Train ────────► Single Final Test Evaluation
                                              (Outputs test metrics & error analysis)
```

1. **Programmatic Selection:** The optimal pipeline is chosen automatically based on highest mean CV Spam F1-score across 12 candidate configurations—never hardcoded.
2. **Data Isolation:** Stratified 80/20 split (`seed=42`) isolates 921 test samples.
3. **Training CV Only:** All model exploration is conducted using **5-Fold Stratified Cross-Validation exclusively on the 3,680 training instances**.
4. **Single Test Evaluation:** The winning model is evaluated once on the isolated test partition, preventing data leakage.

---

## 4. 5-Fold Stratified Cross-Validation Benchmark (Training Partition Only)

Saved automatically as `outputs/cv_experiments.csv`:

| # | Experiment Configuration | Category | `var_smoothing` | CV Accuracy | CV Precision | CV Recall | CV F1-Score |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Baseline (Raw, vs=1e-9) | Baseline | $1.0 \times 10^{-9}$ | $82.07\% \pm 1.24\%$ | $70.12\% \pm 1.52\%$ | $95.03\% \pm 1.12\%$ | $80.69\% \pm 1.19\%$ |
| 2 | Raw vs=1e-5 | Raw Variance | $1.0 \times 10^{-5}$ | $69.62\% \pm 1.05\%$ | $82.59\% \pm 4.14\%$ | $29.24\% \pm 3.80\%$ | $42.99\% \pm 3.73\%$ |
| 3 | Raw vs=1e-2 | Raw Variance | $1.0 \times 10^{-2}$ | $65.71\% \pm 0.65\%$ | $79.59\% \pm 5.61\%$ | $17.66\% \pm 1.93\%$ | $28.81\% \pm 2.41\%$ |
| 4 | Raw vs=0.05 | Raw Variance | $5.0 \times 10^{-2}$ | $65.05\% \pm 0.61\%$ | $78.11\% \pm 5.56\%$ | $15.93\% \pm 2.04\%$ | $26.36\% \pm 2.64\%$ |
| 5 | Log1p vs=1e-9 | Log1p Transform | $1.0 \times 10^{-9}$ | $82.55\% \pm 1.16\%$ | $70.90\% \pm 1.24\%$ | $94.55\% \pm 1.30\%$ | $81.03\% \pm 1.21\%$ |
| 6 | Log1p vs=1e-4 | Log1p Transform | $1.0 \times 10^{-4}$ | $83.67\% \pm 1.01\%$ | $72.24\% \pm 1.12\%$ | $95.10\% \pm 1.30\%$ | $82.11\% \pm 1.08\%$ |
| 7 | Log1p vs=1e-3 | Log1p Transform | $1.0 \times 10^{-3}$ | $84.21\% \pm 0.81\%$ | $72.63\% \pm 1.06\%$ | $96.21\% \pm 0.72\%$ | $82.77\% \pm 0.79\%$ |
| 8 | Log1p vs=1e-2 | Log1p Transform | $1.0 \times 10^{-2}$ | $86.58\% \pm 0.92\%$ | $76.13\% \pm 1.18\%$ | $96.07\% \pm 0.80\%$ | $84.94\% \pm 0.95\%$ |
| **9** | **Log1p vs=0.05 (Selected)** | **Log1p Transform** | **$5.0 \times 10^{-2}$** | **$87.88\% \pm 1.33\%$** | **$79.14\% \pm 2.16\%$** | **$94.14\% \pm 1.13\%$** | **$85.97\% \pm 1.41\%$** |
| 10 | Log1p vs=0.10 | Log1p Transform | $1.0 \times 10^{-1}$ | $87.07\% \pm 1.12\%$ | $78.78\% \pm 2.07\%$ | $92.07\% \pm 0.95\%$ | $84.88\% \pm 1.11\%$ |
| 11 | Log1p + StandardScaler | Scaling | $1.0 \times 10^{-2}$ | $83.18\% \pm 1.20\%$ | $71.62\% \pm 1.36\%$ | $94.97\% \pm 1.35\%$ | $81.65\% \pm 1.25\%$ |
| 12 | Word Features Only (48) | Feature Subset | $5.0 \times 10^{-2}$ | $80.76\% \pm 0.82\%$ | $68.04\% \pm 0.92\%$ | $96.55\% \pm 0.62\%$ | $79.82\% \pm 0.75\%$ |

---

## 5. Folder Structure & Output Artifacts

```
Email_Spam_Classifier/
├── main.py                          # Full pipeline: validation, CV, programmatic selection, test, save
├── data_loader.py                   # Data reading, 58-column schema, duplicate audit, cleaning policy
├── preprocessing.py                 # Feature definitions, Log1p transform, approximate feature proxy
├── model.py                         # GaussianNB pipeline definition, artifact serialization
├── evaluate.py                      # Metrics, confusion matrix, error analysis extraction
├── predict.py                       # CLI inference supporting sample vectors, feature rows, and text
├── requirements.txt                 # Minimum version constraints and exact tested versions
├── README.md                        # Complete project documentation
├── report.md                        # Formal CST413 academic report (10-section structure)
├── tests/                           # Comprehensive automated test suite (24 tests)
│   ├── test_data_loader.py          # Dataset discovery, canonical loading, deduplication audit
│   ├── test_preprocessing.py        # Text cleaning, approximate feature extraction, Log1p
│   ├── test_model.py                # Pipeline construction, training, probabilities, serialization
│   ├── test_pipeline.py             # Cross-validation search, programmatic selection, error analysis
│   └── test_predict.py              # Feature inference, sample vectors, text proxy, error handling
├── data/
│   ├── spambase.data                # Official UCI Spambase dataset (4,601 instances)
│   ├── spambase.names               # Official attribute definitions
│   └── spambase.DOCUMENTATION       # HP Labs dataset documentation
└── outputs/
    ├── metrics.txt                  # Full test set evaluation metrics & baseline comparison
    ├── cv_experiments.csv           # 12-configuration 5-fold cross-validation results table
    ├── test_comparison.csv          # Held-out test set before/after comparison table
    ├── dataset_cleaning_comparison.csv # Side-by-side canonical 4,601 vs 4,204 deduplicated comparison
    ├── error_analysis.csv           # Diagnostic list of test FP and FN errors
    ├── confusion_matrix.png         # Confusion matrix heatmap with counts & percentages
    ├── class_distribution.png       # Canonical dataset class distribution plot
    ├── spambase_gaussian_nb_pipeline.joblib # Full fitted Pipeline
    └── gaussian_nb_model.joblib     # Trained GaussianNB classifier
```

---

## 6. Setup and Execution

### Prerequisites
Python 3.10+ (tested and verified on Python 3.13):
```powershell
pip install -r requirements.txt
```

### 1. Run Automated Test Suite
Run all 24 unit and integration tests with `pytest`:
```powershell
python -m pytest tests/ -v
```

### 2. Run Complete Training and Evaluation Pipeline
```powershell
python main.py
```
This script executes the entire machine learning workflow:
- Audits and loads the canonical UCI Spambase dataset.
- Executes 5-fold Stratified Cross-Validation on the training partition.
- Programmatically selects the best model recipe based on mean CV F1.
- Refits the selected pipeline on the full training partition.
- Evaluates the model on the isolated held-out test partition.
- Runs the dataset cleaning comparison (`outputs/dataset_cleaning_comparison.csv`).
- Extracts misclassified samples to `outputs/error_analysis.csv`.
- Generates all visualization plots and metrics files in `outputs/`.

### 3. Classify Verified Feature Vectors via CLI (Benchmark Path)
Test the classifier directly on verified 57-feature vectors from Spambase:
```powershell
# Classify verified spam sample vector
python predict.py --sample-spam

# Classify verified not-spam sample vector
python predict.py --sample-not-spam
```

You can also pass arbitrary 57 comma-separated numbers:
```powershell
python predict.py --features "0,0.64,0.64,0,...,61,278"
```

### 4. Interactive and Raw Text Prediction (Heuristic Proxy)
```powershell
python predict.py --text "Claim your free money prize from our internet business now!"
```
*Note: Raw text classification uses an approximate heuristic feature extractor. Rigorous benchmark metrics apply to validated 57-feature continuous vectors.*
