# CST413 Machine Learning Assignment 2: Email Spam Classification Using Gaussian Naïve Bayes on UCI Spambase

**Course:** CST413 Machine Learning  
**Assignment:** Assignment 2 — Probabilistic Text & Email Classification  
**Target Algorithm:** Gaussian Naïve Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Dataset:** UCI Spambase Email Dataset (Hewlett-Packard Laboratories)  
**Evaluation Standard:** 20-Mark Comprehensive Academic & Practical Rubric  
**Date:** October 2026  

---

## 1. Title and Objective

### 1.1 Project Title
**An End-to-End Email Spam Classification System on UCI Spambase Implementing Gaussian Naïve Bayes with Log-Transform Distribution Alignment, Variance Regularization, and Leak-Free Cross-Validation.**

### 1.2 Objective
The goal of this assignment is to design, implement, evaluate, and critically analyze an end-to-end machine learning system that classifies email messages as either **Legitimate (Ham / Non-Spam, class 0)** or **Spam (class 1)**. In strict alignment with the assignment specification, the primary classification algorithm is **Gaussian Naïve Bayes (`GaussianNB`)**.

To ensure authentic domain relevance, this project utilizes the benchmark **UCI Spambase** dataset, which consists of real email messages represented by 57 continuous attributes. The key technical objectives are:
1. Implement GaussianNB as the core classifier without substituting non-prescribed algorithms.
2. Investigate the class-conditional Gaussian distribution assumptions on continuous word-frequency percentages and run-length attributes.
3. Diagnose and resolve variance scale disparities across features using a **Log1p transformation ($\ln(1 + x)$)** and variance smoothing regularization (`var_smoothing`).
4. Build a programmatic **Raw Email Feature Extractor** that computes the exact 57 Spambase features from raw email text, satisfying Requirement #2 (Text Preprocessing and Feature Extraction).
5. Enforce a leak-free evaluation methodology using **5-Fold Stratified Cross-Validation on the training partition only**, preserving an untouched held-out test partition for final evaluation.
6. Conduct quantitative and qualitative diagnostic error analyses on misclassified samples.

---

## 2. Dataset Description

### 2.1 Source and Provenance
The system uses the benchmark **UCI Spambase** dataset, created by Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt at **Hewlett-Packard Laboratories** (Palo Alto, California, 1999) and distributed by the UCI Machine Learning Repository:
- **Collection Method:** Legitimate non-spam emails were collected from filed work and personal correspondence of HP engineers; spam emails were gathered from postmaster logs and direct user submissions.
- **Domain Authenticity:** Unlike SMS corpora, Spambase was constructed specifically from **genuine email traffic**, reflecting typical email vocabulary, punctuation markers, and capital letter typing habits.

### 2.2 Raw Characteristics and Data Validation Audit
The dataset was loaded and audited through a rigorous validation process:
- **Source File:** `spambase.data` (58 comma-separated columns: 57 continuous attributes + 1 binary class label).
- **Physical Lines in File:** 4,601
- **Rows Successfully Parsed:** 4,601
- **Malformed / Bad Lines Encountered:** 0
- **Missing / Null Values Removed:** 0
- **Invalid Target Labels:** 0 (all labels strictly binary $\{0, 1\}$).
- **Conflicting Feature Verification:** Evaluated whether identical feature vectors mapped to contradictory labels. **3 conflicting groups (6 rows total)** were identified where identical zero/low vectors had conflicting ground truth labels. These ambiguous rows were filtered out to preserve data integrity.
- **Duplicate Removal:** 391 exact duplicate instances were removed, retaining only the first occurrence.

### 2.3 Sample Counts and Class Distribution
After validation and deduplication, the cleaned dataset contains **4,204 unique email instances**:

| Class Label | Semantic Meaning | Number of Instances | Class Percentage |
| :---: | :---: | :---: | :---: |
| **0** | **Ham (Legitimate Email)** | 2,528 | 60.13% |
| **1** | **Spam (Unsolicited Email)** | 1,676 | 39.87% |
| **Total** | **All Cleaned Instances** | **4,204** | **100.00%** |

The class distribution is balanced compared to typical SMS datasets, with approximately a **1.5:1** ratio (60.13% ham vs. 39.87% spam). A baseline majority-class classifier predicting ham would achieve only 60.13% accuracy, confirming that meaningful probabilistic learning is required.

---

## 3. Preprocessing and Train/Test Splitting

### 3.1 Preprocessing Workflow
1. **Validation & Filtering:** Conflicting feature rows and duplicate entries are removed to prevent noise and data leakage.
2. **Log-Transform Distribution Alignment:** In raw Spambase, feature values span vast dynamic ranges:
   - Word percentages range in $[0, 100]$ with variances $\approx 0.01 - 1.0$.
   - Capital run lengths span $[1, 15841]$ with variances exceeding $3.7 \times 10^5$.
   Applying a **Log1p transformation** ($f(x) = \ln(1 + x)$) stabilizes variances, compresses extreme outliers, and maps heavily right-skewed count/percentage data into near-Gaussian bell-shaped distributions.
3. **Pipeline Encapsulation:** The transformation and classifier are bundled in a scikit-learn `Pipeline`, ensuring identical transformations are applied to both training and test partitions without leakage.

### 3.2 Leak-Free Partitioning Strategy
- **Partition Ratio:** 80% Training ($N = 3,363$) and 20% Held-Out Testing ($N = 841$).
- **Stratified Split:** `stratify=y` is enforced to preserve the 60.13% / 39.87% class ratio in both subsets.
- **Deterministic Seed:** Fixed random seed (`random_state=42`).

```
Cleaned Spambase Dataset: 4,204 instances (2,528 Ham, 1,676 Spam)
                            │
                            ▼
           Stratified Split (80/20, seed=42)
            ┌───────────────┴───────────────┐
            ▼                               ▼
Training Partition: 3,363               Held-Out Test Set: 841
(2,022 Ham, 1,341 Spam)                 (506 Ham, 335 Spam)
            │                                   │
            ▼                                   │ (Strictly isolated;
5-Fold Stratified CV                            │  never seen during
(Hyperparameter tuning on train only)            │  parameter tuning)
            │                                   │
            ▼                                   ▼
Fit Final Selected Model ───────────────► Final Single Evaluation
(Log1p + GaussianNB, vs=0.05)            (Test Metrics & Error Analysis)
```

---

## 4. Feature Extraction: The 57 Spambase Attributes & Raw Email Engine

### 4.1 Schema of the 57 Precomputed Spambase Attributes
The 57 continuous input features fall into three functional groups:

1. **Word-Frequency Percentages (48 attributes, 0–100%):**
   $$x_i = 100 \times \frac{\text{Count of } \text{WORD}_i}{\text{Total Words in Email}}$$
   Target words include spam indicators (`free`, `business`, `money`, `credit`, `remove`, `order`, `000`) and HP work-environment terms (`hp`, `hpl`, `george`, `650`, `lab`, `labs`, `data`, `meeting`).

2. **Character-Frequency Percentages (6 attributes, 0–100%):**
   $$x_i = 100 \times \frac{\text{Count of } \text{CHAR}_i}{\text{Total Characters in Email}}$$
   Target characters: `;`, `(`, `[`, `!`, `$`, `#`. Characters such as `$` and `!` are strong indicators of promotional emphasis and financial lures.

3. **Capital Letter Run-Length Statistics (3 attributes):**
   - `capital_run_length_average`: Mean length of uninterrupted sequences of uppercase letters.
   - `capital_run_length_longest`: Maximum length of an uninterrupted uppercase sequence.
   - `capital_run_length_total`: Total count of uppercase letters across the email.

### 4.2 Raw Email Feature Extraction Engine (`preprocessing.py`)
To satisfy **Requirement #2 (Text preprocessing and feature extraction)** and enable interactive classification of newly typed emails, an extraction engine was constructed in `preprocessing.py`:

```python
import re, numpy as np

def extract_spambase_features(text: str) -> np.ndarray:
    # 1. Word Frequencies (48 attributes)
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    total_words = len(words)
    word_freqs = [
        (100.0 * words.count(w) / total_words) if total_words > 0 else 0.0
        for w in SPAMBASE_WORDS
    ]
    # 2. Character Frequencies (6 attributes)
    total_chars = len(text)
    char_freqs = [
        (100.0 * text.count(c) / total_chars) if total_chars > 0 else 0.0
        for c in SPAMBASE_CHARS
    ]
    # 3. Capital Run-Length Statistics (3 attributes)
    cap_runs = [len(m) for m in re.findall(r"[A-Z]+", text)]
    cap_avg = float(np.mean(cap_runs)) if cap_runs else 1.0
    cap_max = int(max(cap_runs)) if cap_runs else 1
    cap_tot = int(sum(cap_runs)) if cap_runs else 0

    return np.array(word_freqs + char_freqs + [cap_avg, cap_max, cap_tot], dtype=np.float64)
```

This bridge allows users to input arbitrary raw email text via `python predict.py --text "..."`, compute the 57 continuous features dynamically, and obtain calibrated predictions.

---

## 5. Theoretical Foundations of Gaussian Naïve Bayes

### 5.1 Bayes' Theorem
Bayes' theorem calculates the posterior probability of a hypothesis (e.g., email is spam) given observed feature evidence:
$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

#### Numerical Worked Example
Consider an email filtering scenario:
- Prior probability that an email is spam: $P(A) = 0.20$
- Prior probability that an email is legitimate ham: $P(\neg A) = 0.80$
- Probability that an email contains a prominent dollar sign (`$`) given it is spam: $P(B \mid A) = 0.85$
- Probability that an email contains a prominent dollar sign given it is ham: $P(B \mid \neg A) = 0.10$

By the Law of Total Probability:
$$P(B) = P(B \cap A) + P(B \cap \neg A) = P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)$$
$$P(B \cap A) = 0.85 \times 0.20 = 0.17$$
$$P(B \cap \neg A) = 0.10 \times 0.80 = 0.08$$
$$P(B) = 0.17 + 0.08 = 0.25$$

Applying Bayes' theorem:
$$P(A \mid B) = \frac{0.85 \times 0.20}{0.25} = \frac{0.17}{0.25} = 0.68$$
Upon observing feature $B$, the posterior probability that the email is spam rises from 20% to **68%**.

### 5.2 The Naïve Conditional Independence Assumption
For a continuous feature vector $\mathbf{x} = [x_1, x_2, \dots, x_{57}]$, Naïve Bayes assumes that all 57 attributes are **conditionally independent given the class label $C \in \{0, 1\}$**:
$$P(\mathbf{x} \mid C) = \prod_{i=1}^{57} P(x_i \mid C)$$

The posterior class distribution is:
$$P(C \mid \mathbf{x}) \propto P(C) \prod_{i=1}^{57} P(x_i \mid C)$$
Classification selects the Maximum A Posteriori (MAP) class:
$$\hat{y} = \arg\max_{C \in \{0, 1\}} \left[ \ln P(C) + \sum_{i=1}^{57} \ln P(x_i \mid C) \right]$$

### 5.3 The Class-Conditional Gaussian Assumption
Gaussian Naïve Bayes models each continuous attribute $x_i$ as a normal distribution parameterized by class-conditional mean $\mu_{C,i}$ and variance $\sigma_{C,i}^2$:
$$P(x_i \mid C) = \frac{1}{\sqrt{2\pi \sigma_{C,i}^2}} \exp\left( -\frac{(x_i - \mu_{C,i})^2}{2\sigma_{C,i}^2} \right)$$

### 5.4 Variance Disparity and Variance Smoothing Regularization
In raw Spambase, feature variances differ substantially:
- Word percentages have tiny variances ($\sigma^2 \sim 0.01$).
- `capital_run_length_total` has a massive variance ($\sigma^2 \approx 3.7 \times 10^5$).

Because scikit-learn computes `var_smoothing` globally:
$$\sigma_{C,i,\text{smoothed}}^2 = \sigma_{C,i}^2 + \epsilon \cdot \max_{j}(\sigma_j^2)$$
setting `var_smoothing = 0.05` on raw features adds $0.05 \times 3.7 \times 10^5 \approx 18,500$ to all feature variances, drowning out the word-frequency percentages.

**The Solution:**
1. Applying `Log1p` ($x' = \ln(1 + x)$) compresses the variance of capital run-length totals from $10^5$ down to $\approx 3.5$.
2. This aligns feature variances into a common range, allowing `var_smoothing = 0.05` to provide balanced regularization across all 57 features.

---

## 6. Software Architecture and Pipeline Implementation

The codebase is structured into modular components:
- `data_loader.py`: Spambase loading, 58-attribute schema mapping, conflict detection, deduplication, and auditing.
- `preprocessing.py`: Feature schema definitions, Log1p transformer, and the `extract_spambase_features()` raw-email engine.
- `model.py`: GaussianNB pipeline definition, variance smoothing parameters, and artifact persistence.
- `evaluate.py`: Performance metrics computation, confusion matrix plotting, class distribution graphing, and error analysis.
- `main.py`: Orchestrator managing stratified splitting, 5-fold CV, final model fitting, test evaluation, and artifact generation.
- `predict.py`: User-facing CLI inference script supporting raw email text and numeric feature rows.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Leak-free Spambase Pipeline
spambase_pipeline = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, validate=False)),
    ("classifier", GaussianNB(var_smoothing=0.05)),
])
```

---

## 7. Systematic Experiments: 5-Fold Stratified Cross-Validation

All hyperparameter tuning, variance smoothing sweeps, and feature subset investigations were conducted exclusively on the 3,363 training samples using **5-Fold Stratified Cross-Validation**. The test set remained unseen.

The cross-validation results (`outputs/cv_experiments.csv`) across all 12 controlled configurations are reported below:

| # | Experiment Configuration | Category | `var_smoothing` | CV Accuracy (Mean ± Std) | CV Precision (Mean ± Std) | CV Recall (Mean ± Std) | CV F1-Score (Mean ± Std) | Description |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (Raw, vs=1e-9) | Baseline | $1.0 \times 10^{-9}$ | $0.8231 \pm 0.0116$ | $0.7067 \pm 0.0129$ | $0.9515 \pm 0.0127$ | $0.8110 \pm 0.0116$ | Unregularized default on raw features |
| **2** | Raw vs=1e-5 | Raw Variance | $1.0 \times 10^{-5}$ | $0.6830 \pm 0.0108$ | $0.8311 \pm 0.0318$ | $0.2573 \pm 0.0251$ | $0.3923 \pm 0.0311$ | Variance disparity causes signal loss |
| **3** | Raw vs=1e-2 | Raw Variance | $1.0 \times 10^{-2}$ | $0.6497 \pm 0.0146$ | $0.7985 \pm 0.0760$ | $0.1618 \pm 0.0254$ | $0.2687 \pm 0.0386$ | Excessive variance added to word freqs |
| **4** | Raw vs=0.05 | Raw Variance | $5.0 \times 10^{-2}$ | $0.6447 \pm 0.0111$ | $0.7858 \pm 0.0697$ | $0.1499 \pm 0.0171$ | $0.2515 \pm 0.0266$ | Severe recall degradation on raw scale |
| **5** | Log1p vs=1e-9 | Log1p Transform | $1.0 \times 10^{-9}$ | $0.8281 \pm 0.0091$ | $0.7145 \pm 0.0098$ | $0.9478 \pm 0.0169$ | $0.8147 \pm 0.0101$ | Log1p with unregularized variance |
| **6** | Log1p vs=1e-4 | Log1p Transform | $1.0 \times 10^{-4}$ | $0.8415 \pm 0.0105$ | $0.7323 \pm 0.0128$ | $0.9500 \pm 0.0110$ | $0.8270 \pm 0.0106$ | Log1p with light regularization |
| **7** | Log1p vs=1e-3 | Log1p Transform | $1.0 \times 10^{-3}$ | $0.8484 \pm 0.0121$ | $0.7391 \pm 0.0148$ | $0.9582 \pm 0.0076$ | $0.8345 \pm 0.0120$ | Improved log-variance stability |
| **8** | Log1p vs=1e-2 | Log1p Transform | $1.0 \times 10^{-2}$ | $0.8641 \pm 0.0171$ | $0.7625 \pm 0.0238$ | $0.9590 \pm 0.0058$ | $0.8494 \pm 0.0166$ | Steady precision gains |
| **9** | **Log1p vs=0.05 (Selected)** | **Log1p Transform** | **$5.0 \times 10^{-2}$** | **$0.8843 \pm 0.0127$** | **$0.8066 \pm 0.0189$** | **$0.9344 \pm 0.0139$** | **$0.8657 \pm 0.0141$** | **Optimal F1 & precision balance** |
| **10** | Log1p vs=0.10 | Log1p Transform | $1.0 \times 10^{-1}$ | $0.8790 \pm 0.0173$ | $0.8100 \pm 0.0242$ | $0.9105 \pm 0.0140$ | $0.8573 \pm 0.0195$ | Higher precision at expense of recall |
| **11** | Log1p + StandardScaler | Scaling | $1.0 \times 10^{-2}$ | $0.8391 \pm 0.0101$ | $0.7300 \pm 0.0117$ | $0.9470 \pm 0.0132$ | $0.8244 \pm 0.0107$ | Unit-variance scaling |
| **12** | Word Features Only (48) | Feature Subset | $5.0 \times 10^{-2}$ | $0.8162 \pm 0.0118$ | $0.6940 \pm 0.0134$ | $0.9649 \pm 0.0069$ | $0.8073 \pm 0.0110$ | Excluding punctuation & capital stats |

### Key Experimental Findings:
1. **The Log1p Transformation is Critical:** On raw features, smoothing destroys performance (CV F1 falls from 81.10% to 25.15%) due to variance scale disparity. Applying Log1p normalizes feature scales, enabling variance smoothing to boost CV F1 from **81.47% to 86.57%** and precision from **71.45% to 80.66%**.
2. **Punctuation and Capital Runs Matter:** Excluding the 6 character frequencies and 3 run-length features (Exp #12) drops CV F1 by **-5.84%** and precision by **-11.26%**, proving that punctuation (`!`, `$`) and capitalization provide strong predictive signals.
3. **Selected Production Model:** Configuration #9 (`Log1p` + `var_smoothing = 0.05`) achieves the highest CV F1-score ($86.57\%$) with a high recall ($93.44\%$) and strong precision ($80.66\%$).

---

## 8. Final Held-Out Test Set Evaluation

The final selected pipeline (`Log1p` + `GaussianNB(var_smoothing=0.05)`) was refitted on the full 3,363 training samples and evaluated on the held-out test partition ($N = 841$). The unregularized raw baseline was evaluated under identical conditions (`outputs/test_comparison.csv`).

### 8.1 Performance Comparison Table

| Metric | Raw Baseline (`var_smoothing=1e-9`) | Final Model (Log1p + `vs=0.05`) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 82.52% | **87.40%** | **+4.88%** | +5.91% |
| **Spam Precision** | 70.35% | **78.13%** | **+7.79%** | +11.07% |
| **Spam Recall** | 97.01% | **94.93%** | **-2.09%** | -2.15% |
| **Spam F1-Score** | 81.56% | **85.71%** | **+4.16%** | +5.10% |
| **False Positive Rate** | 27.08% (137 FP) | **17.59% (89 FP)** | **-9.49%** | **-35.04% (FP dropped)** |
| **False Negatives** | 10 FN | **17 FN** | +7 FN | — |

### 8.2 Confusion Matrix Breakdown
The held-out test set contains **506 legitimate Ham emails** and **335 Spam emails**:

```
UNREGULARIZED RAW BASELINE:
                     Predicted Ham    Predicted Spam
Actual Ham (506) :        369              137       (FP = 137, FPR = 27.08%)
Actual Spam (335):         10              325       (FN = 10, Recall = 97.01%)

FINAL IMPROVED MODEL (Log1p + vs=0.05):
                     Predicted Ham    Predicted Spam
Actual Ham (506) :        417               89       (FP = 89, FPR = 17.59%)
Actual Spam (335):         17              318       (FN = 17, Recall = 94.93%)
```

- **True Negatives ($TN = 417$):** 82.41% of legitimate emails correctly delivered.
- **False Positives ($FP = 89$):** Dropped from 137 down to 89 (a 35.0% reduction in misfiled legitimate emails).
- **False Negatives ($FN = 17$):** Only 17 out of 335 spam emails bypassed detection.
- **True Positives ($TP = 318$):** 94.93% of all spam emails accurately intercepted.

### 8.3 Classification Report
```
                precision    recall  f1-score   support

Ham (Non-Spam)       0.96      0.82      0.89       506
          Spam       0.78      0.95      0.86       335

      accuracy                           0.87       841
     macro avg       0.87      0.89      0.87       841
  weighted avg       0.89      0.87      0.88       841
```

---

## 9. Discussion and Diagnostic Error Analysis

### 9.1 Qualitative Error Analysis (Test Set Dissections)
From the exported error log (`outputs/error_analysis.csv`), we examine the characteristics of the remaining 89 False Positives and 17 False Negatives:

#### Analysis of False Positives ($FP = 89$):
1. **Capital Letter Bursts in Legitimate Work Emails:**
   - *Sample #2:* `capital_run_length_total = 405.0`, `capital_run_length_longest = 71.0`, `char_freq_! = 0.374`.
   - *Sample #71:* `capital_run_length_total = 1763.0`, `capital_run_length_longest = 185.0`.
   - **Reason:** Legitimate emails containing capitalized programming code, table headers, chemical/biological formulas, or legal disclaimers trigger high capital run-length scores, which the model associates with spam emphasis.
2. **Technical Terms Overlapping with Commercial Jargon:**
   - *Sample #27:* `word_freq_internet = 2.07`, `capital_run_length_total = 64.0`.
   - *Sample #105:* `word_freq_parts = 7.40`, `word_freq_free = 3.70`.
   - **Reason:** Legitimate hardware/software procurement emails frequently discuss "free software", "internet connections", and "replacement parts", triggering high spam likelihoods.

#### Analysis of False Negatives ($FN = 17$):
1. **Low-Capitalization Conversational Spam:**
   - *Sample #113:* `capital_run_length_total = 3.0`, `word_freq_you = 4.76`, zero dollar signs, zero exclamation marks.
   - **Reason:** Sophisticated spam disguised as personal, lower-case conversational messages lacks high capitalization and blatant keywords, blending into the legitimate ham profile.
2. **Deferred Calls-to-Action:**
   - *Sample #77:* `word_freq_will = 1.29`, `capital_run_length_total = 110.0`.
   - **Reason:** Spam messages that ask the recipient to reply or click an unflagged link without using words from the 48 target keywords evade detection.

### 9.2 The Precision-Recall Trade-off in Corporate Email
In enterprise email filtering, **False Positives are substantially more damaging than False Negatives**:
- A user missing a client proposal, flight confirmation, or interview notice can suffer serious consequences.
- The baseline model's 27.08% False Positive Rate (137 good emails blocked) would cause operational failure.
- By introducing `Log1p` and `var_smoothing = 0.05`, False Positives were reduced by **35.0%** (89 FP), while preserving a high **94.93% Recall**.

### 9.3 Critique of Gaussian Normality on Email Attributes
Gaussian Naïve Bayes assumes that each feature follows a symmetric, bell-shaped normal distribution. In Spambase:
- 48 word frequencies and 6 character frequencies are bounded in $[0, 100]$ and heavily zero-inflated (often $>85\%$ zeros).
- Capital run-lengths are strictly positive integers with extreme right-skew.
- The `Log1p` transformation compresses positive skewness and contracts long tails, enabling GaussianNB to achieve **87.40% test accuracy** and an **F1-score of 85.71%**.

### 9.4 Historical Context of the HP Labs Spambase
Spambase was compiled from HP Labs in 1999. Key considerations for modern systems:
- **Personalized Bias:** Terms like `george`, `650`, and `hp` were specific to George Forman's lab. In our experiments, these features act as strong legitimate indicators, which may not generalize to non-HP contexts.
- **Evolving Threats:** Modern spam relies on image-based text, tracking links, and domain spoofing. While Spambase remains a benchmark for feature-based email classification, modern production systems integrate DKIM/DMARC authentication, URL reputation, and deep-learning transformer embeddings.

---

## 10. Conclusion and Future Work

### 10.1 Summary of Accomplishments
1. **Full Assignment Compliance:** Implemented an end-to-end spam classifier strictly using `sklearn.naive_bayes.GaussianNB` on the authentic UCI Spambase email dataset.
2. **Text Preprocessing & Feature Extraction:** Built a dedicated raw-email feature extraction engine (`extract_spambase_features()`) that computes the 57 Spambase features from arbitrary text, satisfying Requirement #2.
3. **Distribution Alignment:** Demonstrated mathematically and empirically that Log1p transformation stabilizes feature variances and satisfies Gaussian distribution requirements, boosting CV F1-score from **81.10% to 86.57%**.
4. **Evaluation Rigor:** Conducted 5-Fold Stratified Cross-Validation on training data only and evaluated once on the isolated held-out test partition.
5. **Practical Prediction CLI:** Delivered an interactive inference CLI supporting raw email classification with calibrated probabilities.

### 10.2 Future Work
1. **Adaptive Probability Thresholding:** Implement an adjustable classification threshold $\tau > 0.5$ on $P(\text{Spam} \mid \mathbf{x})$ to allow administrators to tune false positives down to near-zero levels.
2. **Context-Independent Feature Pruning:** Remove organization-specific tokens (`george`, `650`, `hp`) to evaluate cross-organizational generalization.
3. **Hybrid Ensembles:** Evaluate GaussianNB as an efficient first-stage filter alongside deeper NLP representations.

---

## 11. References
1. Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase Data Set*. Hewlett-Packard Laboratories. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/94/spambase
2. Cranor, L. F., & LaMacchia, B. A. (1998). Spam! *Communications of the ACM*, 41(8), 74–83.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
