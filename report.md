# CST413 Machine Learning Assignment 2: Email and SMS Spam Classification Using Gaussian Naïve Bayes

**Course:** CST413 Machine Learning  
**Assignment:** Assignment 2 — Probabilistic Text Classification  
**Target Algorithm:** Gaussian Naïve Bayes (`sklearn.naive_bayes.GaussianNB`)  
**Evaluation Standard:** 20-Mark Comprehensive Academic & Practical Rubric  
**Date:** October 2026  

---

## 1. Title and Objective

### 1.1 Project Title
**An End-to-End, Leak-Free Spam Classification System Implementing Gaussian Naïve Bayes with Variance Regularization and Rigorous Cross-Validation.**

### 1.2 Objective
The objective of this assignment is to design, implement, evaluate, and critically analyze a machine learning system that classifies text messages as either **Legitimate (Ham, class 0)** or **Spam (class 1)**. In strict compliance with the assignment specification, the primary classification algorithm is **Gaussian Naïve Bayes (`GaussianNB`)**. 

Rather than substituting GaussianNB with naturally discrete or margin-based classifiers (such as Multinomial Naïve Bayes, Support Vector Machines, or Logistic Regression), this study:
1. Implements GaussianNB as the core model.
2. Mathematically investigates its class-conditional normal distribution assumptions on discrete text word counts.
3. Diagnoses and resolves its vulnerability to near-zero variance collapse in sparse feature spaces.
4. Enforces a strict, leak-free evaluation methodology using **5-Fold Stratified Cross-Validation on the training partition only**, reserving a completely isolated held-out test partition for final evaluation.
5. Conducts comprehensive qualitative and quantitative error analyses on misclassified samples.

---

## 2. Dataset Description

### 2.1 Source and Provenance
The system uses the benchmark **SMS Spam Collection** dataset acquired from the UCI Machine Learning Repository (compiled by Tiago A. Almeida, José María Gómez Hidalgo, and Akbo Yamakami). The dataset is composed of real-world mobile text messages tagged as legitimate (ham) or spam, drawn from public mobile phone communication sources, the Grumbletext website, the UK National SMS corpus, and academic research groups.

### 2.2 Raw Characteristics and Data Validation Audit
The raw dataset was read and audited without silently skipping unparsed records:
- **Source File:** `SMSSpamCollection` (tab-separated, no headers, format: `<label>\t<message>`).
- **Total Physical Non-Empty Lines:** 5,574
- **Rows Successfully Parsed:** 5,574
- **Malformed / Bad Lines Encountered:** 0
- **Missing / Null Values Removed:** 0
- **Unrecognized / Invalid Labels Removed:** 0
- **Empty Text Strings Removed:** 0
- **Conflicting Label Verification:** Evaluated whether any identical message text mapped to contradictory labels (`ham` vs. `spam`). **0 conflicting instances were detected**.
- **Exact Duplicate Messages Removed:** 414 duplicate records were identified and deduplicated, retaining only the first unique occurrence.

### 2.3 Sample Counts and Class Distribution
After strict data validation and cleaning, the final dataset consists of **5,160 unique messages**:

| Class Label | Semantic Meaning | Number of Messages | Class Percentage |
| :---: | :---: | :---: | :---: |
| **0** | **Ham (Legitimate)** | 4,518 | 87.56% |
| **1** | **Spam (Unsolicited)** | 642 | 12.44% |
| **Total** | **All Unique Messages** | **5,160** | **100.00%** |

The class distribution exhibits a pronounced class imbalance ratio of approximately **7:1** (87.56% legitimate vs. 12.44% spam). A naïve majority-class dummy classifier would attain an 87.56% accuracy by labeling every message as ham while achieving 0.00% spam recall. Consequently, accuracy is insufficient on its own, and model evaluation must prioritize class-specific **Spam Precision, Spam Recall, and Spam F1-Score**.

### 2.4 Domain Scope and Limitations (SMS vs. Email)
> [!IMPORTANT]
> **Academic Disclosure on Dataset Domain:**
> While this assignment targets email spam classification, the available dataset comprises **SMS text messages**. SMS messages share the fundamental goal of text-based spam detection (filtering unsolicited or predatory communications based on textual cues). However, SMS messages have distinct characteristics:
> - **Message Length:** SMS messages are bounded and concise (often $\le 160$ characters).
> - **Linguistic Style:** Frequent use of mobile abbreviations, phonetic spelling (e.g., "ur", "txt", "fone"), phone numbers, and urgent calls-to-action.
> - **Absence of Email Infrastructure:** Genuine emails contain rich metadata, including RFC 822 / MIME transport headers, sender domain reputations, SPF/DKIM cryptographic signatures, routing hops, HTML DOM structures, and binary file attachments.
> 
> The text vectorization and GaussianNB classification pipeline built here applies identically to email bodies. However, real-world email filtering systems rely heavily on email-specific metadata alongside body text classification.

---

## 3. Text Preprocessing and Train/Test Splitting

### 3.1 Preprocessing Workflow
Text cleaning is deliberately restrained to prevent destroying critical predictive signals:
1. **Whitespace Normalization:** Tabs, newlines, and multiple consecutive whitespace characters are collapsed into single spaces.
2. **Missing Handling:** Null, non-string, or empty entries are detected and filtered.
3. **Case Normalization:** Carried out inside the feature extraction vectorizer (`lowercase=True`) to maintain uniform vocabulary tokens.
4. **Stop-Word Removal:** Evaluated empirically during cross-validation. Removing standard English stop words (`stop_words="english"`) prevents common grammatical tokens from adding high-variance noise across classes.

### 3.2 Leak-Free Partitioning Strategy
To eliminate data leakage, the dataset is split before feature extraction:
- **Split Ratio:** 80% Training ($N = 4,128$) and 20% Held-Out Testing ($N = 1,032$).
- **Stratification:** Stratified sampling (`stratify=df['label']`) is enforced to preserve the exact 87.56% / 12.44% class distribution across both partitions.
- **Random Seed:** Fixed seed (`random_state=42`) for full determinism and reproducibility.

```
Cleaned Dataset: 5,160 messages (4,518 Ham, 642 Spam)
                     │
                     ▼
       Stratified Split (80/20, seed=42)
        ┌────────────┴────────────┐
        ▼                         ▼
Training Set: 4,128 samples     Held-Out Test Set: 1,032 samples
(3,614 Ham, 514 Spam)           (904 Ham, 128 Spam)
        │                                 │
        ▼                                 │ (Strictly isolated;
5-Fold Stratified CV                      │  unseen during feature
(Hyperparameter tuning on train only)      │  and model tuning)
        │                                 │
        ▼                                 ▼
Fit Final Selected Model ─────────► Final Single Evaluation
```

---

## 4. Feature Extraction: Bag-of-Words and Numerical Representations

### 4.1 Bag-of-Words with `CountVectorizer`
Text documents are transformed into numerical feature vectors using the Bag-of-Words (BoW) model implemented via scikit-learn's `CountVectorizer`. For a vocabulary of size $V$, each message $j$ is represented as a vector:
$$\mathbf{x}_j = [x_{j,1}, x_{j,2}, \dots, x_{j,V}]$$
where $x_{j,i}$ is the frequency (occurrence count) of word $i$ in message $j$.

### 4.2 Hyperparameter Constraints
1. **Vocabulary Capacity (`max_features = 2000`):** Retains the 2,000 most informative word tokens by term frequency across the training corpus.
2. **Document Frequency Threshold (`min_df = 2`):** Prunes singleton words (*hapax legomena*) appearing in fewer than 2 messages, eliminating unique typographical errors.
3. **N-gram Range:** Evaluated unigrams `(1, 1)` versus unigrams + bigrams `(1, 2)`.

### 4.3 Dense Array Transformation and Memory Management
Scipy sparse matrices store only non-zero values. However, `sklearn.naive_bayes.GaussianNB` requires continuous mathematical inputs and rejects sparse inputs. The feature matrix must therefore be converted to a dense 2D array via `.toarray()`.

A naive dense matrix conversion of text can exhaust system memory. In this implementation, memory overhead is controlled:
$$\text{Memory Overhead} = N \times V \times \text{Bytes per Float}$$
By enforcing `max_features = 2000` and casting the matrix to 32-bit floating-point numbers (`np.float32`, 4 bytes):
$$\text{Training Memory} = 4,128 \times 2,000 \times 4\text{ bytes} \approx 33.02\text{ MB}$$
$$\text{Testing Memory} = 1,032 \times 2,000 \times 4\text{ bytes} \approx 8.26\text{ MB}$$
This keeps the entire memory footprint under 42 MB, allowing the system to run on commodity CPUs.

---

## 5. Theoretical Foundations of Gaussian Naïve Bayes

### 5.1 Bayes' Theorem
Bayes' theorem provides the principled probabilistic framework for updating the probability of an event given prior knowledge and observed evidence:
$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

#### Numerical Worked Example
Consider a concrete diagnostic scenario:
- Prior probability of spam: $P(A) = 0.20$
- Prior probability of legitimate ham: $P(\neg A) = 1 - 0.20 = 0.80$
- Likelihood that a message contains keyword $B$ ("WINNER") given it is spam: $P(B \mid A) = 0.85$
- Likelihood that keyword $B$ appears in legitimate ham: $P(B \mid \neg A) = 0.10$

By the Law of Total Probability:
$$P(B) = P(B \cap A) + P(B \cap \neg A) = P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)$$
$$P(B \cap A) = 0.85 \times 0.20 = 0.17$$
$$P(B \cap \neg A) = 0.10 \times 0.80 = 0.08$$
$$P(B) = 0.17 + 0.08 = 0.25$$

Applying Bayes' theorem:
$$P(A \mid B) = \frac{0.85 \times 0.20}{0.25} = \frac{0.17}{0.25} = 0.68$$
Upon observing keyword $B$, the posterior probability that the message is spam increases from the 20% prior to **68%**.

### 5.2 The Naïve Conditional Independence Assumption
In text classification with a feature vector $\mathbf{x} = [x_1, x_2, \dots, x_D]$, the joint likelihood $P(\mathbf{x} \mid C)$ is intractable due to token dependencies. The Naïve Bayes model simplifies this by assuming that all features are **conditionally independent given the class label $C$**:
$$P(\mathbf{x} \mid C) = \prod_{i=1}^D P(x_i \mid C)$$

The posterior class probability is expressed as:
$$P(C \mid \mathbf{x}) \propto P(C) \prod_{i=1}^D P(x_i \mid C)$$
Classification follows the Maximum A Posteriori (MAP) decision rule:
$$\hat{y} = \arg\max_{C \in \{0, 1\}} \left[ \ln P(C) + \sum_{i=1}^D \ln P(x_i \mid C) \right]$$

### 5.3 The Class-Conditional Gaussian Assumption
While Multinomial Naïve Bayes models discrete counts using multinomial likelihoods, **Gaussian Naïve Bayes** models each feature $x_i$ as a continuous random variable drawn from a univariate Gaussian (normal) distribution:
$$P(x_i \mid C) = \frac{1}{\sqrt{2\pi \sigma_{C,i}^2}} \exp\left( -\frac{(x_i - \mu_{C,i})^2}{2\sigma_{C,i}^2} \right)$$
where $\mu_{C,i}$ is the empirical sample mean and $\sigma_{C,i}^2$ is the sample variance of feature $i$ conditioned on class $C$.

### 5.4 The Zero-Variance Trap and Variance Smoothing Regularization
In text feature representations, word count matrices are sparse ($\sim 99\%$ zeros). For a word $i$ that appears rarely in training ham messages, its sample variance $\sigma_{\text{ham}, i}^2$ approaches zero:
$$\sigma_{\text{ham}, i}^2 \to 0$$
Under scikit-learn's default setting (`var_smoothing = 1e-9`), the variance denominator remains infinitesimally small. If a test ham message contains even one occurrence of this word ($x_i = 1$), the standardized z-score:
$$z = \frac{1 - \mu_{\text{ham}, i}}{\sigma_{\text{ham}, i}}$$
explodes (often $z > 50$), causing $P(x_i \mid \text{ham})$ to underflow toward zero ($< 10^{-500}$). This single feature dominates the sum of log-likelihoods, forcing the model to falsely classify legitimate messages as spam (producing 165 False Positives and a low 41.49% precision in the unregularized baseline).

To solve this, **variance smoothing regularization** adds a fraction $\epsilon$ of the maximum feature variance across the dataset to all empirical variances:
$$\sigma_{C,i,\text{smoothed}}^2 = \sigma_{C,i}^2 + \epsilon \cdot \max_{j}(\sigma_j^2)$$
Setting `var_smoothing = 0.05` establishes a stable lower variance floor, preventing near-zero variance collapse, neutralizing single-token penalties, and restoring balanced multi-feature probabilistic decisions.

---

## 6. Software Architecture and Pipeline Implementation

The codebase is organized into modular, testable components:
- `data_loader.py`: Dataset discovery, reading, malformed line tracking, conflict detection, deduplication, and auditing.
- `preprocessing.py`: Text cleaning, `CountVectorizer` configuration, and `FunctionTransformer`-based dense float32 conversion.
- `model.py`: GaussianNB model initialization, variance smoothing parameterization, `Pipeline` integration, and serialization.
- `evaluate.py`: Performance metrics computation, confusion matrix plotting, class distribution graphing, and qualitative error extraction.
- `main.py`: Orchestrator managing stratified splitting, 5-fold cross-validation on training data, final model fitting, test evaluation, and artifact export.
- `predict.py`: User-facing interactive and command-line inference script with calibrated probability formatting.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Leak-free end-to-end pipeline
pipeline = Pipeline([
    ("vectorizer", CountVectorizer(lowercase=True, stop_words="english", max_features=2000, min_df=2)),
    ("to_dense", FunctionTransformer(lambda m: m.toarray().astype(np.float32), accept_sparse=True, validate=False)),
    ("classifier", GaussianNB(var_smoothing=0.05)),
])
```

---

## 7. Systematic Experiments: 5-Fold Stratified Cross-Validation

To ensure that the test set remains an independent held-out evaluation set, **all hyperparameter tuning, smoothing sweeps, vocabulary experiments, and feature comparisons were conducted exclusively on the 4,128 training samples using 5-Fold Stratified Cross-Validation**.

The cross-validation results (`outputs/cv_experiments.csv`) across all 12 controlled configurations are reported below:

| # | Experiment Configuration | Category | `var_smoothing` | CV Accuracy (Mean ± Std) | CV Precision (Mean ± Std) | CV Recall (Mean ± Std) | CV F1-Score (Mean ± Std) | Description |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline (vs=1e-9) | Baseline | $1.0 \times 10^{-9}$ | $0.8440 \pm 0.0064$ | $0.4392 \pm 0.0113$ | $0.9125 \pm 0.0162$ | $0.5930 \pm 0.0136$ | Unregularized default GaussianNB |
| **2** | Smoothing vs=1e-5 | Variance Sweep | $1.0 \times 10^{-5}$ | $0.8440 \pm 0.0064$ | $0.4392 \pm 0.0113$ | $0.9125 \pm 0.0162$ | $0.5930 \pm 0.0136$ | Insufficient smoothing |
| **3** | Smoothing vs=1e-2 | Variance Sweep | $1.0 \times 10^{-2}$ | $0.9310 \pm 0.0260$ | $0.6652 \pm 0.0893$ | $0.9455 \pm 0.0190$ | $0.7782 \pm 0.0676$ | Moderate smoothing |
| **4** | **Smoothing vs=0.05** | **Variance Sweep** | **$5.0 \times 10^{-2}$** | **$0.9835 \pm 0.0056$** | **$0.9552 \pm 0.0221$** | **$0.9105 \pm 0.0291$** | **$0.9322 \pm 0.0231$** | **Optimal balanced regularization** |
| **5** | Smoothing vs=0.08 | Variance Sweep | $8.0 \times 10^{-2}$ | $0.9830 \pm 0.0047$ | $0.9804 \pm 0.0128$ | $0.8814 \pm 0.0289$ | $0.9281 \pm 0.0205$ | High precision, lower recall |
| **6** | Vocab max_features=1000 | Vocab Capacity | $5.0 \times 10^{-2}$ | $0.9765 \pm 0.0060$ | $0.8863 \pm 0.0317$ | $0.9319 \pm 0.0162$ | $0.9083 \pm 0.0223$ | Underfitting vocabulary |
| **7** | Vocab max_features=3000 | Vocab Capacity | $5.0 \times 10^{-2}$ | $0.9838 \pm 0.0055$ | $0.9666 \pm 0.0205$ | $0.9008 \pm 0.0269$ | $0.9325 \pm 0.0230$ | Marginal gain (+0.03% F1) at +50% RAM |
| **8** | Min_df=1 (no pruning) | Document Freq | $5.0 \times 10^{-2}$ | $0.9838 \pm 0.0054$ | $0.9557 \pm 0.0267$ | $0.9125 \pm 0.0275$ | $0.9333 \pm 0.0222$ | Retains rare unique words |
| **9** | Min_df=3 (rare pruned) | Document Freq | $5.0 \times 10^{-2}$ | $0.9835 \pm 0.0050$ | $0.9498 \pm 0.0218$ | $0.9164 \pm 0.0218$ | $0.9327 \pm 0.0203$ | Prunes terms appearing $<3$ times |
| **10** | N-grams (1, 2) | N-gram Range | $5.0 \times 10^{-2}$ | $0.9809 \pm 0.0033$ | $0.9888 \pm 0.0101$ | $0.8560 \pm 0.0206$ | $0.9175 \pm 0.0148$ | Sparse bigrams degrade recall (-5.45%) |
| **11** | No Stop Words | Stop-Words | $5.0 \times 10^{-2}$ | $0.9685 \pm 0.0056$ | $0.9921 \pm 0.0065$ | $0.7530 \pm 0.0411$ | $0.8556 \pm 0.0287$ | High-frequency noise degrades recall (-15.75%) |
| **12** | TF-IDF Representation | Feature Type | $2.0 \times 10^{0}$ | $0.9714 \pm 0.0064$ | $0.8692 \pm 0.0343$ | $0.9086 \pm 0.0226$ | $0.8881 \pm 0.0240$ | Scaled vs required for $[0,1]$ features |

### Key Experimental Insights:
1. **Variance Smoothing is Decisive:** Shifting `var_smoothing` from $10^{-9}$ to $0.05$ boosts CV F1-Score from **59.30% to 93.22%** and precision from **43.92% to 95.52%**.
2. **Stop-Word Removal is Essential:** Leaving English stop words in the vocabulary degrades CV recall from $91.05\%$ down to $75.30\%$, because non-informative words introduce high variance across both classes.
3. **Unigrams Outperform Bigrams:** Adding bigrams expands feature sparsity, dropping CV recall from $91.05\%$ to $85.60\%$.
4. **Optimal Vocabulary Selection:** A 2,000-word vocabulary with `min_df = 2` achieves equivalent statistical performance to a 3,000-word vocabulary ($93.22\%$ vs. $93.25\%$ F1) while reducing memory consumption by 33%. Configuration #4 is therefore selected as the final production model.

---

## 8. Final Held-Out Test Set Evaluation

The final selected GaussianNB model (`CountVectorizer(max_features=2000, min_df=2, stop_words="english")` with `var_smoothing=0.05`) was refitted on the full 4,128 training samples and evaluated **once** on the untouched held-out test partition ($N = 1,032$). For direct before/after contrast, the unregularized baseline model was also evaluated under identical conditions (`outputs/test_comparison.csv`).

### 8.1 Performance Comparison Table

| Metric | Baseline Model (`var_smoothing=1e-9`) | Final Regularized Model (`var_smoothing=0.05`) | Absolute Delta ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 82.95% | **98.45%** | **+15.50%** | +18.69% |
| **Spam Precision** | 41.49% | **95.16%** | **+53.67%** | +129.36% |
| **Spam Recall** | 91.41% | **92.19%** | **+0.78%** | +0.85% |
| **Spam F1-Score** | 57.07% | **93.65%** | **+36.58%** | +64.10% |
| **False Positive Rate** | 18.25% (165 FP) | **0.66% (6 FP)** | **-17.59%** | **-96.36% (FP dropped)** |
| **False Negatives** | 11 FN | **10 FN** | **-1 FN** | -9.09% |

### 8.2 Confusion Matrix Breakdown
The held-out test set contains **904 legitimate Ham messages** and **128 Spam messages**:

```
UNREGULARIZED BASELINE (var_smoothing = 1e-9):
                     Predicted Ham    Predicted Spam
Actual Ham (904) :        739              165       (FP = 165, FPR = 18.25%)
Actual Spam (128):         11              117       (FN = 11, Recall = 91.41%)

FINAL REGULARIZED MODEL (var_smoothing = 0.05):
                     Predicted Ham    Predicted Spam
Actual Ham (904) :        898                6       (FP = 6, FPR = 0.66%)
Actual Spam (128):         10              118       (FN = 10, Recall = 92.19%)
```

- **True Negatives ($TN = 898$):** 99.34% of legitimate messages correctly delivered to the inbox.
- **False Positives ($FP = 6$):** Only 6 legitimate messages misrouted to spam (down from 165).
- **False Negatives ($FN = 10$):** Only 10 spam messages escaped detection.
- **True Positives ($TP = 118$):** 92.19% of all spam messages accurately intercepted.

### 8.3 Classification Report
```
              precision    recall  f1-score   support

         Ham       0.99      0.99      0.99       904
        Spam       0.95      0.92      0.94       128

    accuracy                           0.98      1032
   macro avg       0.97      0.96      0.96      1032
weighted avg       0.98      0.98      0.98      1032
```

---

## 9. Discussion and Error Analysis

### 9.1 Qualitative Error Analysis (Test Set Dissections)
From the exported test error audit (`outputs/error_analysis.csv`), we examine the mechanisms behind the remaining 6 False Positives and 10 False Negatives:

#### Analysis of False Positives ($FP = 6$):
1. *"Save yourself the stress. If the person has a dorm account, just send your account details and the money will be sent to you."*
   - **Reason:** Contains dense financial terminology (`account`, `account details`, `money will be sent`), which strongly correlates with financial advance-fee fraud in the training corpus.
2. *"Please protect yourself from e-threats. SIB never asks for sensitive information like Passwords,ATM/SMS PIN thru email. Never share your password with anybody."*
   - **Reason:** A genuine bank security advisory warning users against phishing. Because it explicitly mentions phishing keywords (`Passwords`, `ATM/SMS PIN`, `sensitive information`, `email`), the class-conditional Gaussian likelihood for spam overwhelmed the ham likelihood.
3. *"I (Career Tel) have added u as a contact on INDYAROCKS.COM to send FREE SMS. To remove from phonebook - sms NO to <#>"*
   - **Reason:** Automated invitation containing promotional vocabulary (`FREE SMS`, `contact`, opt-out text).

#### Analysis of False Negatives ($FN = 10$):
1. *"Sorry I missed your call let's talk when you have the time. I'm on 07090201529"*
   - **Reason:** Clever conversational pretext masking a premium-rate callback phone number. The message text is composed almost entirely of common conversational ham words (`sorry`, `missed`, `call`, `talk`, `time`).
2. *"Babe: U want me dont u baby! Im nasty and have a thing 4 filthyguys. Fancy a rude time with a sexy bitch. How about we go slo n hard! Txt XXX SLO(4msgs)"*
   - **Reason:** Adult spam characterized by non-standard colloquial spellings and intentional slang (`dont`, `u`, `slo n hard`, `4msgs`) that bypassed the top 2,000 standard vocabulary tokens.
3. *"Block Breaker now comes in deluxe format with new features and great graphics from T-Mobile. Buy for just £5 by replying GET BBDELUXE and take the challenge"*
   - **Reason:** Uses non-ASCII currency symbols (`£`) and compound keywords (`BBDELUXE`) absent from the unigram training dictionary.

### 9.2 The Precision-Recall Asymmetry in Spam Classification
In spam filtering, **False Positives are substantially more costly than False Negatives**:
- A False Negative means a nuisance spam message reaches the inbox, which a user can easily delete.
- A False Positive means a critical message (e.g., job offer, medical update, password reset, or client communication) is misfiled into the spam folder, potentially causing significant personal or financial harm.

The unregularized baseline model's 18.25% False Positive Rate (165 legitimate messages lost) is unacceptable for production use despite its 91.41% recall. By tuning `var_smoothing = 0.05`, the False Positive Rate drops to **0.66% (6 FP)**, representing a **96.4% reduction in lost legitimate communications** while simultaneously increasing recall to **92.19%**.

### 9.3 Critique of the Gaussian Distributional Assumption on Text
Gaussian Naïve Bayes makes two strong modeling assumptions:
1. **Conditional Independence:** Words are assumed to occur independently given the class. In reality, text exhibits strong syntactic and semantic collocations (e.g., "credit" and "card", "free" and "entry").
2. **Continuous Normality:** Word frequencies are modeled as continuous variables following a symmetric Gaussian bell curve. In practice, bag-of-words features are discrete integers, non-negative, heavily zero-inflated, and positively skewed.

Despite these assumption violations, GaussianNB functions effectively as a linear decision boundary classifier in log-space. When variance smoothing regularizes low-frequency feature variances, the model achieves **98.45% accuracy** and an **F1-score of 93.65%**, confirming that strict normality is not required for robust discriminatory performance.

### 9.4 Generalization to Real-World Email Systems
Applying this model to enterprise email systems requires addressing key domain differences:
- **Feature Diversity:** Email filters combine body text classification with structural header inspection (SPF, DKIM, DMARC), IP address reputation lookups, MIME attachment scanning, and URL domain age analysis.
- **Adversarial Drift:** Spammers frequently employ homoglyphs, zero-width spaces, and image-based text to evade keyword vectorizers.
- **Computational Efficiency:** While MultinomialNB supports sparse computations, GaussianNB requires dense arrays. For vocabularies $>50,000$ tokens, memory requirements scale significantly, making vocabulary pruning (`max_features`) essential for production deployment.

---

## 10. Conclusion and Future Work

### 10.1 Summary of Accomplishments
1. **Assignment Requirements Satisfied:** Built a leak-free, reproducible spam classification system using `sklearn.naive_bayes.GaussianNB` without substituting alternative algorithms.
2. **Methodological Rigor:** Eliminated test set contamination by performing all hyperparameter tuning using **5-Fold Stratified Cross-Validation on training data only**, reserving the test set for a single evaluation.
3. **Variance Collapse Resolved:** Demonstrated mathematically and empirically that variance smoothing (`var_smoothing = 0.05`) resolves near-zero variance collapse, raising Spam Precision from 41.49% to **95.16%** and cutting False Positives from 165 to **6**.
4. **Reproducible Pipeline:** Packaged the system into a scikit-learn `Pipeline` and verified end-to-end execution across CLI and interactive modes.

### 10.2 Future Directions
1. **Sub-word Tokenization:** Integrate Byte-Pair Encoding (BPE) or character n-grams to mitigate out-of-vocabulary evasion and phonetic slang.
2. **Cost-Sensitive Decision Thresholds:** Implement an adjustable classification threshold $\tau > 0.5$ on $P(\text{Spam} \mid \mathbf{x})$ to allow system administrators to further minimize false positives in zero-tolerance enterprise environments.
3. **MIME Feature Integration:** Extend the input pipeline to parse email headers, attachment file types, and HTML structure.

---

## 11. References
1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng)*, 259–262.
2. UCI Machine Learning Repository: *SMS Spam Collection Data Set*. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
