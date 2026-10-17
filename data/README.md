# Dataset Folder

This directory contains the dataset files for the email spam classifier.

**Active Dataset: UCI Spambase (Email Dataset)**
- **Source:** UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/94/spambase)
- **Files:**
  - `spambase.data`: 4,601 email instances with 57 continuous numeric features and 1 binary label.
  - `spambase.names`: Attribute documentation and feature names.
  - `spambase.DOCUMENTATION`: Dataset background and source notes from Hewlett-Packard Laboratories.

Format of `spambase.data`: Comma-separated values, 58 columns per row.
- Columns 0–56: 57 continuous features (48 word frequencies, 6 character frequencies, 3 capital run-length metrics).
- Column 57: Binary class label (`1` = spam, `0` = not spam).
