# Lab3: SMS Spam Classification using Naive Bayes

## Overview
This lab implements a complete SMS spam classification pipeline using Naive Bayes and the Bag-of-Words model. The classifier learns from labeled SMS messages to distinguish between legitimate (ham) and spam messages.

## Requirements (from lab3.pptx)
1. **Data Loading**: Read and parse SMS dataset (spam.csv or SMSSpamCollection format)
2. **Data Splitting**: Train/test split (80/20) with stratified sampling
3. **Feature Representation**: Bag of Words using CountVectorizer (term frequency vectors)
4. **Classification**: Multinomial Naive Bayes with Laplace smoothing (alpha=1)
5. **Evaluation**: Compute confusion matrix, accuracy, precision, recall, F1-score

## Implementation Details

### Algorithm
- **Naive Bayes**: Uses multinomial distribution (counts of word occurrences)
- **Bag of Words**: Each SMS is represented as a vector of word frequencies
- **Laplace Smoothing**: Adds smoothing (alpha=1) to handle zero probabilities

### File Structure
```
lab3/
├── code/
│   ├── run_lab3.py          # Main implementation
│   └── requirements.txt     # Optional dependencies (for sklearn/pandas)
├── SMSSpamCollection        # Sample dataset (tab-separated)
├── evaluation_output.txt    # Sample evaluation results
└── README_lab3.txt         # This file
```

## Usage

### Quick Run (with sample dataset)
```bash
cd lab3
python3 code/run_lab3.py
```

### With custom dataset
```bash
python3 code/run_lab3.py --data /path/to/spam.csv
```

### Features
- **Auto-download**: Script attempts to download dataset from GitHub if local copy not found
- **Flexible parsing**: Supports both tab-separated and comma-separated files
- **Fallback implementation**: If sklearn/pandas unavailable, uses pure-Python implementations
- **Stratified sampling**: Preserves ham/spam ratio in train/test splits

## Evaluation Results

### Sample Run Output
```
Using dataset: /home/lht/dev/study/ai/lab3/SMSSpamCollection

=== Evaluation on test set ===
Accuracy: 1.0

Confusion Matrix (rows=true, cols=pred) labels=
 ['ham', 'spam']
[2, 0]
[0, 1]

Classification Report:
label	precision	recall	f1	support
ham	1.000	1.000	1.000	2
spam	1.000	1.000	1.000	1
```

### Interpretation
- **Accuracy**: 100% (1.0) - All predictions are correct on test set
- **Confusion Matrix**: 
  - Row 0 (ham): 2 correct, 0 misclassified as spam
  - Row 1 (spam): 0 misclassified as ham, 1 correct
- **Per-class metrics**: Both classes have perfect precision, recall, and F1

### Discussion: Impact of Low Spam Recall
If spam recall were low (missing many spam messages), the real-world impact would be:
- **User experience**: Harmful spam messages reach users' inboxes
- **Security risk**: Phishing and scam messages are not caught
- **Business impact**: User trust decreases, more complaints about spam filtering

This underscores why recall for spam detection is critical in production systems.

## Dependencies (Optional)
- `pandas`: For efficient data loading
- `scikit-learn`: For vectorization and classification

If not available, the script uses pure-Python fallbacks.

## Notes
- Sample dataset includes 20 SMS messages for demonstration
- Real evaluation should use the full SMS Spam Collection dataset (~5574 messages)
- Random seed (42) ensures reproducible train/test splits
- Supports binary classification (ham/spam) only
