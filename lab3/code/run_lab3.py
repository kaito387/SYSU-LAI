#!/usr/bin/env python3
"""Lab3: SMS spam classification using Naive Bayes
This script prefers sklearn/pandas if available; otherwise falls back to pure-Python implementations.
"""
import argparse
import os
import sys
import tempfile
import urllib.request
import csv
import re
from collections import defaultdict, Counter
import math

# Try imports
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def download_try(urls, dest):
    for url in urls:
        try:
            print(f"Downloading from {url} ...")
            urllib.request.urlretrieve(url, dest)
            print("Downloaded to", dest)
            return True
        except Exception as e:
            print("Failed to download from", url, "->", e)
    return False


def parse_plain_file(path):
    # supports SMSSpamCollection (tab) or csv with first column label and second text
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
        # heuristic
        sep = '\t' if '\t' in first else ','
        # reset
        f.seek(0)
        reader = csv.reader(f, delimiter=sep)
        for r in reader:
            if not r: continue
            if len(r) >= 2:
                label = r[0]
                text = ','.join(r[1:]) if sep==',' else r[1]
                rows.append((label, text))
    return rows


def load_dataset_fallback(path):
    # try pandas if available
    if HAS_SKLEARN:
        # use original logic
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(path, sep=sep, header=None, engine='python')
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ['label', 'text']
                    return df
            except Exception:
                continue
        try:
            df = pd.read_csv(path, header=0)
            if 'label' in df.columns and 'text' in df.columns:
                return df[['label','text']]
        except Exception:
            pass
        raise ValueError('Unable to parse dataset at '+path)
    else:
        rows = parse_plain_file(path)
        if not rows:
            raise ValueError('Unable to parse dataset at '+path)
        labels = [r[0].strip().lower() for r in rows]
        texts = [r[1] for r in rows]
        return labels, texts


def simple_tokenize(s):
    return re.findall(r"\w+", s.lower())

class SimpleCountVectorizer:
    def fit(self, docs):
        self.vocab_ = {}
        for d in docs:
            for tok in set(simple_tokenize(d)):
                if tok not in self.vocab_:
                    self.vocab_[tok] = len(self.vocab_)
        return self
    def transform(self, docs):
        rows = []
        for d in docs:
            vec = [0]*len(self.vocab_)
            for tok in simple_tokenize(d):
                if tok in self.vocab_:
                    vec[self.vocab_[tok]] += 1
            rows.append(vec)
        return rows
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

class SimpleMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        # X: list of count vectors
        labels = list(set(y))
        self.classes_ = labels
        self.class_count_ = {c:0 for c in labels}
        V = len(X[0]) if X else 0
        self.feature_count_ = {c:[0]*V for c in labels}
        for xi, yi in zip(X,y):
            self.class_count_[yi] += 1
            for i,val in enumerate(xi):
                self.feature_count_[yi][i] += val
        # compute log priors and log probs
        total = sum(self.class_count_.values())
        self.class_log_prior_ = {c:math.log(self.class_count_[c]/total) for c in labels}
        self.feature_log_prob_ = {}
        for c in labels:
            fc = self.feature_count_[c]
            sm = sum(fc)
            denom = sm + self.alpha * V
            self.feature_log_prob_[c] = [math.log((fc_i + self.alpha)/denom) for fc_i in fc]
        return self
    def predict(self, X):
        preds = []
        for xi in X:
            best = None
            best_score = None
            for c in self.classes_:
                score = self.class_log_prior_[c]
                probs = self.feature_log_prob_[c]
                for i,val in enumerate(xi):
                    if val:
                        score += val * probs[i]
                if best is None or score>best_score:
                    best = c; best_score = score
            preds.append(best)
        return preds


def train_test_split_fallback(X, y, test_size=0.2, random_state=42, stratify=None):
    # simple stratified split
    import random
    rng = random.Random(random_state)
    data_by_label = defaultdict(list)
    for xi, yi in zip(X,y):
        data_by_label[yi].append(xi)
    X_train=[]; X_test=[]; y_train=[]; y_test=[]
    for label, items in data_by_label.items():
        n = len(items)
        k = max(1, int(n * test_size))
        idx = list(range(n))
        rng.shuffle(idx)
        test_idx = set(idx[:k])
        for i in range(n):
            if i in test_idx:
                X_test.append(items[i]); y_test.append(label)
            else:
                X_train.append(items[i]); y_train.append(label)
    return X_train, X_test, y_train, y_test


def accuracy_score_fallback(y_true, y_pred):
    correct = sum(1 for a,b in zip(y_true,y_pred) if a==b)
    return correct/len(y_true) if y_true else 0.0

def confusion_matrix_fallback(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true)|set(y_pred)))
    idx = {l:i for i,l in enumerate(labels)}
    M = [[0]*len(labels) for _ in labels]
    for a,b in zip(y_true,y_pred):
        M[idx[a]][idx[b]] += 1
    return M, labels

def classification_report_fallback(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true)|set(y_pred)))
    idx = {l:i for i,l in enumerate(labels)}
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    for a,b in zip(y_true,y_pred):
        support[a]+=1
        if a==b:
            tp[a]+=1
        else:
            fp[b]+=1
            fn[a]+=1
    lines = []
    for l in labels:
        p = tp[l]/(tp[l]+fp[l]) if (tp[l]+fp[l])>0 else 0.0
        r = tp[l]/support[l] if support[l]>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        lines.append((l, p, r, f1, support[l]))
    return lines


def print_report(labels_metrics):
    print('label\tprecision\trecall\tf1\tsupport')
    for l,p,r,f1,s in labels_metrics:
        print(f"{l}\t{p:.3f}\t{r:.3f}\t{f1:.3f}\t{s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', help='Path to spam dataset (csv/tsv). If omitted, looks for lab3/spam.csv or downloads known mirrors.')
    args = p.parse_args()

    candidates = []
    if args.data:
        candidates.append(args.data)
    candidates.extend([
        '/home/lht/dev/study/ai/lab3/spam.csv',
        '/home/lht/dev/study/ai/lab3/SMSSpamCollection',
        './lab3/spam.csv',
        './lab3/SMSSpamCollection'
    ])

    urls = [
        'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv',
        'https://raw.githubusercontent.com/epfml/ML_course/master/labs/ex11_text_classification/data/smsspamcollection/SMSSpamCollection',
        'https://raw.githubusercontent.com/abkonochin/smsspamcollection/master/SMSSpamCollection'
    ]

    data_path = None
    for c in candidates:
        if c and os.path.exists(c):
            data_path = c
            print('Using dataset:', data_path)
            break
    if data_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.tsv')
        tmp.close()
        ok = download_try(urls, tmp.name)
        if ok:
            data_path = tmp.name
        else:
            print('Failed to obtain dataset. Please place spam.csv (or SMSSpamCollection) in lab3/ and rerun.')
            sys.exit(2)

    if HAS_SKLEARN:
        df = load_dataset_fallback(data_path)
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip()!='']
        X = df['text'].values
        y = df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

        vec = CountVectorizer()
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        print('\n=== Evaluation on test set ===')
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('\nConfusion Matrix (rows=true, cols=pred):')
        print(confusion_matrix(y_test, y_pred))
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
    else:
        labels, texts = load_dataset_fallback(data_path)
        # normalize
        y = [l.strip().lower() for l in labels]
        X = texts
        X_train, X_test, y_train, y_test = train_test_split_fallback(X,y,test_size=0.2,random_state=42,stratify=y)

        vec = SimpleCountVectorizer()
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        clf = SimpleMultinomialNB()
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        print('\n=== Evaluation on test set ===')
        print('Accuracy:', accuracy_score_fallback(y_test, y_pred))
        M, labs = confusion_matrix_fallback(y_test, y_pred)
        print('\nConfusion Matrix (rows=true, cols=pred) labels=\n', labs)
        for row in M:
            print(row)
        print('\nClassification Report:')
        rep = classification_report_fallback(y_test, y_pred, labels=labs)
        print_report(rep)

if __name__ == '__main__':
    main()
