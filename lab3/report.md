---
title: "人工智能实验报告"
subtitle: "中山大学计算机学院本科生实验报告"
author:
  - "课程名称：Artificial Intelligence"
  - "学号：24344064"
  - "姓名：廖海涛"
date: ""
---

# 实验题目

**Lab3：基于朴素贝叶斯的短信垃圾分类**

本实验实现一个基于多项式朴素贝叶斯（Multinomial Naive Bayes）和词袋模型（Bag of Words）的短信垃圾分类器，能够从已标注的 SMS 数据中学习，自动区分正常短信（ham）和垃圾短信（spam）。

---

# 实验内容

## 1. 算法原理

### 1.1 整体流程

1. **数据加载**：读取 SMS Spam Collection 数据集，提取短信文本和对应标签（ham/spam）
2. **训练/测试集划分**：使用分层抽样将数据集划分为训练集（80%）和测试集（20%），保证两者中 ham/spam 的比例一致
3. **文本向量化**：使用词袋模型将每条短信转换为一个固定长度的词频向量
4. **模型训练**：使用多项式朴素贝叶斯分类器进行训练，采用拉普拉斯平滑（$\alpha=1$）处理零概率问题
5. **预测 & 评估**：在测试集上进行预测，计算准确率、混淆矩阵、精确率、召回率和 F1 分数

### 1.2 文本向量化：词袋模型 (Bag of Words)

词袋模型将每条短信转换为一个**固定长度的数值向量**，向量的每一维对应词汇表中的一个词，值为该词在这条短信中出现的次数。

设词汇表为 $V = \{w_1, w_2, \dots, w_{|V|}\}$（所有短信中出现过的去重词汇）。对第 $i$ 条短信 $d_i$，其向量表示为：

$$\mathbf{x}^{(i)} = [c_{i1}, c_{i2}, \dots, c_{i|V|}]$$

其中 $c_{ij}$ = 词 $w_j$ 在短信 $d_i$ 中出现的次数。

**举例**：

| 短信 | go | until | free | win |
|------|-----|-------|------|-----|
| "go go until" | 2 | 1 | 0 | 0 |
| "free win free" | 0 | 0 | 2 | 1 |

每条短信从变长文本变成了等长向量，可以被数学模型直接处理。

### 1.3 朴素贝叶斯分类器 (Naive Bayes)

**贝叶斯定理**：给定一条短信 $\mathbf{x}$，判断它属于哪个类别 $y \in \{\text{ham}, \text{spam}\}$：

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}$$

- $P(y \mid \mathbf{x})$：**后验概率** — 看到短信后，它是 ham/spam 的概率
- $P(y)$：**先验概率** — 训练集中 ham/spam 各自的比例
- $P(\mathbf{x} \mid y)$：**似然** — 在已知类别下，出现该短信内容的概率
- $P(\mathbf{x})$：**证据** — 对所有类别相同，可忽略

**朴素假设 (Naive Assumption)**：假设所有词之间**条件独立**（给定类别后，每个词出现与否互不影响）：

$$P(\mathbf{x} \mid y) = \prod_{j=1}^{|V|} P(w_j \mid y)^{c_j}$$

**多项式朴素贝叶斯 (Multinomial NB)**：使用多项分布建模，每个词的出现次数 $c_j$ 服从多项分布：

$$P(w_j \mid y) = \frac{\text{类别 } y \text{ 中词 } w_j \text{ 的总出现次数}}{\text{类别 } y \text{ 中所有词的总出现次数}} = \frac{N_{yj}}{\sum_k N_{yk}}$$

**分类决策**：

$$\hat{y} = \arg\max_{y} \; P(y) \cdot \prod_{j=1}^{|V|} P(w_j \mid y)^{c_j}$$

### 1.4 拉普拉斯平滑 (Laplace Smoothing)

**问题**：如果词 $w_j$ 在训练集的 spam 类别中从未出现，则 $P(w_j \mid \text{spam}) = 0$，导致整个后验概率变为 0 —— 一条短信只要包含任何一个"没见过的词"就会被判为概率为 0。

**解决方案**：给每个计数加上一个伪计数 $\alpha$：

$$P(w_j \mid y) = \frac{N_{yj} + \alpha}{\sum_k N_{yk} + \alpha \cdot |V|}$$

- $\alpha = 1$ 即为拉普拉斯平滑
- 分子 $+\alpha$：假装每个词至少出现了 $\alpha$ 次
- 分母 $+\alpha \cdot |V|$：保证概率之和仍为 1

### 1.5 对数空间计算 (Log-Space)

直接连乘很多小于 1 的概率会导致**数值下溢**（浮点数精度不够，结果变成 0）。利用 $\log(a \cdot b) = \log a + \log b$ 将乘法转为加法：

$$\log P(y \mid \mathbf{x}) \propto \log P(y) + \sum_{j=1}^{|V|} c_j \cdot \log P(w_j \mid y)$$

### 1.7 评估指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **准确率** Accuracy | $\dfrac{TP + TN}{TP + TN + FP + FN}$ | 预测正确的比例 |
| **精确率** Precision | $\dfrac{TP}{TP + FP}$ | 预测为 spam 的里面，多少真的是 spam |
| **召回率** Recall | $\dfrac{TP}{TP + FN}$ | 真正的 spam 中，有多少被找出来了 |
| **F1 分数** | $\dfrac{2 \cdot P \cdot R}{P + R}$ | 精确率和召回率的调和平均 |

### 1.8 完整数学流程总结

1. **词汇表构建**：从训练短信中提取所有去重词 → $V = \{w_1, \dots, w_{|V|}\}$
2. **文本向量化**：每条短信 → 词频向量 $\mathbf{x} = [c_1, \dots, c_{|V|}]$
3. **计算先验概率（对数）**：$\log P(y) = \log\frac{\text{count}(y)}{\text{total}}$
4. **计算条件概率（对数+平滑）**：$\log P(w_j \mid y) = \log\frac{N_{yj} + \alpha}{\sum_k N_{yk} + \alpha|V|}$
5. **预测新短信**：$\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{j: c_j > 0} c_j \cdot \log P(w_j \mid y) \right]$
6. **评估**：在保留的测试集上计算准确率、混淆矩阵、精确率/召回率/F1

## 2. 关键代码展示

### 2.1 自实现朴素贝叶斯分类器

```python
class SimpleMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        labels = list(set(y))
        self.classes_ = labels
        self.class_count_ = {c: 0 for c in labels}
        V = len(X[0]) if X else 0
        self.feature_count_ = {c: [0] * V for c in labels}

        for xi, yi in zip(X, y):
            self.class_count_[yi] += 1
            for i, val in enumerate(xi):
                self.feature_count_[yi][i] += val

        # 计算对数先验概率
        total = sum(self.class_count_.values())
        self.class_log_prior_ = {
            c: math.log(self.class_count_[c] / total) for c in labels
        }

        # 计算对数条件概率（含拉普拉斯平滑）
        self.feature_log_prob_ = {}
        for c in labels:
            fc = self.feature_count_[c]
            sm = sum(fc)
            denom = sm + self.alpha * V          # 分母加平滑
            self.feature_log_prob_[c] = [
                math.log((fc_i + self.alpha) / denom)
                for fc_i in fc
            ]
        return self

    def predict(self, X):
        preds = []
        for xi in X:
            best, best_score = None, None
            for c in self.classes_:
                score = self.class_log_prior_[c]  # log P(y)
                probs = self.feature_log_prob_[c]
                for i, val in enumerate(xi):
                    if val:
                        score += val * probs[i]    # val * log P(w_i|y)
                if best is None or score > best_score:
                    best, best_score = c, score
            preds.append(best)
        return preds
```

### 2.2 自实现词袋向量化器

```python
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
            vec = [0] * len(self.vocab_)
            for tok in simple_tokenize(d):
                if tok in self.vocab_:
                    vec[self.vocab_[tok]] += 1
            rows.append(vec)
        return rows
```

### 2.3 分层抽样划分

```python
def train_test_split_fallback(X, y, test_size=0.2, random_state=42, stratify=None):
    rng = random.Random(random_state)
    data_by_label = defaultdict(list)
    for xi, yi in zip(X, y):
        data_by_label[yi].append(xi)

    X_train, X_test, y_train, y_test = [], [], [], []
    for label, items in data_by_label.items():
        n = len(items)
        k = max(1, int(n * test_size))      # 每组 20% 给测试
        idx = list(range(n))
        rng.shuffle(idx)
        test_idx = set(idx[:k])
        for i in range(n):
            if i in test_idx:
                X_test.append(items[i]); y_test.append(label)
            else:
                X_train.append(items[i]); y_train.append(label)
    return X_train, X_test, y_train, y_test
```

## 3. 创新点 & 优化

### 3.1 对数空间计算

在 `predict` 方法中使用对数空间计算，将概率连乘转换为对数加法，有效避免浮点数下溢问题：

$$\log P(y \mid \mathbf{x}) \propto \log P(y) + \sum_{j: c_j > 0} c_j \cdot \log P(w_j \mid y)$$

并且只累加出现过的词（`if val:`），利用文本数据的稀疏性提升计算效率。

---

# 实验结果及分析

## 1. 实验结果展示

实验在 SMS Spam Collection 样本数据集上运行，使用 80/20 分层抽样划分训练集和测试集，拉普拉斯平滑参数 $\alpha = 1$。

### 运行输出

```
Using dataset: spam.csv

=== Evaluation on test set ===
Accuracy: 0.981149012567325

Confusion Matrix (rows=true, cols=pred) labels=
 ['ham', 'spam']
[959, 6]
[15, 134]

Classification Report:
label   precision       recall  f1      support
ham     0.985   0.994   0.989   965
spam    0.957   0.899   0.927   149
```

### 结果可视化

**混淆矩阵**：

|  | 预测=ham | 预测=spam |
|--|---------|----------|
| **实际=ham** | 959 (TN) | 6 (FP) |
| **实际=spam** | 15 (FN) | 134 (TP) |

**各指标汇总**：

| 类别 | 精确率 (Precision) | 召回率 (Recall) | F1 分数 | 样本数 |
|------|-------------------|-----------------|---------|--------|
| ham | 0.985 | 0.994 | 0.989 | 965 |
| spam | 0.957 | 0.899 | 0.927 | 149 |

## 2. 评测指标分析

### 2.1 整体性能

- **准确率 = 98.11%**：测试集上大部分短信被正确分类，模型在该数据集上表现良好。

### 2.2 垃圾短信召回率的重要性

在垃圾短信检测的实际应用中，**召回率（Recall）尤为关键**：
- 低召回率意味着大量垃圾短信漏网，进入用户收件箱
- 可能导致钓鱼、诈骗短信未被拦截，带来安全风险
- 用户信任度下降，对垃圾过滤功能产生不满

因此，在生产系统中通常会在保证较高精确率的前提下，尽可能提升召回率。

---

# 参考资料

- SMS Spam Collection Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset