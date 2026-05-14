# Lab3：基于朴素贝叶斯的短信垃圾分类 — 原理详解

## 1. 整体流程

```
原始短信文本
    │
    ▼
┌──────────────┐
│  数据加载     │  读取 spam.csv，提取 label（ham/spam）和 text
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  训练/测试    │  80% 训练，20% 测试，分层抽样保证类别比例一致
│  集划分       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  文本向量化   │  Bag of Words → 词频向量（CountVectorizer）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  模型训练     │  多项式朴素贝叶斯 + 拉普拉斯平滑（α=1）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  预测 & 评估  │  准确率、混淆矩阵、精确率、召回率、F1
└──────────────┘
```

---

## 2. 文本向量化：词袋模型 (Bag of Words)

### 2.1 核心思想

把每条短信转换成一个**固定长度的数值向量**，向量的每一维对应词汇表中的一个词，值为该词在这条短信中出现的次数。

### 2.2 数学表示

设词汇表为 $V = \{w_1, w_2, \dots, w_{|V|}\}$（所有短信中出现过的去重词汇）。

对第 $i$ 条短信 $d_i$，其向量表示为：

$$\mathbf{x}^{(i)} = [c_{i1}, c_{i2}, \dots, c_{i|V|}]$$

其中 $c_{ij}$ = 词 $w_j$ 在短信 $d_i$ 中出现的次数。

### 2.3 代码对应（`SimpleCountVectorizer`）

```python
# fit: 构建词汇表（遍历所有训练短信，收集去重词汇）
vocab_ = {"go": 0, "until": 1, "jurong": 2, ...}

# transform: 将每条短信转为词频向量
"go go until"  →  [2, 1, 0, ...]
#                  ↑  ↑  ↑
#                 go until jurong
```

### 2.4 举例

| 短信 | go | until | free | win |
|------|-----|-------|------|-----|
| "go go until" | 2 | 1 | 0 | 0 |
| "free win free" | 0 | 0 | 2 | 1 |

每条短信从变长文本变成了等长向量，可以被数学模型直接处理。

---

## 3. 朴素贝叶斯分类器 (Naive Bayes)

### 3.1 贝叶斯定理

给定一条短信 $\mathbf{x}$，我们要判断它属于哪个类别 $y \in \{\text{ham}, \text{spam}\}$：

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}$$

- $P(y \mid \mathbf{x})$：**后验概率** — 看到这条短信后，它是 ham/spam 的概率
- $P(y)$：**先验概率** — 训练集中 ham/spam 各自的比例
- $P(\mathbf{x} \mid y)$：**似然** — 在已知类别下，出现这条短信内容的概率
- $P(\mathbf{x})$：**证据** — 对所有类别相同，可以忽略

### 3.2 朴素假设 (Naive Assumption)

直接计算 $P(\mathbf{x} \mid y)$ 需要 $P(w_1, w_2, \dots, w_n \mid y)$ —— 所有词的联合概率，复杂度爆炸。

**朴素贝叶斯的核心简化**：假设所有词之间**条件独立**（给定类别后，每个词出现与否互不影响）：

$$P(\mathbf{x} \mid y) = \prod_{j=1}^{|V|} P(w_j \mid y)^{c_j}$$

> $\quad$"朴素"（Naive）就体现在这个独立性假设上。虽然现实中词与词之间显然不独立（比如 "New" 后面经常跟 "York"），但实践中这个假设依然能取得很好的分类效果。

### 3.3 多项式朴素贝叶斯 (Multinomial NB)

对于文本分类，使用**多项分布**建模——每个词的出现次数 $c_j$ 服从多项分布：

$$P(w_j \mid y) = \frac{\text{类别 } y \text{ 中词 } w_j \text{ 的总出现次数}}{\text{类别 } y \text{ 中所有词的总出现次数}} = \frac{N_{yj}}{\sum_k N_{yk}}$$

### 3.4 分类决策

对于一条新短信 $\mathbf{x}$，预测它属于哪个类别：

$$\hat{y} = \arg\max_{y} \; P(y) \cdot \prod_{j=1}^{|V|} P(w_j \mid y)^{c_j}$$

---

## 4. 拉普拉斯平滑 (Laplace Smoothing)

### 4.1 问题：零概率

如果词 $w_j$ 在训练集的 spam 类别中从未出现，那么 $P(w_j \mid \text{spam}) = 0$。一旦乘积中某个因子为 0，整个后验概率就变成 0——一条短信只要包含任何一个"没见过的词"就会被判为概率为 0。

### 4.2 解决方案

给每个计数加上一个小的伪计数 $\alpha$：

$$P(w_j \mid y) = \frac{N_{yj} + \alpha}{\sum_k N_{yk} + \alpha \cdot |V|}$$

- $\alpha = 1$ 即为拉普拉斯平滑（代码中 `SimpleMultinomialNB(alpha=1.0)`）
- 分子 $+\alpha$：假装每个词至少出现了 $\alpha$ 次
- 分母 $+\alpha \cdot |V|$：保证概率之和仍为 1

### 4.3 代码对应

```python
# 计算每个类别的平滑对数概率
for c in labels:
    fc = self.feature_count_[c]          # 类别 c 下每个词的计数
    sm = sum(fc)                          # 类别 c 下所有词的总数
    denom = sm + self.alpha * V           # 分母加平滑
    self.feature_log_prob_[c] = [
        math.log((fc_i + self.alpha) / denom)   # 取对数
        for fc_i in fc
    ]
```

---

## 5. 对数空间计算 (Log-Space)

### 5.1 为什么用对数

直接连乘很多小于 1 的概率会导致**数值下溢**（浮点数精度不够，结果变成 0）。

利用 $\log(a \cdot b) = \log a + \log b$，把乘法转为加法：

$$\log P(y \mid \mathbf{x}) \propto \log P(y) + \sum_{j=1}^{|V|} c_j \cdot \log P(w_j \mid y)$$

### 5.2 代码对应

```python
# predict 方法中
score = self.class_log_prior_[c]          # log P(y)
for i, val in enumerate(xi):
    if val:                                # 只累加出现过的词
        score += val * probs[i]            # val * log P(w_i | y)
```

---

## 6. 训练/测试集划分（分层抽样）

### 6.1 为什么要分层 (Stratified)

如果随机划分，可能训练集里 spam 很少而测试集里 spam 很多，导致模型评估不准确。

分层抽样保证：
- 训练集中 ham:spam 比例 ≈ 原始比例
- 测试集中 ham:spam 比例 ≈ 原始比例

### 6.2 代码对应（`train_test_split_fallback`）

```python
# 按标签分组 → 每组内部随机打乱 → 取 20% 作测试集
data_by_label = {"ham": [...], "spam": [...]}
for label, items in data_by_label.items():
    k = max(1, int(len(items) * 0.2))   # 每组 20% 给测试
    # ... 划分 ...
```

---

## 7. 评估指标

### 7.1 混淆矩阵

|  | 预测=ham | 预测=spam |
|--|---------|----------|
| **实际=ham** | TN（真阴性） | FP（假阳性） |
| **实际=spam** | FN（假阴性） | TP（真阳性） |

### 7.2 四个核心指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **准确率** Accuracy | $\dfrac{TP + TN}{TP + TN + FP + FN}$ | 预测正确的比例 |
| **精确率** Precision | $\dfrac{TP}{TP + FP}$ | 预测为 spam 的里面，多少真的是 spam |
| **召回率** Recall | $\dfrac{TP}{TP + FN}$ | 真正的 spam 中，有多少被找出来了 |
| **F1 分数** | $\dfrac{2 \cdot P \cdot R}{P + R}$ | 精确率和召回率的调和平均 |

---

## 8. 完整数学流程总结

1. **词汇表构建**：从训练短信中提取所有去重词 → $V = \{w_1, \dots, w_{|V|}\}$

2. **文本向量化**：每条短信 → 词频向量 $\mathbf{x} = [c_1, \dots, c_{|V|}]$

3. **计算先验概率**（对数）：
   $$\log P(y) = \log\frac{\text{count}(y)}{\text{total}}$$

4. **计算条件概率**（对数 + 平滑）：
   $$\log P(w_j \mid y) = \log\frac{N_{yj} + \alpha}{\sum_k N_{yk} + \alpha|V|}$$

5. **预测新短信**：
   $$\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{j: c_j > 0} c_j \cdot \log P(w_j \mid y) \right]$$

6. **评估**：在保留的测试集上计算准确率、混淆矩阵、精确率/召回率/F1。

---

## 9. 为什么朴素贝叶斯适合文本分类？

- **高维稀疏数据友好**：词汇表很大（几千到几万维），但每条短信只包含少量词，计算效率高
- **训练极快**：只需遍历一次数据统计词频，时间复杂度 $O(N \cdot L)$
- **可解释性强**：可以直观看出哪些词对 spam/ham 判别贡献最大
- **小样本也能工作**：拉普拉斯平滑缓解了数据稀疏问题
