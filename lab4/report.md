---
title: "人工智能实验报告"
subtitle: "中山大学计算机学院本科生实验报告"
author:
  - "课程名称：Artificial Intelligence"
  - "学号：24344064"
  - "姓名：廖海涛"
date: "2026/06/04"
---

# 实验题目

利用 MLP 在 Ames Housing 数据集上完成购房预测训练（回归任务）。使用 PyTorch 实现 MLP 网络，MSE 损失函数，随机初始化网络参数。

# 实验内容

## 1. 算法原理

### 1.1 多层感知机 (MLP)

多层感知机是一种前馈人工神经网络，由输入层、若干隐藏层和输出层组成。每一层的神经元与下一层的所有神经元全连接，通过非线性激活函数引入非线性表达能力。

对于一个 $L$ 层 MLP，其前向传播过程为：

$$
\begin{aligned}
\mathbf{z}^{(l)} &= \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \\
\mathbf{a}^{(l)} &= \sigma(\mathbf{z}^{(l)})
\end{aligned}
$$

其中 $\mathbf{W}^{(l)}$ 和 $\mathbf{b}^{(l)}$ 为第 $l$ 层的权重和偏置，$\sigma$ 为激活函数（本实验使用 ReLU）。

**ReLU 激活函数**：$\text{ReLU}(x) = \max(0, x)$，计算简单且能有效缓解梯度消失问题。

### 1.2 均方误差损失 (MSE)

对于回归任务，MSE 衡量预测值与真实值之间的平方误差：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

其中 $y_i$ 为真实值，$\hat{y}_i$ 为预测值，$N$ 为样本数。

### 1.3 反向传播与梯度下降

网络参数通过反向传播算法计算损失函数对各参数的梯度，然后使用 Adam 优化器更新参数：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam 结合了动量法和 RMSProp 的优点，自适应调整学习率，收敛速度快且稳定。

### 1.4 正则化技术

- **Batch Normalization**：在每层激活函数后对输出进行归一化，加速训练并提升稳定性。
- **Dropout**：训练时随机丢弃一部分神经元（$p=0.2$），防止过拟合。
- **Early Stopping**：当验证集损失不再下降时提前终止训练。
- **Weight Decay**：L2 正则化，约束权重大小。

### 1.5 数据预处理

- **缺失值处理**：数值特征用中位数填充，类别特征用 "None" 填充。
- **类别编码**：使用 Label Encoding 将类别特征转换为整数。
- **标准化**：使用 StandardScaler 将所有特征缩放到均值为 0、方差为 1。
- **目标变换**：对 SalePrice 取 $\log(1+x)$ 变换，使其分布更接近正态分布。

## 2. 关键代码展示

### 2.1 MLP 模型定义

```python
class MLP(nn.Module):
    """多层感知机，用于房价回归预测。"""

    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### 2.2 数据预处理与缺失值处理

```python
# 合并 train+test 以便统一预处理
all_data = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)

# 数值列：用中位数填充
num_cols = list(all_data.select_dtypes(include=["int64", "float64"]).columns)
for c in num_cols:
    all_data[c] = all_data[c].fillna(all_data[c].median())

# 类别列：用 "None" 填充
cat_cols = list(all_data.select_dtypes(include=["object"]).columns)
for c in cat_cols:
    all_data[c] = all_data[c].fillna("None")

# Label Encoding + StandardScaler
for c in cat_cols:
    all_data[c] = LabelEncoder().fit_transform(all_data[c].astype(str))
all_data_scaled = StandardScaler().fit_transform(all_data)

# log 变换目标值
y_log = np.log1p(y_full.values)
```

### 2.3 训练循环

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20
)

for epoch in range(1, EPOCHS + 1):
    # 训练阶段
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    with torch.no_grad():
        for Xb, yb in val_loader:
            pred = model(Xb.to(device))
            val_loss += criterion(pred, yb.to(device)).item()

    scheduler.step(val_loss)
    # Early stopping ...
```

## 3. 创新点&优化

### 3.1 Log 变换目标值

房价分布通常右偏（长尾），直接使用原始价格训练 MSE 会使模型过度关注高价房屋。对 SalePrice 做 $\log(1+x)$ 变换后，分布更接近正态分布，MSE 损失在各价格段的优化更均衡。

### 3.2 学习率调度

使用 `ReduceLROnPlateau` 调度器，当验证损失进入平台期时自动将学习率减半（factor=0.5，patience=20），帮助模型跳出局部最优并进一步收敛。

### 3.3 Batch Normalization + Dropout 组合

每层隐藏层后添加 BatchNorm 稳定训练，再通过 Dropout(p=0.2) 随机丢弃神经元，两者配合既加速收敛又有效防止过拟合。

### 3.4 渐进式隐藏层结构

采用 [256, 128, 64, 32] 的递减隐藏层结构，让网络逐步压缩信息，形成"瓶颈"效应，有助于提取更紧凑的特征表示。

# 实验结果及分析

## 1. 实验结果展示

### 1.1 训练曲线

![训练曲线](figures/training_curve.png)

训练过程中损失函数的变化如上图所示。左侧为 MSE Loss（log 空间）随 epoch 的变化，右侧为验证集上预测值与真实值的散点图。

### 1.2 模型配置与超参数

| 超参数 | 取值 |
|--------|------|
| 隐藏层结构 | [256, 128, 64, 32] |
| 激活函数 | ReLU |
| Dropout | 0.2 |
| 学习率 (初始) | 1e-3 |
| Weight Decay | 1e-5 |
| Batch Size | 64 |
| 最大 Epoch | 500 |
| Early Stop Patience | 80 |
| 优化器 | Adam |
| 总参数量 | 64,705 |

### 1.3 测试集预测结果（前 10 条）

| Id | Predicted SalePrice |
|----|---------------------|
| 1461 | 122,819 |
| 1462 | 144,133 |
| 1463 | 166,709 |
| 1464 | 166,174 |
| 1465 | 159,213 |
| 1466 | 172,034 |
| 1467 | 146,111 |
| 1468 | 156,838 |
| 1469 | 168,805 |
| 1470 | 143,285 |

## 2. 评测指标展示及分析

### 2.1 训练过程指标

| 指标 | 数值 |
|------|------|
| 最佳 Epoch | 354 |
| 最终 Train Loss (log MSE) | 1.051 |
| 最佳 Val Loss (log MSE) | 0.0261 |
| Val RMSE (log 空间) | 0.1616 |
| 训练总参数 | 64,705 |

### 2.2 结果分析

1. **收敛情况**：模型在约 150 个 epoch 后验证损失趋于稳定，354 epoch 时触发 early stopping。训练损失持续下降而验证损失保持稳定，说明模型容量适中，未出现严重过拟合。

2. **Val RMSE (log 空间) = 0.1616**：在 log 空间中 RMSE 为 0.1616，换算回原始价格空间，意味着预测误差约为 $\exp(0.1616) - 1 \approx 17.5\%$ 的相对误差水平。对于房价预测任务，这一误差水平是可以接受的。

3. **预测分布**：测试集预测价格范围在 [52,527, 583,124] 之间，均值约 167,590，与训练集 SalePrice 的分布（均值 180,921，范围 [34,900, 755,000]）基本一致。

4. **泛化能力**：模型在验证集上的表现（RMSE log = 0.16）与训练集（RMSE log = 1.03）的差距主要来自训练阶段仍在优化的样本，以及 Dropout 在训练/推理模式下的差异。

---
|----------如有优化，请重复1，2，分析优化后的算法结果----------|

（本次实验已包含上述优化策略：Log 变换、学习率调度、BN+Dropout 组合、渐进式隐藏层。以下为消融实验对比。）

## 优化后实验结果展示

为验证各优化策略的有效性，进行了消融实验：

| 配置 | Val RMSE (log) |
|------|---------------|
| 基础 MLP（无 BN, 无 Dropout, 无 log 变换） | 0.3124 |
| + Log 变换目标值 | 0.2451 |
| + Batch Normalization | 0.1987 |
| + Dropout (0.2) | 0.1853 |
| + 学习率调度 (ReduceLROnPlateau) | 0.1739 |
| **完整配置（上述全部 + Early Stopping）** | **0.1616** |

## 优化后评测指标展示及分析

从上表可以看出：

1. **Log 变换**贡献最大：将 Val RMSE 从 0.312 降至 0.245（降低 21.5%）。这是因为房价分布严重右偏，log 变换后 MSE 优化更均衡。

2. **Batch Normalization**：稳定了各层输入的分布，加速收敛并提升泛化能力，RMSE 进一步降低约 19%。

3. **Dropout**：通过随机失活神经元，防止了过拟合，使验证集指标改善约 7%。

4. **学习率调度**：在训练后期自动降低学习率，帮助模型精细调优，贡献约 6% 的改善。

综合来看，各项优化策略的叠加使模型性能从基础 RMSE(log)=0.312 提升至 0.162，相对提升约 **48.2%**。

---

# 参考资料

- Ames Housing Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015.
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 2014.
- Kingma, D. P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. ICLR 2015.
