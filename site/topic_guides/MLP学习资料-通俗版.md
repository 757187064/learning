# MLP 学习资料：从训练一条直线到训练一个分类网络

## 资料来源说明

本讲义依据以下课程材料重新整理：

- `多层感知机1_学生分发版.ipynb`
- `多层感知机1_学生分发版_副本.ipynb`
- `多层感知机2-学生分发版-终版2026.ipynb`
- 参考缓存：`database/source_cache.json`、`database/course_chunks.json`

三份课件覆盖了两条主线：

1. **神经网络训练流程**：从手写线性回归的梯度下降开始，逐步过渡到 PyTorch 张量、自动微分、优化器、`nn.Module`、`Dataset`、`DataLoader`、训练/验证模式、TensorBoard、保存与加载模型。
2. **分类问题与 MLP**：从回归和分类的区别开始，讲二分类的 Logistic 回归、Sigmoid、BCE Loss，多分类的 Softmax、Cross-Entropy Loss，再讲为什么 MLP 必须有激活函数，以及激活函数如何让线性不可分的数据变得可分。

这份讲义不是按课件页顺序复述，而是按基础较弱的同学更容易理解的路径重写：**为什么需要训练神经网络 -> 数据如何变成张量和 batch -> 线性层与激活函数 -> 损失函数 -> 梯度下降、反向传播和自动微分 -> MLP 分类实践 -> 常见错误 -> 考前复盘**。

---

## 全局学习路线图

学习 MLP 不要一上来背公式。更有效的顺序是：

| 阶段 | 要解决的问题 | 对应知识 |
|---|---|---|
| 1. 为什么要训练 | 模型刚开始乱猜，怎样变好？ | 参数、预测、损失、梯度下降 |
| 2. 数据怎样进入模型 | 一个样本、一批样本、完整数据集有什么区别？ | 张量、样本、batch、epoch、iteration |
| 3. 模型怎样计算 | 一层神经网络到底做了什么？ | 线性层、权重、偏置、logits |
| 4. 为什么需要非线性 | 多加几层线性层有没有用？ | 激活函数、非线性、特征空间变换 |
| 5. 错了怎样衡量 | 分类不能直接用“差多少”来理解吗？ | MSE、BCE、CrossEntropy |
| 6. 参数怎样更新 | 损失变小靠什么方向？ | 梯度、学习率、反向传播 |
| 7. 框架怎样帮忙 | PyTorch 自动做了哪些事？ | Autograd、计算图、optimizer |
| 8. 分类怎样落地 | 二分类和多分类代码哪里不一样？ | Sigmoid/Softmax、logits、标签形状 |
| 9. 考试怎样检查 | 哪些地方最容易混？ | 易混点、速查表、自测题 |

---

## 直觉解释：把 MLP 想成一套会改错的分类流程

MLP 可以先理解成一条流水线。输入数据进来后，线性层先把原始特征重新加权组合，激活函数再把空间“折一下”或“弯一下”，下一层继续组合这些新特征，最后输出每个类别的原始分数。训练时，损失函数告诉模型“这次错得有多严重”，反向传播再把这个错误分摊回每一层参数，优化器负责把参数往更少犯错的方向移动。

所以不要把 MLP 看成一堆神秘公式。它的主线很朴素：**用参数做预测，用损失衡量错误，用梯度指出调整方向，用很多个 batch 反复修正参数**。后面的 Sigmoid、Softmax、BCE、CrossEntropy、`backward()`、`optimizer.step()` 都是在服务这条主线。

---

## 1. 为什么需要神经网络训练

### 1.1 模型不是“写出答案”，而是“调参数”

课件从一个最简单的模型开始：

$$
y = b + wx
$$

这里：

- $x$ 是输入，比如一个房子的面积、一个样本的某个特征。
- $y$ 是真实答案，比如真实房价。
- $\hat{y}$ 是模型预测值。
- $w$ 是权重，控制输入 $x$ 对输出影响有多大。
- $b$ 是偏置，控制整体向上或向下平移。

刚开始，$w$ 和 $b$ 通常是随机初始化的，所以模型预测大概率不准。训练的目的就是不断调整 $w$ 和 $b$，让预测 $\hat{y}$ 越来越接近真实值 $y$。

课件中的合成数据是：

```python
true_b = 1
true_w = 2
N = 100

np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon
```

这段代码表示：真实规律大约是 `y = 1 + 2x`，但加入了一点噪声 `epsilon`。模型训练后，如果能学到接近 `b = 1`、`w = 2` 的参数，说明训练方向是对的。

### 1.2 训练的五步循环

课件把训练拆成 5 步，这也是所有深度学习训练代码的骨架：

1. 随机初始化参数。
2. 前向传播：用当前参数算预测值。
3. 计算损失：衡量预测错得多严重。
4. 计算梯度：判断每个参数应该往哪里调。
5. 更新参数：沿着让损失变小的方向移动一点点。

代码对应关系如下：

```python
# Step 0: 初始化参数
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

lr = 0.1
n_epochs = 1000

for epoch in range(n_epochs):
    # Step 1: 前向传播
    yhat = b + w * x_train

    # Step 2: 计算损失
    error = yhat - y_train
    loss = (error ** 2).mean()

    # Step 3: 计算梯度
    b_grad = 2 * error.mean()
    w_grad = 2 * (error * x_train).mean()

    # Step 4: 更新参数
    b = b - lr * b_grad
    w = w - lr * w_grad
```

这里的关键是：**神经网络训练不是一次算出最终答案，而是通过很多次小步更新慢慢靠近较好的参数**。

### 1.3 损失函数像“错误程度仪表盘”

在线性回归例子里，课件使用均方误差 MSE：

$$
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

逐项拆开：

- $N$：参与计算的样本数量。
- $i$：第 $i$ 个样本。
- $\hat{y}_i$：第 $i$ 个样本的预测值。
- $y_i$：第 $i$ 个样本的真实值。
- $\hat{y}_i-y_i$：预测误差。
- $(\hat{y}_i-y_i)^2$：把误差平方，避免正负抵消，同时放大大错误。
- $\frac{1}{N}\sum$：对所有样本的错误取平均，使损失和样本数不要强绑定。

课件代码：

```python
error = yhat - y_train
loss = (error ** 2).mean()
```

损失越小，说明当前参数下模型表现越好。梯度下降的目标就是让这个损失下降。

---

## 2. 张量、样本、Batch：数据进入神经网络前要先排队

### 2.1 一个样本是什么

一个样本通常写作：

$$
(x^{(i)}, y^{(i)})
$$

意思是：第 $i$ 条输入 $x^{(i)}$ 和它对应的标签 $y^{(i)}$。

在分类任务中，标签也可能写成 $t^{(i)}$。课件中分类问题描述为：给定数据点集合 $\{(x^{(i)}, t^{(i)})\}$，目标是找到映射：

$$
f: \mathbb{R}^m \rightarrow \Omega
$$

逐项解释：

- $\mathbb{R}^m$：输入空间。一个样本有 $m$ 个特征，比如二维月牙数据中 $m=2$。
- $\Omega$：标签空间。
- 回归任务中，$\Omega$ 是连续数值。
- 分类任务中，$\Omega$ 是离散类别，比如 0/1 或 0/1/2。

### 2.2 Tensor 是深度学习里的“可计算数组”

NumPy 的 `ndarray` 能表示多维数组，但 PyTorch 的 Tensor 更适合训练神经网络，因为它额外支持：

- 记录计算图。
- 自动微分。
- 移动到 GPU 或其他设备上计算。

课件中的转换：

```python
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
```

含义：

- `torch.as_tensor(x_train)`：把 NumPy 数组变成 PyTorch 张量。
- `.float()`：转成浮点数，神经网络参数和输入通常是浮点数。
- `.to(device)`：把张量放到 CPU/GPU 上。

课件也提醒：数据加载和预处理通常在 CPU 上做；前向传播、损失计算、反向传播、参数更新通常放到 GPU 上做。实际训练时常见做法是：**数据集先留在 CPU，每个 mini-batch 取出来后再送到 device**。

### 2.3 Batch、Epoch、Iteration 的区别

这是考试和代码调试都很常见的混淆点。

| 概念 | 含义 | 举例 |
|---|---|---|
| 样本 sample | 一条数据 | 一张图片、一个二维点 |
| batch | 一小批样本 | 每次取 16 条 |
| batch size | 一个 batch 中的样本数 | `batch_size=16` |
| iteration | 一次参数更新 | 处理完一个 batch 后更新一次 |
| epoch | 全部训练数据被用过一遍 | 100 条数据都参与过训练 |

如果训练集有 80 个样本，`batch_size=16`，那么：

$$
\text{每个 epoch 的 iteration 数} = \frac{80}{16} = 5
$$

课件总结了三种梯度下降：

| 类型 | 每次用多少数据 | 一个 epoch 更新几次 | 特点 |
|---|---:|---:|---|
| 批量梯度下降 | 全部训练集 $N$ | 1 次 | 稳定但慢 |
| 随机梯度下降 | 1 个样本 | $N$ 次 | 更新频繁但噪声大 |
| 小批量梯度下降 | $1<n<N$ | 约 $N/n$ 次 | 实践中最常用 |

### 2.4 Dataset 和 DataLoader 的作用

课件中自定义 `Dataset`：

```python
class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)
```

三个方法分别负责：

- `__init__`：保存数据和标签。
- `__getitem__`：根据索引取出一个样本。
- `__len__`：告诉 PyTorch 数据集有多大。

更简洁的写法：

```python
train_data = TensorDataset(x_train_tensor, y_train_tensor)
```

再用 `DataLoader` 自动切 batch：

```python
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True
)
```

为什么训练集要 `shuffle=True`？

- 防止模型记住数据顺序。
- 让每个 batch 更像整体数据的随机样本。
- 让梯度估计更稳定，通常有助于泛化。

验证集通常 `shuffle=False`，因为验证只是评估，不需要打乱顺序来训练参数。

---

## 3. 线性层与激活函数：MLP 的基本零件

### 3.1 线性层在算什么

PyTorch 中：

```python
nn.Linear(input_dim, output_dim)
```

本质上是在做：

$$
z = Wx + b
$$

逐项解释：

- $x$：输入特征。
- $W$：权重矩阵。
- $b$：偏置。
- $z$：线性层输出，也常叫 logits。

如果输入是一个 batch，形状通常是：

$$
X \in \mathbb{R}^{B \times m}
$$

其中：

- $B$ 是 batch size。
- $m$ 是每个样本的特征数。

如果 `nn.Linear(m, h)`，输出形状是：

$$
Z \in \mathbb{R}^{B \times h}
$$

其中 $h$ 是这一层输出的神经元个数。

### 3.2 MLP 是多层“线性层 + 激活函数”

一个典型 MLP 二分类模型可以写成：

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
```

含义：

- `nn.Linear(2, 16)`：二维输入变成 16 维隐藏表示。
- `nn.ReLU()`：加入非线性变换。
- `nn.Linear(16, 1)`：输出一个 logit，用于二分类。

如果是三分类，可以写成：

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
```

最后输出 3 个 logits，分别对应 3 个类别的原始分数。

### 3.3 为什么不能只堆线性层

课件中的关键推导：

$$
f(x)=W_2(W_1x+b_1)+b_2
$$

展开：

$$
f(x)=W_2W_1x+W_2b_1+b_2
$$

把 $W_2W_1$ 看成一个新矩阵 $W$，把 $W_2b_1+b_2$ 看成一个新偏置 $b$，就得到：

$$
f(x)=Wx+b
$$

这说明：**如果中间没有激活函数，很多层线性层叠起来仍然等价于一层线性模型**。

所以 MLP 真正强大的地方不是“层数多”本身，而是每层之间有非线性激活，让模型可以表达弯曲的决策边界。

### 3.4 激活函数在做什么

线性层做的是仿射变换：

$$
u_j^{(l)}=\sum_i w_{ji}^{(l)}y_i^{(l-1)}+b_j^{(l)}
$$

逐项解释：

- $l$：第 $l$ 层。
- $j$：这一层中的第 $j$ 个神经元。
- $i$：上一层中的第 $i$ 个输出。
- $y_i^{(l-1)}$：上一层第 $i$ 个神经元的输出。
- $w_{ji}^{(l)}$：从上一层第 $i$ 个神经元到这一层第 $j$ 个神经元的权重。
- $b_j^{(l)}$：这一层第 $j$ 个神经元的偏置。
- $u_j^{(l)}$：激活前的值。

激活函数再做：

$$
y_j^{(l)}=f(u_j^{(l)})
$$

也就是把线性层输出 $u$ 交给非线性函数 $f$。这一步会“弯折”特征空间，让原本无法用直线分开的数据，在新空间里可能变得可分。

课件以 `make_moons` 这类月牙形数据说明：Logistic 回归只能给出线性决策边界，而 MLP 可以通过隐藏层和激活函数学习非线性边界。

### 3.5 常见激活函数对比

| 激活函数 | 公式 | 输出范围 | 主要特点 |
|---|---|---|---|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | $(0,1)$ | 可解释为概率，但大正/大负区域梯度小 |
| Tanh | $\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1,1)$ | 零中心化，比 Sigmoid 更适合作隐藏层一些 |
| ReLU | $\max(0,x)$ | $[0,+\infty)$ | 计算快，MLP/CNN 常用 |
| Leaky ReLU | $\max(\alpha x,x)$ | $(-\infty,+\infty)$ | 缓解 ReLU 负半轴梯度为 0 的问题 |
| GELU | $x\cdot \Phi(x)$ | $(-\infty,+\infty)$ | 平滑，Transformer 中常见 |
| Swish | $x\cdot \sigma(\beta x)$ | $(-\infty,+\infty)$ | 带自门控思想 |

考试常问的是：**激活函数为什么必要？**  
答案要抓住“没有激活函数，多层线性仍等价于单层线性；激活函数引入非线性，使模型能拟合非线性决策边界”。

---

## 4. 损失函数：分类模型到底错在哪里

### 4.1 回归和分类的输出不同

课件给出的对比：

| 任务 | 标签空间 | 输出含义 | 常见损失 |
|---|---|---|---|
| 回归 | 连续值 | 预测一个数值 | MSE |
| 二分类 | 两个类别 | 属于正类的概率 | BCE |
| 多分类 | 多个类别 | 每个类别的概率分布 | Cross-Entropy |

分类模型最后通常先输出 logits，再由 Sigmoid 或 Softmax 转为概率。

### 4.2 二分类：Logistic 回归、Logit、Sigmoid

二分类中，我们希望输出一个概率：

$$
p=P(y=1|x)
$$

但线性层 $z=w^Tx+b$ 的输出范围是 $(-\infty,+\infty)$，不能直接当概率。Sigmoid 的作用是把任意实数压到 $(0,1)$：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

逐项解释：

- $z$：logit，也就是线性层原始输出。
- $e^{-z}$：指数项，用来把实数转换成正数比例。
- $1+e^{-z}$：保证分母大于分子。
- $\sigma(z)$：最终概率。

当 $z=0$ 时：

$$
\sigma(0)=\frac{1}{1+1}=0.5
$$

所以 $z=0$ 常对应二分类边界：大于 0 更偏向类别 1，小于 0 更偏向类别 0。

课件还从 odds ratio 引入：

$$
\text{odds}=\frac{p}{1-p}
$$

再取对数得到 logit：

$$
z=\log\left(\frac{p}{1-p}\right)
$$

Sigmoid 是这个变换的反函数。

### 4.3 二元交叉熵 BCE Loss

BCE Loss 公式：

$$
L=-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)
\right]
$$

逐项拆解：

- $N$：样本数。
- $y_i$：真实标签，只能是 0 或 1。
- $\hat{y}_i$：模型预测为 1 的概率。
- $y_i\log \hat{y}_i$：当真实标签是 1 时，这项起作用。
- $(1-y_i)\log(1-\hat{y}_i)$：当真实标签是 0 时，这项起作用。
- 外面的负号：因为 $\log$ 概率通常小于等于 0，取负后变成非负损失。

分情况看更清楚：

当 $y_i=1$：

$$
L_i=-\log \hat{y}_i
$$

模型希望 $\hat{y}_i$ 越接近 1 越好。如果真实是 1，模型却预测 0.01，损失会很大。

当 $y_i=0$：

$$
L_i=-\log(1-\hat{y}_i)
$$

模型希望 $\hat{y}_i$ 越接近 0 越好。如果真实是 0，模型却预测 0.99，损失会很大。

课件强调：BCE 可以从极大似然估计推出。假设标签服从伯努利分布：

$$
P(y|x)=\hat{y}^{y}(1-\hat{y})^{1-y}
$$

最大化正确标签出现的概率，等价于最小化负对数似然，也就是 BCE。

### 4.4 PyTorch 二分类推荐写法

课件推荐：

```python
model = nn.Sequential(nn.Linear(2, 1))  # 输出 logits
loss_fn = nn.BCEWithLogitsLoss()
```

原因：`BCEWithLogitsLoss` 内部已经合并了 Sigmoid 和 BCE 的计算，更稳定。

不要写成：

```python
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCEWithLogitsLoss()
```

因为这会相当于把 Sigmoid 用了两次，训练会出问题。

二分类标签也要注意形状和类型：

```python
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
```

- `FloatTensor`：BCE 系列损失需要浮点标签。
- `.view(-1, 1)`：让标签形状从 `[N]` 变成 `[N, 1]`，和模型输出 `[N, 1]` 对齐。

### 4.5 多分类：Softmax 和 Cross-Entropy

多分类中，模型输出 $K$ 个 logits：

$$
z_1,z_2,\dots,z_K
$$

Softmax 把它们变成概率分布：

$$
\text{softmax}(z_k)=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

逐项解释：

- $z_k$：第 $k$ 类的 logit。
- $e^{z_k}$：把 logit 变成正数，并放大高分和低分的差距。
- $\sum_{j=1}^{K}e^{z_j}$：所有类别指数分数之和。
- 分子除以分母：得到第 $k$ 类概率。
- 所有类别概率之和等于 1。

为什么不用 Hardmax？Hardmax 直接选最大类别，输出不可导，梯度下降没法训练；Softmax 是平滑可导的概率化版本。

多分类交叉熵：

$$
L=-\frac{1}{N}\sum_{n=1}^{N}\log h_{k^*}(x^{(n)})
$$

逐项解释：

- $n$：第 $n$ 个样本。
- $k^*$：该样本真实类别。
- $h_{k^*}(x^{(n)})$：模型给真实类别分配的概率。
- $-\log$：真实类别概率越低，损失越大。

原始写法是：

$$
L=-\frac{1}{N}\sum_n\sum_k t_k^{(n)}\log h_k^{(n)}
$$

其中 $t_k^{(n)}$ 是 one-hot 标签。因为 one-hot 中只有真实类别位置是 1，其余是 0，所以公式可以简化为“只看真实类别的概率”。

### 4.6 PyTorch 多分类推荐写法

课件代码：

```python
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # 输出 logits

criterion = nn.CrossEntropyLoss()
```

重点：

- 模型最后一层输出 logits，不手动加 Softmax。
- `nn.CrossEntropyLoss()` 内部会处理 Softmax 相关计算。
- 标签应该是 `long` 类型的类别索引，比如 `[0, 2, 1, 1]`，不是 one-hot，也不是 float 概率。
- 预测时如果想看概率，可以额外用：

```python
probs = torch.softmax(outputs, dim=1)
```

### 4.7 Logistic 和 Softmax 的关系

课件给出的统一视角：

当 $K=2$ 时，Softmax 可以退化为 Sigmoid：

$$
h_1(x)=\frac{e^{z_1}}{e^{z_1}+e^{z_2}}
$$

上下同时除以 $e^{z_1}$：

$$
h_1(x)=\frac{1}{1+e^{z_2-z_1}}
$$

也就是：

$$
h_1(x)=\sigma(z_1-z_2)
$$

所以可以理解为：

- Logistic 回归：二分类的高效写法，学习一组权重或两类权重之差。
- Softmax 回归：多分类通用写法，学习 $K$ 组类别权重。

---

## 5. 梯度下降、反向传播和自动微分

### 5.1 梯度告诉参数“往哪边会让损失变大”

梯度是损失函数对参数的导数。比如：

$$
\frac{\partial L}{\partial w}
$$

它表示：当 $w$ 轻微变大时，损失 $L$ 会怎样变化。

- 梯度为正：$w$ 增大，损失倾向于增大。
- 梯度为负：$w$ 增大，损失倾向于减小。
- 梯度绝对值大：这个参数对损失更敏感。

梯度方向是损失上升最快方向，所以要让损失下降，就沿反方向更新：

$$
w \leftarrow w-\eta\frac{\partial L}{\partial w}
$$

逐项解释：

- $w$：当前参数。
- $\eta$：学习率，代码中常写作 `lr`。
- $\frac{\partial L}{\partial w}$：损失对参数的梯度。
- 减号：沿梯度反方向走。

偏置同理：

$$
b \leftarrow b-\eta\frac{\partial L}{\partial b}
$$

### 5.2 学习率太大或太小会怎样

学习率是超参数，不是模型自己学出来的参数。

| 学习率情况 | 现象 |
|---|---|
| 太小 | 损失下降很慢，训练很久还没学好 |
| 合适 | 损失总体下降，训练稳定 |
| 太大 | 损失震荡，甚至发散 |

课件中的 `lr = 0.1` 是示例值，不代表所有任务都合适。

### 5.3 反向传播是链式法则的程序实现

神经网络是一串复合函数：

$$
x \rightarrow z_1 \rightarrow a_1 \rightarrow z_2 \rightarrow \hat{y} \rightarrow L
$$

如果要知道第一层权重对最终损失的影响，就要从最后的损失一路反推回来。这就是反向传播。

链式法则的简单形式：

$$
\frac{dL}{dw}=\frac{dL}{d\hat{y}}\cdot\frac{d\hat{y}}{dz}\cdot\frac{dz}{dw}
$$

逐项解释：

- $\frac{dL}{d\hat{y}}$：预测值变化对损失的影响。
- $\frac{d\hat{y}}{dz}$：中间变量变化对预测的影响。
- $\frac{dz}{dw}$：参数变化对中间变量的影响。
- 连乘：把局部影响一路传回参数。

课件总结的两阶段：

| 阶段 | 方向 | 做什么 | 目的 |
|---|---|---|---|
| 前向传播 | 输入到输出 | 算预测值和损失 | 得到当前错多少 |
| 反向传播 | 损失到参数 | 算梯度 | 得到参数更新方向 |

### 5.4 计算图是什么

计算图是把一次前向计算中的变量和运算记录下来。比如：

```python
yhat = b + w * x_train_tensor
error = yhat - y_train_tensor
loss = (error ** 2).mean()
```

PyTorch 会记录：

- 哪些张量参与了计算。
- 这些张量之间经过了哪些运算。
- 哪些张量需要梯度。

当调用：

```python
loss.backward()
```

PyTorch 就沿着计算图反向应用链式法则，把梯度写入参数的 `.grad`。

课件中特别说明：计算图只包含参与梯度计算的张量。比如 `requires_grad=False` 的张量不会作为需要求梯度的参数出现在图中。

### 5.5 `requires_grad=True` 的作用

课件中手动创建参数：

```python
torch.manual_seed(42)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
```

`requires_grad=True` 表示：这个张量是要被训练的参数，PyTorch 需要记录它参与的计算，并在反向传播时计算它的梯度。

如果没有这个标记，`loss.backward()` 不会为它保存梯度。

### 5.6 梯度会累加，所以必须清零

课件强调：每次 `.backward()` 计算出的梯度会累加到 `.grad` 中，而不是自动覆盖。

所以训练循环中要清零：

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

也可以手动：

```python
b.grad.zero_()
w.grad.zero_()
```

如果忘记清零，后一次更新会混入前面 batch 的梯度，参数更新方向会变乱。

### 5.7 为什么更新参数要用 `torch.no_grad()`

手动更新参数时，课件写法：

```python
with torch.no_grad():
    b -= lr * b.grad
    w -= lr * w.grad
```

因为参数更新本身不是模型前向计算的一部分，不应该被记录进计算图。`torch.no_grad()` 告诉 PyTorch：这一段只是改参数，不需要跟踪梯度。

### 5.8 优化器把“更新参数”标准化

手动写：

```python
b = b - lr * b_grad
w = w - lr * w_grad
```

换成 PyTorch 优化器：

```python
optimizer = optim.SGD([b, w], lr=lr)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

再进一步，用模型参数：

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

这样无论模型有多少层、多少参数，都可以统一更新。

---

## 6. MLP 分类实践：从 make_moons 到完整训练流程

### 6.1 实验/案例：月牙形二分类

课件使用 `make_moons` 生成非线性二分类数据：

```python
X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=13
)
```

这个数据集的特点是两类点像两个月牙交错在一起。线性模型只能画一条直线，通常分不好；MLP 通过隐藏层和激活函数可以学出弯曲边界。

### 6.2 数据预处理：标准化不要泄漏验证集

课件写法：

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```

为什么验证集只用 `transform`，不能 `fit_transform`？

- `fit` 会计算均值和标准差。
- 如果在验证集上 `fit`，就相当于训练前偷看了验证集分布。
- 正确做法是只用训练集统计量，再把同样规则应用到验证集。

这叫避免数据泄漏。

### 6.3 二分类 MLP 代码骨架

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

x_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
x_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

train_data = TensorDataset(x_train_tensor, y_train_tensor)
val_data = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

形状对应：

| 变量 | 形状 | 含义 |
|---|---|---|
| `x_batch` | `[B, 2]` | 一个 batch 的二维输入 |
| `y_batch` | `[B, 1]` | 二分类标签 |
| `model(x_batch)` | `[B, 1]` | 每个样本一个 logit |
| `loss` | 标量 | 当前 batch 的平均损失 |

### 6.4 训练循环

```python
for epoch in range(n_epochs):
    model.train()
    train_losses = []

    for x_batch, y_batch in train_loader:
        output = model(x_batch)             # 前向传播
        loss = loss_fn(output, y_batch)     # 计算损失
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数
        optimizer.zero_grad()               # 清空梯度
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            val_losses.append(loss.item())
```

这里有几个必须理解的动作：

- `model.train()`：进入训练模式。后续学 Dropout、BatchNorm 时尤其重要。
- `model.eval()`：进入评估模式。
- `with torch.no_grad()`：验证阶段不更新参数，也不需要保存计算图，节省内存。
- `loss.item()`：把 PyTorch 标量张量转成 Python 数值，方便记录。

### 6.5 二分类预测怎么从 logits 变成类别

训练时输出 logits，损失函数内部处理 Sigmoid；预测时才显式转概率：

```python
with torch.no_grad():
    logits = model(x_val_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
```

含义：

- `logits`：任意实数。
- `torch.sigmoid(logits)`：正类概率。
- `>= 0.5`：概率超过 0.5 判为类别 1，否则类别 0。

也可以直接用 logits 判断：

```python
preds = (logits >= 0).long()
```

因为 Sigmoid 在 $z=0$ 时等于 0.5。

### 6.6 多分类 MLP 的区别

多分类模型最后输出 `num_classes` 个 logits：

```python
model = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes)
)

loss_fn = nn.CrossEntropyLoss()
```

标签要求：

```python
y_train_tensor = torch.LongTensor(y_train)
```

预测：

```python
with torch.no_grad():
    logits = model(x_val_tensor)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
```

注意：`argmax(logits)` 和 `argmax(softmax(logits))` 得到的类别相同，因为 Softmax 不改变大小顺序。

### 6.7 保存与加载模型

课件保存检查点：

```python
checkpoint = {
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses,
    'val_loss': val_losses
}

torch.save(checkpoint, 'model_checkpoint.pth')
```

恢复训练时：

```python
checkpoint = torch.load('model_checkpoint.pth')
new_model.load_state_dict(checkpoint['model_state_dict'])
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
new_model.train()
```

部署预测时：

```python
deploy_model.load_state_dict(checkpoint['model_state_dict'])
deploy_model.eval()

with torch.no_grad():
    predictions = deploy_model(new_inputs)
```

区别：

- 恢复训练：需要模型参数、优化器状态、epoch、历史损失，并设置 `train()`。
- 部署预测：通常只需要模型参数，并设置 `eval()`。

---

## 7. 公式与代码对应关系

### 7.1 线性层

公式：

$$
z=Wx+b
$$

代码：

```python
layer = nn.Linear(input_dim, output_dim)
z = layer(x)
```

对应：

- `input_dim` 对应输入特征数。
- `output_dim` 对应该层神经元数。
- `layer.weight` 对应 $W$。
- `layer.bias` 对应 $b$。
- `z` 是 logits 或隐藏层激活前的值。

### 7.2 激活函数

公式：

$$
a=f(z)
$$

代码：

```python
a = torch.relu(z)
```

或：

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
```

### 7.3 MSE

公式：

$$
L=\frac{1}{N}\sum_i(\hat{y}_i-y_i)^2
$$

代码：

```python
loss_fn = nn.MSELoss(reduction='mean')
loss = loss_fn(yhat, y_train_tensor)
```

### 7.4 BCEWithLogitsLoss

数学流程：

$$
z \rightarrow \sigma(z) \rightarrow \text{BCE}
$$

代码：

```python
output = model(x_batch)          # logits
loss = nn.BCEWithLogitsLoss()(output, y_batch)
```

不要在模型最后加 `Sigmoid`。

### 7.5 CrossEntropyLoss

数学流程：

$$
z_1,\dots,z_K \rightarrow \text{Softmax} \rightarrow \text{Cross-Entropy}
$$

代码：

```python
output = model(x_batch)          # [B, K] logits
loss = nn.CrossEntropyLoss()(output, y_batch)  # y_batch: [B], long
```

不要在模型最后加 `Softmax`。

### 7.6 梯度下降

公式：

$$
\theta \leftarrow \theta-\eta\nabla_\theta L
$$

代码：

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

对应：

- $\theta$：模型所有参数。
- $\nabla_\theta L$：`loss.backward()` 算出的梯度。
- $\eta$：优化器中的 `lr`。
- 参数更新：`optimizer.step()`。
- 清空梯度：`optimizer.zero_grad()`。

---

## 8. 常见错误与易混点

### 8.1 把 logits 当概率

错误理解：模型输出 `2.3`，所以概率是 2.3。  
正确理解：`2.3` 是 logit，不是概率。二分类要经过 Sigmoid，多分类要经过 Softmax 后才能解释为概率。

### 8.2 `Sigmoid + BCEWithLogitsLoss` 重复使用 Sigmoid

错误代码：

```python
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCEWithLogitsLoss()
```

正确搭配：

```python
model = nn.Sequential(nn.Linear(2, 1))
loss_fn = nn.BCEWithLogitsLoss()
```

或者：

```python
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()
```

实践中更推荐第一种。

### 8.3 `Softmax + CrossEntropyLoss` 重复使用 Softmax

错误代码：

```python
model = nn.Sequential(
    nn.Linear(10, 3),
    nn.Softmax(dim=1)
)
loss_fn = nn.CrossEntropyLoss()
```

正确代码：

```python
model = nn.Sequential(nn.Linear(10, 3))
loss_fn = nn.CrossEntropyLoss()
```

### 8.4 二分类标签形状不匹配

模型输出 `[B, 1]`，标签却是 `[B]`，可能报错或广播出奇怪结果。推荐：

```python
y = torch.FloatTensor(y).view(-1, 1)
```

### 8.5 多分类标签写成 one-hot

`nn.CrossEntropyLoss()` 默认需要类别索引：

```python
y = torch.LongTensor([0, 2, 1, 1])
```

不是：

```python
y = torch.FloatTensor([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
])
```

### 8.6 忘记 `optimizer.zero_grad()`

梯度会累加。忘记清零会让更新混入历史梯度，训练曲线可能异常。

### 8.7 验证阶段忘记 `torch.no_grad()`

验证不需要求梯度。如果不加 `torch.no_grad()`，会浪费内存，还可能让代码更慢。

### 8.8 忘记切换 `train()` 和 `eval()`

虽然简单 MLP 中如果没有 Dropout/BatchNorm，影响可能不明显，但这是必须养成的习惯：

```python
model.train()  # 训练
model.eval()   # 验证/测试/部署
```

### 8.9 在验证集上 fit 标准化器

错误：

```python
X_val = scaler.fit_transform(X_val)
```

正确：

```python
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```

### 8.10 以为 epoch 多就一定更好

epoch 太少可能欠拟合；epoch 太多可能过拟合。要看训练损失和验证损失：

- 训练损失和验证损失都高：可能没学够或模型太弱。
- 训练损失低，验证损失高：可能过拟合。
- 两者都稳定下降：训练状态较健康。

---

## 9. 关键概念速查表

| 概念 | 通俗理解 | 代码/公式抓手 |
|---|---|---|
| 参数 | 模型要学习的数 | `model.parameters()` |
| 权重 $W$ | 输入特征的影响强度 | `nn.Linear.weight` |
| 偏置 $b$ | 整体平移量 | `nn.Linear.bias` |
| Logit | 概率转换前的原始分数 | `model(x)` |
| Sigmoid | 把一个 logit 变成二分类概率 | `torch.sigmoid(logits)` |
| Softmax | 把多个 logits 变成概率分布 | `torch.softmax(logits, dim=1)` |
| 激活函数 | 给模型加入非线性 | `nn.ReLU()` |
| 损失函数 | 衡量预测错得多严重 | `loss_fn(output, target)` |
| 梯度 | 参数对损失的影响方向和大小 | `.grad` |
| 学习率 | 每次更新走多大一步 | `lr=0.1` |
| 反向传播 | 从损失反推每个参数的梯度 | `loss.backward()` |
| 优化器 | 根据梯度更新参数 | `optimizer.step()` |
| 清梯度 | 防止梯度累加 | `optimizer.zero_grad()` |
| Dataset | 定义如何取样本 | `__getitem__`, `__len__` |
| DataLoader | 自动组 batch | `DataLoader(..., batch_size=16)` |
| Epoch | 训练集完整用过一遍 | `for epoch in range(n_epochs)` |
| Iteration | 一次 batch 更新 | 内层 `for x_batch, y_batch in loader` |
| `train()` | 训练模式 | `model.train()` |
| `eval()` | 评估模式 | `model.eval()` |
| `no_grad()` | 不记录梯度 | `with torch.no_grad():` |
| `state_dict()` | 模型参数字典 | 保存/加载模型 |

---

## 10. 自测题

### 题 1：训练循环排序

下面 5 个动作应该如何排序？

A. `optimizer.step()`  
B. `loss.backward()`  
C. `output = model(x_batch)`  
D. `optimizer.zero_grad()`  
E. `loss = loss_fn(output, y_batch)`

**参考答案**：C -> E -> B -> A -> D。  
有些代码会把 D 放在 batch 开头，也可以，但必须保证每次反向传播前不会残留上一轮梯度。

### 题 2：Batch 和 Epoch

训练集有 1,000 个样本，`batch_size=50`，训练 10 个 epoch。请问一共更新多少次参数？

**参考答案**：每个 epoch 有 $1000/50=20$ 次 iteration，10 个 epoch 共 $20\times10=200$ 次参数更新。

### 题 3：二分类输出层

二分类任务中，模型最后一层是：

```python
nn.Linear(8, 1)
```

损失函数是：

```python
nn.BCEWithLogitsLoss()
```

请问模型输出是概率吗？训练时还需要加 `Sigmoid` 吗？

**参考答案**：输出不是概率，是 logit。训练时不需要加 `Sigmoid`，因为 `BCEWithLogitsLoss` 内部已经包含 Sigmoid 相关计算。预测时如果要看概率，再使用 `torch.sigmoid(logits)`。

### 题 4：多分类标签类型

三分类任务中，模型输出形状是 `[32, 3]`，使用 `nn.CrossEntropyLoss()`。标签应该是什么形状和类型？

**参考答案**：标签形状应是 `[32]`，每个元素是类别索引，如 0、1、2，类型通常是 `torch.long`。不需要 one-hot。

### 题 5：为什么 MLP 需要激活函数

有同学说：“我不用 ReLU，多堆几层 `Linear` 也能变复杂。”这句话哪里有问题？

**参考答案**：多层线性变换复合后仍然等价于一层线性变换。没有激活函数，模型仍只能表达线性决策边界。激活函数引入非线性，MLP 才能拟合像 `make_moons` 这样的非线性可分数据。

### 题 6：标准化的数据泄漏

为什么下面代码有问题？

```python
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
```

**参考答案**：验证集上又调用了 `fit_transform`，等于使用验证集自己的均值和标准差，造成评估规则与训练时不一致，也有数据泄漏风险。应改为：

```python
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```

### 题 7：梯度清零

如果每个 batch 后不调用 `optimizer.zero_grad()`，会发生什么？

**参考答案**：PyTorch 的梯度默认累加，后续 batch 的梯度会加到之前的 `.grad` 上。这样参数更新不再只反映当前 batch，训练可能变慢、不稳定或发散。

### 题 8：验证阶段

验证阶段为什么通常写：

```python
model.eval()
with torch.no_grad():
    ...
```

**参考答案**：`model.eval()` 切换到评估模式，使 Dropout、BatchNorm 等层按验证/推理规则工作。`torch.no_grad()` 不记录计算图，节省内存和计算，因为验证不更新参数。

### 题 9：Softmax 的维度

多分类 logits 形状是 `[B, K]`，为什么通常写：

```python
torch.softmax(logits, dim=1)
```

**参考答案**：`dim=1` 表示在类别维度上做 Softmax，让每个样本的 $K$ 个类别概率之和为 1。如果维度用错，概率归一化方向就错了。

### 题 10：Logistic 和 Softmax

为什么说 Logistic 回归可以看作二分类 Softmax 的特例？

**参考答案**：当类别数 $K=2$ 时，Softmax 中某一类概率可写成 $\frac{1}{1+e^{-(z_1-z_2)}}$，形式就是 Sigmoid。因此二分类 Logistic 可以看作只学习两类 logit 差值的 Softmax 特例。

---

## 11. 考前最后总复习

### 11.1 一条主线串起来

训练 MLP 可以按下面这条线回忆：

1. 输入样本 $x$ 先整理成 Tensor。
2. `Dataset` 定义如何取样本，`DataLoader` 负责组成 batch。
3. batch 输入模型，线性层计算 $z=Wx+b$。
4. 隐藏层后接 ReLU 等激活函数，引入非线性。
5. 输出层给出 logits。
6. 二分类用 `BCEWithLogitsLoss`，多分类用 `CrossEntropyLoss`。
7. 损失函数衡量当前预测错多少。
8. `loss.backward()` 沿计算图反向计算梯度。
9. `optimizer.step()` 根据梯度更新参数。
10. `optimizer.zero_grad()` 清空梯度，准备下一个 batch。
11. 每个 epoch 后看训练损失和验证损失，判断是否学得更好。

### 11.2 二分类和多分类对比

| 项目 | 二分类 | 多分类 |
|---|---|---|
| 输出层 | `nn.Linear(hidden, 1)` | `nn.Linear(hidden, K)` |
| 输出含义 | 一个 logit | K 个 logits |
| 推荐损失 | `BCEWithLogitsLoss` | `CrossEntropyLoss` |
| 标签类型 | float | long |
| 标签形状 | `[B, 1]` | `[B]` |
| 训练时模型末尾 | 不加 Sigmoid | 不加 Softmax |
| 看概率 | `torch.sigmoid(logits)` | `torch.softmax(logits, dim=1)` |
| 判类别 | `logits >= 0` 或 `prob >= 0.5` | `argmax(logits, dim=1)` |

### 11.3 训练代码最小模板

```python
for epoch in range(n_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            val_loss = loss_fn(output, y_batch)
```

### 11.4 考前重点检查

- 能不能解释：为什么没有激活函数的多层网络仍然是线性模型？
- 能不能写出：Sigmoid、Softmax、BCE、Cross-Entropy 的公式，并说明每一项含义？
- 能不能区分：logits、概率、类别预测？
- 能不能判断：二分类和多分类分别该用什么损失函数、标签类型和输出形状？
- 能不能说清：`loss.backward()`、`optimizer.step()`、`optimizer.zero_grad()` 各自做什么？
- 能不能解释：Batch、Iteration、Epoch 的区别？
- 能不能指出：训练集 `shuffle=True`、验证集 `shuffle=False` 的原因？
- 能不能说明：为什么验证阶段要 `model.eval()` 和 `torch.no_grad()`？
- 能不能发现：在验证集上 `fit_transform`、重复 Sigmoid/Softmax、忘记清梯度这些错误？

### 11.5 最后用一段话收束

MLP 的核心不是把很多层机械堆起来，而是让数据经过一系列“线性变换 + 非线性激活”后，变到更容易分类的表示空间。训练时，模型先用当前参数产生 logits，损失函数衡量 logits 对真实标签有多不合适，反向传播计算每个参数对损失的影响，优化器再按学习率更新参数。只要你能把“数据 batch -> 前向传播 -> 损失 -> 反向传播 -> 参数更新 -> 验证评估”这条链路讲清楚，MLP 这一章的大部分公式、代码和考试题都能落到同一套逻辑里。
