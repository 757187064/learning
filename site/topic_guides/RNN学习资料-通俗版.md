# RNN 学习资料（通俗版）

## 资料来源说明

本讲义根据课程课件 `/Users/sakiko/Public/deeplearing/课件和笔记/课件/循环神经网络-学生分发版-终版2026.ipynb` 重新整理，内容覆盖课件中的：

- 序列数据与循环神经网络
- `nn.RNNCell`、`nn.RNN` 的输入输出
- `batch_first`、`output`、`h_n` 的 shape
- 堆叠 RNN、双向 RNN、深度双向 RNN
- RNN 的梯度消失和梯度爆炸
- GRU 与 LSTM
- Padding 与 Packing
- 1D 卷积、TCN
- IMDB 情感分类综合案例

写作风格参考了用户样例 `/Users/sakiko/Documents/New project 2/RNN学习资料-详细版.md` 的教学节奏：先建立直觉，再解释符号、公式、shape 和代码。但本文重新组织了语言、例子和章节，不照抄原文。

---

## 全局学习路线图

这一章容易乱，是因为它同时出现了三个层面的东西：

1. 数据层面：序列、时间步、batch、padding
2. 模型层面：RNN、GRU、LSTM、双向、堆叠
3. 代码层面：`output`、`h_n`、`PackedSequence`、`batch_first`

建议按下面这条线学：

```text
序列数据为什么不能当普通向量
    ↓
一个时间步：x_t 和 h_{t-1} 如何变成 h_t
    ↓
一条序列：RNN 如何一步一步读完整段输入
    ↓
batch 和 shape：N、L、F、H 分别是什么
    ↓
RNN 输出：output 和 h_n 到底差在哪里
    ↓
堆叠 / 双向：层数和方向数如何改变 shape
    ↓
长期依赖问题：为什么普通 RNN 会忘远处信息
    ↓
GRU：用两个门控制记忆更新
    ↓
LSTM：用 cell state 单独管理长期记忆
    ↓
Padding / Packing：真实文本长短不一时怎么喂给模型
    ↓
1D 卷积 / TCN：另一类处理序列的方法
    ↓
情感分类实践：Embedding → LSTM → Dropout → FC
    ↓
考前复盘：易混点、自测题、速查表
```

---

## 1. 序列数据为什么不能当普通向量

### 1.1 什么是序列数据

序列数据就是一组有先后顺序的数据。顺序不是装饰，而是含义的一部分。

常见例子：

- 一句话：第 1 个词、第 2 个词、第 3 个词
- 一段语音：前 10ms、后 10ms、再后 10ms
- 股票价格：昨天、今天、明天
- 传感器记录：第 1 秒、第 2 秒、第 3 秒

如果把一句话里的词打乱，意思可能完全改变：

```text
I do not like this movie.
I like this movie, not bad.
```

两句话都可能出现 `I`、`like`、`movie`、`not`，但情感倾向不同。区别不只在词本身，还在词出现的位置和前后关系。

### 1.2 普通向量方法的问题

如果把文本简单看成一个普通向量，比如“这个词出现了几次、那个词出现了几次”，模型会更关注词有没有出现，却很难自然理解顺序。

普通 MLP 的典型限制：

- 输入维度通常固定，不适合直接处理长度变化很大的句子。
- 它没有内置的时间方向，不知道哪个词先出现、哪个词后出现。
- 它不能自然表达“读到后面时要参考前面内容”。

CNN 可以看局部模式，比如连续几个词组成的片段，但普通卷积也不是专门为“一步一步携带历史信息”设计的。

RNN 这一章的核心问题可以概括为：

**模型怎样一边读当前内容，一边保留前面已经读过的信息。**

---

## 2. 一个时间步：RNN 单元在做什么

先不要急着看一整句话，先看最小动作：只处理一个时间步。

在第 `t` 个时间步，RNN 会拿到两样东西：

- `x_t`：当前输入。比如当前这个词的向量。
- `h_{t-1}`：上一个时间步留下来的隐藏状态，也就是前面内容的压缩表示。

RNN 要输出：

- `h_t`：当前时间步更新后的隐藏状态。

### 2.1 标准 RNN 公式

标准 RNN 的隐藏状态更新可以写成：

```text
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
```

也常写成 PyTorch 更接近的形式：

```text
h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
```

### 2.2 公式逐项拆解

| 符号 | 含义 | 直觉 |
|:---|:---|:---|
| `x_t` | 当前时间步的输入向量 | 当前读到的词、当前时刻的观测值 |
| `h_{t-1}` | 上一时间步隐藏状态 | 之前读过内容留下的记忆 |
| `W_xh` / `W_ih` | 输入到隐藏状态的权重 | 决定当前输入怎样影响新状态 |
| `W_hh` | 隐藏状态到隐藏状态的权重 | 决定旧记忆怎样延续到新状态 |
| `b_h` / `b_ih` / `b_hh` | 偏置 | 给变换增加可调的平移量 |
| `tanh` | 激活函数 | 把状态压到 `(-1, 1)` 附近，增加非线性 |
| `h_t` | 新隐藏状态 | 当前输入和旧记忆融合后的结果 |

把公式翻译成学习时更容易懂的动作：

```text
当前输入先变换一下
+ 旧隐藏状态也变换一下
+ 偏置
再经过 tanh
= 新隐藏状态
```

### 2.3 和课件代码的对应关系

课件中用 `rnn_step` 演示了单步 RNN：

```python
def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):
    h_t = np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    return h_t
```

对应关系：

| 代码 | 公式 | 含义 |
|:---|:---|:---|
| `x_t` | `x_t` | 当前时间步输入 |
| `h_prev` | `h_{t-1}` | 上一步隐藏状态 |
| `W_xh @ x_t` | `W_xh x_t` | 当前输入的贡献 |
| `W_hh @ h_prev` | `W_hh h_{t-1}` | 历史信息的贡献 |
| `np.tanh(...)` | `tanh(...)` | 得到新隐藏状态 |

课件里的示例序列有 3 个时间步，每一步输入都是 3 维向量，隐藏状态是 4 维向量：

```text
input_size = 3
hidden_size = 4

x_t      shape: (3,)
h_prev   shape: (4,)
h_t      shape: (4,)
```

---

## 3. 一条序列：RNN 怎样读完整段输入

一条序列不是只做一次 `rnn_step`，而是反复做。

假设一句话有 5 个词：

```text
x_1, x_2, x_3, x_4, x_5
```

RNN 的处理过程是：

```text
h_1 = RNNCell(x_1, h_0)
h_2 = RNNCell(x_2, h_1)
h_3 = RNNCell(x_3, h_2)
h_4 = RNNCell(x_4, h_3)
h_5 = RNNCell(x_5, h_4)
```

这里最重要的是：同一个 RNN 单元在每个时间步重复使用。它不是第 1 步一套参数、第 2 步又一套参数，而是所有时间步共享同一组参数。

### 3.1 参数共享是什么意思

参数共享就是：

```text
每个时间步都用同一套 W_xh、W_hh、b_h。
```

这样做有两个好处：

- 序列长度可以变化。因为同一套规则可以重复用很多次。
- 参数量不会随着序列长度线性增加。10 个词和 100 个词不是分别训练 10 套和 100 套权重。

### 3.2 隐藏状态为什么像“记忆”

`h_5` 直接依赖 `h_4`，而 `h_4` 依赖 `h_3`，`h_3` 又依赖 `h_2`。所以 `h_5` 虽然是最后一步的状态，但它间接包含了前面很多步的信息。

这就是 RNN 能处理序列的根本原因：

```text
当前状态 = 当前输入 + 过去状态的延续
```

---

## 4. batch 和 shape：N、L、F、H 怎么看

RNN 的难点经常不是公式，而是 shape。先把四个字母固定下来：

| 符号 | 英文 | 中文 | 例子 |
|:---|:---|:---|:---|
| `N` | batch size | 一个 batch 有多少条样本 | 一次喂 64 条评论 |
| `L` | sequence length | 每条序列有多少个时间步 | 一条评论截断到最多 500 个词 |
| `F` | feature size / input size | 每个时间步的输入维度 | 每个词用 128 维向量表示 |
| `H` | hidden size | 隐藏状态维度 | LSTM 隐藏层 256 维 |

### 4.1 从一条序列开始看

如果一条句子有 5 个词，每个词已经表示成 100 维向量：

```text
一条序列 shape = (L, F) = (5, 100)
```

意思是：

- 有 5 个时间步。
- 每个时间步是 100 维。

### 4.2 再加入 batch

训练时通常一次输入多条句子。比如一次输入 32 条，每条 5 个词，每个词 100 维：

```text
batch shape = (N, L, F) = (32, 5, 100)
```

意思是：

- 第 1 维 `32`：这一批有 32 条句子。
- 第 2 维 `5`：每条句子有 5 个时间步。
- 第 3 维 `100`：每个时间步是 100 维向量。

### 4.3 `batch_first=True` 到底控制什么

PyTorch 的 `nn.RNN`、`nn.GRU`、`nn.LSTM` 默认输入是：

```text
(L, N, F)
```

也就是 sequence 在前，batch 在中间。

但 DataLoader 里更常见、更符合直觉的是：

```text
(N, L, F)
```

所以常常设置：

```python
nn.RNN(input_size=F, hidden_size=H, batch_first=True)
nn.GRU(input_size=F, hidden_size=H, batch_first=True)
nn.LSTM(input_size=F, hidden_size=H, batch_first=True)
```

设置后，普通张量输入输出的 batch 维度就放在最前面。

要注意：`batch_first=True` 只改变输入和 `output` 的维度顺序，不改变 `h_n` 的维度顺序。`h_n` 仍然是：

```text
(num_layers * num_directions, N, H)
```

这是考试和写代码时非常容易错的地方。

---

## 5. `nn.RNNCell` 和 `nn.RNN` 的区别

课件专门比较了这两个接口。

### 5.1 `nn.RNNCell`：一次只处理一个时间步

`nn.RNNCell` 输入：

```text
x_t     (N, F)
h_prev  (N, H)
```

输出：

```text
h_t     (N, H)
```

它只负责一步，循环要自己写。适合自定义循环逻辑，比如某一步要加注意力、要根据条件改变输入、要做自回归生成。

### 5.2 `nn.RNN`：直接处理整条序列

如果设置 `batch_first=True`，`nn.RNN` 输入：

```text
x       (N, L, F)
```

输出：

```text
output  (N, L, H)
h_n     (num_layers * num_directions, N, H)
```

它内部已经帮你按时间步循环完。标准序列建模时更常用。

### 5.3 对比表

| 对比项 | `nn.RNNCell` | `nn.RNN` |
|:---|:---|:---|
| 处理粒度 | 单个时间步 | 整条序列 |
| 是否手写循环 | 需要 | 不需要 |
| 当前输入 shape | `(N, F)` | `(N, L, F)` 或 `(L, N, F)` |
| 隐藏状态 | 自己传入、自己保存 | 模块返回 `h_n` |
| 适合场景 | 自定义复杂流程 | 标准 RNN/GRU/LSTM 层 |

---

## 6. RNN 输出：`output` 和 `h_n`

这是本章最重要的 shape 之一。

假设：

```python
rnn = nn.RNN(input_size=F, hidden_size=H, batch_first=True)
output, h_n = rnn(x)
```

输入：

```text
x       (N, L, F)
```

输出：

```text
output  (N, L, H)
h_n     (1, N, H)        # 单层单向时
```

### 6.1 `output` 是什么

`output` 保存每个时间步的隐藏状态。

如果序列长度 `L=5`，它相当于保存：

```text
h_1, h_2, h_3, h_4, h_5
```

所以：

```text
output[:, 0, :]   第 1 个时间步的隐藏状态
output[:, 1, :]   第 2 个时间步的隐藏状态
output[:, -1, :]  最后一个时间步的隐藏状态
```

适合用 `output` 的任务：

- 每个词都要判断类别：命名实体识别、词性标注
- 每个时间步都要输出预测：时间序列逐步预测
- 注意力机制需要看所有时间步的表示

### 6.2 `h_n` 是什么

`h_n` 保存最终隐藏状态。单层单向时：

```text
h_n shape = (1, N, H)
h_n[-1] shape = (N, H)
```

适合用 `h_n` 的任务：

- 一整句话只判断一个标签：情感分类
- 一段时间序列只判断一个类别：动作识别、异常检测
- 编码器要把整条序列压缩成一个向量

### 6.3 单层单向时的关系

在单层单向 RNN/GRU 中，通常有：

```python
torch.allclose(output[:, -1, :], h_n[-1])
```

课件中的 shape 验证也展示了这一点。

但这句话有前提：单层、单向、没有 packing 干扰最后时间步含义。

### 6.4 双向时为什么不能直接照搬

双向 RNN 中：

```text
output shape = (N, L, 2H)
h_n shape    = (2, N, H)        # 单层双向时
```

`output[:, -1, :]` 的含义是：

```text
[正向在最后一个位置的状态 | 反向在最后一个位置的状态]
```

但反向 RNN 是从右往左读的。反向的“最终状态”对应的是原序列最左边的位置，不是 `output[:, -1, :]` 里的反向部分。

所以双向分类时更稳妥的取法是：

```python
final = torch.cat([h_n[-2], h_n[-1]], dim=1)
```

含义：

- `h_n[-2]`：最后一层正向最终状态
- `h_n[-1]`：最后一层反向最终状态
- 拼接后 shape 是 `(N, 2H)`

---

## 7. 堆叠 RNN：多层时 shape 怎么变

堆叠 RNN 就是把多层 RNN 叠起来：

```text
第 1 层 RNN 的 output → 第 2 层 RNN 的输入
第 2 层 RNN 的 output → 第 3 层 RNN 的输入
```

课件中的示例：

```python
stacked_rnn = nn.RNN(
    input_size=F,
    hidden_size=H,
    num_layers=3,
    batch_first=True,
    bidirectional=False
)

output, final_hidden = stacked_rnn(x)
```

如果：

```text
x shape = (N, L, F)
```

那么单向 3 层 RNN：

```text
output       shape = (N, L, H)
final_hidden shape = (3, N, H)
```

解释：

- `output` 只保存最后一层所有时间步的隐藏状态。
- `final_hidden[0]` 是第 1 层最后时刻状态。
- `final_hidden[1]` 是第 2 层最后时刻状态。
- `final_hidden[2]` 是第 3 层最后时刻状态。
- 做整体分类时，常取 `final_hidden[-1]`。

### 7.1 堆叠的代价

堆叠层数增加，表达能力可能增强，但代价也明显：

- 参数量增加。
- 计算更慢。
- 梯度传播路径更长。
- 更容易过拟合。

课件提醒：两层堆叠 RNN 计算代价已经不低，因为它既有时间方向的顺序依赖，又有层间依赖。

---

## 8. 双向 RNN：两个方向一起读

单向 RNN 只能从左到右读：

```text
x_1 → x_2 → x_3 → ... → x_L
```

双向 RNN 同时做两件事：

```text
正向：x_1 → x_2 → ... → x_L
反向：x_L → x_{L-1} → ... → x_1
```

然后把两个方向的隐藏状态拼接起来。

### 8.1 双向的 shape

课件中的双向示例：

```python
bi_rnn = nn.RNN(
    input_size=F,
    hidden_size=H,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)

output_bi, hidden_bi = bi_rnn(x)
```

如果：

```text
x shape = (N, L, F)
```

那么：

```text
output_bi shape = (N, L, 2H)
hidden_bi shape = (2, N, H)
```

最后一维为什么是 `2H`？

```text
正向 H 维 + 反向 H 维 = 2H 维
```

### 8.2 双向适合什么任务

适合：

- 情感分类：整条评论已经完整给出，可以同时看前文和后文。
- 命名实体识别：判断当前词是不是人名、地名时，后文很有帮助。
- 离线文本分析：不要求实时逐字生成。

不适合：

- 实时生成。
- 在线预测时不能看到未来信息的任务。
- 严格因果建模，比如预测下一个 token 时不能偷看后文。

### 8.3 多层双向时的隐藏状态

如果：

```python
num_layers = 3
bidirectional = True
```

那么：

```text
h_n shape = (num_layers * 2, N, H) = (6, N, H)
```

索引大致按“层和方向”排列。最后一层两个方向通常是：

```python
forward_last = h_n[-2]
backward_last = h_n[-1]
final = torch.cat([forward_last, backward_last], dim=1)
```

---

## 9. 长期依赖问题：普通 RNN 为什么会忘远处信息

普通 RNN 理论上可以把早期信息一路传到后面，但训练时会遇到梯度问题。

### 9.1 从公式看问题

RNN 的状态更新：

```text
h_k = tanh(W_xh x_k + W_hh h_{k-1} + b_h)
```

训练时，如果最终损失 `L` 依赖最后状态 `h_n`，那么早期状态 `h_t` 对最后损失的影响要经过很多步传播。

课件中给出链式法则：

```text
∂L/∂W_hh = Σ_{t=1}^{n} (∂L/∂h_n) · (∂h_n/∂h_t) · (∂h_t/∂W_hh)
```

危险项是：

```text
∂h_n/∂h_t
```

它表示第 `t` 步的信息要传到第 `n` 步，中间跨了多少步。

### 9.2 连乘为什么危险

根据链式法则：

```text
∂h_n/∂h_t = Π_{k=t+1}^{n} ∂h_k/∂h_{k-1}
```

这是一个连乘。只要每一步的传递系数略小于 1，乘很多次就会接近 0：

```text
0.9^10  ≈ 0.35
0.9^50  ≈ 0.005
0.9^100 ≈ 0.00003
```

这就是梯度消失：远处的信息对当前损失的影响越来越弱，模型很难学到长距离关系。

如果每一步的系数略大于 1，乘很多次就会变得很大，这就是梯度爆炸。

### 9.3 直觉解释

想象一句很长的评论，开头说：

```text
Although the first half was boring, ...
```

后面很久才说：

```text
the ending was brilliant.
```

判断情感时，模型必须把前后关系合起来。如果普通 RNN 在长距离传播中逐渐丢掉了前面的转折信息，就可能只抓住局部词，判断错误。

GRU 和 LSTM 的出现，就是为了让信息流动更可控，减轻这种长距离传递困难。

---

## 10. GRU：用两个门控制记忆更新

GRU，全称 Gated Recurrent Unit，课件中强调它的核心思想是引入门控机制控制信息流动。

普通 RNN 每一步都直接用 `tanh(...)` 生成新状态。GRU 不这么粗放，它会先问两个问题：

1. 生成新候选状态时，要参考多少过去？
2. 最终状态里，旧记忆和新候选各占多少？

### 10.1 GRU 的两个门

| 门 | 符号 | 激活函数 | 作用 |
|:---|:---|:---|:---|
| 重置门 | `r_t` | sigmoid | 控制候选状态计算时使用多少旧记忆 |
| 更新门 | `z_t` | sigmoid | 控制最终状态中新旧信息的混合比例 |

sigmoid 的输出在 0 到 1 之间，所以门可以理解成“比例控制器”。

### 10.2 常见 GRU 公式

一组常见写法是：

```text
r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

### 10.3 公式逐项拆解

| 符号 | 含义 | 学习时怎么理解 |
|:---|:---|:---|
| `r_t` | reset gate，重置门 | 决定旧记忆参与候选状态计算的程度 |
| `z_t` | update gate，更新门 | 决定旧状态保留多少、新候选采纳多少 |
| `n_t` | candidate hidden state，候选状态 | 当前时刻可能采用的新记忆 |
| `⊙` | 按元素相乘 | 每个维度单独控制比例 |
| `h_{t-1}` | 旧隐藏状态 | 过去信息 |
| `h_t` | 新隐藏状态 | 旧状态和候选状态混合后的结果 |

最后一行最关键：

```text
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

如果某个维度上 `z_t` 接近 1：

```text
h_t ≈ h_{t-1}
```

说明模型倾向于保留旧记忆。

如果某个维度上 `z_t` 接近 0：

```text
h_t ≈ n_t
```

说明模型倾向于采用新候选。

### 10.4 GRU 的 shape

使用 `nn.GRU` 时，如果：

```python
gru = nn.GRU(input_size=F, hidden_size=H, batch_first=True)
output, h_n = gru(x)
```

输入输出和 RNN 很像：

```text
x       (N, L, F)
output  (N, L, H)
h_n     (num_layers * num_directions, N, H)
```

GRU 没有 LSTM 的 `cell state`，所以只返回：

```text
output, h_n
```

---

## 11. LSTM：把长期记忆单独管理

LSTM，全称 Long Short-Term Memory。它比 GRU 更复杂，因为它有两套状态：

- `h_t`：hidden state，隐藏状态，更像当前对外输出的表示。
- `C_t`：cell state，细胞状态，更像长期记忆通道。

课件中强调：LSTM 的核心是“旧的忘记多少，新的输入多少”。

### 11.1 LSTM 的三个门

| 门 | 符号 | 激活函数 | 作用 |
|:---|:---|:---|:---|
| 遗忘门 | `f_t` | sigmoid | 控制旧 cell state 保留多少 |
| 输入门 | `i_t` | sigmoid | 控制新信息写入多少 |
| 输出门 | `o_t` | sigmoid | 控制 cell state 中多少内容变成隐藏状态 |

### 11.2 LSTM 公式

课件中的常见写法：

```text
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Ĉ_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ Ĉ_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

### 11.3 公式逐项拆解

| 符号 | 含义 | 直觉 |
|:---|:---|:---|
| `[h_{t-1}, x_t]` | 拼接旧隐藏状态和当前输入 | 同时看过去理解和当前新内容 |
| `f_t` | 遗忘门 | 旧长期记忆哪些维度继续留着 |
| `i_t` | 输入门 | 新候选内容哪些维度可以写入 |
| `Ĉ_t` | 候选 cell state | 当前输入产生的新记忆草稿 |
| `C_t` | 新 cell state | 长期记忆更新后的结果 |
| `o_t` | 输出门 | 长期记忆中哪些内容拿出来形成 `h_t` |
| `h_t` | 新 hidden state | 当前时间步对外输出的短期表示 |

### 11.4 LSTM 的更新逻辑

可以按三步看：

1. 用 `f_t ⊙ C_{t-1}` 决定旧长期记忆保留多少。
2. 用 `i_t ⊙ Ĉ_t` 决定新内容写入多少。
3. 用 `o_t ⊙ tanh(C_t)` 决定当前隐藏状态输出多少。

LSTM 比普通 RNN 更适合长序列，是因为 `C_t` 提供了一条相对平稳的信息通道，旧信息不必每一步都被完全重写。

### 11.5 LSTM 的 shape

使用 `nn.LSTM`：

```python
lstm = nn.LSTM(input_size=F, hidden_size=H, batch_first=True)
output, (h_n, c_n) = lstm(x)
```

如果单层单向：

```text
x       (N, L, F)
output  (N, L, H)
h_n     (1, N, H)
c_n     (1, N, H)
```

如果多层双向：

```text
output  (N, L, 2H)
h_n     (num_layers * 2, N, H)
c_n     (num_layers * 2, N, H)
```

LSTM 和 GRU/RNN 返回值的区别：

```text
RNN/GRU:  output, h_n
LSTM:     output, (h_n, c_n)
```

---

## 12. RNN、GRU、LSTM 模型对比表

| 模型 | 状态数量 | 门控数量 | 参数量 | 长期依赖能力 | 训练速度 | 适合场景 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| RNN | 1 个 `h_t` | 0 | 少 | 弱 | 快 | 短序列、教学理解、简单基线 |
| GRU | 1 个 `h_t` | 2 | 中 | 中等偏强 | 中 | 数据量不大、希望比 RNN 稳定 |
| LSTM | `h_t` + `C_t` | 3 | 多 | 强 | 较慢 | 长文本、长时间序列、经典序列任务 |

不要简单理解成 LSTM 永远最好。更合理的选择方式：

- 想快速建立基线：RNN 或 GRU。
- 序列较长但模型不想太重：GRU。
- 长期依赖明显、需要更稳定记忆：LSTM。
- 数据量很大、任务复杂：还可以考虑 Transformer。

---

## 13. Padding 和 Packing：真实文本长度不一样怎么办

现实里的文本长度不可能整齐一致：

```text
I love it.                         长度 3
This movie is really wonderful.    长度 5
Bad.                               长度 1
```

但一个 batch 需要组成规则张量。于是有两个概念：

- Padding：补齐。
- Packing：打包，告诉 RNN 哪些位置是真实内容。

### 13.1 Padding 是什么

把短序列补到同样长度：

```text
[I, love, it, <PAD>, <PAD>]
[This, movie, is, really, wonderful]
[Bad, <PAD>, <PAD>, <PAD>, <PAD>]
```

如果转成词 ID，`<PAD>` 常用 0：

```text
[12, 45, 88, 0, 0]
[31, 29,  5, 7, 9]
[66,  0,  0, 0, 0]
```

Padding 后可以得到统一 shape：

```text
padded shape = (N, max_L)
```

如果已经过 Embedding：

```text
embedded shape = (N, max_L, embed_dim)
```

### 13.2 只 padding 的问题

课件强调：填充值可能影响 RNN 隐藏状态。即使输入是 0，RNN 还有权重和偏置，后续隐藏状态仍可能变化。

问题包括：

- 浪费计算：模型处理了大量 `<PAD>`。
- 干扰最终状态：短句后面的 padding 位置可能继续改变隐藏状态。
- 最后时间步不再等于真实句子的最后一个词。

所以短序列分类时，直接取 `output[:, -1, :]` 很危险，因为 `-1` 可能对应 `<PAD>`，不是原句最后一个真实词。

### 13.3 Packing 是什么

Packing 会把真实长度告诉 PyTorch，让 RNN/LSTM 跳过 padding 位置。

课件中有两种方式：

```python
rnn_utils.pack_sequence(seq_tensors, enforce_sorted=False)
```

或者先 padding，再打包：

```python
packed = rnn_utils.pack_padded_sequence(
    padded,
    lengths_sorted.cpu(),
    batch_first=True,
    enforce_sorted=True
)
```

PackedSequence 不是普通规则张量，它主要包含：

| 属性 | 含义 |
|:---|:---|
| `data` | 所有有效 token 拼接起来的数据 |
| `batch_sizes` | 每个时间步还有多少条序列没有结束 |
| `sorted_indices` | 排序后的索引 |
| `unsorted_indices` | 恢复原顺序的索引 |

### 13.4 和课件 `collate_fn` 的对应关系

课件的 IMDB 案例中，`collate_fn` 做了这些事：

```text
原始 batch
    ↓
分离 texts 和 labels
    ↓
计算每条文本真实长度 lengths
    ↓
按长度降序排序
    ↓
pad_sequence 补齐
    ↓
pack_padded_sequence 打包
    ↓
返回 packed, labels_sorted, lengths_sorted
```

对应代码要点：

```python
texts, labels = zip(*batch)
lengths = torch.tensor([len(t) for t in texts])
sorted_indices = torch.argsort(lengths, descending=True)

padded = rnn_utils.pad_sequence(
    texts_sorted,
    batch_first=True,
    padding_value=0
)

packed = rnn_utils.pack_padded_sequence(
    padded,
    lengths_sorted.cpu(),
    batch_first=True,
    enforce_sorted=True
)
```

特别注意：排序文本时，标签也必须用同样顺序排序，否则输入和标签会错位。

```python
labels_sorted = labels[sorted_indices]
```

### 13.5 Packing 下的 shape

假设一个 batch 有 64 条评论，每条长度不同。

Padding 后：

```text
padded shape = (64, max_seq_len_in_batch)
```

打包后：

```text
packed.data shape = (total_valid_tokens,)
packed.batch_sizes shape = (max_seq_len_in_batch,)
```

如果先对 `packed.data` 做 Embedding：

```text
embedded_data shape = (total_valid_tokens, embed_dim)
```

再重新构造 PackedSequence，送入 LSTM：

```python
output, (hidden, cell) = self.lstm(embedded)
```

这时 `hidden` 仍然是规则张量：

```text
hidden shape = (num_layers * num_directions, batch, hidden_dim)
```

所以情感分类案例直接取：

```python
final_hidden = hidden[-1]
```

---

## 14. 1D 卷积和 TCN：序列建模的另一条路

RNN 是按时间一步一步处理。1D 卷积则是用卷积核在序列维度上滑动。

### 14.1 1D 卷积看什么

图像卷积在高度和宽度上滑动；1D 卷积只沿一个维度滑动，通常是时间或序列长度。

PyTorch 中 `Conv1d` 常用输入：

```text
(N, C, L)
```

含义：

- `N`：batch size
- `C`：channel 数，也可以理解成每个时间步的特征通道数
- `L`：序列长度

注意这和 RNN 常用的 `(N, L, F)` 不同。若要把 Embedding 后的文本送入 `Conv1d`，常需要转置：

```python
embedded = embedding(x)          # (N, L, embed_dim)
conv_in = embedded.transpose(1, 2)  # (N, embed_dim, L)
```

### 14.2 课件中的移动平均例子

课件用温度序列演示 1D 卷积：

```python
temperatures = torch.tensor([...]).float()
weight = torch.ones(window_size) * 0.2
temp_input = temperatures.view(1, 1, -1)
```

这里：

```text
temp_input shape = (1, 1, L)
```

含义是：

- 1 条样本
- 1 个通道
- 长度为 `L` 的温度序列

卷积核像一个滑动窗口，局部地看连续几天的温度。

### 14.3 1D 卷积 vs RNN

| 对比项 | 1D 卷积 | RNN / GRU / LSTM |
|:---|:---|:---|
| 处理方式 | 局部窗口滑动 | 按时间步递推 |
| 并行性 | 强，可以并行算很多位置 | 弱，后一步依赖前一步 |
| 局部模式 | 很擅长 | 也能学，但不如卷积直接 |
| 长程依赖 | 普通卷积较弱 | LSTM/GRU 更自然 |
| 输入 shape | `(N, C, L)` | `(N, L, F)` 或 `(L, N, F)` |
| 常见任务 | 局部模式识别、快速序列分类 | 文本、语音、时间序列建模 |

### 14.4 TCN 是什么

TCN 是 Temporal Convolutional Network，时序卷积网络。它用卷积处理序列，但通过一些设计增强长程建模能力。

课件中提到三项关键技术：

| 技术 | 解决的问题 | 含义 |
|:---|:---|:---|
| 因果卷积 | 防止偷看未来 | 当前位置只依赖当前和过去 |
| 空洞卷积 | 扩大感受野 | 间隔采样，让卷积看得更远 |
| 残差连接 | 深层训练困难 | 帮助梯度传播，稳定训练 |

模型选择时可以这样判断：

- 短序列、局部模式明显：普通 1D 卷积。
- 长序列、希望并行、又要看远一些：TCN。
- 需要逐步更新状态、变长处理自然：RNN/GRU/LSTM。
- 大规模文本理解：Transformer 往往更强。

---

## 15. 情感分类实践：IMDB 案例完整串起来

课件的综合案例是 IMDB 电影评论二分类：

```text
输入：一段电影评论文本
输出：正面 or 负面
```

模型结构：

```text
文本
  ↓
分词 + 词表编码
  ↓
PackedSequence
  ↓
Embedding
  ↓
LSTM
  ↓
取最后隐藏状态
  ↓
Dropout
  ↓
Linear
  ↓
logit
  ↓
sigmoid 后得到正面概率
```

### 15.1 超参数

课件中的关键配置：

| 参数 | 值 | 含义 |
|:---|:---:|:---|
| `MAX_SEQ_LEN` | 500 | 评论最多保留 500 个词 |
| `MIN_WORD_FREQ` | 5 | 低频词变成 `<UNK>` |
| `BATCH_SIZE` | 64 | 每批 64 条评论 |
| `EMBED_DIM` | 128 | 每个词变成 128 维向量 |
| `HIDDEN_DIM` | 256 | LSTM 隐藏状态 256 维 |
| `NUM_LAYERS` | 1 | 单层 LSTM |
| `DROPOUT` | 0.5 | Dropout 概率 |
| `BIDIRECTIONAL` | False | 单向 LSTM |
| `LEARNING_RATE` | 0.001 | Adam 初始学习率 |
| `WEIGHT_DECAY` | `1e-5` | L2 正则化 |
| `CLIP_GRAD_NORM` | 5.0 | 梯度裁剪阈值 |
| `PATIENCE` | 3 | 早停耐心值 |

### 15.2 数据流程

课件中的数据流程：

1. 下载并读取 Stanford IMDB 数据集。
2. 原始标签转成二分类：负面为 0，正面为 1。
3. 从训练集划分 20% 作为验证集，并用 `stratify` 保持正负比例。
4. 用训练集和验证集文本构建词表，测试集不参与建词表，避免信息泄露。
5. `Dataset.__getitem__` 返回变长整数序列和标签。
6. `DataLoader` 用自定义 `collate_fn` 生成 PackedSequence。

### 15.3 词表和编码

课件中 `Vocab` 保留两个特殊 token：

```text
<PAD> = 0
<UNK> = 1
```

编码过程：

```python
tokens = self.tokenize(text)[:max_len]
indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
return torch.tensor(indices, dtype=torch.long)
```

shape：

```text
单条文本编码后: (seq_len,)
```

这里还没有 Embedding，所以每个词只是一个整数 ID。

### 15.4 模型定义和代码对应

课件中的模型：

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
            dropout=0 if num_layers == 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, 1)
```

几个关键点：

| 代码 | 含义 |
|:---|:---|
| `nn.Embedding(vocab_size, embed_dim, padding_idx=0)` | 把词 ID 变成词向量；`<PAD>` 的向量保持为 0 |
| `input_size=embed_dim` | LSTM 每个时间步吃到的是词向量 |
| `hidden_size=hidden_dim` | LSTM 隐藏状态维度 |
| `batch_first=False` | 因为 PackedSequence 内部按 time-first 组织 |
| `bidirectional=bidirectional` | 是否双向 |
| `fc = nn.Linear(..., 1)` | 输出一个二分类 logit |

### 15.5 PackedSequence 进模型后的 shape

课件里没有直接对 PackedSequence 做普通 embedding，而是先取出 `data`：

```python
embedded_data = self.embedding(packed_input.data)
```

shape 变化：

```text
packed_input.data      (total_valid_tokens,)
embedded_data          (total_valid_tokens, embed_dim)
```

然后重新构造 PackedSequence：

```python
embedded = nn.utils.rnn.PackedSequence(
    embedded_data,
    packed_input.batch_sizes,
    packed_input.sorted_indices,
    packed_input.unsorted_indices
)
```

再送进 LSTM：

```python
output, (hidden, cell) = self.lstm(embedded)
```

返回：

```text
output  PackedSequence，包含所有有效时间步输出
hidden  (num_layers * num_directions, batch, hidden_dim)
cell    (num_layers * num_directions, batch, hidden_dim)
```

单层单向时：

```text
hidden shape = (1, batch, 256)
cell shape   = (1, batch, 256)
```

最终取：

```python
final_hidden = hidden[-1]      # (batch, hidden_dim)
```

如果改成双向：

```python
final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
```

shape：

```text
(batch, hidden_dim * 2)
```

最后分类：

```python
dropped = self.dropout(final_hidden)
logits = self.fc(dropped).squeeze(-1)
```

shape：

```text
final_hidden  (batch, hidden_dim)
dropped       (batch, hidden_dim)
fc output     (batch, 1)
logits        (batch,)
```

### 15.6 为什么输出是 logit，不是概率

模型最后输出 `logits`：

```text
logits shape = (batch,)
```

它不是 0 到 1 的概率，而是未归一化分数。

训练时使用：

```python
criterion = nn.BCEWithLogitsLoss()
```

这个损失函数内部已经包含 sigmoid 和二元交叉熵，比手动 `sigmoid + BCELoss` 更稳定。

预测时才做：

```python
probs = torch.sigmoid(outputs)
preds = (probs > 0.5).float()
```

### 15.7 训练流程

课件中的训练函数可以拆成固定流程：

```text
model.train()
for packed_texts, labels, lengths in train_loader:
    packed_texts, labels 移到设备
    outputs = model(packed_texts)
    loss = BCEWithLogitsLoss(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
```

每一步的作用：

| 步骤 | 代码 | 作用 |
|:---|:---|:---|
| 训练模式 | `model.train()` | 开启 Dropout |
| 前向传播 | `outputs = model(packed_texts)` | 得到每条评论的 logit |
| 计算损失 | `criterion(outputs, labels)` | 衡量预测和标签差距 |
| 清梯度 | `optimizer.zero_grad()` | 避免梯度累加 |
| 反向传播 | `loss.backward()` | 计算梯度 |
| 梯度裁剪 | `clip_grad_norm_` | 防止梯度爆炸 |
| 参数更新 | `optimizer.step()` | 更新模型参数 |

### 15.8 验证与测试流程

评估时：

```python
model.eval()
with torch.no_grad():
    outputs = model(packed_texts)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
```

`model.eval()` 会关闭 Dropout，`torch.no_grad()` 会节省显存并加速。

课件使用的指标：

| 指标 | 含义 |
|:---|:---|
| Accuracy | 总体预测正确比例 |
| Precision | 预测为正面里有多少真的是正面 |
| Recall | 真实正面里有多少被找出来 |
| F1 | Precision 和 Recall 的调和平均 |
| Confusion Matrix | TP、TN、FP、FN 的具体数量 |

### 15.9 实验结果和改进方向

课件给出的基准对比：

| 模型 | IMDB 准确率 |
|:---|:---:|
| 单层 LSTM + 随机初始化 Embedding | 约 85% |
| 单层 LSTM + GloVe 预训练词向量 | 约 88% |
| 双向 LSTM | 约 89% |
| BERT-base | 约 94% |

可以改进：

| 改进方向 | 具体做法 | 预期作用 |
|:---|:---|:---|
| 预训练词向量 | 用 GloVe/Word2Vec 初始化 Embedding | 利用外部语料知识 |
| 双向 LSTM | 设置 `BIDIRECTIONAL=True` | 同时看前文和后文 |
| 调整 Dropout | 增大或加入 embedding dropout | 缓解过拟合 |
| 减小 hidden_dim | 例如从 256 改为 128 | 降低模型容量 |
| 更早早停 | 减小 `PATIENCE` | 防止后期过拟合 |
| Attention | 在时间步上学习关注重点词 | 提升解释性和效果 |

---

## 16. 易混点集中整理

### 16.1 `batch_first=True` 会不会改变 `h_n`

不会。

如果 `batch_first=True`：

```text
input   (N, L, F)
output  (N, L, H)
h_n     (num_layers * num_directions, N, H)
```

`h_n` 仍然不是 `(N, ...)` 开头。

### 16.2 `output[:, -1, :]` 总能做分类吗

不总是。

它适合单层单向、没有 padding 干扰、最后一个时间步就是真实最后 token 的情况。

以下情况要谨慎：

- batch 内有 padding，`-1` 可能是 `<PAD>`。
- 双向 RNN，`output[:, -1, :]` 的反向部分不是反向最终状态。
- 使用 packing 时，直接在普通 padded output 上取最后位置可能不等于真实长度位置。

### 16.3 GRU 和 LSTM 最大区别是什么

GRU：

```text
只有 h_t，一个状态；两个门。
```

LSTM：

```text
有 h_t 和 C_t，两个状态；三个门。
```

GRU 更简洁，LSTM 记忆管理更细。

### 16.4 Padding 和 Packing 的区别

| 概念 | 解决什么 | 是否改变数据形状 |
|:---|:---|:---|
| Padding | 把不同长度补成统一长度 | 变成规则张量 |
| Packing | 告诉 RNN 哪些是真实长度 | 变成 PackedSequence |

Padding 是为了凑成 batch；Packing 是为了不让模型认真处理 `<PAD>`。

### 16.5 LSTM 的 `output`、`h_n`、`c_n`

```python
output, (h_n, c_n) = lstm(x)
```

| 返回值 | 含义 |
|:---|:---|
| `output` | 最后一层所有时间步的 hidden state |
| `h_n` | 每层每方向最终 hidden state |
| `c_n` | 每层每方向最终 cell state |

做分类通常用 `h_n`，不是直接用 `c_n`。

---

## 17. 关键概念速查表

| 概念 | 快速解释 | 常见 shape |
|:---|:---|:---|
| 时间步 | 序列中的一个位置 | 第 `t` 个词 |
| `x_t` | 当前时间步输入 | `(N, F)` |
| `h_t` | 当前隐藏状态 | `(N, H)` |
| `N` | batch size | 64 |
| `L` | 序列长度 | 500 |
| `F` | 输入特征维度 | 128 |
| `H` | 隐藏状态维度 | 256 |
| `output` | 所有时间步的输出 | `(N, L, H)` 或 `(N, L, 2H)` |
| `h_n` | 最终隐藏状态 | `(layers * directions, N, H)` |
| `c_n` | LSTM 最终 cell state | `(layers * directions, N, H)` |
| `batch_first` | batch 是否放第一维 | 影响 input 和 output |
| `bidirectional` | 是否双向 | 输出特征维变成 `2H` |
| `num_layers` | 堆叠层数 | `h_n` 第一维变大 |
| Padding | 补齐短序列 | `(N, max_L)` |
| Packing | 打包真实 token | `PackedSequence` |
| Embedding | 词 ID 转向量 | `(N, L)` → `(N, L, E)` |
| Logit | 未经 sigmoid 的分类分数 | `(N,)` |

---

## 18. 自测题

### 题 1：`output` 和 `h_n`

有代码：

```python
rnn = nn.RNN(input_size=10, hidden_size=16, batch_first=True)
x = torch.randn(32, 5, 10)
output, h_n = rnn(x)
```

问：

1. `output.shape` 是什么？
2. `h_n.shape` 是什么？
3. `output[:, -1, :].shape` 是什么？

答案：

```text
output.shape = (32, 5, 16)
h_n.shape = (1, 32, 16)
output[:, -1, :].shape = (32, 16)
```

### 题 2：`batch_first`

如果：

```python
rnn = nn.RNN(10, 16, batch_first=False)
x = torch.randn(32, 5, 10)
```

直接 `rnn(x)` 对吗？为什么？

答案：

不对。`batch_first=False` 时，RNN 期待输入是 `(L, N, F)`，也就是 `(5, 32, 10)`。现在的 `x` 是 `(N, L, F)`，需要：

```python
x = x.transpose(0, 1)
```

或者创建模型时设置：

```python
batch_first=True
```

### 题 3：双向 RNN 的 shape

有代码：

```python
rnn = nn.RNN(32, 64, batch_first=True, bidirectional=True)
x = torch.randn(8, 20, 32)
output, h_n = rnn(x)
```

问：

1. `output.shape` 是什么？
2. `h_n.shape` 是什么？
3. 如果做分类，如何拼接两个方向的最终状态？

答案：

```text
output.shape = (8, 20, 128)
h_n.shape = (2, 8, 64)
```

分类可写：

```python
final = torch.cat([h_n[-2], h_n[-1]], dim=1)
```

`final.shape = (8, 128)`。

### 题 4：多层双向 LSTM

有代码：

```python
lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=3,
    batch_first=True,
    bidirectional=True
)
x = torch.randn(64, 50, 100)
output, (h_n, c_n) = lstm(x)
```

问：

1. `output.shape` 是什么？
2. `h_n.shape` 是什么？
3. `c_n.shape` 是什么？

答案：

```text
output.shape = (64, 50, 512)
h_n.shape = (6, 64, 256)
c_n.shape = (6, 64, 256)
```

因为 `num_layers * num_directions = 3 * 2 = 6`。

### 题 5：GRU 和 LSTM

下面说法哪个正确？

A. GRU 和 LSTM 都返回 `(output, (h_n, c_n))`。  
B. GRU 有 cell state，LSTM 没有。  
C. GRU 通常返回 `output, h_n`，LSTM 返回 `output, (h_n, c_n)`。  
D. RNN、GRU、LSTM 都有三个门。

答案：C。

解释：

- GRU 没有单独的 `c_n`。
- LSTM 有 `h_n` 和 `c_n`。
- 普通 RNN 没有门控。

### 题 6：Padding 和最后时间步

一个 batch 中两条句子真实长度分别是 5 和 3，padding 到长度 5。使用普通 padded input 直接进入 LSTM，然后取：

```python
output[:, -1, :]
```

对第二条句子来说，取到的是第几个位置？它一定是真实最后词吗？

答案：

取到 padding 后的第 5 个位置，不一定是真实最后词。第二条句子真实长度是 3，第 4、5 个位置是 `<PAD>`。所以直接取 `-1` 会取到 padding 位置的输出，可能不适合作为句子表示。

更合适的方法：

- 使用 `pack_padded_sequence`，再取 `h_n`。
- 或者根据真实长度 `lengths` 去 gather 每条序列的最后有效位置。

### 题 7：PackedSequence 的 `data`

`PackedSequence.data` 的 shape 是：

```text
(total_valid_tokens,)
```

这里的 `total_valid_tokens` 是什么意思？

答案：

它是这个 batch 里所有真实 token 的总数，不包含 padding。例如 3 条序列真实长度分别是 4、2、1，那么：

```text
total_valid_tokens = 4 + 2 + 1 = 7
```

### 题 8：IMDB 案例里的输出

课件中 `SentimentLSTM.forward` 最后返回：

```python
logits = self.fc(dropped).squeeze(-1)
```

如果 batch size 是 64，那么 `logits.shape` 是什么？训练时为什么不用先手动 sigmoid？

答案：

```text
logits.shape = (64,)
```

训练时使用 `nn.BCEWithLogitsLoss()`，它内部已经包含 sigmoid 和二元交叉熵，数值更稳定，所以不用先手动 sigmoid。

### 题 9：1D 卷积输入 shape

Embedding 后文本 shape 是：

```text
(N, L, E)
```

要送入 `nn.Conv1d(in_channels=E, ...)`，通常要怎么变？

答案：

需要转成：

```text
(N, E, L)
```

代码：

```python
conv_input = embedded.transpose(1, 2)
```

### 题 10：长期依赖

为什么普通 RNN 容易出现梯度消失？

答案：

因为远处时间步到当前时间步的梯度传播包含很多步偏导数连乘：

```text
∂h_n/∂h_t = Π ∂h_k/∂h_{k-1}
```

如果每一步传递系数小于 1，乘很多次后会趋近于 0，早期信息对当前损失的影响变得很弱，模型就难学到长距离依赖。

---

## 19. 最后总复习

这一章可以压缩成一条主线：

```text
序列有顺序
→ 模型需要记住前文
→ RNN 用隐藏状态 h_t 传递历史
→ 普通 RNN 长距离传播不稳定
→ GRU 用门控制新旧记忆混合
→ LSTM 用 C_t 单独维护长期记忆
→ 真实文本长度不同，需要 Padding/Packing
→ 做分类时常取最终隐藏状态
→ 1D 卷积和 TCN 是另一类序列建模方法
```

考前最需要稳住的内容：

1. `N、L、F、H` 的含义。
2. `batch_first=True` 只影响普通输入和 `output`，不改变 `h_n`。
3. `output` 是所有时间步，`h_n` 是最终隐藏状态。
4. 单向单层时 `output[:, -1, :]` 常等于 `h_n[-1]`，但双向、padding、packing 场景要重新判断。
5. GRU 有两个门，没有 `c_n`。
6. LSTM 有三个门，返回 `output, (h_n, c_n)`。
7. Padding 是补齐，Packing 是跳过无效 padding。
8. IMDB 案例的训练链路是：词表编码 → PackedSequence → Embedding → LSTM → hidden → Dropout → Linear → logits → BCEWithLogitsLoss。
9. RNN/GRU/LSTM 输入常见为 `(N, L, F)`，但 `Conv1d` 常见为 `(N, C, L)`。
10. 长期依赖问题的数学根源是跨时间步梯度连乘。

如果复习时间很短，优先做三件事：

- 手写一遍 `nn.RNN / nn.GRU / nn.LSTM` 的输入输出 shape。
- 分清单向、双向、多层时 `h_n` 第一维怎么来的。
- 把 IMDB 案例的 forward 过程按 shape 说出来。

能把这三件事讲清楚，RNN 这一章的大部分考试题和代码阅读题就有基础了。
