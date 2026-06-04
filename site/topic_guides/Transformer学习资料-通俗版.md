# Transformer 学习资料：通俗版

## 资料来源说明

本讲义根据本课程三份 Transformer 课件重新整理：

- `注意力机制及Transformer1-学生分发版-终版2026.ipynb`：Seq2Seq、Encoder-Decoder、Attention、Q/K/V、Scaled Dot-Product Attention、Source Mask、Multi-Head Attention、Self-Attention、Target Mask、位置编码。
- `注意力机制及Transformer2-学生分发版-终版2026.ipynb`：Narrow Attention、Transformer Encoder、Transformer Decoder、LayerNorm、IMDB 情感分类训练实践、PyTorch 内置 Transformer、ViT、DistilBERT 微调。
- `注意力机制及Transformer3-学生分发版-终版2026(3).ipynb`：词嵌入演进、BERT 与 GPT、HuggingFace Transformers、预训练与微调、现代 Transformer 架构演进。

我还参考了复习网站数据库中的 `source_cache.json` 和 `course_chunks.json` 来核对课件覆盖范围。正文不是对已有 `site/guides` 讲义模板的改写，而是按“基础不牢的学生如何逐步建立理解”的路线重新组织。

---

## 全局学习路线图

Transformer 这章最容易学散，因为它有很多名词：Q、K、V、mask、multi-head、position encoding、encoder、decoder、BERT、GPT、ViT。不要一上来就背公式。更好的路线是：

| 学习阶段 | 要解决的问题 | 学到的核心 |
|---|---|---|
| 1. Seq2Seq 为什么有瓶颈 | 为什么早期 Encoder-Decoder 不够用 | 一个固定上下文向量装不下长序列 |
| 2. Attention 的直觉 | 模型如何在生成不同词时看输入的不同部分 | 动态加权看源序列 |
| 3. Q/K/V 从哪里来 | 注意力到底在匹配什么、取什么 | Q 表示查询需求，K 表示匹配标签，V 表示可取信息 |
| 4. Scaled Dot-Product | 公式每一步在干什么 | `QK^T` 打分，除以 `sqrt(d_k)` 稳定数值，softmax 变权重，再乘 V 取信息 |
| 5. Multi-Head | 为什么不止算一次 attention | 多个头在不同子空间学不同关系 |
| 6. 位置编码 | 为什么 Self-Attention 还需要位置 | Attention 本身不懂顺序，需要额外告诉它位置 |
| 7. Mask 的三种作用 | 为什么有些位置要被遮住 | 忽略 PAD、防止 Decoder 偷看未来、构造 BERT 填空任务 |
| 8. Encoder/Decoder 结构 | 完整 Transformer 如何拼起来 | Encoder 做理解，Decoder 做生成，Cross-Attention 连接二者 |
| 9. 训练与推理 | 训练时和真正生成时有什么区别 | 训练可并行看标签前缀，推理要一步一步生成 |
| 10. BERT/GPT/ViT | 同一个 Transformer 为什么能做不同任务 | Encoder-only、Decoder-only、图像 patch 序列化 |
| 11. 考前复盘 | 怎么避免常见混淆 | 用 shape、来源、mask 方向、任务目标来判断 |

---

## 1. Seq2Seq 为什么会有瓶颈

Seq2Seq 的任务是“一个序列进来，另一个序列出去”。典型例子：

- 机器翻译：英文句子 -> 中文句子
- 文本摘要：长文章 -> 短摘要
- 语音识别：音频帧序列 -> 文字序列
- 对话系统：用户输入 -> 模型回复

早期 Seq2Seq 常用 RNN/LSTM 做 Encoder-Decoder：

1. Encoder 从左到右读完整个源序列。
2. Encoder 把读到的信息压缩成最后一个隐藏状态，也叫上下文向量。
3. Decoder 拿这个上下文向量作为起点，一个词一个词生成目标序列。

问题出在第 2 步。假设源句子很长，模型要把所有词、顺序、语法、语义都塞进一个固定长度向量里。短句还能勉强，长句就很难。这个固定向量就像一张很小的纸，句子越长，越容易丢重点。

### 一个翻译例子

如果源句是：

```text
The agreement signed by the two companies last year was finally cancelled.
```

Decoder 生成中文前半句时可能要关注 `agreement`、`signed`，生成后半句时要关注 `finally cancelled`。如果它每一步都只能看同一个压缩向量，就很难在不同生成位置拿到不同重点。

这就是 Attention 要解决的核心：不要把输入全部压成一个向量，而是在每次生成时，让 Decoder 重新选择源序列中最相关的位置。

---

## 2. Attention 的直觉

Attention 的核心想法很朴素：生成当前词时，不必平均看整句，而是给输入序列每个位置分配一个权重。

例如翻译 “I love deep learning”：

| Decoder 当前要生成 | 更应该关注源句哪些词 |
|---|---|
| “我” | `I` |
| “喜欢” | `love` |
| “深度” | `deep` |
| “学习” | `learning` |

这不是硬规则，而是模型通过训练学出来的软权重。每个源词都会得到一个 0 到 1 之间的权重，所有权重加起来为 1。权重大，说明当前生成步骤更依赖这个源词。

所以 Attention 可以理解成三步：

1. 当前要生成的位置提出一个“查询”。
2. 源序列每个位置拿出自己的“可匹配信息”。
3. 匹配程度高的位置权重大，再把这些位置携带的信息加权求和。

这三步引出 Q/K/V。

---

## 3. Q/K/V 从哪里来

Q/K/V 是 Attention 里最容易被背混的概念。可以先用查资料的场景理解：

- Query：我现在想找什么。
- Key：每份资料贴出来的关键词，用来和 Query 匹配。
- Value：资料真正的内容，匹配上以后要取走的信息。

### 3.1 在传统 Encoder-Decoder Attention 中

| 符号 | 来源 | 含义 |
|---|---|---|
| Q | Decoder 当前隐藏状态 | 当前生成位置想知道什么 |
| K | Encoder 所有隐藏状态经过线性变换 | 源序列每个位置能被怎样匹配 |
| V | Encoder 所有隐藏状态经过另一组线性变换 | 源序列每个位置真正提供什么信息 |

Key 和 Value 都来自 Encoder，但不是同一个东西。它们来自同一批源序列表示，却经过不同线性层，所以数值通常不同。

### 3.2 在 Self-Attention 中

Self-Attention 的特别之处是 Q、K、V 都来自同一个序列。

假设输入是 `x = [x_1, x_2, ..., x_L]`，每个位置都有一个 `d_model` 维向量。模型用三组线性层得到：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

这表示同一个词向量会被投影成三个角色：

- 当 Q：它作为“提问者”，去找其他位置。
- 当 K：它作为“被匹配对象”，告诉别人自己适合被什么查询匹配。
- 当 V：它作为“信息内容”，被别人根据权重取走。

### 3.3 Cross-Attention 中 Q/K/V 的来源

在 Transformer Decoder 的 Cross-Attention 里：

| 符号 | 来源 |
|---|---|
| Q | Decoder 已经经过 Masked Self-Attention 的状态 |
| K | Encoder 最后一层输出 |
| V | Encoder 最后一层输出 |

这就是 Decoder 用自己的生成状态去查询 Encoder 的源句信息。

---

## 4. Scaled Dot-Product Attention 公式逐项拆解

标准公式是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

不要把它当成一个整体死背。把它拆开就很清楚。

### 4.1 输入 shape

先不看 batch 和 head，只看一个 attention：

| 张量 | shape | 含义 |
|---|---|---|
| Q | `[n, d_k]` | 有 n 个查询位置，每个查询是 `d_k` 维 |
| K | `[m, d_k]` | 有 m 个可匹配位置，每个 key 是 `d_k` 维 |
| V | `[m, d_v]` | 有 m 个可取信息位置，每个 value 是 `d_v` 维 |

注意：K 和 V 的序列长度都是 `m`，因为每个 key 对应一个 value。

### 4.2 第一步：`QK^T` 计算匹配分数

$$
scores = QK^\top
$$

shape 变化：

```text
Q:     [n, d_k]
K^T:   [d_k, m]
scores:[n, m]
```

`scores[i, j]` 表示第 i 个 Query 和第 j 个 Key 的匹配程度。点积越大，方向越接近、数值也越强，说明越相关。

### 4.3 第二步：除以 `sqrt(d_k)`

$$
scaled\_scores = \frac{QK^\top}{\sqrt{d_k}}
$$

为什么不是除以 `d_k`，也不是不除？因为点积是 `d_k` 个乘积项相加。如果 Q 和 K 的每一维方差大约为 1，那么点积的方差会随 `d_k` 增大，大约是 `d_k`。维度越高，点积分数越容易变得很大。

分数太大时，softmax 会变得极端：

```text
softmax([1, 2, 3])       -> 比较平滑
softmax([10, 20, 30])    -> 最大项接近 1，其他接近 0
```

一旦 softmax 太接近 one-hot，很多位置的梯度会很小，训练不稳定。除以 `sqrt(d_k)` 相当于把点积的标准差拉回到比较合适的尺度。

### 4.4 第三步：softmax 变成注意力权重

$$
\alpha = \text{softmax}(scaled\_scores)
$$

softmax 是沿着 Key 的维度做的，也就是每一行归一化：

```text
scores shape: [n, m]
alpha shape:  [n, m]
每一行 alpha[i, :] 的和为 1
```

第 i 个 Query 会对 m 个 Key 各分配一个权重。

### 4.5 第四步：乘以 V，取加权信息

$$
output = \alpha V
$$

shape 变化：

```text
alpha:  [n, m]
V:      [m, d_v]
output: [n, d_v]
```

每个输出位置都是所有 Value 的加权和。权重来自 Q 和 K 的匹配，内容来自 V。

### 4.6 带 batch 和多头时的常见 shape

实际代码里通常有 batch 和 head：

```text
X:      [batch, seq_len, d_model]
Q/K/V:  [batch, seq_len, d_model]
切头后: [batch, n_heads, seq_len, d_k]
scores: [batch, n_heads, query_len, key_len]
alpha:  [batch, n_heads, query_len, key_len]
context:[batch, n_heads, query_len, d_k]
拼回:   [batch, query_len, d_model]
```

如果是 Self-Attention，`query_len = key_len = seq_len`。如果是 Cross-Attention，`query_len` 来自目标序列长度，`key_len` 来自源序列长度。

---

## 5. Multi-Head Attention 为什么有用

一个 attention 头只产生一种“关注模式”。但语言里的关系很多：

- 主语和谓语的关系
- 代词和它指代对象的关系
- 当前词和相邻词的局部关系
- 句首词和句尾词的长距离关系
- 标点、否定词、修饰词对语义的影响

一个头很难同时把这些关系都学好。Multi-Head Attention 就是让多个头并行工作，每个头在自己的子空间里学一种或几种关系。

公式：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W_O
$$

其中：

$$
\text{head}_i=\text{Attention}(QW_Q^{(i)},KW_K^{(i)},VW_V^{(i)})
$$

### 5.1 Wide Attention 和 Narrow Attention

课件特别强调了 Narrow Attention。

| 实现方式 | 做法 | 特点 |
|---|---|---|
| Wide Attention | 每个头都看完整 `d_model`，每个头输出也较大 | 容易理解，但计算量大 |
| Narrow Attention | 先投影到 `d_model`，再切成 `h` 段，每头 `d_k=d_model/h` | 工业模型常用，计算更省 |

Narrow Attention 里有一个关键顺序：先投影，再切片。

如果先把原始输入切片，每个头一开始只能看到原始特征的一部分；如果先用 `W_Q/W_K/W_V` 把完整输入投影，再切成多个头，每个切片已经是完整输入特征的线性组合，信息更丰富。

### 5.2 代码对应关系

课件里的多头注意力代码核心是：

```python
Q = self.W_q(query)
K = self.W_k(key)
V = self.W_v(value)

Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

output, attn_weights = self.attention(Q, K, V, mask)

output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
output = self.W_o(output)
```

对应关系：

| 代码 | 数学含义 | shape 变化 |
|---|---|---|
| `self.W_q(query)` | 生成 Q | `[B,L,D] -> [B,L,D]` |
| `view(..., n_heads, d_k)` | 切成多个头 | `[B,L,D] -> [B,L,H,d_k]` |
| `transpose(1, 2)` | 把 head 维提前，方便并行算 | `[B,L,H,d_k] -> [B,H,L,d_k]` |
| `self.attention(Q,K,V,mask)` | 每个头独立算 attention | 输出 `[B,H,L,d_k]` |
| `transpose(...).view(...)` | 拼回所有头 | `[B,H,L,d_k] -> [B,L,D]` |
| `self.W_o(output)` | 多头结果再融合 | `[B,L,D] -> [B,L,D]` |

---

## 6. 位置编码：为什么 Attention 需要“顺序提示”

Self-Attention 有一个优点：所有位置可以并行计算。它也因此有一个问题：只看 attention 公式，模型不知道词的先后顺序。

如果没有位置编码，下面两个序列对 Self-Attention 来说很容易混淆：

```text
我 喜欢 学习
学习 喜欢 我
```

词集合一样，但顺序完全不同，意思也不同。RNN 天然按顺序读，Transformer 不按顺序读，所以要额外加入位置编码。

### 6.1 经典正弦位置编码

Transformer 原始论文使用：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

逐项解释：

| 符号 | 含义 |
|---|---|
| `pos` | 当前词的位置，0、1、2、... |
| `i` | 维度编号的一半 |
| `2i` | 偶数维度用 sin |
| `2i+1` | 奇数维度用 cos |
| `d_model` | 词向量维度 |
| `10000^{2i/d_model}` | 控制不同维度的频率 |

低维变化快，高维变化慢。这样模型既能区分相邻位置，也能感知较长距离。

### 6.2 代码对应关系

课件中的位置编码实现：

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, d_model, 2).float() *
    (-np.log(10000.0) / d_model)
)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)
self.register_buffer("pe", pe)
```

shape 对应：

```text
position: [max_len, 1]
div_term: [d_model/2]
pe:       [max_len, d_model]
pe unsqueeze 后: [1, max_len, d_model]
x:        [batch, seq_len, d_model]
x + pe[:, :seq_len, :]: [batch, seq_len, d_model]
```

`register_buffer` 的意思是：位置编码跟着模型保存和移动设备，但不是可训练参数。

### 6.3 RoPE 了解到什么程度

课件也提到现代大模型常用 RoPE。它和正弦位置编码的区别：

| 对比 | Sinusoidal PE | RoPE |
|---|---|---|
| 操作位置 | 加到输入 embedding 上 | 旋转 Q/K 向量 |
| 位置类型 | 绝对位置更明显 | 相对位置关系更直接 |
| 常见模型 | 原始 Transformer、早期 BERT/GPT | LLaMA、Qwen、DeepSeek 等 |
| 考试重点 | 公式、为什么需要位置 | 知道现代模型常用即可 |

---

## 7. Mask 的三种作用

Mask 的本质是：在 softmax 前，把某些位置的分数设成一个极小值，比如 `-inf` 或 `-1e9`。这样 softmax 后，这些位置权重接近 0。

### 7.1 Padding Mask：忽略补齐位置

batch 训练时，句子长度不同，要补 PAD：

```text
句子 A: [101, 234, 456, 0,   0]
句子 B: [101, 567, 890, 123, 0]
```

PAD 只是为了凑 shape，不是真实词。模型不应该关注它。

课件代码：

```python
def create_padding_mask(seq, pad_idx=0):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask
```

shape：

```text
seq:  [batch, seq_len]
mask: [batch, 1, 1, seq_len]
```

为什么要变成 `[batch, 1, 1, seq_len]`？因为 attention 分数通常是 `[batch, heads, query_len, key_len]`，mask 要能广播到 `heads` 和 `query_len` 上。

### 7.2 Causal Mask / Target Mask：防止生成时看未来

Decoder 生成第 t 个词时，只能看自己和前面的词，不能看后面的真实答案。否则训练时模型会作弊。

典型 mask：

```text
1 = 可以看，0 = 不能看

位置0: [1, 0, 0, 0, 0]
位置1: [1, 1, 0, 0, 0]
位置2: [1, 1, 1, 0, 0]
位置3: [1, 1, 1, 1, 0]
位置4: [1, 1, 1, 1, 1]
```

它也叫 subsequent mask、look-ahead mask、causal mask。名字不同，核心都是遮住未来位置。

### 7.3 MLM Mask：BERT 预训练中的“填空遮盖”

BERT 的 `[MASK]` 和 attention 里的 padding/causal mask 不是同一种东西。BERT 的 MLM mask 是训练任务的一部分：随机遮住一部分 token，让模型根据左右上下文猜回来。

课件中 BERT 的做法：

- 随机选择约 15% 的 token。
- 其中多数替换为 `[MASK]`。
- 少数替换为随机 token 或保持不变。
- 模型目标是预测原始 token。

区别要记清：

| mask 类型 | 用在哪里 | 作用 |
|---|---|---|
| Padding Mask | Encoder、Decoder、Cross-Attention 都可能用 | 不看 PAD |
| Causal/Target Mask | Decoder Self-Attention、GPT | 不看未来 |
| MLM Mask | BERT 预训练输入 | 制造填空任务 |

---

## 8. Encoder 和 Decoder 结构

### 8.1 Transformer Encoder

一个 Encoder Layer 通常包含：

```text
输入 x
  -> Multi-Head Self-Attention
  -> Add & Norm
  -> Feed Forward Network
  -> Add & Norm
输出
```

课件中的 EncoderLayer 使用了 norm-first 风格：

```python
norm_query = self.norm1(query)
self.self_attn_heads.init_keys(norm_query)
states = self.self_attn_heads(norm_query, mask)
att = query + self.drop1(states)

norm_att = self.norm2(att)
out = self.ffn(norm_att)
out = att + self.drop2(out)
```

对应理解：

| 组件 | 作用 |
|---|---|
| Self-Attention | 让每个词和整句其他词交互 |
| FFN | 对每个位置独立做非线性变换 |
| Residual | 保留原始信息，让梯度更容易传 |
| LayerNorm | 稳定每层输入输出 |
| Dropout | 训练时正则化，减少过拟合 |

Encoder 的输入输出 shape 通常不变：

```text
输入: [batch, src_len, d_model]
输出: [batch, src_len, d_model]
```

如果是分类任务，可以只用 Encoder，因为目标不是生成序列，而是把整句表示拿去分类。例如 IMDB 情感分类就是 Encoder-only 加分类头。

### 8.2 Transformer Decoder

一个 Decoder Layer 比 Encoder 多一个 Cross-Attention：

```text
目标序列输入 y
  -> Masked Self-Attention
  -> Add & Norm
  -> Cross-Attention, Q 来自 Decoder, K/V 来自 Encoder
  -> Add & Norm
  -> Feed Forward Network
  -> Add & Norm
输出
```

Decoder 三个子层：

| 子层 | Q 来源 | K/V 来源 | mask |
|---|---|---|---|
| Masked Self-Attention | 目标序列 | 目标序列 | target padding + causal |
| Cross-Attention | Decoder 第一子层输出 | Encoder 输出 | source padding |
| FFN | 每个目标位置 | 每个目标位置 | 通常不需要 attention mask |

输入输出 shape：

```text
Encoder 输出 memory: [batch, src_len, d_model]
Decoder 输入 target: [batch, tgt_len, d_model]
Decoder 输出:        [batch, tgt_len, d_model]
最终 logits:         [batch, tgt_len, vocab_size]
```

### 8.3 Encoder 和 Decoder 的分工

| 结构 | 擅长 | 常见模型 |
|---|---|---|
| Encoder-only | 理解、分类、抽取、检索、序列标注 | BERT、RoBERTa、DistilBERT |
| Decoder-only | 续写、对话、代码生成、自回归文本生成 | GPT 系列、LLaMA、Qwen、DeepSeek |
| Encoder-Decoder | 翻译、摘要、输入输出都较明确的生成任务 | 原始 Transformer、T5、BART |

---

## 9. Transformer 训练流程

### 9.1 从零训练一个文本分类 Transformer

以课件里的 IMDB 情感分类为例，任务是评论 -> 正面/负面，所以只需要 Encoder。

流程：

1. 文本清洗和分词。
2. 建词表，把 token 转为 id。
3. 对句子 padding/truncation，得到 `[batch, seq_len]`。
4. Embedding 层把 id 转成 `[batch, seq_len, d_model]`。
5. 加位置编码。
6. 送入多层 Transformer Encoder。
7. 取整句表示。常见做法是取 `[CLS]` 位置，或者对所有 token pooling。
8. 接线性分类头，输出 `[batch, num_classes]`。
9. 用交叉熵损失训练。

shape 总览：

```text
token ids:     [B, L]
embedding:     [B, L, D]
encoder out:   [B, L, D]
sentence vec:  [B, D]
logits:        [B, C]
labels:        [B]
```

课件中从零训练的 Transformer 在 IMDB 上约 84% 准确率，DistilBERT 微调能到约 90% 到 91%。这说明预训练不是换了完全不同的架构，而是让模型先在大规模语料上学到通用语言知识。

### 9.2 Seq2Seq Transformer 的训练

以翻译为例，训练时有源句和目标句：

```text
源句 src: 我 喜欢 深度 学习
目标 tgt: I like deep learning
```

训练流程：

1. `src` 送入 Encoder，得到 `memory`。
2. `tgt` 右移一位作为 Decoder 输入。例如输入 `<BOS> I like deep`，目标是 `I like deep learning <EOS>`。
3. Decoder 用 causal mask，保证每个位置只能看目标前缀。
4. Decoder Cross-Attention 查询 Encoder 的 `memory`。
5. 输出每个位置的词表 logits。
6. 对每个非 PAD 目标位置计算交叉熵。

训练时可以并行算所有目标位置，因为 causal mask 已经保证每个位置不能看未来。

### 9.3 预训练加微调

以 DistilBERT 微调 IMDB 为例：

1. 加载预训练 tokenizer。
2. tokenizer 输出：
   - `input_ids`: token id 序列
   - `attention_mask`: 真实 token 为 1，padding 为 0
   - `labels`: 分类标签
3. 加载预训练模型并换成二分类头。
4. 用较小学习率微调，例如课件中用 `1e-5`。
5. 用验证集选模型，测试集只做最终评估。

代码中的关键字段：

```python
encoding = tokenizer(
    text,
    add_special_tokens=True,
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

return {
    "input_ids": encoding["input_ids"].flatten(),
    "attention_mask": encoding["attention_mask"].flatten(),
    "labels": torch.tensor(label, dtype=torch.long)
}
```

这里的 `attention_mask` 是 padding mask，不是 BERT 的 `[MASK]` 预训练遮盖。

---

## 10. Transformer 推理流程

训练和推理最容易混淆。

### 10.1 分类任务推理

分类任务很简单：

```text
输入整句 -> tokenizer -> encoder -> 分类头 -> 类别概率
```

一次前向传播就能出结果，不需要逐词生成。

### 10.2 生成任务推理

生成任务必须一步一步来：

1. 输入源句，Encoder 先算好 `memory`。
2. Decoder 输入 `<BOS>`。
3. 得到第一个词的概率分布。
4. 选出一个词，可以用 greedy、beam search、top-k、top-p 等策略。
5. 把新词接到 Decoder 输入后面。
6. 重复直到生成 `<EOS>` 或达到最大长度。

推理时没有真实标签可以喂给 Decoder，所以不能用 teacher forcing。训练时用目标前缀帮助学习，推理时只能用模型自己已经生成的前缀。

### 10.3 GPT 的推理

GPT 是 Decoder-only。没有 Encoder 输入时，它就是根据 prompt 自回归生成：

```text
prompt -> 预测下一个 token -> 拼回 prompt -> 再预测下一个 token -> ...
```

Causal mask 始终保证当前位置只能看前文。

---

## 11. BERT、GPT、ViT 差异

### 11.1 BERT：Encoder-only，偏理解

BERT 用 Transformer Encoder。它的注意力是双向的，一个 token 可以看左右两边。预训练任务主要是 MLM，也就是把一部分词遮住再预测。

适合：

- 文本分类
- 情感分析
- 命名实体识别
- 阅读理解
- 句子相似度

BERT 常用 `[CLS]` 位置的输出表示整句，再接分类头。

### 11.2 GPT：Decoder-only，偏生成

GPT 用 Transformer Decoder 的自回归部分。它不使用 Encoder-Decoder Cross-Attention，核心是 masked self-attention。

预训练任务：

```text
给前面的 token，预测下一个 token
```

适合：

- 文本续写
- 对话
- 代码生成
- 长文本生成
- 指令跟随

### 11.3 ViT：把图像变成序列

Vision Transformer 的关键是把图片切成 patch：

```text
图片 -> 切成小块 patches -> 每个 patch 展平成向量 -> 线性投影成 token -> 加位置编码 -> Transformer Encoder
```

例如图片大小 `[H, W, C]`，patch 大小 `P x P`：

```text
patch 数量 = (H/P) * (W/P)
每个 patch 原始维度 = P * P * C
投影后每个 patch = d_model 维
序列 shape = [batch, num_patches + 1, d_model]
```

多出来的 `+1` 通常是 `[CLS]` token，用来做图像分类。

### 11.4 模型对比表

| 模型 | 架构 | 注意力可见范围 | 预训练目标 | 输入形态 | 输出常见形式 | 代表任务 |
|---|---|---|---|---|---|---|
| 原始 Transformer | Encoder-Decoder | Encoder 双向，Decoder 因果 | 翻译监督训练 | 源序列 + 目标前缀 | 目标序列 | 翻译、摘要 |
| BERT | Encoder-only | 双向 | MLM，早期还有 NSP | 文本 token | 分类、抽取、token 标签 | 理解类任务 |
| GPT | Decoder-only | 只能看前文 | 预测下一个 token | prompt token | 下一个 token 分布 | 生成类任务 |
| ViT | Encoder-only | patch 之间双向 | 图像分类或大规模预训练 | 图像 patches | 类别或视觉特征 | 图像分类、视觉任务 |
| DistilBERT | Encoder-only | 双向 | 蒸馏 BERT | 文本 token | 分类、理解表示 | 轻量微调 |

---

## 12. 输入输出 shape 总表

| 场景 | 输入 | 中间张量 | 输出 |
|---|---|---|---|
| Embedding | `[B, L]` token ids | 查表 | `[B, L, D]` |
| 位置编码 | `[B, L, D]` | 加 `pe[:, :L, :]` | `[B, L, D]` |
| 单头 Attention | Q `[B, n, d_k]`, K `[B, m, d_k]`, V `[B, m, d_v]` | scores `[B, n, m]` | `[B, n, d_v]` |
| 多头 Attention | `[B, L, D]` | `[B, H, L, d_k]` | `[B, L, D]` |
| Encoder Layer | `[B, src_len, D]` | self-attn + FFN | `[B, src_len, D]` |
| Decoder Layer | tgt `[B, tgt_len, D]`, memory `[B, src_len, D]` | masked self-attn + cross-attn + FFN | `[B, tgt_len, D]` |
| 分类头 | `[B, L, D]` | 取 CLS 或 pooling 得 `[B, D]` | `[B, C]` |
| 语言模型头 | `[B, L, D]` | linear 到词表 | `[B, L, vocab_size]` |

---

## 13. 易混点集中讲清

### 13.1 K 和 V 都来自 Encoder，为什么要分开

K 用来算匹配分数，V 用来提供内容。匹配依据和取出的内容可以不同，所以用不同线性层。类比查书：目录标题用于检索，正文内容用于阅读。

### 13.2 Self-Attention 和 Cross-Attention 的区别

| 类型 | Q | K/V |
|---|---|---|
| Self-Attention | 当前序列 | 当前序列 |
| Cross-Attention | Decoder 序列 | Encoder 序列 |

判断方法：如果 Q 和 K/V 来自同一批序列，就是 self；如果 Q 来自目标端、K/V 来自源端，就是 cross。

### 13.3 Source Mask 和 Target Mask 的区别

Source Mask 通常是 padding mask，用来忽略源句 PAD。Target Mask 通常包含 padding mask 和 causal mask，既忽略 PAD，也遮住未来。

### 13.4 `[MASK]` 和 attention mask 的区别

`[MASK]` 是一个实际 token，放进输入里让 BERT 做填空。`attention_mask` 通常是 0/1 标记，告诉 attention 哪些位置不该看。

### 13.5 Encoder-only 为什么能分类

分类不需要逐词生成，只需要理解输入。Encoder 的输出已经融合了全句信息，取 `[CLS]` 或 pooling 后接分类头即可。

### 13.6 GPT 为什么不能直接双向看

GPT 的目标是生成下一个 token。如果训练时能看右边答案，它就学不到真正的续写能力。因此必须用 causal mask。

### 13.7 位置编码不是可有可无

Attention 分数只看向量相似度，不天然知道顺序。没有位置编码，词序信息会严重不足。

### 13.8 `sqrt(d_k)` 里的 `d_k` 是每个头的维度

多头时缩放用的是每个 head 的 key 维度 `d_k = d_model / n_heads`，不是整个 `d_model`。

### 13.9 LayerNorm 和 BatchNorm 为什么不同

Transformer 常用 LayerNorm，因为它对每个 token 的特征维度做归一化，不依赖 batch 统计；序列长度不同、padding 多时，BatchNorm 的统计容易被污染。

---

## 14. 关键概念速查表

| 概念 | 快速解释 | 考试常问点 |
|---|---|---|
| Seq2Seq | 输入序列映射到输出序列 | 机器翻译、摘要等 |
| Encoder | 编码源序列 | 输出 memory |
| Decoder | 根据目标前缀生成输出 | 训练和推理不同 |
| Attention | 动态加权取信息 | Q/K 匹配，V 提供内容 |
| Q | Query，查询 | Decoder 当前状态或当前序列投影 |
| K | Key，匹配依据 | 和 Q 点积得到分数 |
| V | Value，信息内容 | 被注意力权重加权求和 |
| `QK^T` | 原始匹配分数 | shape 是 `[query_len, key_len]` |
| `sqrt(d_k)` | 缩放因子 | 防止 softmax 饱和 |
| softmax | 分数转权重 | 沿 key_len 方向归一化 |
| Multi-Head | 多个 attention 并行 | 不同头学不同关系 |
| Narrow Attention | 先投影再切头 | 大模型常用 |
| Positional Encoding | 注入顺序 | sin/cos 公式 |
| Padding Mask | 忽略 PAD | softmax 前填 `-inf` |
| Causal Mask | 遮住未来 | Decoder/GPT 必需 |
| MLM Mask | BERT 填空训练 | 和 attention mask 区分 |
| Encoder-only | 只用编码器 | BERT、分类理解 |
| Decoder-only | 只用解码器自回归部分 | GPT、生成 |
| Encoder-Decoder | 编码源序列再解码目标序列 | 翻译、摘要 |
| ViT | 图像 patch 当 token | 图像变序列 |

---

## 15. 自测题

### 题 1：Q/K/V 来源判断

在翻译模型 Decoder 的 Cross-Attention 中，Q、K、V 分别来自哪里？为什么 K 和 V 都来自 Encoder 却不能说它们完全一样？

参考答案：Q 来自 Decoder 当前层前一子层输出，K/V 来自 Encoder 输出。K 用于匹配，V 用于提供内容，二者通常经过不同线性层，所以数值和作用不同。

### 题 2：`sqrt(d_k)` 的作用

如果把 scaled dot-product attention 里的除以 `sqrt(d_k)` 去掉，维度很大时可能发生什么？

参考答案：点积分数方差随 `d_k` 增大，softmax 容易极端化，注意力权重接近 one-hot，梯度变小，训练更不稳定。

### 题 3：mask 方向

Decoder 生成第 3 个 token 时，能不能看到第 4 个 token？训练时目标句完整给了模型，为什么仍然不能看？

参考答案：不能看。训练时虽然目标句已知，但模型要学习真实生成过程，只能基于前缀预测后续。如果能看未来，会泄漏答案。

### 题 4：padding mask shape

输入 token shape 是 `[batch, seq_len]`，为什么 padding mask 常做成 `[batch, 1, 1, seq_len]`？

参考答案：attention scores 常是 `[batch, heads, query_len, key_len]`。mask 需要在 head 和 query 维度上广播，只在 key_len 维度标记哪些位置不能被关注。

### 题 5：位置编码必要性

Self-Attention 已经能看全句，为什么还需要位置编码？

参考答案：看全句不等于知道顺序。Attention 本身对位置排列不敏感，需要通过位置编码告诉模型每个 token 在序列中的位置。

### 题 6：Encoder 和 Decoder 混淆

为什么 IMDB 情感分类只用 Encoder 就够了，而机器翻译通常需要 Decoder？

参考答案：情感分类是输入序列到类别，不需要逐词生成；Encoder 提取整句表示后接分类头即可。翻译是输入序列到输出序列，需要 Decoder 自回归生成目标句。

### 题 7：BERT 和 GPT 的 attention 差异

BERT 和 GPT 都是 Transformer，为什么 BERT 可以看右边词，GPT 不可以？

参考答案：BERT 做理解和 MLM 填空，目标是利用双向上下文；GPT 做自回归生成，目标是预测下一个 token，不能看未来答案。

### 题 8：Multi-Head 的 shape

若 `d_model=512`，`n_heads=8`，每个头的 `d_k` 是多少？缩放时除以什么？

参考答案：`d_k=512/8=64`，缩放时除以 `sqrt(64)=8`。

### 题 9：ViT 输入

ViT 为什么能用 Transformer 处理图像？它的“序列”从哪里来？

参考答案：ViT 把图像切成固定大小 patches，每个 patch 展平并投影成 token，再加位置编码，形成 patch token 序列。

### 题 10：训练和推理

Seq2Seq Transformer 训练时 Decoder 可以一次处理整个目标序列，为什么推理时还要一个 token 一个 token 生成？

参考答案：训练时有完整目标序列，可通过 causal mask 并行计算每个位置的预测；推理时没有未来 token，只能先生成一个，再把它作为下一步输入。

---

## 16. 最后总复习

把 Transformer 串起来，可以用下面这条线：

```text
Seq2Seq 任务
  -> 早期 Encoder-Decoder 把源句压成一个向量
  -> 长句信息丢失，形成瓶颈
  -> Attention 让 Decoder 每一步动态看源句不同位置
  -> Q/K/V 把“查什么、怎么匹配、取什么”分开
  -> Scaled Dot-Product 用 QK^T 打分，除 sqrt(d_k)，softmax，再乘 V
  -> Self-Attention 让一个序列内部所有位置互相看
  -> Multi-Head 让多个头学习不同关系
  -> 位置编码补上顺序信息
  -> mask 控制哪些位置不能被看
  -> Encoder 负责理解，Decoder 负责生成
  -> BERT 用 Encoder 做双向理解
  -> GPT 用 Decoder-only 做自回归生成
  -> ViT 把图像 patch 当作 token 序列
```

考前检查自己是否真正理解，可以问三类问题：

1. 来源问题：Q/K/V 分别从哪来？mask 用在哪？位置编码加在哪？
2. shape 问题：`QK^T` 后 shape 是什么？多头切分后 shape 是什么？logits shape 是什么？
3. 任务问题：这个任务是理解、生成，还是输入输出序列转换？应该用 Encoder-only、Decoder-only，还是 Encoder-Decoder？

如果这三类问题都能答清楚，Transformer 的主干就稳了。剩下的 LayerNorm、Dropout、预训练、HuggingFace、RoPE、GQA、FlashAttention 等内容，都是在这个主干上继续增强稳定性、效率和效果。
