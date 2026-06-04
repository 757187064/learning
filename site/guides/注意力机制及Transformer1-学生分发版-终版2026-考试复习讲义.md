# 注意力机制及Transformer1-学生分发版-终版2026：期末考试复习讲义

> 使用方式：先通读“学习路线”，再按章节背“必记句子”，最后用每章自测题检查。这里不是课件原文搬运，而是按考试复习顺序重写。

## 一、这份课件先解决什么问题

这一类课件从序列建模瓶颈进入注意力机制。复习时先把 Q/K/V、注意力权重、mask、位置编码讲顺，再看 Transformer 编码器和解码器。

考试常考注意力公式的每一项含义、Self-Attention 与 Cross-Attention 的区别、以及 mask 为什么存在。

## 二、学习路线

1. **第六章 Seq2Seq、注意力机制与Transformer基础**：先抓 Seq2Seq、Encoder-Decoder、Attention、Q/K/V、Self-Attention。
2. **第六章 Seq2Seq、注意力机制与Transformer基础 / 第1节 序列到序列（Seq2Seq）**：先抓 Seq2Seq、Attention。
3. **第六章 Seq2Seq、注意力机制与Transformer基础 / 第2节 Encoder-Decoder 架构**：先抓 Encoder-Decoder、Teacher Forcing、注意力机制、Attention、上下文向量。
4. **第六章 Seq2Seq、注意力机制与Transformer基础 / 第3节 注意力机制（Attention Mechanism）**：先抓 注意力机制、上下文向量、Attention、Softmax、归一化。
5. **第六章 Seq2Seq、注意力机制与Transformer基础 / 第4节 Self-Attention（自注意力机制）**：先抓 注意力机制、Attention、Query、Key、Value。
6. **第六章 Seq2Seq、注意力机制与Transformer基础 / 第5节 位置编码（Positional Encoding）**：先抓 位置编码、Positional Encoding、Transformer、RNN、Attention。

## 三、章节详解

## 第 1 部分：第六章 Seq2Seq、注意力机制与Transformer基础

### 1. 本节先看什么

这一节先把 **Seq2Seq、Encoder-Decoder、Attention、Q/K/V** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Seq2Seq | 编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。 | 常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。 | 普通 Seq2Seq 容易受固定长度上下文瓶颈影响。 | Seq2Seq = Encoder + Decoder。 |
| Encoder-Decoder | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问... |
| Attention | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问... |
| Q/K/V | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| Positional Encoding | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |
| 注意力机制 | 根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 | 常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 | 注意力不是简单平均，而是按相关性加权。 | Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。 |
| 位置编码 | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问... |
| Softmax | 把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 | 常考多分类输出、注意力权重为什么能加权求和。 | Softmax 是对一组数整体归一化，不是逐个独立压缩。 | Softmax 输出非负且总和为 1。 |
| 梯度消失 | 反向传播经过很多层或很多时间步时，梯度可能不断变小，导致前面层或早期时间步很难学到东西。 | 常考深层网络/RNN 为什么训练困难，以及 LSTM、GRU、残差连接如何缓解。 | 梯度消失不是过拟合；它是参数难以有效更新的问题。 | 多个小于 1 的因子连乘，会让梯度趋近 0。 |
| RNN | 按时间步处理序列，用隐藏状态把前面信息传到后面。 | 常考隐藏状态、序列建模、梯度消失/爆炸。 | RNN 不是一次性把所有时间步完全独立处理。 | 当前输出依赖当前输入和上一时刻隐藏状态。 |
| Teacher Forcing | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练时用真实标签辅助学习 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练... |
| Query | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练时用真实标签辅助学习 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练... |
| Key | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练时用真实标签辅助学习 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练... |
| Value | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练时用真实标签辅助学习 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练... |

### 3. 像考试答案一样组织语言

- **Seq2Seq**：编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。 考试写作时要补一句：常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。 易错点是：普通 Seq2Seq 容易受固定长度上下文瓶颈影响。
- **Encoder-Decoder**：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Attention**：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Q/K/V**：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **Positional Encoding**：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。
- **注意力机制**：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 考试写作时要补一句：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 易错点是：注意力不是简单平均，而是按相关性加权。

### 4. 本节自测

- 判断：Seq2Seq只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Encoder-Decoder只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Q/K/V只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Positional Encoding只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：注意力机制只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Seq2Seq = Encoder + Decoder。
- 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
- 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
- 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
- 自注意力：同一序列内部做注意力。
- 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
- Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
- 第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
- Softmax 输出非负且总和为 1。

## 第 2 部分：第六章 Seq2Seq、注意力机制与Transformer基础 / 第1节 序列到序列（Seq2Seq）

### 1. 本节先看什么

这一节先把 **Seq2Seq、Attention** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Seq2Seq | 编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。 | 常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。 | 普通 Seq2Seq 容易受固定长度上下文瓶颈影响。 | Seq2Seq = Encoder + Decoder。 |
| Attention | 第1节 序列到序列（Seq2Seq）；一、什么是 Seq2Seq？；Seq2Seq（Sequence to Sequence），顾名思义，是一种将一个序列映射到另一个序列的深度学习模型。；与之前我们处理的问题不同：；图像分类：输入一张图 -> 输出一个类别（单输入 -> 单输出） | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第1节 序列到序列（Seq2Seq）；一、什么是 Seq2Seq？；Seq2Seq（Sequence to Sequence），顾名思义，是一种将一个序列映射到另一个序列的深度学习模型。；与之前我们处理的问题不同：；图像分类：输入一张图 -... |

### 3. 像考试答案一样组织语言

- **Seq2Seq**：编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。 考试写作时要补一句：常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。 易错点是：普通 Seq2Seq 容易受固定长度上下文瓶颈影响。
- **Attention**：第1节 序列到序列（Seq2Seq）；一、什么是 Seq2Seq？；Seq2Seq（Sequence to Sequence），顾名思义，是一种将一个序列映射到另一个序列的深度学习模型。；与之前我们处理的问题不同：；图像分类：输入一张图 -> 输出一个类别（单输入 -> 单输出） 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。

### 4. 本节自测

- 判断：Seq2Seq只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Seq2Seq = Encoder + Decoder。
- 第1节 序列到序列（Seq2Seq）；一、什么是 Seq2Seq？；Seq2Seq（Sequence to Sequence），顾名思义，是一种将一个序列映射到另一个序列的深度学习模型。；与之前我们处理的问题不同：；图像分类：输入一张图 -...

## 第 3 部分：第六章 Seq2Seq、注意力机制与Transformer基础 / 第2节 Encoder-Decoder 架构

### 1. 本节先看什么

这一节先把 **Encoder-Decoder、Teacher Forcing、注意力机制、Attention** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Encoder-Decoder | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核... |
| Teacher Forcing | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核... |
| 注意力机制 | 根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 | 常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 | 注意力不是简单平均，而是按相关性加权。 | Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。 |
| Attention | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核... |
| 上下文向量 | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核... |

### 3. 像考试答案一样组织语言

- **Encoder-Decoder**：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Teacher Forcing**：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **注意力机制**：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 考试写作时要补一句：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 易错点是：注意力不是简单平均，而是按相关性加权。
- **Attention**：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **上下文向量**：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。

### 4. 本节自测

- 判断：Encoder-Decoder只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Teacher Forcing只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：注意力机制只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：上下文向量只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
- 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
- Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
- 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
- 第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...

## 第 4 部分：第六章 Seq2Seq、注意力机制与Transformer基础 / 第3节 注意力机制（Attention Mechanism）

### 1. 本节先看什么

这一节先把 **注意力机制、上下文向量、Attention、Softmax** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| 注意力机制 | 根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 | 常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 | 注意力不是简单平均，而是按相关性加权。 | Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。 |
| 上下文向量 | 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 三、Query、Key、Value（Q/K/V）；注意力... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向... |
| Attention | 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 三、Query、Key、Value（Q/K/V）；注意力... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向... |
| Softmax | 把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 | 常考多分类输出、注意力权重为什么能加权求和。 | Softmax 是对一组数整体归一化，不是逐个独立压缩。 | Softmax 输出非负且总和为 1。 |
| 归一化 | 把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 | 常考图像预处理为什么要缩放像素值。 | 归一化不等于 BatchNorm，前者多是输入预处理。 | 图像常先把像素值缩放到 0-1。 |
| Query | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decoder 的当前隐藏状态 / "我想查什么" / 是 五、为什么... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decod... |
| Key | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decoder 的当前隐藏状态 / "我想查什么" / 是 五、为什么... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decod... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |
| Encoder-Decoder | 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向... |
| Value | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decoder 的当前隐藏状态 / "我想查什么" / 是 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decod... |
| Q/K/V | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decoder 的当前隐藏状态 / "我想查什么" / 是 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decod... |
| Scaled Dot-Product Attention | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decoder 的当前隐藏状态 / "我想查什么" / 是 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 / 来源 / 作用 / 是否经过线性变换；Query（Q） / Decod... |
| 反向传播 | 五、为什么要除以 $\sqrt{d_k}$？；1. 从相似度到点积：为什么用点积计算注意力分数？；在注意力机制中，我们需要衡量一个Query（查询）与每个Key（键）的"匹配程度"。最常用的方法是点积（Dot Product），但让我们从更基础的余弦相似度Cosine Similarity说起。；余... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 五、为什么要除以 $\sqrt{d_k}$？；1. 从相似度到点积：为什么用点积计算注意力分数？；在注意力机制中，我们需要衡量一个Query（查询）与每个Key（键）的"匹配程度"。最常用的方法是点积（Dot Product），但让我们从更... |
| 标准化 | 把数据按均值和方差调整到更稳定的尺度，帮助模型训练。 | 常考归一化和标准化的区别、为什么图像输入要预处理。 | 不要把标准化简单等同于除以 255；标准化通常涉及均值和标准差。 | 标准化常写作 (x-mean)/std。 |
| 梯度消失 | 反向传播经过很多层或很多时间步时，梯度可能不断变小，导致前面层或早期时间步很难学到东西。 | 常考深层网络/RNN 为什么训练困难，以及 LSTM、GRU、残差连接如何缓解。 | 梯度消失不是过拟合；它是参数难以有效更新的问题。 | 多个小于 1 的因子连乘，会让梯度趋近 0。 |
| Batch | 一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 | 常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 | Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。 | Batch 是一批样本；Epoch 是全训练集完整训练一遍。 |

### 3. 像考试答案一样组织语言

- **注意力机制**：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 考试写作时要补一句：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 易错点是：注意力不是简单平均，而是按相关性加权。
- **上下文向量**：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 三、Query、Key、Value（Q/K/V）；注意力... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Attention**：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 三、Query、Key、Value（Q/K/V）；注意力... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Softmax**：把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 考试写作时要补一句：常考多分类输出、注意力权重为什么能加权求和。 易错点是：Softmax 是对一组数整体归一化，不是逐个独立压缩。
- **归一化**：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 考试写作时要补一句：常考图像预处理为什么要缩放像素值。 易错点是：归一化不等于 BatchNorm，前者多是输入预处理。
- **Query**：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decoder 的当前隐藏状态 | "我想查什么" | 是 五、为什么... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Key**：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decoder 的当前隐藏状态 | "我想查什么" | 是 五、为什么... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：注意力机制只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：上下文向量只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Softmax只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：归一化只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Query只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Key只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
- 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向...
- 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向...
- Softmax 输出非负且总和为 1。
- 图像常先把像素值缩放到 0-1。
- 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...
- 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
- 第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向...
- 三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...

## 第 5 部分：第六章 Seq2Seq、注意力机制与Transformer基础 / 第4节 Self-Attention（自注意力机制）

### 1. 本节先看什么

这一节先把 **注意力机制、Attention、Query、Key** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| 注意力机制 | 根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 | 常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 | 注意力不是简单平均，而是按相关性加权。 | Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。 |
| Attention | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](... |
| Query | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](... |
| Key | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](... |
| Value | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |
| Cross-Attention | Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。 | 常考 Transformer 解码器中 Cross-Attention 的输入来源。 | 不要和 Self-Attention 混成同一来源。 | 交叉注意力：Q 与 K/V 来源不同。 |
| RNN | 按时间步处理序列，用隐藏状态把前面信息传到后面。 | 常考隐藏状态、序列建模、梯度消失/爆炸。 | RNN 不是一次性把所有时间步完全独立处理。 | 当前输出依赖当前输入和上一时刻隐藏状态。 |
| 上下文向量 | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](... |
| Target Mask | 五、Target Mask（目标掩码 / 后续掩码）；为了确保 Decoder 不"偷看"未来，我们需要一种特殊的掩码：Target Mask。；Target Mask 的核心思想；Target Mask 是一个上三角掩码（Upper-Triangular Mask）：；位置: 0 1 2 3 4 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 五、Target Mask（目标掩码 / 后续掩码）；为了确保 Decoder 不"偷看"未来，我们需要一种特殊的掩码：Target Mask。；Target Mask 的核心思想；Target Mask 是一个上三角掩码（Upper-Tr... |
| Subsequent Mask | 五、Target Mask（目标掩码 / 后续掩码）；为了确保 Decoder 不"偷看"未来，我们需要一种特殊的掩码：Target Mask。；Target Mask 的核心思想；Target Mask 是一个上三角掩码（Upper-Triangular Mask）：；位置: 0 1 2 3 4 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 五、Target Mask（目标掩码 / 后续掩码）；为了确保 Decoder 不"偷看"未来，我们需要一种特殊的掩码：Target Mask。；Target Mask 的核心思想；Target Mask 是一个上三角掩码（Upper-Tr... |

### 3. 像考试答案一样组织语言

- **注意力机制**：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。 考试写作时要补一句：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。 易错点是：注意力不是简单平均，而是按相关性加权。
- **Attention**：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Query**：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Key**：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Value**：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。
- **Cross-Attention**：Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。 考试写作时要补一句：常考 Transformer 解码器中 Cross-Attention 的输入来源。 易错点是：不要和 Self-Attention 混成同一来源。

### 4. 本节自测

- 判断：注意力机制只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Query只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Key只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Value只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Cross-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
- 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
- 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
- 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
- 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
- 自注意力：同一序列内部做注意力。
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
- 交叉注意力：Q 与 K/V 来源不同。
- 当前输出依赖当前输入和上一时刻隐藏状态。
- 第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...

## 第 6 部分：第六章 Seq2Seq、注意力机制与Transformer基础 / 第5节 位置编码（Positional Encoding）

### 1. 本节先看什么

这一节先把 **位置编码、Positional Encoding、Transformer、RNN** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| 位置编码 | 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 二、位置编码的设计推导；设计一个好的位置编码，需要... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编... |
| Positional Encoding | 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 二、位置编码的设计推导；设计一个好的位置编码，需要... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |
| RNN | 按时间步处理序列，用隐藏状态把前面信息传到后面。 | 常考隐藏状态、序列建模、梯度消失/爆炸。 | RNN 不是一次性把所有时间步完全独立处理。 | 当前输出依赖当前输入和上一时刻隐藏状态。 |
| Attention | 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| RoPE | 一种旋转位置编码，把位置信息注入向量表示，常用于现代 Transformer。 | 如果课件讲到，主要考它属于位置编码思想，不会要求复杂推导。 | 不要把 RoPE 写成普通绝对位置编号相加。 | RoPE 用旋转方式表达相对位置信息。 |

### 3. 像考试答案一样组织语言

- **位置编码**：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 二、位置编码的设计推导；设计一个好的位置编码，需要... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Positional Encoding**：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 二、位置编码的设计推导；设计一个好的位置编码，需要... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。
- **RNN**：按时间步处理序列，用隐藏状态把前面信息传到后面。 考试写作时要补一句：常考隐藏状态、序列建模、梯度消失/爆炸。 易错点是：RNN 不是一次性把所有时间步完全独立处理。
- **Attention**：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **RoPE**：一种旋转位置编码，把位置信息注入向量表示，常用于现代 Transformer。 考试写作时要补一句：如果课件讲到，主要考它属于位置编码思想，不会要求复杂推导。 易错点是：不要把 RoPE 写成普通绝对位置编号相加。

### 4. 本节自测

- 判断：位置编码只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Positional Encoding只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：RNN只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：RoPE只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编...
- 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
- 当前输出依赖当前输入和上一时刻隐藏状态。
- 第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编...
- 自注意力：同一序列内部做注意力。
- RoPE 用旋转方式表达相对位置信息。

## 四、考前总复盘

考前不要平均用力。优先检查下面这些问题：

1. 每个核心概念能不能用一句话说明“它是什么”。
2. 能不能说出它在模型结构、训练流程或数据处理中的位置。
3. 能不能说出一个最常见的错误说法。
4. 遇到选择题时，能不能判断选项是在混淆概念、夸大作用，还是写反了训练/推理阶段。

## 五、打印建议

<style>@media print { @page { margin: 8mm; } body { font-size: 11pt; line-height: 1.35; } h1, h2, h3 { page-break-after: avoid; } table { font-size: 9pt; border-collapse: collapse; } th, td { padding: 3px 5px; border: 1px solid #ddd; vertical-align: top; } }</style>
