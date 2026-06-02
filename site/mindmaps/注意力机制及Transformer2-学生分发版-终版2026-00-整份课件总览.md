# 注意力机制及Transformer2-学生分发版-终版2026 / 总览
## 第六章（二）Self-Attention 与完整 Transformer 架构
### Batch
- 通俗解释：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3....
### BatchNorm
- 通俗解释：在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。
- 考试怎么考：常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。
- 易错点：推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。
- 必记：BatchNorm：标准化激活，稳定训练，加快收敛。
### LSTM
- 通俗解释：通过输入门、遗忘门、输出门和细胞状态保存长期信息。
- 考试怎么考：常考门控作用，以及为什么能缓解长序列梯度问题。
- 易错点：不要漏掉细胞状态是 LSTM 的关键通道。
- 必记：LSTM：输入门、遗忘门、输出门、细胞状态。
### Attention
- 通俗解释：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3....
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### Cross-Attention
- 通俗解释：Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。
- 考试怎么考：常考 Transformer 解码器中 Cross-Attention 的输入来源。
- 易错点：不要和 Self-Attention 混成同一来源。
- 必记：交叉注意力：Q 与 K/V 来源不同。
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
### Encoder-Decoder
- 通俗解释：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3....
### 归一化
- 通俗解释：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。
- 考试怎么考：常考图像预处理为什么要缩放像素值。
- 易错点：归一化不等于 BatchNorm，前者多是输入预处理。
- 必记：图像常先把像素值缩放到 0-1。
### Dropout
- 通俗解释：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。
- 考试怎么考：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。
- 易错点：Dropout 不是提高模型容量，而是正则化。
- 必记：Dropout：训练随机失活，推理正常使用。
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第1节 回顾与过渡
### Softmax
- 通俗解释：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
### Dropout
- 通俗解释：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。
- 考试怎么考：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。
- 易错点：Dropout 不是提高模型容量，而是正则化。
- 必记：Dropout：训练随机失活，推理正常使用。
### 残差连接
- 通俗解释：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
### RNN
- 通俗解释：按时间步处理序列，用隐藏状态把前面信息传到后面。
- 考试怎么考：常考隐藏状态、序列建模、梯度消失/爆炸。
- 易错点：RNN 不是一次性把所有时间步完全独立处理。
- 必记：当前输出依赖当前输入和上一时刻隐藏状态。
### Seq2Seq
- 通俗解释：编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。
- 考试怎么考：常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。
- 易错点：普通 Seq2Seq 容易受固定长度上下文瓶颈影响。
- 必记：Seq2Seq = Encoder + Decoder。
### Attention
- 通俗解释：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
### Q/K/V
- 通俗解释：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
### 上下文向量
- 通俗解释：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### Positional Encoding
- 通俗解释：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第2节 Narrow Attention 深度解析
### Batch
- 通俗解释：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wide Attention | 完整 $d_{model}$ |...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wid...
### Softmax
- 通俗解释：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wide Attention | 完整 $d_{model}$ |...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wid...
### Attention
- 通俗解释：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wide Attention | 完整 $d_{model}$ |...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wid...
### Multi-Head Attention
- 通俗解释：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wide Attention | 完整 $d_{model}$ |...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wid...
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第3节 Transformer Encoder（多层堆叠）
### Softmax
- 通俗解释：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
### MLP
- 通俗解释：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
### 归一化
- 通俗解释：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。
- 考试怎么考：常考图像预处理为什么要缩放像素值。
- 易错点：归一化不等于 BatchNorm，前者多是输入预处理。
- 必记：图像常先把像素值缩放到 0-1。
### Dropout
- 通俗解释：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。
- 考试怎么考：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。
- 易错点：Dropout 不是提高模型容量，而是正则化。
- 必记：Dropout：训练随机失活，推理正常使用。
### 残差连接
- 通俗解释：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
### Attention
- 通俗解释：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
### Multi-Head Attention
- 通俗解释：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### 位置编码
- 通俗解释：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第4节 Transformer Decoder
### Attention
- 通俗解释：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
### Query
- 通俗解释：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
### Key
- 通俗解释：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
### Value
- 通俗解释：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### Cross-Attention
- 通俗解释：Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。
- 考试怎么考：常考 Transformer 解码器中 Cross-Attention 的输入来源。
- 易错点：不要和 Self-Attention 混成同一来源。
- 必记：交叉注意力：Q 与 K/V 来源不同。
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第5节 Layer Normalization 深度讲解
### Batch
- 通俗解释：第5节 Layer Normalization 深度讲解；一、为什么 Transformer 不用 BatchNorm？；在深度学习中，Batch Normalization（BatchNorm）曾经是稳定训练的标配。但在 Transformer 中，研究者选择了 Layer Normalizati...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第5节 Layer Normalization 深度讲解；一、为什么 Transformer 不用 BatchNorm？；在深度学习中，Batch Normalization（BatchNorm）曾经是稳定训练的标配。但在 Transfor...
### 填充
- 通俗解释：在输入边界补值，让卷积能处理边缘或保持输出尺寸。
- 考试怎么考：常考 Same 卷积为何需要 padding。
- 易错点：填充不是增加有效信息，而是控制边界和尺寸。
- 必记：Padding 用于控制输出尺寸和边界信息。
### 归一化
- 通俗解释：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。
- 考试怎么考：常考图像预处理为什么要缩放像素值。
- 易错点：归一化不等于 BatchNorm，前者多是输入预处理。
- 必记：图像常先把像素值缩放到 0-1。
### BatchNorm
- 通俗解释：在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。
- 考试怎么考：常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。
- 易错点：推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。
- 必记：BatchNorm：标准化激活，稳定训练，加快收敛。
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）
### LSTM
- 通俗解释：通过输入门、遗忘门、输出门和细胞状态保存长期信息。
- 考试怎么考：常考门控作用，以及为什么能缓解长序列梯度问题。
- 易错点：不要漏掉细胞状态是 LSTM 的关键通道。
- 必记：LSTM：输入门、遗忘门、输出门、细胞状态。
### Attention
- 通俗解释：第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）；一、任务介绍：IMDB 电影评论情感分析；在上一章（循环神经网络）中，我们使用 LSTM 对 IMDB 电影评论进行情感分类（正面/负面）。本章，我们将用 Transformer 解决同一个任务，对比两者的效果差异...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）；一、任务介绍：IMDB 电影评论情感分析；在上一章（循环神经网络）中，我们使用 LSTM 对 IMDB 电影评论进行情感分类（正面/负面）。本章，我们将用 T...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第7节 PyTorch 内置 Transformer
### Batch
- 通俗解释：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
### 激活函数
- 通俗解释：给线性变换加入非线性，使网络能拟合非线性关系。
- 考试怎么考：常考没有激活函数时多层线性网络仍等价于线性模型。
- 易错点：不要把激活函数说成只改变维度，它主要提供非线性。
- 必记：仿射变换 + 激活函数 = 神经网络层的基本模式。
### 填充
- 通俗解释：在输入边界补值，让卷积能处理边缘或保持输出尺寸。
- 考试怎么考：常考 Same 卷积为何需要 padding。
- 易错点：填充不是增加有效信息，而是控制边界和尺寸。
- 必记：Padding 用于控制输出尺寸和边界信息。
### 归一化
- 通俗解释：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。
- 考试怎么考：常考图像预处理为什么要缩放像素值。
- 易错点：归一化不等于 BatchNorm，前者多是输入预处理。
- 必记：图像常先把像素值缩放到 0-1。
### Dropout
- 通俗解释：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。
- 考试怎么考：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。
- 易错点：Dropout 不是提高模型容量，而是正则化。
- 必记：Dropout：训练随机失活，推理正常使用。
### Padding
- 通俗解释：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
### Key
- 通俗解释：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
### Subsequent Mask
- 通俗解释：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
### 位置编码
- 通俗解释：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉
### MLP
- 通俗解释：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti...
### 卷积
- 通俗解释：用小窗口在图像局部滑动，对局部像素和卷积核权重相乘求和，提取边缘、纹理等局部特征。
- 考试怎么考：常考输出尺寸、卷积核/步长/填充的作用，以及 CNN 实际常用互相关。
- 易错点：不要把 CNN 说成全连接；卷积核心是局部连接和参数共享。
- 必记：输出尺寸：(输入尺寸 + 2P - K) / S + 1。
### 卷积核
- 通俗解释：卷积层中可学习的小矩阵/滤波器，用来扫描局部区域并提取特征。
- 考试怎么考：常考卷积核大小、通道数、参数共享和输出尺寸计算。
- 易错点：卷积核参数会训练更新，不是手工固定模板。
- 必记：卷积核大小 K、步长 S、填充 P 共同决定输出尺寸。
### 步长
- 通俗解释：卷积核或池化窗口每次移动的间隔。
- 考试怎么考：常考步长变大时输出尺寸如何变化。
- 易错点：步长越大，输出空间尺寸通常越小。
- 必记：Stride 控制滑动间隔。
### 感受野
- 通俗解释：某一层一个神经元能看到原始输入图像的区域大小。
- 考试怎么考：常考层数、卷积核大小、步长如何影响感受野。
- 易错点：步长变大通常会让前一层对应感受野扩大，不是缩小。
- 必记：层越深，感受野通常越大；多层小卷积能逐步扩大感受野。
### 池化
- 通俗解释：对局部区域做最大值或平均值汇总，降低空间尺寸并增强一定平移不变性。
- 考试怎么考：常考池化作用、是否有可学习参数、与卷积的区别。
- 易错点：池化不是全连接，也不是用来增加参数量。
- 必记：池化：降采样、压缩空间尺寸、保留显著信息。
### ResNet
- 通俗解释：用残差/跳跃连接让网络学习 F(x)+x，缓解深层网络退化和梯度传播困难。
- 考试怎么考：常考残差连接为什么能训练更深网络。
- 易错点：残差连接不是单纯增加层数，而是改变信息和梯度路径。
- 必记：ResNet 核心：y = F(x) + x。
### Attention
- 通俗解释：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### 位置编码
- 通俗解释：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti...
## 第六章（二）Self-Attention 与完整 Transformer 架构 / 第10节 预训练模型：DistilBERT 微调实战
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
### LSTM
- 通俗解释：通过输入门、遗忘门、输出门和细胞状态保存长期信息。
- 考试怎么考：常考门控作用，以及为什么能缓解长序列梯度问题。
- 易错点：不要漏掉细胞状态是 LSTM 的关键通道。
- 必记：LSTM：输入门、遗忘门、输出门、细胞状态。
### 学习率
- 通俗解释：预训练模型 vs 从零训练：对比总结；指标 | 我们的 Transformer (v6) | DistilBERT (v8/v10)；参数量 | ~4.7M | ~66M；预训练数据 | 无 | Wikipedia + BookCorpus（数十亿词）；训练方式 | 随机初始化 | 预训练权重 +...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：预训练模型 vs 从零训练：对比总结；指标 | 我们的 Transformer (v6) | DistilBERT (v8/v10)；参数量 | ~4.7M | ~66M；预训练数据 | 无 | Wikipedia + BookCorpus...
### Epoch
- 通俗解释：预训练模型 vs 从零训练：对比总结；指标 | 我们的 Transformer (v6) | DistilBERT (v8/v10)；参数量 | ~4.7M | ~66M；预训练数据 | 无 | Wikipedia + BookCorpus（数十亿词）；训练方式 | 随机初始化 | 预训练权重 +...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：预训练模型 vs 从零训练：对比总结；指标 | 我们的 Transformer (v6) | DistilBERT (v8/v10)；参数量 | ~4.7M | ~66M；预训练数据 | 无 | Wikipedia + BookCorpus...
### 早停
- 通俗解释：预训练模型 vs 从零训练：对比总结；指标 | 我们的 Transformer (v6) | DistilBERT (v8/v10)；参数量 | ~4.7M | ~66M；预训练数据 | 无 | Wikipedia + BookCorpus（数十亿词）；训练方式 | 随机初始化 | 预训练权重 +...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：预训练模型 vs 从零训练：对比总结；指标 | 我们的 Transformer (v6) | DistilBERT (v8/v10)；参数量 | ~4.7M | ~66M；预训练数据 | 无 | Wikipedia + BookCorpus...
