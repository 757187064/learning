# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构
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
