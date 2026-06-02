# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构 / 第3节 Transformer Encoder（多层堆叠）
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
