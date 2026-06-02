# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构 / 第4节 Transformer Decoder
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
