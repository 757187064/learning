# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构 / 第10节 预训练模型：DistilBERT 微调实战
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
