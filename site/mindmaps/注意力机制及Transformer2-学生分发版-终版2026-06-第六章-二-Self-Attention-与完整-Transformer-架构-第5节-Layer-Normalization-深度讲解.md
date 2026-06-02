# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构 / 第5节 Layer Normalization 深度讲解
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
