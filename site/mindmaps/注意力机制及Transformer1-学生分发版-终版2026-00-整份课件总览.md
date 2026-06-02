# 注意力机制及Transformer1-学生分发版-终版2026 / 总览
## 第六章 Seq2Seq、注意力机制与Transformer基础
### Seq2Seq
- 通俗解释：编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。
- 考试怎么考：常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。
- 易错点：普通 Seq2Seq 容易受固定长度上下文瓶颈影响。
- 必记：Seq2Seq = Encoder + Decoder。
### Encoder-Decoder
- 通俗解释：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
### Attention
- 通俗解释：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
### Q/K/V
- 通俗解释：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### Positional Encoding
- 通俗解释：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
### 注意力机制
- 通俗解释：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。
- 考试怎么考：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。
- 易错点：注意力不是简单平均，而是按相关性加权。
- 必记：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
### 位置编码
- 通俗解释：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问题到Q/K/V，再到Scaled Dot-Product；4...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第六章 Seq2Seq、注意力机制与Transformer基础；1. Seq2Seq（序列到序列）：理解序列映射任务的本质；2. Encoder-Decoder架构：编码器-解码器的工作机制；3. 注意力机制（Attention）：从瓶颈问...
### Softmax
- 通俗解释：1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练时用真实标签辅助学习
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：1. Seq2Seq：序列到序列的映射框架，输入和输出都是序列，长度可以不同；2. Encoder-Decoder：；Encoder：将源序列压缩为上下文向量；Decoder：将上下文向量解码为目标序列；Teacher Forcing：训练...
## 第六章 Seq2Seq、注意力机制与Transformer基础 / 第1节 序列到序列（Seq2Seq）
### Seq2Seq
- 通俗解释：编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。
- 考试怎么考：常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。
- 易错点：普通 Seq2Seq 容易受固定长度上下文瓶颈影响。
- 必记：Seq2Seq = Encoder + Decoder。
### Attention
- 通俗解释：第1节 序列到序列（Seq2Seq）；一、什么是 Seq2Seq？；Seq2Seq（Sequence to Sequence），顾名思义，是一种将一个序列映射到另一个序列的深度学习模型。；与之前我们处理的问题不同：；图像分类：输入一张图 -> 输出一个类别（单输入 -> 单输出）
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第1节 序列到序列（Seq2Seq）；一、什么是 Seq2Seq？；Seq2Seq（Sequence to Sequence），顾名思义，是一种将一个序列映射到另一个序列的深度学习模型。；与之前我们处理的问题不同：；图像分类：输入一张图 -...
## 第六章 Seq2Seq、注意力机制与Transformer基础 / 第2节 Encoder-Decoder 架构
### 隐藏状态
- 通俗解释：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
### Encoder-Decoder
- 通俗解释：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
### Teacher Forcing
- 通俗解释：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
### 注意力机制
- 通俗解释：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。
- 考试怎么考：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。
- 易错点：注意力不是简单平均，而是按相关性加权。
- 必记：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
### Attention
- 通俗解释：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
### 上下文向量
- 通俗解释：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核心任务：将源序列（Source Sequence）编码成一个...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第2节 Encoder-Decoder 架构；为了完成"序列 -> 序列"的映射任务，研究者们提出了一种优雅的架构：编码器-解码器架构（Encoder-Decoder Architecture）。；一、Encoder（编码器）：理解输入；核...
## 第六章 Seq2Seq、注意力机制与Transformer基础 / 第3节 注意力机制（Attention Mechanism）
### 注意力机制
- 通俗解释：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。
- 考试怎么考：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。
- 易错点：注意力不是简单平均，而是按相关性加权。
- 必记：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
### 上下文向量
- 通俗解释：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 三、Query、Key、Value（Q/K/V）；注意力...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向...
### Attention
- 通俗解释：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量 三、Query、Key、Value（Q/K/V）；注意力...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向...
### Softmax
- 通俗解释：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decoder 的当前隐藏状态 | "我想查什么" | 是 五、为什么...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...
### 归一化
- 通俗解释：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。
- 考试怎么考：常考图像预处理为什么要缩放像素值。
- 易错点：归一化不等于 BatchNorm，前者多是输入预处理。
- 必记：图像常先把像素值缩放到 0-1。
### Query
- 通俗解释：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decoder 的当前隐藏状态 | "我想查什么" | 是 五、为什么...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...
### Key
- 通俗解释：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decoder 的当前隐藏状态 | "我想查什么" | 是 五、为什么...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
### Encoder-Decoder
- 通俗解释：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向量
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第3节 注意力机制（Attention Mechanism）；一、为什么需要注意力机制？；回顾 Encoder-Decoder 的瓶颈问题：；整个源序列被压缩成一个上下文向量；Decoder 在生成每一个目标词时，都只能依赖这同一个上下文向...
### 隐藏状态
- 通俗解释：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decoder 的当前隐藏状态 | "我想查什么" | 是
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：三、Query、Key、Value（Q/K/V）；注意力机制引入了三个核心概念：Query（查询）、Key（键）、Value（值）。；2. Q/K/V 的来源；概念 | 来源 | 作用 | 是否经过线性变换；Query（Q） | Decod...
## 第六章 Seq2Seq、注意力机制与Transformer基础 / 第4节 Self-Attention（自注意力机制）
### 注意力机制
- 通俗解释：根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。
- 考试怎么考：常考 Q/K/V 含义、注意力权重、上下文向量计算流程。
- 易错点：注意力不是简单平均，而是按相关性加权。
- 必记：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
### Attention
- 通俗解释：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
### Query
- 通俗解释：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
### Key
- 通俗解释：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
### Value
- 通俗解释：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](https://arxiv.org/abs/1706.037...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第4节 Self-Attention（自注意力机制）；一、Attention Is All You Need；2017年，Google 的研究团队发表了一篇具有里程碑意义的论文：；["Attention Is All You Need"](...
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
### Cross-Attention
- 通俗解释：Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。
- 考试怎么考：常考 Transformer 解码器中 Cross-Attention 的输入来源。
- 易错点：不要和 Self-Attention 混成同一来源。
- 必记：交叉注意力：Q 与 K/V 来源不同。
### 卷积
- 通俗解释：用小窗口在图像局部滑动，对局部像素和卷积核权重相乘求和，提取边缘、纹理等局部特征。
- 考试怎么考：常考输出尺寸、卷积核/步长/填充的作用，以及 CNN 实际常用互相关。
- 易错点：不要把 CNN 说成全连接；卷积核心是局部连接和参数共享。
- 必记：输出尺寸：(输入尺寸 + 2P - K) / S + 1。
### RNN
- 通俗解释：按时间步处理序列，用隐藏状态把前面信息传到后面。
- 考试怎么考：常考隐藏状态、序列建模、梯度消失/爆炸。
- 易错点：RNN 不是一次性把所有时间步完全独立处理。
- 必记：当前输出依赖当前输入和上一时刻隐藏状态。
## 第六章 Seq2Seq、注意力机制与Transformer基础 / 第5节 位置编码（Positional Encoding）
### 位置编码
- 通俗解释：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 二、位置编码的设计推导；设计一个好的位置编码，需要...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编...
### Positional Encoding
- 通俗解释：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序 二、位置编码的设计推导；设计一个好的位置编码，需要...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编...
### Transformer
- 通俗解释：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。
- 考试怎么考：常考编码器/解码器结构、多头注意力和位置编码。
- 易错点：Transformer 不依赖 RNN 的逐步递归来建模序列。
- 必记：Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
### RNN
- 通俗解释：按时间步处理序列，用隐藏状态把前面信息传到后面。
- 考试怎么考：常考隐藏状态、序列建模、梯度消失/爆炸。
- 易错点：RNN 不是一次性把所有时间步完全独立处理。
- 必记：当前输出依赖当前输入和上一时刻隐藏状态。
### Attention
- 通俗解释：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编码了顺序
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：第5节 位置编码（Positional Encoding）；一、为什么需要位置编码？；Self-Attention 有一个致命的缺陷：它完全丢失了序列的顺序信息。；对比 RNN：；RNN 是顺序计算的：h_t 依赖于 h_{t-1}，天然编...
### Self-Attention
- 通俗解释：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。
- 考试怎么考：常考它和 Cross-Attention 的区别。
- 易错点：Self 不是只看自己一个位置，而是同一序列内部互相看。
- 必记：自注意力：同一序列内部做注意力。
### RoPE
- 通俗解释：一种旋转位置编码，把位置信息注入向量表示，常用于现代 Transformer。
- 考试怎么考：如果课件讲到，主要考它属于位置编码思想，不会要求复杂推导。
- 易错点：不要把 RoPE 写成普通绝对位置编号相加。
- 必记：RoPE 用旋转方式表达相对位置信息。
### 嵌入
- 通俗解释：四、现代大模型中的旋转位置编码（RoPE）；前面我们学习了经典的 sin/cos 位置编码，这是 2017 年原始 Transformer 论文中的方案。但在现代大语言模型（如 LLaMA、ChatGLM、DeepSeek）中，广泛使用的是一种更先进的位置编码——旋转位置编码（RoPE, Rotar...
- 考试怎么考：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。
- 易错点：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- 必记：四、现代大模型中的旋转位置编码（RoPE）；前面我们学习了经典的 sin/cos 位置编码，这是 2017 年原始 Transformer 论文中的方案。但在现代大语言模型（如 LLaMA、ChatGLM、DeepSeek）中，广泛使用的是...
