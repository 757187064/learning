# 注意力机制及Transformer2-学生分发版-终版2026：期末考试复习讲义

> 使用方式：先通读“学习路线”，再按章节背“必记句子”，最后用每章自测题检查。这里不是课件原文搬运，而是按考试复习顺序重写。

## 一、这份课件先解决什么问题

这一类课件从序列建模瓶颈进入注意力机制。复习时先把 Q/K/V、注意力权重、mask、位置编码讲顺，再看 Transformer 编码器和解码器。

考试常考注意力公式的每一项含义、Self-Attention 与 Cross-Attention 的区别、以及 mask 为什么存在。

## 二、学习路线

1. **第六章（二）Self-Attention 与完整 Transformer 架构**：先抓 BatchNorm、Attention、Self-Attention、Cross-Attention、Transformer。
2. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第1节 回顾与过渡**：先抓 Softmax、Dropout、RNN、Seq2Seq、Attention。
3. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第2节 Narrow Attention 深度解析**：先抓 Batch、Softmax、Attention、Multi-Head Attention。
4. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第3节 Transformer Encoder（多层堆叠）**：先抓 Softmax、归一化、Dropout、Attention、Multi-Head Attention。
5. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第4节 Transformer Decoder**：先抓 Attention、Query、Key、Value、Self-Attention。
6. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第5节 Layer Normalization 深度讲解**：先抓 Batch、归一化、BatchNorm、Transformer。
7. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）**：先抓 Attention、Self-Attention、Transformer。
8. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第7节 PyTorch 内置 Transformer**：先抓 Batch、激活函数、归一化、Dropout、Key。
9. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉**：先抓 Attention、Self-Attention、位置编码、Transformer。
10. **第六章（二）Self-Attention 与完整 Transformer 架构 / 第10节 预训练模型：DistilBERT 微调实战**：先抓 Transformer、学习率、Epoch。

## 三、章节详解

## 第 1 部分：第六章（二）Self-Attention 与完整 Transformer 架构

### 1. 本节先看什么

这一节先把 **BatchNorm、Attention、Self-Attention、Cross-Attention** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| BatchNorm | 在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。 | 常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。 | 推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。 | BatchNorm：标准化激活，稳定训练，加快收敛。 |
| Attention | 第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3.... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| Cross-Attention | Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。 | 常考 Transformer 解码器中 Cross-Attention 的输入来源。 | 不要和 Self-Attention 混成同一来源。 | 交叉注意力：Q 与 K/V 来源不同。 |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |
| Encoder-Decoder | 第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3.... |
| Batch | 一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 | 常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 | Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。 | Batch 是一批样本；Epoch 是全训练集完整训练一遍。 |
| 归一化 | 把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 | 常考图像预处理为什么要缩放像素值。 | 归一化不等于 BatchNorm，前者多是输入预处理。 | 图像常先把像素值缩放到 0-1。 |
| Dropout | 训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 | 常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 | Dropout 不是提高模型容量，而是正则化。 | Dropout：训练随机失活，推理正常使用。 |
| Query | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T5 全部使用 Narrow Attention；2. Sub... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T... |
| Key | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T5 全部使用 Narrow Attention；2. Sub... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T... |
| Value | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T5 全部使用 Narrow Attention；2. Sub... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T... |
| Positional Encoding | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T5 全部使用 Narrow Attention；2. Sub... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T... |

### 3. 像考试答案一样组织语言

- **BatchNorm**：在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。 考试写作时要补一句：常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。 易错点是：推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。
- **Attention**：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **Cross-Attention**：Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。 考试写作时要补一句：常考 Transformer 解码器中 Cross-Attention 的输入来源。 易错点是：不要和 Self-Attention 混成同一来源。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。
- **Encoder-Decoder**：第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3. Transformer Encoder：从单层到多层堆叠；4... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Batch**：一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 考试写作时要补一句：常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 易错点是：Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。
- **归一化**：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 考试写作时要补一句：常考图像预处理为什么要缩放像素值。 易错点是：归一化不等于 BatchNorm，前者多是输入预处理。

### 4. 本节自测

- 判断：BatchNorm只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Cross-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Encoder-Decoder只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Batch只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：归一化只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- BatchNorm：标准化激活，稳定训练，加快收敛。
- 第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3....
- 自注意力：同一序列内部做注意力。
- 交叉注意力：Q 与 K/V 来源不同。
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
- 第六章（二）Self-Attention 与完整 Transformer 架构；1. 回顾与过渡：承接上一章，明确本章学习目标；2. Narrow Attention 深度解析：投影切分（chunking）原理——为什么大模型都用它；3....
- Batch 是一批样本；Epoch 是全训练集完整训练一遍。
- 图像常先把像素值缩放到 0-1。
- Dropout：训练随机失活，推理正常使用。
- 1. Narrow Attention：；核心原则：先投影，再切片（chunk the projections, not the inputs）；每个投影值是所有原始特征的线性组合，确保每个头都能访问完整信息；工业标准：GPT、BERT、T...

## 第 2 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第1节 回顾与过渡

### 1. 本节先看什么

这一节先把 **Softmax、Dropout、RNN、Seq2Seq** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Softmax | 把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 | 常考多分类输出、注意力权重为什么能加权求和。 | Softmax 是对一组数整体归一化，不是逐个独立压缩。 | Softmax 输出非负且总和为 1。 |
| Dropout | 训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 | 常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 | Dropout 不是提高模型容量，而是正则化。 | Dropout：训练随机失活，推理正常使用。 |
| RNN | 按时间步处理序列，用隐藏状态把前面信息传到后面。 | 常考隐藏状态、序列建模、梯度消失/爆炸。 | RNN 不是一次性把所有时间步完全独立处理。 | 当前输出依赖当前输入和上一时刻隐藏状态。 |
| Seq2Seq | 编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。 | 常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。 | 普通 Seq2Seq 容易受固定长度上下文瓶颈影响。 | Seq2Seq = Encoder + Decoder。 |
| Attention | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下文向量 → Decoder | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下... |
| Q/K/V | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下文向量 → Decoder | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下... |
| 上下文向量 | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下文向量 → Decoder | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| Positional Encoding | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下文向量 → Decoder | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 / 核心内容 / 关键公式/概念；Seq2Seq / 序列到序列的映射框架 / Encoder → 上下... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Softmax**：把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 考试写作时要补一句：常考多分类输出、注意力权重为什么能加权求和。 易错点是：Softmax 是对一组数整体归一化，不是逐个独立压缩。
- **Dropout**：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 考试写作时要补一句：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 易错点是：Dropout 不是提高模型容量，而是正则化。
- **RNN**：按时间步处理序列，用隐藏状态把前面信息传到后面。 考试写作时要补一句：常考隐藏状态、序列建模、梯度消失/爆炸。 易错点是：RNN 不是一次性把所有时间步完全独立处理。
- **Seq2Seq**：编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。 考试写作时要补一句：常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。 易错点是：普通 Seq2Seq 容易受固定长度上下文瓶颈影响。
- **Attention**：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Q/K/V**：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **上下文向量**：第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下文向量 → Decoder 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。

### 4. 本节自测

- 判断：Softmax只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Dropout只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：RNN只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Seq2Seq只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Q/K/V只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：上下文向量只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Softmax 输出非负且总和为 1。
- Dropout：训练随机失活，推理正常使用。
- 当前输出依赖当前输入和上一时刻隐藏状态。
- Seq2Seq = Encoder + Decoder。
- 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
- 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
- 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
- 自注意力：同一序列内部做注意力。
- 第1节 回顾与过渡；一、上一章我们学到了什么？；在上一章中，我们沿着 Transformer 的发展脉络，学习了以下核心内容：；知识点 | 核心内容 | 关键公式/概念；Seq2Seq | 序列到序列的映射框架 | Encoder → 上下...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 3 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第2节 Narrow Attention 深度解析

### 1. 本节先看什么

这一节先把 **Batch、Softmax、Attention、Multi-Head Attention** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Batch | 一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 | 常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 | Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。 | Batch 是一批样本；Epoch 是全训练集完整训练一遍。 |
| Softmax | 把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 | 常考多分类输出、注意力权重为什么能加权求和。 | Softmax 是对一组数整体归一化，不是逐个独立压缩。 | Softmax 输出非负且总和为 1。 |
| Attention | 第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 / 每个头的输入 / 优点 / 缺点；Wide Attention / 完整 $d_{model}$ /... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 / 每个头的输入 / 优点 / 缺点；Wid... |
| Multi-Head Attention | 第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 / 每个头的输入 / 优点 / 缺点；Wide Attention / 完整 $d_{model}$ /... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 / 每个头的输入 / 优点 / 缺点；Wid... |

### 3. 像考试答案一样组织语言

- **Batch**：一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 考试写作时要补一句：常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 易错点是：Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。
- **Softmax**：把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 考试写作时要补一句：常考多分类输出、注意力权重为什么能加权求和。 易错点是：Softmax 是对一组数整体归一化，不是逐个独立压缩。
- **Attention**：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wide Attention | 完整 $d_{model}$ |... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Multi-Head Attention**：第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wide Attention | 完整 $d_{model}$ |... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。

### 4. 本节自测

- 判断：Batch只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Softmax只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Multi-Head Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Batch 是一批样本；Epoch 是全训练集完整训练一遍。
- Softmax 输出非负且总和为 1。
- 第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wid...
- 第2节 Narrow Attention 深度解析；一、从 Wide 到 Narrow：为什么需要改变？；在上一章中，我们介绍了 Multi-Head Attention 的两种实现方式：；方式 | 每个头的输入 | 优点 | 缺点；Wid...

## 第 4 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第3节 Transformer Encoder（多层堆叠）

### 1. 本节先看什么

这一节先把 **Softmax、归一化、Dropout、Attention** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Softmax | 把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 | 常考多分类输出、注意力权重为什么能加权求和。 | Softmax 是对一组数整体归一化，不是逐个独立压缩。 | Softmax 输出非负且总和为 1。 |
| 归一化 | 把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 | 常考图像预处理为什么要缩放像素值。 | 归一化不等于 BatchNorm，前者多是输入预处理。 | 图像常先把像素值缩放到 0-1。 |
| Dropout | 训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 | 常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 | Dropout 不是提高模型容量，而是正则化。 | Dropout：训练随机失活，推理正常使用。 |
| Attention | 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关... |
| Multi-Head Attention | 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| 位置编码 | 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Softmax**：把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。 考试写作时要补一句：常考多分类输出、注意力权重为什么能加权求和。 易错点是：Softmax 是对一组数整体归一化，不是逐个独立压缩。
- **归一化**：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 考试写作时要补一句：常考图像预处理为什么要缩放像素值。 易错点是：归一化不等于 BatchNorm，前者多是输入预处理。
- **Dropout**：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 考试写作时要补一句：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 易错点是：Dropout 不是提高模型容量，而是正则化。
- **Attention**：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Multi-Head Attention**：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **位置编码**：第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关注其他所有位置；2. Feed-Forward Networ... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：Softmax只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：归一化只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Dropout只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Multi-Head Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：位置编码只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Softmax 输出非负且总和为 1。
- 图像常先把像素值缩放到 0-1。
- Dropout：训练随机失活，推理正常使用。
- 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
- 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
- 自注意力：同一序列内部做注意力。
- 第3节 Transformer Encoder（多层堆叠）；一、从单层到多层：为什么要堆叠？；一个 Encoder Layer 包含两个 Sub-Layers：；1. Multi-Head Self-Attention：让序列中的每个位置关...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 5 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第4节 Transformer Decoder

### 1. 本节先看什么

这一节先把 **Attention、Query、Key、Value** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Attention | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decoder；Sub-Layer 1 / Multi-Head Se... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decod... |
| Query | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decoder；Sub-Layer 1 / Multi-Head Se... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decod... |
| Key | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decoder；Sub-Layer 1 / Multi-Head Se... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decod... |
| Value | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decoder；Sub-Layer 1 / Multi-Head Se... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer / Encoder / Decod... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| Cross-Attention | Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。 | 常考 Transformer 解码器中 Cross-Attention 的输入来源。 | 不要和 Self-Attention 混成同一来源。 | 交叉注意力：Q 与 K/V 来源不同。 |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Attention**：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Query**：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Key**：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Value**：第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decoder；Sub-Layer 1 | Multi-Head Se... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **Cross-Attention**：Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。 考试写作时要补一句：常考 Transformer 解码器中 Cross-Attention 的输入来源。 易错点是：不要和 Self-Attention 混成同一来源。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Query只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Key只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Value只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Cross-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
- 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
- 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
- 第4节 Transformer Decoder；一、Decoder 比 Encoder 多什么？；Decoder 的任务是生成目标序列，因此它比 Encoder 多了一层 Sub-Layer：；Layer | Encoder | Decod...
- 自注意力：同一序列内部做注意力。
- 交叉注意力：Q 与 K/V 来源不同。
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 6 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第5节 Layer Normalization 深度讲解

### 1. 本节先看什么

这一节先把 **Batch、归一化、BatchNorm、Transformer** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Batch | 一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 | 常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 | Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。 | Batch 是一批样本；Epoch 是全训练集完整训练一遍。 |
| 归一化 | 把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 | 常考图像预处理为什么要缩放像素值。 | 归一化不等于 BatchNorm，前者多是输入预处理。 | 图像常先把像素值缩放到 0-1。 |
| BatchNorm | 在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。 | 常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。 | 推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。 | BatchNorm：标准化激活，稳定训练，加快收敛。 |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Batch**：一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 考试写作时要补一句：常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 易错点是：Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。
- **归一化**：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 考试写作时要补一句：常考图像预处理为什么要缩放像素值。 易错点是：归一化不等于 BatchNorm，前者多是输入预处理。
- **BatchNorm**：在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。 考试写作时要补一句：常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。 易错点是：推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：Batch只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：归一化只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：BatchNorm只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Batch 是一批样本；Epoch 是全训练集完整训练一遍。
- 图像常先把像素值缩放到 0-1。
- BatchNorm：标准化激活，稳定训练，加快收敛。
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 7 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）

### 1. 本节先看什么

这一节先把 **Attention、Self-Attention、Transformer** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Attention | 第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）；一、任务介绍：IMDB 电影评论情感分析；在上一章（循环神经网络）中，我们使用 LSTM 对 IMDB 电影评论进行情感分类（正面/负面）。本章，我们将用 Transformer 解决同一个任务，对比两者的效果差异... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）；一、任务介绍：IMDB 电影评论情感分析；在上一章（循环神经网络）中，我们使用 LSTM 对 IMDB 电影评论进行情感分类（正面/负面）。本章，我们将用 T... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Attention**：第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）；一、任务介绍：IMDB 电影评论情感分析；在上一章（循环神经网络）中，我们使用 LSTM 对 IMDB 电影评论进行情感分类（正面/负面）。本章，我们将用 Transformer 解决同一个任务，对比两者的效果差异... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- 第6节 完整 Transformer 架构 + 端到端训练（IMDB 情感分类）；一、任务介绍：IMDB 电影评论情感分析；在上一章（循环神经网络）中，我们使用 LSTM 对 IMDB 电影评论进行情感分类（正面/负面）。本章，我们将用 T...
- 自注意力：同一序列内部做注意力。
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 8 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第7节 PyTorch 内置 Transformer

### 1. 本节先看什么

这一节先把 **Batch、激活函数、归一化、Dropout** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Batch | 一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 | 常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 | Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。 | Batch 是一批样本；Epoch 是全训练集完整训练一遍。 |
| 激活函数 | 给线性变换加入非线性，使网络能拟合非线性关系。 | 常考没有激活函数时多层线性网络仍等价于线性模型。 | 不要把激活函数说成只改变维度，它主要提供非线性。 | 仿射变换 + 激活函数 = 神经网络层的基本模式。 |
| 归一化 | 把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 | 常考图像预处理为什么要缩放像素值。 | 归一化不等于 BatchNorm，前者多是输入预处理。 | 图像常先把像素值缩放到 0-1。 |
| Dropout | 训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 | 常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 | Dropout 不是提高模型容量，而是正则化。 | Dropout：训练随机失活，推理正常使用。 |
| Key | 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T... |
| Subsequent Mask | 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T... |
| 位置编码 | 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Batch**：一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。 考试写作时要补一句：常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。 易错点是：Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。
- **激活函数**：给线性变换加入非线性，使网络能拟合非线性关系。 考试写作时要补一句：常考没有激活函数时多层线性网络仍等价于线性模型。 易错点是：不要把激活函数说成只改变维度，它主要提供非线性。
- **归一化**：把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。 考试写作时要补一句：常考图像预处理为什么要缩放像素值。 易错点是：归一化不等于 BatchNorm，前者多是输入预处理。
- **Dropout**：训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。 考试写作时要补一句：常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。 易错点是：Dropout 不是提高模型容量，而是正则化。
- **Key**：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Subsequent Mask**：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **位置编码**：第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.Transformer。了解它的接口和特点非常重要。；PyTo... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：Batch只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：激活函数只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：归一化只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Dropout只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Key只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Subsequent Mask只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：位置编码只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Batch 是一批样本；Epoch 是全训练集完整训练一遍。
- 仿射变换 + 激活函数 = 神经网络层的基本模式。
- 图像常先把像素值缩放到 0-1。
- Dropout：训练随机失活，推理正常使用。
- 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
- 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
- 第7节 PyTorch 内置 Transformer；一、为什么需要了解 PyTorch 内置实现？；到目前为止，我们用自定义代码实现了 Transformer 的各个组件。但在实际工作中，你更可能直接使用 PyTorch 提供的 nn.T...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 9 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉

### 1. 本节先看什么

这一节先把 **Attention、Self-Attention、位置编码、Transformer** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Attention | 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti... |
| Self-Attention | Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 | 常考它和 Cross-Attention 的区别。 | Self 不是只看自己一个位置，而是同一序列内部互相看。 | 自注意力：同一序列内部做注意力。 |
| 位置编码 | 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了... | 会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 | 不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。 | 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti... |
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |

### 3. 像考试答案一样组织语言

- **Attention**：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Self-Attention**：Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。 考试写作时要补一句：常考它和 Cross-Attention 的区别。 易错点是：Self 不是只看自己一个位置，而是同一序列内部互相看。
- **位置编码**：第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attention——并不局限于文本。2020 年，Google 提出了... 考试写作时要补一句：会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。 易错点是：不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。
- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。

### 4. 本节自测

- 判断：Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Self-Attention只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：位置编码只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti...
- 自注意力：同一序列内部做注意力。
- 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉；一、从 NLP 到 CV：Transformer 的跨界；Transformer 最初为 NLP 设计，但其核心思想——Self-Attenti...
- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。

## 第 10 部分：第六章（二）Self-Attention 与完整 Transformer 架构 / 第10节 预训练模型：DistilBERT 微调实战

### 1. 本节先看什么

这一节先把 **Transformer、学习率、Epoch** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？

### 2. 核心知识点表

| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |
|---|---|---|---|---|
| Transformer | 以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 | 常考编码器/解码器结构、多头注意力和位置编码。 | Transformer 不依赖 RNN 的逐步递归来建模序列。 | Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。 |
| 学习率 | 控制每次参数沿梯度方向更新的步子大小。学习率太大容易震荡甚至发散，太小则收敛很慢。 | 常考学习率过大/过小的训练现象，以及学习率调度的目的。 | 学习率不是越大越好；也不是模型学到的参数。 | 学习率决定更新步长。 |
| Epoch | 训练集被模型完整看过一遍，叫一个 epoch。一个 epoch 内通常包含很多个 batch 更新。 | 常考 Epoch、Batch、Iteration 的区别，或判断训练轮数增加对欠拟合/过拟合的影响。 | Epoch 不是一次参数更新；一次更新通常对应一个 batch。 | 1 个 Epoch = 全部训练样本被用过一遍。 |

### 3. 像考试答案一样组织语言

- **Transformer**：以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。 考试写作时要补一句：常考编码器/解码器结构、多头注意力和位置编码。 易错点是：Transformer 不依赖 RNN 的逐步递归来建模序列。
- **学习率**：控制每次参数沿梯度方向更新的步子大小。学习率太大容易震荡甚至发散，太小则收敛很慢。 考试写作时要补一句：常考学习率过大/过小的训练现象，以及学习率调度的目的。 易错点是：学习率不是越大越好；也不是模型学到的参数。
- **Epoch**：训练集被模型完整看过一遍，叫一个 epoch。一个 epoch 内通常包含很多个 batch 更新。 考试写作时要补一句：常考 Epoch、Batch、Iteration 的区别，或判断训练轮数增加对欠拟合/过拟合的影响。 易错点是：Epoch 不是一次参数更新；一次更新通常对应一个 batch。

### 4. 本节自测

- 判断：Transformer只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：学习率只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）
- 判断：Epoch只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）

### 5. 本节速记

- Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。
- 学习率决定更新步长。
- 1 个 Epoch = 全部训练样本被用过一遍。

## 四、考前总复盘

考前不要平均用力。优先检查下面这些问题：

1. 每个核心概念能不能用一句话说明“它是什么”。
2. 能不能说出它在模型结构、训练流程或数据处理中的位置。
3. 能不能说出一个最常见的错误说法。
4. 遇到选择题时，能不能判断选项是在混淆概念、夸大作用，还是写反了训练/推理阶段。

## 五、打印建议

<style>@media print { @page { margin: 8mm; } body { font-size: 11pt; line-height: 1.35; } h1, h2, h3 { page-break-after: avoid; } table { font-size: 9pt; border-collapse: collapse; } th, td { padding: 3px 5px; border: 1px solid #ddd; vertical-align: top; } }</style>
