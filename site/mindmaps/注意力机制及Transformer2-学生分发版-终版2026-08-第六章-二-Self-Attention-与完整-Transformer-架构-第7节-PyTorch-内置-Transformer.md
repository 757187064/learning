# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构 / 第7节 PyTorch 内置 Transformer
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
