# 注意力机制及Transformer2-学生分发版-终版2026 / 第六章（二）Self-Attention 与完整 Transformer 架构 / 第8节 Vision Transformer（ViT）：Transformer 进军计算机视觉
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
