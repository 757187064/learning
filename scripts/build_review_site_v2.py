#!/usr/bin/env python3
import hashlib
import html
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "深度学习复习网站"
DATABASE = OUT / "database"
SITE = OUT / "site"
ASSETS = SITE / "assets"
IMAGE_OUT = SITE / "image"
MINDMAP_OUT = SITE / "mindmaps"
GUIDE_OUT = SITE / "guides"
CACHE_VERSION = 6


KEY_TERMS = [
    "梯度下降", "损失函数", "学习率", "Epoch", "Batch", "反向传播", "计算图", "自动微分",
    "Logistic", "Sigmoid", "Softmax", "交叉熵", "BCEWithLogitsLoss", "CrossEntropyLoss",
    "MLP", "多层感知机", "激活函数", "仿射变换", "卷积", "互相关", "卷积核", "步长", "填充",
    "Valid", "Same", "Full", "感受野", "池化", "NCHW", "NHWC", "归一化", "标准化",
    "LeNet", "AlexNet", "ImageFolder", "WeightedRandomSampler", "类别不平衡",
    "梯度消失", "梯度爆炸", "梯度裁剪", "Xavier", "Kaiming", "BatchNorm", "Dropout",
    "数据增强", "权重衰减", "早停", "优化器", "Momentum", "Nesterov", "Adagrad", "RMSProp",
    "Adam", "学习率调度器", "ReduceLROnPlateau", "超参数", "消融实验",
    "VGG", "GoogLeNet", "Inception", "ResNet", "残差连接", "迁移学习",
    "RNN", "隐藏状态", "双向RNN", "堆叠RNN", "GRU", "LSTM", "Padding", "Packing",
    "PackedSequence", "1D卷积", "TCN", "Seq2Seq", "Encoder-Decoder", "Teacher Forcing",
    "注意力机制", "Attention", "Query", "Key", "Value", "Q/K/V", "上下文向量",
    "Scaled Dot-Product Attention", "sqrt(d_k)", "Source Mask", "Multi-Head Attention",
    "Self-Attention", "Cross-Attention", "Target Mask", "Subsequent Mask",
    "位置编码", "Positional Encoding", "Transformer", "RoPE",
    "图学习", "图神经网络", "节点", "边", "邻接矩阵", "度矩阵", "拉普拉斯矩阵",
    "谱图理论", "图卷积", "GCN", "GAT", "GraphSAGE", "消息传递", "聚合",
    "同质图", "异质图", "节点分类", "图分类", "链路预测", "随机游走",
    "PageRank", "嵌入", "网络表示学习", "Vertex", "Edge",
]

BAD_QUESTION_PATTERNS = [
    "核心词是",
    "这句话描述的核心词",
    "必须抓住的两个要点",
    "围绕“",
    "围绕\"",
    "若考试围绕",
    "若题目围绕",
    "若训练代码围绕",
    "若围绕",
    "关系最直接的知识点",
    "最适合作为考试标准答案",
]

NOISE_PATTERNS = [
    "import ", "warnings.", "plt.", "torch.", "np.", "rcParams", "dark_background",
    "本章内容", "通过本练习，你将", "练习目标", "|:----", "答案解析", "回答正确",
    "分值1分", "class ", "def ", "return ", "from ",
    "北京航空航天", "北京航", "UNIVERSITY", "233740", "2026年05月26日",
]

TERM_GUIDE = {
    "卷积": ("用小窗口在图像局部滑动，对局部像素和卷积核权重相乘求和，提取边缘、纹理等局部特征。", "常考输出尺寸、卷积核/步长/填充的作用，以及 CNN 实际常用互相关。", "不要把 CNN 说成全连接；卷积核心是局部连接和参数共享。", "输出尺寸：(输入尺寸 + 2P - K) / S + 1。"),
    "互相关": ("形式接近卷积，但不翻转卷积核；深度学习框架里的卷积层通常实际做互相关。", "常考 CNN 中是否需要翻转卷积核，或为什么工程上用互相关。", "不要因为名字叫卷积层，就默认一定执行数学卷积的核翻转。", "CNN 训练会学习卷积核参数，因此互相关也能学到有效特征。"),
    "Valid": ("不填充边界，只在卷积核完全落入输入时计算，输出尺寸会变小。", "常和 Same、Full 对比考输出大小。", "Valid 不是大量填充，恰好是不填充。", "Valid：无填充，输出通常最小。"),
    "Same": ("通过适当填充，让卷积输出空间尺寸尽量和输入保持一致。", "常考给定输入、卷积核、步长时需要多少 padding。", "Same 的目标是保持尺寸，不是扩大特征图。", "步长为 1 时，Same 卷积常让 H、W 不变。"),
    "Full": ("在边界外进行更充分的填充，使卷积核可以覆盖到输入边缘外侧，输出会变大。", "常和 Valid、Same 放在一起判断哪种输出最大。", "Full 不是常规 CNN 默认选择，考试多用于概念区分。", "Full：充分填充，输出通常最大。"),
    "感受野": ("某一层一个神经元能看到原始输入图像的区域大小。", "常考层数、卷积核大小、步长如何影响感受野。", "步长变大通常会让前一层对应感受野扩大，不是缩小。", "层越深，感受野通常越大；多层小卷积能逐步扩大感受野。"),
    "池化": ("对局部区域做最大值或平均值汇总，降低空间尺寸并增强一定平移不变性。", "常考池化作用、是否有可学习参数、与卷积的区别。", "池化不是全连接，也不是用来增加参数量。", "池化：降采样、压缩空间尺寸、保留显著信息。"),
    "卷积核": ("卷积层中可学习的小矩阵/滤波器，用来扫描局部区域并提取特征。", "常考卷积核大小、通道数、参数共享和输出尺寸计算。", "卷积核参数会训练更新，不是手工固定模板。", "卷积核大小 K、步长 S、填充 P 共同决定输出尺寸。"),
    "步长": ("卷积核或池化窗口每次移动的间隔。", "常考步长变大时输出尺寸如何变化。", "步长越大，输出空间尺寸通常越小。", "Stride 控制滑动间隔。"),
    "填充": ("在输入边界补值，让卷积能处理边缘或保持输出尺寸。", "常考 Same 卷积为何需要 padding。", "填充不是增加有效信息，而是控制边界和尺寸。", "Padding 用于控制输出尺寸和边界信息。"),
    "标准化": ("把数据按均值和方差调整到更稳定的尺度，帮助模型训练。", "常考归一化和标准化的区别、为什么图像输入要预处理。", "不要把标准化简单等同于除以 255；标准化通常涉及均值和标准差。", "标准化常写作 (x-mean)/std。"),
    "归一化": ("把数值缩放到较统一范围，例如图像像素从 0-255 缩放到 0-1。", "常考图像预处理为什么要缩放像素值。", "归一化不等于 BatchNorm，前者多是输入预处理。", "图像常先把像素值缩放到 0-1。"),
    "激活函数": ("给线性变换加入非线性，使网络能拟合非线性关系。", "常考没有激活函数时多层线性网络仍等价于线性模型。", "不要把激活函数说成只改变维度，它主要提供非线性。", "仿射变换 + 激活函数 = 神经网络层的基本模式。"),
    "BatchNorm": ("在小批量上标准化中间激活，再用可学习参数恢复表达能力，帮助训练更稳定。", "常考训练/推理阶段统计量不同，以及放在卷积/全连接和激活附近。", "推理阶段一般使用训练中累计的均值方差，不依赖当前 batch。", "BatchNorm：标准化激活，稳定训练，加快收敛。"),
    "Dropout": ("训练时随机丢弃一部分神经元，迫使网络不要过度依赖某些特征。", "常考训练阶段启用、推理阶段关闭，以及它缓解过拟合。", "Dropout 不是提高模型容量，而是正则化。", "Dropout：训练随机失活，推理正常使用。"),
    "Batch": ("一次送入模型并一起计算损失的一组样本。它决定每次参数更新看到多少样本，也影响显存占用和训练稳定性。", "常考 batch size、mini-batch、epoch、iteration 的区别，也会结合张量形状中的 N 维度判断。", "Batch 不是序列长度；在 RNN 中常见形状里 N 是样本数，L 才是时间步长度。", "Batch 是一批样本；Epoch 是全训练集完整训练一遍。"),
    "Epoch": ("训练集被模型完整看过一遍，叫一个 epoch。一个 epoch 内通常包含很多个 batch 更新。", "常考 Epoch、Batch、Iteration 的区别，或判断训练轮数增加对欠拟合/过拟合的影响。", "Epoch 不是一次参数更新；一次更新通常对应一个 batch。", "1 个 Epoch = 全部训练样本被用过一遍。"),
    "学习率": ("控制每次参数沿梯度方向更新的步子大小。学习率太大容易震荡甚至发散，太小则收敛很慢。", "常考学习率过大/过小的训练现象，以及学习率调度的目的。", "学习率不是越大越好；也不是模型学到的参数。", "学习率决定更新步长。"),
    "损失函数": ("把模型预测和真实标签之间的差距变成一个可优化的数值，反向传播就是从损失开始计算梯度。", "常考输出层与损失函数的匹配，例如多分类常配 CrossEntropyLoss，二分类可配 BCEWithLogitsLoss。", "不要把准确率当作训练损失；也不要把 BCE 和 CE 的适用场景混用。", "损失函数告诉模型“错在哪里”，优化器负责“怎么改参数”。"),
    "Sigmoid": ("把任意实数压到 0 到 1 之间，可解释为概率或门控强度。GRU/LSTM 的门通常用它决定保留多少信息。", "常考输出范围、二分类概率、以及门控结构中为什么适合控制开关。", "Sigmoid 不只属于分类输出，在 RNN 门控里也很重要。", "Sigmoid 输出在 0 到 1；越接近 1 越保留，越接近 0 越抑制。"),
    "Softmax": ("把一组分数转成和为 1 的概率分布，常用于多分类或注意力权重。", "常考多分类输出、注意力权重为什么能加权求和。", "Softmax 是对一组数整体归一化，不是逐个独立压缩。", "Softmax 输出非负且总和为 1。"),
    "梯度消失": ("反向传播经过很多层或很多时间步时，梯度可能不断变小，导致前面层或早期时间步很难学到东西。", "常考深层网络/RNN 为什么训练困难，以及 LSTM、GRU、残差连接如何缓解。", "梯度消失不是过拟合；它是参数难以有效更新的问题。", "多个小于 1 的因子连乘，会让梯度趋近 0。"),
    "梯度爆炸": ("梯度在反向传播中变得过大，使参数更新剧烈，训练损失可能震荡或变成 NaN。", "常考梯度裁剪为什么能缓解 RNN 训练不稳定。", "Dropout 主要缓解过拟合，不是解决梯度爆炸的标准答案。", "梯度裁剪限制梯度范数，防止更新步子过大。"),
    "梯度裁剪": ("当梯度范数超过阈值时按比例缩小，常用于稳定 RNN 等容易梯度爆炸的训练。", "常考它处理的是梯度爆炸，而不是梯度消失。", "裁剪不是把梯度直接清零，而是限制其大小。", "Gradient Clipping：限制梯度大小，稳定训练。"),
    "Adam": ("结合动量和自适应学习率思想，为不同参数调整更新幅度。", "常和 SGD、Momentum、RMSProp 对比优化器特点。", "Adam 方便但不等于一定泛化最好，仍需调学习率等超参数。", "Adam = 一阶动量 + 二阶矩估计。"),
    "RMSProp": ("用梯度平方的指数滑动平均调整学习率，缓解 Adagrad 学习率过快衰减。", "常考它和 Adagrad 的区别。", "不要写成简单固定学习率下降。", "RMSProp 使用指数滑动平均保存近期梯度尺度。"),
    "VGG": ("大量使用 3x3 小卷积核堆叠构建深层网络。", "常考小卷积核堆叠为什么能扩大感受野并减少参数。", "不要把 VGG 的特点写成复杂多分支结构。", "VGG：多层小卷积核堆叠。"),
    "ResNet": ("用残差/跳跃连接让网络学习 F(x)+x，缓解深层网络退化和梯度传播困难。", "常考残差连接为什么能训练更深网络。", "残差连接不是单纯增加层数，而是改变信息和梯度路径。", "ResNet 核心：y = F(x) + x。"),
    "RNN": ("按时间步处理序列，用隐藏状态把前面信息传到后面。", "常考隐藏状态、序列建模、梯度消失/爆炸。", "RNN 不是一次性把所有时间步完全独立处理。", "当前输出依赖当前输入和上一时刻隐藏状态。"),
    "GRU": ("用更新门和重置门控制历史信息保留与遗忘，是简化版门控循环网络。", "常和 LSTM 对比门结构和参数量。", "GRU 没有单独的细胞状态 c_t。", "GRU：更新门 + 重置门。"),
    "LSTM": ("通过输入门、遗忘门、输出门和细胞状态保存长期信息。", "常考门控作用，以及为什么能缓解长序列梯度问题。", "不要漏掉细胞状态是 LSTM 的关键通道。", "LSTM：输入门、遗忘门、输出门、细胞状态。"),
    "Padding": ("把不同长度的序列补到同一长度，方便组成 batch 输入模型。", "常考为什么需要 padding，以及它和 packing 的关系。", "Padding 补出来的位置不是真实信息，训练时要避免把它当有效时间步。", "Padding：补齐长度；Packing：跳过无效补位。"),
    "Packing": ("把变长序列的有效部分打包，让 RNN 少处理 padding 位置。", "常考 pack_padded_sequence 的目的和使用前提。", "Packing 不是改变序列含义，而是提高变长序列处理效率。", "Packing 让 RNN 聚焦真实时间步。"),
    "PackedSequence": ("PyTorch 中表示变长序列打包后的对象，内部只保留有效时间步，交给 RNN 处理时可以跳过 padding。", "常考 pack_padded_sequence 与 pad_packed_sequence 的前后关系。", "PackedSequence 不是普通张量；使用前后要注意 lengths、batch_first 和是否排序。", "先 padding 组成 batch，再 packing 跳过补位，必要时再 pad 回来。"),
    "1D卷积": ("在序列的一维时间/位置方向上滑动卷积核，提取局部 n-gram 或短期模式。", "常和 RNN 对比：1D 卷积并行提取局部模式，RNN 逐步传递隐藏状态。", "1D 卷积适合局部依赖，不等于天然拥有长期记忆。", "1D 卷积：沿序列维度提取局部模式。"),
    "隐藏状态": ("RNN 在每个时间步保存的当前记忆，用来把历史信息传给后续时间步。", "常考 h_t 与 h_{t-1} 的关系，以及 output 和 h_n 的区别。", "隐藏状态不是模型参数，而是随输入序列变化的中间表示。", "h_t = 当前输入 + 过去记忆的综合表示。"),
    "双向RNN": ("同时从前往后和从后往前读序列，让当前位置利用左右两侧上下文。", "常考它为什么比单向 RNN 能看到更多上下文。", "双向 RNN 不适合严格只能用过去信息的实时预测场景。", "双向 RNN = 正向隐藏状态 + 反向隐藏状态。"),
    "堆叠RNN": ("把多层 RNN 叠起来，上一层的输出序列作为下一层输入。", "常考堆叠带来更强表达能力，也增加训练难度。", "堆叠不是时间步变多，而是层数变深。", "堆叠 RNN：时间方向 + 层方向都要传播。"),
    "注意力机制": ("根据 Query 与 Key 的匹配程度，给 Value 分配权重并加权求和。", "常考 Q/K/V 含义、注意力权重、上下文向量计算流程。", "注意力不是简单平均，而是按相关性加权。", "Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。"),
    "Self-Attention": ("Q、K、V 来自同一个序列，让序列内部不同位置相互建立联系。", "常考它和 Cross-Attention 的区别。", "Self 不是只看自己一个位置，而是同一序列内部互相看。", "自注意力：同一序列内部做注意力。"),
    "Cross-Attention": ("Query 来自一个序列，Key/Value 来自另一个序列，常用于解码器关注编码器输出。", "常考 Transformer 解码器中 Cross-Attention 的输入来源。", "不要和 Self-Attention 混成同一来源。", "交叉注意力：Q 与 K/V 来源不同。"),
    "Transformer": ("以注意力为核心的序列模型，用多头注意力、前馈网络、残差和归一化堆叠编码器/解码器。", "常考编码器/解码器结构、多头注意力和位置编码。", "Transformer 不依赖 RNN 的逐步递归来建模序列。", "Transformer 核心：Attention + FFN + 残差 + Norm + 位置编码。"),
    "sqrt(d_k)": ("缩放点积注意力中的除数，用来避免 QK 点积过大导致 softmax 过于尖锐。", "常考为什么要缩放，以及 d_k 是 Key/Query 的维度。", "不要把它当成可学习参数。", "QK^T 除以 sqrt(d_k) 是为了稳定 softmax。"),
    "RoPE": ("一种旋转位置编码，把位置信息注入向量表示，常用于现代 Transformer。", "如果课件讲到，主要考它属于位置编码思想，不会要求复杂推导。", "不要把 RoPE 写成普通绝对位置编号相加。", "RoPE 用旋转方式表达相对位置信息。"),
    "Seq2Seq": ("编码器把输入序列压成表示，解码器再生成输出序列，常用于翻译、问答等序列到序列任务。", "常考 Encoder-Decoder 结构、Teacher Forcing 和注意力机制为何被引入。", "普通 Seq2Seq 容易受固定长度上下文瓶颈影响。", "Seq2Seq = Encoder + Decoder。"),
    "图学习": ("研究如何在图结构数据上学习表示和完成预测，核心是同时利用节点特征与关系结构。", "常考图数据适用场景、节点/边/图级任务。", "不要只看单个样本特征，图学习关键是关系。", "图学习 = 特征信息 + 结构信息。"),
    "节点": ("图中的对象或实体，可以带有节点特征。", "常考节点分类任务：预测每个节点的类别。", "节点特征和边关系是两类信息，不要混在一起。", "节点表示对象，边表示对象之间的关系。"),
    "边": ("连接两个节点的关系，可以是有向/无向、带权/不带权。", "常考边方向、权重和邻接矩阵的对应关系。", "有向图的邻接矩阵通常不对称。", "边表示关系，权重表示关系强弱。"),
    "Vertex": ("图中的节点，也就是图结构里的对象或实体。", "常考 Vertex/Node 与 Edge 的区别。", "Vertex 是节点，不是边。", "Vertex/Node 表示对象；Edge 表示关系。"),
    "Edge": ("图中的边，表示两个节点之间的关系，可以有方向和权重。", "常考 Edge 与邻接矩阵元素的对应关系。", "Edge 不是节点特征本身，而是节点之间的连接。", "Edge 表示关系，A_ij 常记录边或边权。"),
    "邻接矩阵": ("用矩阵记录节点之间是否相连或连接权重。", "常考根据图画矩阵，或根据矩阵判断邻居。", "无向图邻接矩阵通常对称；有向图不一定对称。", "A_ij 表示 i 到 j 是否有边或边权。"),
    "度矩阵": ("对角矩阵，对角线记录每个节点的度。", "常和拉普拉斯矩阵 L=D-A 一起考。", "度矩阵不是邻接矩阵，它只放在对角线上。", "D_ii = 节点 i 的度。"),
    "拉普拉斯矩阵": ("用度矩阵和邻接矩阵构造，反映图的连接结构。", "常考 L=D-A 以及它在谱图方法中的作用。", "不要把 L 和 A 混用。", "图拉普拉斯：L = D - A。"),
    "GCN": ("图卷积网络，通过邻接关系聚合邻居节点特征，再更新节点表示。", "常考 GCN 为什么能利用图结构、聚合邻居信息的基本流程。", "GCN 不只看节点自身特征，还看邻居。", "GCN：邻居聚合 + 线性变换 + 非线性。"),
    "GAT": ("图注意力网络，给不同邻居分配不同注意力权重再聚合。", "常和 GCN 对比：GCN 多按固定归一化聚合，GAT 学习邻居重要性。", "GAT 的重点是邻居权重可学习，不是所有邻居同等重要。", "GAT：用注意力学习邻居权重。"),
    "GraphSAGE": ("通过采样邻居并聚合来生成节点表示，适合较大图的归纳学习。", "常考采样聚合思想，以及和一次性全图训练的区别。", "GraphSAGE 不是简单查表保存每个节点向量。", "GraphSAGE：采样邻居，聚合表示。"),
    "消息传递": ("节点从邻居接收信息、聚合信息、更新自身表示，是很多图神经网络的统一视角。", "常考消息、聚合、更新三个步骤。", "不要只写公式，考试要说明信息沿边传播。", "消息传递：邻居发消息，节点聚合并更新。"),
}


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def compact(text: str, max_len: int = 260) -> str:
    text = re.sub(r"\s+", " ", strip_html(text))
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("，。；:： ") + "..."


def clean_for_study(text: str, max_len: int = 180) -> str:
    text = strip_html(text)
    lines = []
    for raw in text.splitlines():
        line = raw.strip(" >#-\t|")
        if not line:
            continue
        if any(p in line for p in NOISE_PATTERNS):
            continue
        if len(line) < 8 and not re.search(r"[A-Za-z]{2,}|\d", line):
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue
        lines.append(line)
    text = "；".join(lines[:5]) if lines else compact(text, max_len)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#+\s*", "", text)
    return compact(text, max_len)


def infer_course_family(title: str) -> str:
    if "注意力" in title or "Transformer" in title:
        return "transformer"
    if "循环神经" in title or "RNN" in title or "LSTM" in title or "GRU" in title or "序列" in title:
        return "rnn"
    if "图学习" in title or "图神经" in title or "图卷积" in title or "节点" in title or "邻接" in title:
        return "graph"
    if "卷积" in title and "图卷积" not in title:
        return "cnn"
    if "多层感知机" in title:
        return "mlp"
    return "general"


def term_allowed(term, family, chapter_title=""):
    title = chapter_title or ""
    common = {
        "梯度下降", "损失函数", "学习率", "Epoch", "Batch", "反向传播", "计算图", "自动微分",
        "Logistic", "Sigmoid", "Softmax", "交叉熵", "BCEWithLogitsLoss", "CrossEntropyLoss",
        "激活函数", "仿射变换", "归一化", "标准化", "Dropout", "BatchNorm", "梯度消失",
        "梯度爆炸", "梯度裁剪", "优化器", "Adam", "RMSProp", "Momentum", "权重衰减",
        "数据增强", "超参数", "消融实验",
    }
    by_family = {
        "cnn": {"卷积", "互相关", "卷积核", "步长", "填充", "Valid", "Same", "Full", "感受野", "池化", "NCHW", "NHWC", "LeNet", "AlexNet", "VGG", "GoogLeNet", "Inception", "ResNet", "残差连接", "迁移学习", "ImageFolder", "WeightedRandomSampler", "类别不平衡"},
        "mlp": {"MLP", "多层感知机", "Xavier", "Kaiming"},
        "rnn": {"RNN", "隐藏状态", "双向RNN", "堆叠RNN", "GRU", "LSTM", "Padding", "Packing", "PackedSequence", "1D卷积", "TCN", "Seq2Seq", "Encoder-Decoder", "Teacher Forcing"},
        "transformer": {"RNN", "Seq2Seq", "Encoder-Decoder", "Teacher Forcing", "注意力机制", "Attention", "Query", "Key", "Value", "Q/K/V", "上下文向量", "Scaled Dot-Product Attention", "sqrt(d_k)", "Source Mask", "Multi-Head Attention", "Self-Attention", "Cross-Attention", "Target Mask", "Subsequent Mask", "位置编码", "Positional Encoding", "Transformer", "RoPE"},
        "graph": {"图学习", "图神经网络", "节点", "边", "邻接矩阵", "度矩阵", "拉普拉斯矩阵", "谱图理论", "图卷积", "GCN", "GAT", "GraphSAGE", "消息传递", "聚合", "同质图", "异质图", "节点分类", "图分类", "链路预测", "随机游走", "PageRank", "嵌入", "网络表示学习", "Vertex", "Edge"},
    }
    if term in common or term in by_family.get(family, set()):
        return True
    if family == "rnn" and term in {"卷积", "卷积核", "感受野"} and "1D卷积" in title:
        return True
    if family == "transformer" and term in {"LSTM", "GRU"} and re.search(r"RNN|序列|Seq2Seq|过渡|瓶颈", title):
        return True
    return family == "general"


def generic_row(term: str, evidence: str, family: str):
    if term in TERM_GUIDE:
        plain, exam, pitfall, memory = TERM_GUIDE[term]
        return plain, exam, pitfall, memory
    base = clean_for_study(evidence, 150)
    if family == "cnn":
        exam = f"会放在 CNN 结构、输出尺寸、参数共享或训练流程里考，重点判断它在卷积网络中起什么作用。"
        pitfall = "不要只背名词；要能说清它和卷积层、池化、损失函数或训练阶段的关系。"
    elif family == "mlp":
        exam = "会结合前向传播、损失函数、梯度下降、激活函数或分类输出层一起考。"
        pitfall = "不要把线性变换、非线性激活、损失函数和优化器的职责混在一起。"
    elif family == "rnn":
        exam = "会结合序列输入、隐藏状态、长距离依赖或门控结构考。"
        pitfall = "不要把序列模型当成普通独立样本分类；时间步之间有信息传递。"
    elif family == "transformer":
        exam = "会结合 Q/K/V、注意力权重、位置编码、mask 或编码器解码器结构考。"
        pitfall = "不要只背公式；要能说明每个张量来自哪里、为什么要 mask 或缩放。"
    elif family == "graph":
        exam = "会结合节点、边、邻接矩阵、图任务类型或邻居聚合流程考。"
        pitfall = "不要只看节点自身特征；图学习的重点是把关系结构一起纳入模型。"
    else:
        exam = "会考定义、作用、使用位置、和相近概念的区别。"
        pitfall = "不要脱离课件上下文孤立背诵。"
    plain = base or f"{term}是本节需要掌握的核心概念，要结合课件中的定义、作用和使用场景理解。"
    memory = TERM_GUIDE.get(term, (None, None, None, None))[3] or plain
    return plain, exam, pitfall, compact(memory, 120)


def is_study_chunk(chunk):
    if chunk.get("type") == "code":
        return False
    text = chunk.get("text", "")
    if any(p in text for p in NOISE_PATTERNS[:8]):
        useful = [t for t in chunk.get("terms", []) if t in TERM_GUIDE]
        return bool(useful) and len(clean_for_study(text, 100)) > 20
    return len(clean_for_study(text, 80)) > 20


def extract_heading_terms(chunk):
    values = list(chunk.get("heading_path") or []) + [chunk.get("title", "")]
    terms = []
    for value in values:
        clean = re.sub(r"第[一二三四五六七八九十\d]+\s*[章节]", "", value)
        for part in re.split(r"[：:、/|\-—\s]+", clean):
            part = part.strip(" #（）()[]【】")
            if 2 <= len(part) <= 18 and not part.isdigit():
                terms.append(part)
    return terms


def is_noise_heading(title):
    clean = re.sub(r"\s+", "", title or "")
    if len(clean) < 2:
        return True
    return any(p.replace(" ", "") in clean for p in [
        "北京航空航天", "北京航宫航天", "UNIVERSITY", "NOUNIVERSITY", "ANGUNIVERGITY",
        "233740", "2026年05月26日", "BITA",
    ])


def heading_level(line: str):
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not match:
        return None
    title = strip_html(match.group(2)).strip(" #")
    if not title:
        return None
    return len(match.group(1)), title


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def terms_in(text: str):
    found = []
    for term in KEY_TERMS:
        if term in {"GAT", "GCN", "GRU", "RNN", "LSTM", "MLP", "TCN"}:
            if re.search(rf"\b{term}\b", text, flags=re.I):
                found.append(term)
            continue
        if term == "Batch":
            if re.search(r"\bBatch\b|批量|小批量", text, flags=re.I):
                found.append(term)
            continue
        if term in {"Same", "Valid", "Full"}:
            if re.search(rf"\b{term}\b", text, flags=re.I) and re.search(r"(卷积|convolution|padding|填充)", text, flags=re.I):
                found.append(term)
            continue
        if term == "RoPE":
            if re.search(r"RoPE|旋转位置|rotary", text, flags=re.I) and re.search(r"(位置|Transformer|attention|编码|rotary)", text, flags=re.I):
                found.append(term)
            continue
        if term == "边":
            if re.search(r"(边\s*[（(]?\s*Edge|Edge\s*[）)]?\s*边|节点.*边|边.*节点)", text, flags=re.I):
                found.append(term)
            continue
        if term == "Edge":
            if re.search(r"\bEdge\b", text, flags=re.I):
                found.append(term)
            continue
        if term == "Vertex":
            if re.search(r"\bVertex\b", text, flags=re.I):
                found.append(term)
            continue
        if re.search(re.escape(term), text, flags=re.I):
            found.append(term)
    return found


def load_cache():
    path = DATABASE / "source_cache.json"
    if not path.exists():
        return {"version": CACHE_VERSION, "files": {}}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": CACHE_VERSION, "files": {}}
    if cache.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "files": {}}
    cache.setdefault("files", {})
    return cache


def extract_notebook(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    manifest = {
        "file": path.name,
        "kind": "ipynb",
        "size_bytes": path.stat().st_size,
        "cell_count": len(cells),
        "markdown_cells": sum(1 for c in cells if c.get("cell_type") == "markdown"),
        "code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
    }
    chunks = []
    path_stack = []
    for i, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell.get("cell_type") == "markdown":
            for raw_line in source.splitlines():
                parsed = heading_level(raw_line)
                if parsed:
                    level, title = parsed
                    path_stack = path_stack[: level - 1]
                    path_stack.append(title)
        clean = strip_html(source)
        if not clean:
            continue
        title = path_stack[-1] if path_stack else path.stem
        chunks.append({
            "id": f"{path.stem}::cell-{i + 1}",
            "file": path.name,
            "kind": "ipynb",
            "cell_index": i + 1,
            "slide_index": None,
            "type": cell.get("cell_type", "unknown"),
            "title": title,
            "heading_path": path_stack[-4:],
            "text": clean,
            "summary": compact(clean, 680),
            "terms": terms_in(clean),
        })
    return manifest, chunks


def ensure_ocr_binary():
    swiftc = shutil.which("swiftc")
    swift = OUT / "scripts" / "ocr_image.swift"
    binary = Path("/tmp") / "review_site_ocr_image"
    if not swiftc or not swift.exists():
        return None
    if binary.exists() and binary.stat().st_mtime >= swift.stat().st_mtime:
        return binary
    try:
        subprocess.run([swiftc, str(swift), "-o", str(binary)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return binary
    except Exception:
        return None


def image_target_for_slide(zf, slide_index):
    rel_name = f"ppt/slides/_rels/slide{slide_index}.xml.rels"
    if rel_name not in zf.namelist():
        return None
    xml = zf.read(rel_name).decode("utf-8", errors="ignore")
    matches = re.findall(r'Target="([^"]+\.(?:jpg|jpeg|png))"', xml, flags=re.I)
    if not matches:
        return None
    target = matches[0]
    if target.startswith("../"):
        return "ppt/" + target[3:]
    if target.startswith("/"):
        return target.lstrip("/")
    return "ppt/slides/" + target


def ocr_slide_image(zf, image_name, tmp_path, ocr_binary):
    if not ocr_binary or image_name not in zf.namelist():
        return ""
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(zf.read(image_name))
    try:
        result = subprocess.run([str(ocr_binary), str(tmp_path)], check=False, capture_output=True, text=True, timeout=25)
    except Exception:
        return ""
    return strip_html(result.stdout)


def graph_fallback_text(path_name, slide_index):
    if "第一部分" in path_name:
        return f"图学习课件第一部分，第 {slide_index} 页。复习重点围绕图数据的基本概念、节点、边、邻接矩阵、度矩阵、拉普拉斯矩阵、图任务类型和图学习算法概述。"
    if "第二部分" in path_name:
        return f"图学习课件第二部分，第 {slide_index} 页。复习重点围绕图神经网络、图卷积、GCN、GAT、GraphSAGE、消息传递、邻居聚合和节点表示学习。"
    return f"{path_name} 第 {slide_index} 页，图片型 PPT 页面。"


def extract_pptx(path: Path):
    try:
        from pptx import Presentation
    except Exception:
        manifest = {
            "file": path.name,
            "kind": "pptx",
            "size_bytes": path.stat().st_size,
            "slide_count": 0,
            "warning": "python-pptx unavailable; skipped extraction",
        }
        return manifest, []

    prs = Presentation(str(path))
    manifest = {
        "file": path.name,
        "kind": "pptx",
        "size_bytes": path.stat().st_size,
        "slide_count": len(prs.slides),
    }
    chunks = []
    ocr_binary = ensure_ocr_binary()
    zf = zipfile.ZipFile(path)
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        title = f"第 {i} 页"
        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue
            value = strip_html(shape.text)
            if not value:
                continue
            if title == f"第 {i} 页":
                title = value.splitlines()[0][:80]
            texts.append(value)
        try:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                texts.append("备注：" + strip_html(notes))
        except Exception:
            pass
        if not texts:
            image_name = image_target_for_slide(zf, i)
            if image_name:
                tmp = Path("/tmp") / "review_site_ppt_ocr" / f"{file_hash(path)[:12]}-{i}{Path(image_name).suffix}"
                ocr_text = ocr_slide_image(zf, image_name, tmp, ocr_binary)
                if ocr_text:
                    texts.append(ocr_text)
                    title = ocr_text.splitlines()[0][:80]
        if not texts:
            fallback = graph_fallback_text(path.name, i)
            texts.append(fallback)
            title = fallback.split("。")[0]
        clean = strip_html("\n".join(texts))
        if not clean:
            continue
        chunks.append({
            "id": f"{path.stem}::slide-{i}",
            "file": path.name,
            "kind": "pptx",
            "cell_index": None,
            "slide_index": i,
            "type": "slide",
            "title": title,
            "heading_path": [title],
            "text": clean,
            "summary": compact(clean, 680),
            "terms": terms_in(clean),
        })
    return manifest, chunks


def extract_sources():
    cache = load_cache()
    manifest = []
    chunks = []
    files = sorted(list(ROOT.glob("*.ipynb")) + list(ROOT.glob("*.pptx")))
    new_cache = {"version": CACHE_VERSION, "files": {}}
    for path in files:
        sha = file_hash(path)
        cached = cache.get("files", {}).get(path.name)
        cache_usable = cached and cached.get("sha256") == sha
        if cache_usable and path.suffix.lower() == ".pptx" and not cached.get("chunks"):
            cache_usable = False
        if cache_usable:
            item_manifest = cached["manifest"]
            item_chunks = cached["chunks"]
        else:
            if path.suffix.lower() == ".pptx":
                item_manifest, item_chunks = extract_pptx(path)
            else:
                item_manifest, item_chunks = extract_notebook(path)
            item_manifest["sha256"] = sha
        item_manifest["sha256"] = sha
        for chunk in item_chunks:
            chunk["terms"] = terms_in(chunk.get("text", ""))
            chunk["summary"] = compact(chunk.get("text", ""), 680)
        manifest.append(item_manifest)
        chunks.extend(item_chunks)
        new_cache["files"][path.name] = {
            "sha256": sha,
            "manifest": item_manifest,
            "chunks": item_chunks,
        }
    return manifest, chunks, new_cache


def build_term_index(chunks):
    term_data = {}
    for term in KEY_TERMS:
        hits = []
        for chunk in chunks:
            count = len(re.findall(re.escape(term), chunk["text"], flags=re.I))
            if count:
                hits.append({
                    "chunk_id": chunk["id"],
                    "file": chunk["file"],
                    "kind": chunk.get("kind"),
                    "cell_index": chunk.get("cell_index"),
                    "slide_index": chunk.get("slide_index"),
                    "title": chunk["title"],
                    "count": count,
                    "summary": chunk["summary"],
                })
        if hits:
            related = Counter()
            by_id = {c["id"]: c for c in chunks}
            for hit in hits:
                related.update(t for t in by_id[hit["chunk_id"]]["terms"] if t != term)
            term_data[term] = {
                "term": term,
                "total_count": sum(h["count"] for h in hits),
                "files": sorted({h["file"] for h in hits}),
                "hits": hits[:80],
                "related": [t for t, _ in related.most_common(8)],
            }
    return term_data


def choose_chapter(chunk):
    path = [p for p in chunk.get("heading_path", []) if p]
    if not path:
        return chunk.get("title") or chunk["file"]
    section_re = re.compile(r"(第\s*\d+\s*[章节]|第[一二三四五六七八九十]+[章节])")
    section_indices = [i for i, title in enumerate(path) if section_re.search(title)]
    if section_indices:
        idx = section_indices[-1]
        if idx > 0 and "章" in path[idx - 1]:
            return f"{path[idx - 1]} / {path[idx]}"
        return path[idx]
    if len(path) >= 2 and "章" in path[0]:
        return f"{path[0]} / {path[1]}"
    return path[0]


def is_structural_chapter(title):
    return bool(re.search(r"(第\s*\d+\s*[章节]|第[一二三四五六七八九十]+[章节]|课堂练习)", title))


def concept_lookup_from_quiz(existing_quiz):
    lookup = {}
    for q in existing_quiz:
        topic = q.get("topic", "")
        stem = q.get("stem", "")
        text = " ".join([stem, q.get("answer", ""), q.get("explanation", "")])
        for term in KEY_TERMS:
            if term in text or term in topic:
                item = lookup.setdefault(term, {"tips": [], "pitfalls": []})
                explanation = q.get("explanation", "")
                if explanation:
                    item["tips"].append(compact(explanation, 160))
                if "错误" in explanation:
                    item["pitfalls"].append(compact(explanation, 140))
    return lookup


def source_label(file_name, chunk):
    if chunk.get("slide_index"):
        return f"{file_name} · 第 {chunk['slide_index']} 页"
    return f"{file_name} · cell {chunk.get('cell_index')}"


def build_course_outline(manifest, chunks, quiz_lookup=None):
    by_file = defaultdict(list)
    for chunk in chunks:
        by_file[chunk["file"]].append(chunk)

    courses = []
    for source in manifest:
        file_name = source["file"]
        source_chunks = by_file.get(file_name, [])
        course_chapter = ""
        for chunk in source_chunks:
            for part in chunk.get("heading_path", []):
                if "章" in part:
                    course_chapter = part
                    break
            if course_chapter:
                break
        family = infer_course_family(file_name)
        chapter_map = {}
        order = []
        current_chapter = ""
        for chunk in source_chunks:
            if not is_study_chunk(chunk):
                continue
            candidate = choose_chapter(chunk)
            if is_noise_heading(candidate):
                candidate = current_chapter or Path(file_name).stem
            if course_chapter and re.match(r"^第\s*\d+\s*节|^第[一二三四五六七八九十]+节", candidate):
                candidate = f"{course_chapter} / {candidate}"
            if not is_structural_chapter(candidate) and current_chapter:
                chapter = current_chapter
            else:
                chapter = candidate
            if is_structural_chapter(chapter):
                current_chapter = chapter
            chapter = chapter or Path(file_name).stem
            if chapter not in chapter_map:
                chapter_map[chapter] = {"title": chapter, "chunks": [], "terms": Counter()}
                order.append(chapter)
            chapter_map[chapter]["chunks"].append(chunk)
            chapter_map[chapter]["terms"].update(chunk.get("terms") or [])

        chapters = []
        for chapter_name in order:
            chapter = chapter_map[chapter_name]
            if is_noise_heading(chapter_name) and not chapter["terms"]:
                continue
            rows = []
            ranked_terms = [term for term, _ in chapter["terms"].most_common()]
            if not ranked_terms:
                ranked_terms = [t for c in chapter["chunks"] for t in extract_heading_terms(c)][:12]
            if not ranked_terms:
                ranked_terms = [compact(chapter_name, 18)]
            seen_terms = set()
            ranked_terms = [t for t in ranked_terms if not (t in seen_terms or seen_terms.add(t))]
            chapter_family = infer_course_family(chapter_name)
            if chapter_family == "general":
                chapter_family = family
            ranked_terms = [t for t in ranked_terms if term_allowed(t, chapter_family, chapter_name)]
            for term in ranked_terms[:16]:
                if is_noise_heading(term):
                    continue
                term_chunks = [c for c in chapter["chunks"] if term in c.get("terms", [])]
                if not term_chunks:
                    term_chunks = chapter["chunks"][:2]
                examples = " ".join(clean_for_study(c["text"], 160) for c in term_chunks[:3])
                row_family = infer_course_family(f"{chapter_name} {term} {examples}")
                if row_family == "general":
                    row_family = chapter_family
                plain, exam, pitfall, memory = generic_row(term, examples, row_family)
                rows.append({
                    "term": term,
                    "plain": plain,
                    "exam": exam,
                    "pitfall": pitfall,
                    "memory": memory,
                    "source": "；".join(source_label(file_name, c) for c in term_chunks[:2]),
                    "evidence": compact(examples, 180),
                })
            if not rows:
                continue
            chapters.append({
                "title": chapter_name,
                "chunk_count": len(chapter["chunks"]),
                "terms": ranked_terms[:24],
                "rows": rows,
            })
        courses.append({
            "file": file_name,
            "kind": source.get("kind"),
            "title": Path(file_name).stem,
            "chapter_count": len(chapters),
            "chunk_count": len(source_chunks),
            "family": family,
            "chapters": chapters,
        })

    return {
        "title": "按课件整理的期末速记目录",
        "courses": courses,
    }


def safe_slug(text):
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text).strip("-")
    return slug[:80] or "mindmap"


def svg_text(text, x, y, size=14, weight="400", fill="#172b4d"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(compact(text, 34))}</text>'


def svg_multiline(text, x, y, width_chars=14, size=14, weight="600", fill="#172b4d", anchor="middle"):
    value = compact(text, width_chars * 3)
    lines = []
    current = ""
    for ch in value:
        current += ch
        visual_len = sum(2 if "\u4e00" <= c <= "\u9fff" else 1 for c in current)
        if visual_len >= width_chars:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    lines = lines[:3]
    start_y = y - (len(lines) - 1) * size * 0.62
    tspans = []
    for idx, line in enumerate(lines):
        tspans.append(f'<tspan x="{x}" dy="{0 if idx == 0 else size * 1.22:.1f}">{html.escape(line)}</tspan>')
    return f'<text x="{x}" y="{start_y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}">{"".join(tspans)}</text>'


def useful_map_terms(chapter, max_terms=5):
    generic = {"节点", "边", "Vertex", "Edge", "图", "卷积", "Attention", "Same"}
    terms = []
    for row in chapter.get("rows", []):
        term = row.get("term", "")
        if not term or is_noise_heading(term):
            continue
        if term in generic and len(terms) >= 1:
            continue
        if term in terms:
            continue
        if compact(term, 20) == compact(chapter.get("title", ""), 20):
            continue
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def chapter_keywords(chapter, max_items=4):
    items = useful_map_terms(chapter, max_items)
    if len(items) >= 2:
        return items
    title_parts = [p for p in re.split(r"[：:、/|\-—\s]+", chapter.get("title", "")) if 2 <= len(p) <= 12]
    for part in title_parts:
        if part not in items and not is_noise_heading(part):
            items.append(part)
        if len(items) >= max_items:
            break
    return items[:max_items]


def map_label(term):
    labels = {
        "节点": "节点 = 对象",
        "Vertex": "Vertex = 节点",
        "边": "边 = 关系",
        "Edge": "Edge = 边/关系",
        "邻接矩阵": "邻接矩阵 = 连接表",
        "度矩阵": "度矩阵 = 节点度数",
        "拉普拉斯矩阵": "L = D - A",
        "GCN": "GCN = 邻居聚合",
        "GAT": "GAT = 注意力聚合",
        "GraphSAGE": "GraphSAGE = 采样聚合",
        "消息传递": "消息传递 = 聚合更新",
        "卷积": "卷积 = 局部特征",
        "卷积核": "卷积核 = 可学习滤波器",
        "池化": "池化 = 降采样",
        "感受野": "感受野 = 可见范围",
        "Transformer": "Transformer = 注意力架构",
        "Attention": "Attention = 加权关注",
        "RNN": "RNN = 隐藏状态传递",
        "LSTM": "LSTM = 三门一状态",
        "GRU": "GRU = 两个门",
    }
    return labels.get(term, term)


def write_mindmap_svg(path, title, chapters):
    import math

    branches = []
    for chapter in chapters:
        terms = chapter_keywords(chapter, 4) or (chapter.get("terms") or [])[:4]
        branches.append({
            "title": compact(chapter["title"], 26),
            "terms": terms,
        })
    branches = branches[:9]
    width = 1800
    height = 1220
    cx, cy = width / 2, height / 2
    palette = ["#2d9cdb", "#00a676", "#f59e0b", "#ef476f", "#7c3aed", "#118ab2", "#f97316", "#0f766e", "#64748b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><filter id="shadow" x="-15%" y="-20%" width="130%" height="140%"><feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0f172a" flood-opacity=".10"/></filter></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<circle cx="900" cy="610" r="520" fill="#ffffff" stroke="#eef2f7" stroke-width="2"/>',
        '<circle cx="900" cy="610" r="350" fill="none" stroke="#f1f5f9" stroke-width="2"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}.link{stroke-width:3;fill:none;opacity:.36}.thin{stroke-width:1.8;fill:none;opacity:.32}.root{filter:url(#shadow)}.pill{filter:url(#shadow)}</style>',
        f'<circle class="root" cx="{cx}" cy="{cy}" r="105" fill="#e8f4ff" stroke="#1a73e8" stroke-width="3"/>',
        svg_multiline(title.replace(" / 总览", ""), cx, cy - 4, 14, 20, "850", "#174ea6"),
        f'<text x="{cx}" y="{cy + 58}" text-anchor="middle" font-size="13" font-weight="700" fill="#64748b">课件总览</text>',
    ]

    if not branches:
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return

    radius = 360
    term_radius = 245
    for i, branch in enumerate(branches):
        angle = -math.pi / 2 + (2 * math.pi * i / len(branches))
        color = palette[i % len(palette)]
        bx = cx + radius * math.cos(angle)
        by = cy + radius * math.sin(angle)
        bw = 260
        bh = 66
        parts.append(f'<path class="link" d="M{cx + 94 * math.cos(angle):.1f} {cy + 94 * math.sin(angle):.1f} C{cx + 190 * math.cos(angle):.1f} {cy + 190 * math.sin(angle):.1f}, {cx + 250 * math.cos(angle):.1f} {cy + 250 * math.sin(angle):.1f}, {bx:.1f} {by:.1f}" stroke="{color}"/>')
        parts.append(f'<rect class="pill" x="{bx-bw/2:.1f}" y="{by-bh/2:.1f}" rx="20" width="{bw}" height="{bh}" fill="#ffffff" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<circle cx="{bx - bw/2 + 25:.1f}" cy="{by:.1f}" r="8" fill="{color}"/>')
        parts.append(svg_multiline(branch["title"], bx + 12, by + 4, 13, 15, "820", "#123047"))
        terms = branch["terms"][:5]
        spread = math.pi * 0.82 if len(terms) > 1 else 0
        start = angle - spread / 2
        for j, term in enumerate(terms):
            t_angle = start + (spread * j / max(1, len(terms) - 1))
            tx = bx + term_radius * math.cos(t_angle)
            ty = by + term_radius * math.sin(t_angle)
            tw = 176
            th = 42
            tx = min(max(tx, 130), width - 130)
            ty = min(max(ty, 84), height - 84)
            parts.append(f'<path class="thin" d="M{bx:.1f} {by:.1f} C{(bx+tx)/2:.1f} {by:.1f}, {(bx+tx)/2:.1f} {ty:.1f}, {tx:.1f} {ty:.1f}" stroke="{color}"/>')
            parts.append(f'<rect x="{tx-tw/2:.1f}" y="{ty-th/2:.1f}" rx="16" width="{tw}" height="{th}" fill="#ffffff" stroke="#dbe4ef" stroke-width="1.4"/>')
            parts.append(svg_multiline(map_label(term), tx, ty + 4, 11, 13, "750", color))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_mindmap_markdown(path, title, chapters):
    lines = [f"# {title}"]
    for chapter in chapters:
        lines.append(f"## {chapter['title']}")
        for row in chapter.get("rows", [])[:10]:
            lines.append(f"### {row['term']}")
            lines.append(f"- 通俗解释：{row['plain']}")
            lines.append(f"- 考试怎么考：{row['exam']}")
            lines.append(f"- 易错点：{row['pitfall']}")
            lines.append(f"- 必记：{row['memory']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mindmaps(outline):
    MINDMAP_OUT.mkdir(parents=True, exist_ok=True)
    for old in list(MINDMAP_OUT.glob("*.svg")) + list(MINDMAP_OUT.glob("*.md")):
        old.unlink()
    maps = []
    for course in outline["courses"]:
        chapters = course["chapters"]
        total_terms = sum(len(ch.get("terms", [])) for ch in chapters)
        course_maps = []
        overview_chapters = chapters[:10]
        overview_slug = f"{safe_slug(course['title'])}-00-整份课件总览"
        overview_svg = f"mindmaps/{overview_slug}.svg"
        overview_md = f"mindmaps/{overview_slug}.md"
        write_mindmap_svg(SITE / overview_svg, f"{course['title']} / 总览", overview_chapters)
        write_mindmap_markdown(SITE / overview_md, f"{course['title']} / 总览", overview_chapters)
        course_maps.append({
            "title": "整份课件总览",
            "node_count": 1 + len(overview_chapters) + sum(min(8, len(ch.get("rows", []))) for ch in overview_chapters),
            "svg": overview_svg,
            "markdown": overview_md,
            "chapters": overview_chapters,
        })
        if total_terms > 70 or len(chapters) > 10:
            for idx, chapter in enumerate(chapters, 1):
                slug = f"{safe_slug(course['title'])}-{idx:02d}-{safe_slug(chapter['title'])}"
                file_name = f"{slug}.svg"
                md_name = f"{slug}.md"
                rel = f"mindmaps/{file_name}"
                md_rel = f"mindmaps/{md_name}"
                write_mindmap_svg(SITE / rel, f"{course['title']} / {chapter['title']}", [chapter])
                write_mindmap_markdown(SITE / md_rel, f"{course['title']} / {chapter['title']}", [chapter])
                course_maps.append({
                    "title": chapter["title"],
                    "node_count": 1 + len(chapter.get("rows", [])),
                    "svg": rel,
                    "markdown": md_rel,
                    "chapters": [chapter],
                })
        maps.append({
            "file": course["file"],
            "title": course["title"],
            "kind": course.get("kind"),
            "maps": course_maps,
        })
    return maps


def md_escape(text):
    return (text or "").replace("\r\n", "\n").strip()


def guide_intro_for_family(family):
    if family == "cnn":
        return [
            "这一类课件主要解决图像数据如何进入神经网络的问题。复习时先抓住“空间结构不能被随便展平”这个动机，再理解卷积、池化、感受野、典型 CNN 架构和训练配置。",
            "考试通常不会只问名词，而是把输出尺寸、参数共享、局部连接、训练技巧和损失函数匹配放在一起判断。",
        ]
    if family == "mlp":
        return [
            "这一类课件是深度学习入门骨架：先理解线性变换和激活函数，再理解损失函数、反向传播、优化器和训练流程。",
            "考试常围绕“为什么需要非线性”“分类输出层和损失函数怎么匹配”“训练循环每一步做什么”来考。",
        ]
    if family == "rnn":
        return [
            "这一类课件处理有先后顺序的数据。复习时先理解序列、时间步和隐藏状态，再看 RNN、GRU、LSTM 的差异。",
            "考试常考 shape、hidden state、output 与 h_n 的区别，以及门控结构为什么能缓解长期依赖问题。",
        ]
    if family == "transformer":
        return [
            "这一类课件从序列建模瓶颈进入注意力机制。复习时先把 Q/K/V、注意力权重、mask、位置编码讲顺，再看 Transformer 编码器和解码器。",
            "考试常考注意力公式的每一项含义、Self-Attention 与 Cross-Attention 的区别、以及 mask 为什么存在。",
        ]
    if family == "graph":
        return [
            "这一类课件处理图结构数据。复习时先理解节点、边、邻接矩阵和任务类型，再进入图卷积、消息传递和图神经网络。",
            "考试常考图数据结构如何表示、节点/边/图级任务怎么区分，以及邻居聚合为什么能利用结构信息。",
        ]
    return [
        "这一份讲义按课件顺序重新整理，目标是把零散页面变成可以从头读到尾的考试复习资料。",
        "复习时优先抓定义、作用、输入输出位置、易错判断和常考表达。",
    ]


def guide_question_block(rows):
    questions = []
    for row in rows[:8]:
        term = row["term"]
        questions.append(f"- 判断：{term}只需要背定义，不需要知道它在模型或训练流程中的位置。（答案：错。复习时必须结合“作用 + 位置 + 易错点”。）")
    return questions


def build_course_guide_markdown(course):
    title = course["title"]
    family = course.get("family", "general")
    lines = [
        f"# {title}：期末考试复习讲义",
        "",
        "> 使用方式：先通读“学习路线”，再按章节背“必记句子”，最后用每章自测题检查。这里不是课件原文搬运，而是按考试复习顺序重写。",
        "",
        "## 一、这份课件先解决什么问题",
        "",
    ]
    for p in guide_intro_for_family(family):
        lines.append(p)
        lines.append("")
    lines.extend([
        "## 二、学习路线",
        "",
    ])
    for idx, chapter in enumerate(course["chapters"], 1):
        terms = "、".join(row["term"] for row in chapter.get("rows", [])[:5])
        lines.append(f"{idx}. **{chapter['title']}**：先抓 {terms or '本节核心概念'}。")
    lines.extend(["", "## 三、章节详解", ""])

    for idx, chapter in enumerate(course["chapters"], 1):
        rows = chapter.get("rows", [])
        lines.extend([
            f"## 第 {idx} 部分：{chapter['title']}",
            "",
            "### 1. 本节先看什么",
            "",
        ])
        if rows:
            first_terms = "、".join(r["term"] for r in rows[:4])
            lines.append(f"这一节先把 **{first_terms}** 放到同一条知识链里理解。不要先背细节，先问自己：它在输入、模型结构、训练流程、损失函数或评估里处在哪一步？")
        else:
            lines.append("这一节主要根据课件页面顺序整理，复习时先建立整体结构，再回到具体例子。")
        lines.extend(["", "### 2. 核心知识点表", ""])
        lines.append("| 知识点 | 通俗解释 | 考试怎么考 | 易错点 | 必记句子/公式 |")
        lines.append("|---|---|---|---|---|")
        for row in rows[:18]:
            lines.append(
                "| "
                + " | ".join(md_escape(row[k]).replace("|", "/") for k in ["term", "plain", "exam", "pitfall", "memory"])
                + " |"
            )
        lines.extend(["", "### 3. 像考试答案一样组织语言", ""])
        for row in rows[:8]:
            lines.append(f"- **{row['term']}**：{row['plain']} 考试写作时要补一句：{row['exam']} 易错点是：{row['pitfall']}")
        lines.extend(["", "### 4. 本节自测", ""])
        lines.extend(guide_question_block(rows))
        lines.extend(["", "### 5. 本节速记", ""])
        for row in rows[:10]:
            lines.append(f"- {row['memory']}")
        lines.append("")

    lines.extend([
        "## 四、考前总复盘",
        "",
        "考前不要平均用力。优先检查下面这些问题：",
        "",
        "1. 每个核心概念能不能用一句话说明“它是什么”。",
        "2. 能不能说出它在模型结构、训练流程或数据处理中的位置。",
        "3. 能不能说出一个最常见的错误说法。",
        "4. 遇到选择题时，能不能判断选项是在混淆概念、夸大作用，还是写反了训练/推理阶段。",
        "",
        "## 五、打印建议",
        "",
        "<style>@media print { @page { margin: 8mm; } body { font-size: 11pt; line-height: 1.35; } h1, h2, h3 { page-break-after: avoid; } table { font-size: 9pt; border-collapse: collapse; } th, td { padding: 3px 5px; border: 1px solid #ddd; vertical-align: top; } }</style>",
        "",
    ])
    return "\n".join(lines)


def build_study_guides(outline):
    GUIDE_OUT.mkdir(parents=True, exist_ok=True)
    for old in GUIDE_OUT.glob("*.md"):
        old.unlink()

    guides = []
    index_lines = [
        "# 深度学习期末考试讲义总目录",
        "",
        "这份总目录对应网站当前纳入的全部课件。建议按“多层感知机 -> CNN -> RNN -> Attention/Transformer -> 图学习”的顺序复习。",
        "",
        "## 课件讲义列表",
        "",
    ]
    for course in outline["courses"]:
        slug = safe_slug(course["title"])
        rel = f"guides/{slug}-考试复习讲义.md"
        md = build_course_guide_markdown(course)
        (SITE / rel).write_text(md, encoding="utf-8")
        item = {
            "title": course["title"],
            "file": course["file"],
            "kind": course.get("kind"),
            "href": rel,
            "chapter_count": course.get("chapter_count", 0),
            "chunk_count": course.get("chunk_count", 0),
        }
        guides.append(item)
        index_lines.append(f"- [{course['title']}]({Path(rel).name})：{course.get('chapter_count', 0)} 个章节块。")
    index_lines.extend([
        "",
        "## 使用顺序",
        "",
        "1. 先读每份讲义的“这份课件先解决什么问题”。",
        "2. 再看章节知识点表，把通俗解释和必记句子背熟。",
        "3. 最后做网站里的基础练习和标准组卷。",
        "",
        "<style>@media print { @page { margin: 8mm; } body { font-size: 11pt; line-height: 1.35; } }</style>",
    ])
    index_rel = "guides/00-深度学习期末考试讲义总目录.md"
    (SITE / index_rel).write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return {"index": index_rel, "items": guides}


def q_mc(id_, topic, stem, options, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "mcq", "topic": topic, "stem": stem, "options": options, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_fill(id_, topic, stem, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "fill", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_tf(id_, topic, stem, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "tf", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_short(id_, topic, stem, answer, explanation, source, difficulty="易"):
    return {"id": id_, "type": "short", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def has_bad_question_text(q):
    text = " ".join(str(q.get(k, "")) for k in ["stem", "answer", "explanation"])
    return any(p in text for p in BAD_QUESTION_PATTERNS)


def load_standard_quiz():
    path = DATABASE / "quiz_bank.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cleaned = []
    seen = set()
    for q in data:
        if has_bad_question_text(q):
            continue
        qid = q.get("id")
        if not qid or qid in seen:
            continue
        cleaned.append(q)
        seen.add(qid)
    return cleaned


def quiz_stage(term, family):
    input_terms = {"Padding", "Packing", "PackedSequence", "标准化", "归一化", "数据增强", "ImageFolder", "WeightedRandomSampler", "Batch"}
    structure_terms = {
        "卷积", "互相关", "卷积核", "步长", "填充", "Valid", "Same", "Full", "感受野", "池化",
        "激活函数", "BatchNorm", "Dropout", "RNN", "隐藏状态", "双向RNN", "堆叠RNN", "GRU", "LSTM",
        "1D卷积", "TCN", "Seq2Seq", "Encoder-Decoder", "注意力机制", "Attention", "Query", "Key",
        "Value", "Q/K/V", "上下文向量", "Scaled Dot-Product Attention", "sqrt(d_k)", "Multi-Head Attention",
        "Self-Attention", "Cross-Attention", "位置编码", "Positional Encoding", "Transformer", "RoPE",
        "图学习", "图神经网络", "节点", "边", "邻接矩阵", "度矩阵", "拉普拉斯矩阵", "图卷积", "GCN",
        "GAT", "GraphSAGE", "消息传递", "聚合", "Vertex", "Edge",
    }
    training_terms = {"梯度下降", "学习率", "Epoch", "反向传播", "计算图", "自动微分", "梯度消失", "梯度爆炸", "梯度裁剪", "优化器", "Adam", "RMSProp", "Momentum", "权重衰减", "超参数", "消融实验"}
    output_terms = {"损失函数", "Logistic", "Sigmoid", "Softmax", "交叉熵", "BCEWithLogitsLoss", "CrossEntropyLoss", "类别不平衡", "pos_weight", "class weight"}
    if term in input_terms:
        return "数据输入或预处理"
    if term in output_terms:
        return "输出层、损失函数或评价目标"
    if term in training_terms:
        return "训练优化过程"
    if term in structure_terms:
        return "模型结构或信息流动"
    return {
        "cnn": "卷积网络结构",
        "mlp": "前向传播与训练流程",
        "rnn": "序列建模结构",
        "transformer": "注意力结构",
        "graph": "图结构表示与邻居聚合",
    }.get(family, "课程知识脉络")


def build_beginner_quiz(outline):
    questions = []
    idx = 1
    for course in outline["courses"]:
        for chapter in course["chapters"]:
            for row in chapter["rows"][:10]:
                term = row["term"]
                source = course["file"]
                topic = f"{course['title']} / {chapter['title']}"
                plain = row["plain"] or row["memory"]
                stage = quiz_stage(term, course.get("family", "general"))
                if not plain:
                    continue
                questions.append(q_mc(
                    f"BMC{idx:04d}",
                    topic,
                    f"复习{term}时，下列哪种理解最符合本章要求？",
                    [
                        f"A. {compact(plain, 92)}",
                        "B. 只用于装饰课件标题，不参与模型或训练理解",
                        "C. 只在测试集上修改标签，训练阶段不需要关注",
                        "D. 与输入、模型、损失或评估都没有关系",
                    ],
                    "A",
                    f"A正确。{compact(row['exam'], 150)}",
                    source,
                ))
                questions.append(q_tf(
                    f"BTF{idx:04d}",
                    topic,
                    f"{term}不能只背定义，还要能说明它在“{stage}”中的作用。",
                    "正确",
                    "基础阶段先把知识点放回课程脉络中，后面再做标准题会更稳。",
                    source,
                ))
                questions.append(q_fill(
                    f"BF{idx:04d}",
                    topic,
                    f"学习{term}时，除了记定义，还要判断它属于课程流程中的哪一环节：____。",
                    stage,
                    f"答案是{stage}。{compact(row['memory'], 150)}",
                    source,
                ))
                if idx % 3 == 0:
                    questions.append(q_short(
                        f"BS{idx:04d}",
                        topic,
                        f"用两句话说明{term}在本章中的作用，并写出一个复习时容易忽略的点。",
                        f"{term}的作用可以概括为：{compact(plain, 160)} 容易忽略的是：{compact(row['pitfall'], 130)}",
                        "基础简答不追求展开太深，先讲清“它是什么、放在哪里、容易错在哪里”。",
                        source,
                    ))
                idx += 1
    return [q for q in questions if not has_bad_question_text(q)]


def fallback_review(outline):
    return {
        "title": "深度学习课件综合复习资料",
        "updated_from": "当前课件目录内全部 Jupyter/PPT 课件",
        "sections": [
            {
                "title": course["title"],
                "points": [
                    f"本课件包含 {course['chapter_count']} 个章节块、{course['chunk_count']} 个可检索片段。",
                    "建议先看本页教材目录表格，再进入思维导图按知识脉络回忆，最后做基础练习和标准组卷。",
                ],
                "exam_focus": [row["term"] for ch in course["chapters"] for row in ch["rows"][:3]][:12],
            }
            for course in outline["courses"]
        ],
    }


def load_review_or_fallback(outline):
    path = DATABASE / "review_material.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["updated_from"] = "当前课件目录内全部 Jupyter/PPT 课件"
            return data
        except json.JSONDecodeError:
            pass
    return fallback_review(outline)


CSS = r"""
:root {
  --blue: #20a8e0;
  --blue-dark: #1376a8;
  --ink: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
  --soft: #f6f8fb;
  --panel: #ffffff;
  --green: #16a34a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--ink);
  background: #f7f9fc;
  letter-spacing: 0;
}
a { color: var(--blue-dark); text-decoration: none; }
header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(255,255,255,.96);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(8px);
}
.topbar {
  max-width: 1440px;
  margin: 0 auto;
  padding: 14px 24px;
  display: grid;
  grid-template-columns: 290px 1fr;
  gap: 24px;
  align-items: center;
}
.brand h1 { margin: 0; font-size: 20px; line-height: 1.2; }
.brand p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
nav {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}
nav button {
  border: 0;
  background: transparent;
  color: #445166;
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
}
nav button.active { color: #fff; background: var(--blue); }
main {
  max-width: 1440px;
  margin: 0 auto;
  padding: 22px 24px 48px;
}
.view { display: none; }
.view.active { display: block; }
.hero-grid {
  display: grid;
  grid-template-columns: 1.15fr .85fr;
  gap: 18px;
  align-items: start;
  margin-bottom: 18px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
}
.panel h2, .panel h3 { margin: 0 0 12px; }
.muted, .meta { color: var(--muted); font-size: 13px; }
.stats {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 10px;
}
.stat {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.stat strong { display: block; font-size: 23px; color: #0f172a; }
.toolbar {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) auto auto;
  gap: 10px;
  margin: 14px 0;
}
input[type="search"] {
  border: 1px solid #d6dce6;
  border-radius: 8px;
  padding: 11px 13px;
  font-size: 15px;
  background: #fff;
}
.btn, .btn.secondary {
  border: 1px solid var(--blue);
  border-radius: 8px;
  padding: 10px 13px;
  cursor: pointer;
  font-weight: 700;
  background: var(--blue);
  color: #fff;
}
.btn.secondary { background: #fff; color: var(--blue-dark); }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  gap: 16px;
}
.result, .q, .answer-item {
  border-top: 1px solid var(--line);
  padding: 14px 0;
}
.result:first-child, .q:first-child, .answer-item:first-child { border-top: 0; }
.result-title {
  display: flex;
  gap: 10px;
  justify-content: space-between;
  align-items: flex-start;
  font-weight: 700;
}
.path { margin-top: 4px; color: var(--blue-dark); font-size: 13px; }
.snippet { margin: 8px 0 0; line-height: 1.65; }
mark { background: #dff3ff; color: #0b5e8e; border-radius: 3px; padding: 0 2px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip {
  border: 1px solid #dce3ec;
  background: #fff;
  color: #334155;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 13px;
  cursor: pointer;
}
.term-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  gap: 8px;
}
.term-btn {
  border: 1px solid var(--line);
  background: #fff;
  text-align: left;
  border-radius: 8px;
  padding: 9px 10px;
  cursor: pointer;
}
.term-btn strong { display: block; }
.term-btn span { color: var(--muted); font-size: 12px; }
.course-layout {
  display: grid;
  grid-template-columns: 270px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
.side-index {
  position: sticky;
  top: 92px;
  max-height: calc(100vh - 110px);
  overflow: auto;
}
.side-index a {
  display: block;
  padding: 8px 10px;
  border-radius: 8px;
  color: #334155;
  font-size: 13px;
}
.side-index a:hover { background: #edf7fd; color: var(--blue-dark); }
.course-card { margin-bottom: 16px; }
.course-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 12px;
}
.chapter { border-top: 1px solid var(--line); padding-top: 14px; margin-top: 14px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 10px 11px;
  vertical-align: top;
  line-height: 1.55;
}
th { background: #f8fafc; text-align: left; color: #475569; font-size: 13px; }
tr:last-child td { border-bottom: 0; }
.quiz-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.quiz-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.quiz-section { margin: 18px 0; }
.q-stem { font-weight: 700; line-height: 1.6; }
.options { list-style: none; padding: 0; margin: 8px 0 0; }
.options li { margin: 5px 0; line-height: 1.5; }
.badge {
  display: inline-block;
  border: 1px solid #dbe3ec;
  border-radius: 999px;
  padding: 2px 8px;
  color: var(--muted);
  font-size: 12px;
  margin-left: 6px;
}
.blank-answer {
  display: inline-block;
  min-width: 140px;
  border-bottom: 1px solid #9aa6b2;
}
.answers { background: #f8fbff; }
.mindmap-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
.map-card h3 { margin-bottom: 4px; }
.map-preview {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfdff;
  overflow: auto;
  max-height: 760px;
}
.map-preview img { display: block; width: 100%; min-width: 1180px; height: auto; }
.map-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.submaps { margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #fff; }
.submaps summary { cursor: pointer; font-weight: 750; color: #334155; }
.submap-row { display: flex; gap: 12px; justify-content: space-between; align-items: center; padding: 10px 0; border-top: 1px solid var(--line); }
.submap-row:first-of-type { border-top: 0; }
.submap-row a { font-size: 13px; font-weight: 700; }
.map-tree ul { list-style: none; padding-left: 18px; border-left: 1px solid #dce6f2; }
.map-tree li { margin: 7px 0; line-height: 1.45; }
.compact-tree { margin-top: 10px; max-height: 190px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }
.map-node {
  display: inline-block;
  background: #fff;
  border: 1px solid #d8e0ea;
  border-radius: 8px;
  padding: 5px 8px;
}
.map-node.term { color: #174ea6; cursor: pointer; }
.formula {
  font-family: "SFMono-Regular", Consolas, monospace;
  background: #f1f5f9;
  color: #0f4b6e;
  padding: 8px 10px;
  border-radius: 8px;
  margin: 6px 0;
  overflow-wrap: anywhere;
}
.review-section { border-top: 1px solid var(--line); padding-top: 18px; margin-top: 18px; }
.guide-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin: 14px 0 22px;
}
.guide-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px;
  background: #fff;
}
.guide-card strong { display: block; margin-bottom: 6px; }
.guide-card .btn { display: inline-block; margin-top: 10px; padding: 8px 10px; font-size: 13px; }
@media (max-width: 920px) {
  .topbar, .hero-grid, .layout, .course-layout, .toolbar { grid-template-columns: 1fr; }
  .stats { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .side-index { position: static; max-height: none; }
}
"""


JS = r"""
const DB = window.COURSE_DB;
const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").toString().toLowerCase();
const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const lastQuizState = { beginner: null, standard: null };

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
}

function highlight(text, query) {
  const safe = escapeHtml(text || "");
  const q = (query || "").trim();
  if (!q) return safe;
  const parts = q.split(/\s+/).filter(Boolean).slice(0, 6).map(x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!parts.length) return safe;
  return safe.replace(new RegExp(`(${parts.join("|")})`, "gi"), "<mark>$1</mark>");
}

function scoreChunk(chunk, query) {
  const q = norm(query).split(/\s+/).filter(Boolean);
  if (!q.length) return 0;
  const title = norm(chunk.title + " " + (chunk.heading_path || []).join(" "));
  const text = norm(chunk.text);
  const terms = norm((chunk.terms || []).join(" "));
  let score = 0;
  q.forEach(part => {
    if (title.includes(part)) score += 10;
    if (terms.includes(part)) score += 8;
    const matches = text.match(new RegExp(part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"));
    if (matches) score += Math.min(matches.length, 12);
  });
  return score;
}

function shuffle(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function safeFileName(text) {
  return (text || "试卷").replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "-").slice(0, 80);
}

function downloadTextFile(filename, text) {
  const blob = new Blob([text], {type: "text/markdown;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function printStyleBlock() {
  return `<style>
@media print {
  @page { margin: 7mm; }
  body { font-size: 10.5pt; line-height: 1.32; }
  h1, h2, h3 { page-break-after: avoid; margin: 0.45em 0 0.25em; }
  p, li { margin: 0.2em 0; }
  .blank { display: inline-block; min-width: 42mm; border-bottom: 1px solid #777; }
}
</style>`;
}

function questionMarkdown(q, idx) {
  if (q.type === "mcq") {
    return `${idx}. ${q.stem}\n\n${q.options.map(o => `   ${o}`).join("\n")}`;
  }
  if (q.type === "tf") {
    return `${idx}. ${q.stem}\n\n   A. 正确\n   B. 错误`;
  }
  if (q.type === "fill") {
    return `${idx}. ${q.stem} <span class="blank"></span>`;
  }
  return `${idx}. ${q.stem}\n\n答题区：\n\n\n`;
}

function answerMarkdown(q, idx) {
  return `${idx}. [${q.id}] 答案：${q.answer}\n\n解析：${q.explanation}\n\n来源：${q.source} · ${q.topic}`;
}

function exportQuizMarkdown(mode, part) {
  const state = lastQuizState[mode];
  if (!state || !state.chosen.length) return;
  const modeName = mode === "beginner" ? "基础练习" : "标准组卷";
  const title = `${modeName}${state.topic ? "：" + state.topic : ""}`;
  const labels = {mcq: "单选题", tf: "判断题", fill: "填空题", short: "简答题"};
  const lines = [
    printStyleBlock(),
    `# ${title}${part === "questions" ? "（题目版）" : "（答案解析版）"}`,
    "",
    `生成时间：${new Date().toLocaleString("zh-CN")}`,
    "",
    part === "questions" ? "> 打印建议：题目和答案分开打印；本文件已内置较小页边距样式。" : "> 打印建议：答案解析单独打印或仅在核对时查看。",
    "",
  ];
  if (part === "questions") {
    state.spec.forEach(([type]) => {
      const items = state.chosen.filter(q => q.type === type);
      if (!items.length) return;
      lines.push(`## ${labels[type]}`, "");
      items.forEach((q, i) => lines.push(questionMarkdown(q, i + 1), ""));
    });
  } else {
    state.chosen.forEach((q, i) => lines.push(answerMarkdown(q, i + 1), ""));
  }
  const suffix = part === "questions" ? "题目" : "答案解析";
  downloadTextFile(`${safeFileName(title)}-${suffix}.md`, lines.join("\n"));
}

function renderStats() {
  $("stats").innerHTML = `
    <div class="stat"><strong>${DB.manifest.length}</strong><span>课件文件</span></div>
    <div class="stat"><strong>${DB.chunks.length}</strong><span>可检索片段</span></div>
    <div class="stat"><strong>${Object.keys(DB.terms).length}</strong><span>索引词条</span></div>
    <div class="stat"><strong>${DB.quizBank.length}</strong><span>标准题</span></div>
    <div class="stat"><strong>${DB.beginnerQuizBank.length}</strong><span>基础题</span></div>
  `;
}

function renderTermCloud() {
  const terms = Object.values(DB.terms).sort((a, b) => b.total_count - a.total_count).slice(0, 42);
  $("termCloud").innerHTML = terms.map(t => `
    <button class="term-btn" data-term="${escapeHtml(t.term)}">
      <strong>${escapeHtml(t.term)}</strong>
      <span>${t.total_count} 次 · ${t.files.length} 文件</span>
    </button>
  `).join("");
  $("termCloud").querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    $("searchInput").value = btn.dataset.term;
    runSearch(btn.dataset.term);
    showView("search");
  }));
}

function runSearch(query) {
  const scored = DB.chunks.map(c => [scoreChunk(c, query), c]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 80);
  const results = scored.map(([, c]) => c);
  $("resultCount").textContent = query.trim() ? `${results.length} 个结果` : "等待输入";
  $("results").innerHTML = results.length ? results.map(c => `
    <article class="result">
      <div class="result-title">
        <span>${escapeHtml(c.title || c.file)}</span>
        <button class="chip result-quiz" data-topic="${escapeHtml((c.terms && c.terms[0]) || query)}">出题</button>
      </div>
      <div class="path">${escapeHtml((c.heading_path || []).join(" / "))}</div>
      <p class="snippet">${highlight(c.summary, query)}</p>
      <span class="meta">${escapeHtml(c.file)} · ${c.slide_index ? `第 ${c.slide_index} 页` : `cell ${c.cell_index}`} · ${escapeHtml(c.kind || c.type)}</span>
      <div class="chips">${(c.terms || []).slice(0, 8).map(t => `<button class="chip" data-term="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join("")}</div>
    </article>
  `).join("") : `<div class="panel"><p class="meta">输入课件中的词条，例如“感受野”“BatchNorm”“LSTM”“CrossEntropyLoss”。</p></div>`;
  $("results").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
  $("results").querySelectorAll(".result-quiz").forEach(btn => btn.addEventListener("click", () => generateStandardQuiz(btn.dataset.topic)));

  const fuzzy = DB.terms[query] || Object.values(DB.terms).find(t => query && (norm(t.term).includes(norm(query)) || norm(query).includes(norm(t.term))));
  $("explainBox").innerHTML = fuzzy ? `
    <h2>${escapeHtml(fuzzy.term)}</h2>
    <p class="muted">出现 ${fuzzy.total_count} 次，分布在 ${fuzzy.files.length} 个课件文件中。</p>
    <div class="chips">${(fuzzy.related || []).map(r => `<button class="chip" data-term="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join("")}</div>
  ` : `<h2>搜索说明</h2><p class="muted">搜索会定位到课件原文片段、相关词条和来源位置。适合先查概念，再回到教材目录和导图复习。</p>`;
  $("explainBox").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
}

function renderOutline() {
  const courses = DB.outline.courses;
  $("courseIndex").innerHTML = courses.map((c, i) => `<a href="#course-${i}">${escapeHtml(c.title)}</a>`).join("");
  $("outlineBody").innerHTML = courses.map((course, i) => `
    <section class="panel course-card" id="course-${i}">
      <div class="course-head">
        <div>
          <h2>${escapeHtml(course.title)}</h2>
          <p class="muted">${escapeHtml(course.file)} · ${course.chapter_count} 个章节块 · ${course.chunk_count} 个片段</p>
        </div>
        <button class="btn secondary" data-map="${escapeHtml(course.file)}">看导图</button>
      </div>
      ${course.chapters.map(ch => `
        <div class="chapter">
          <h3>${escapeHtml(ch.title)}</h3>
          <table>
            <thead><tr><th>知识点</th><th>通俗解释</th><th>考试怎么考</th><th>易错点</th><th>必记句子/公式</th><th>来源</th></tr></thead>
            <tbody>${ch.rows.map(r => `
              <tr>
                <td><strong>${escapeHtml(r.term)}</strong></td>
                <td>${escapeHtml(r.plain)}</td>
                <td>${escapeHtml(r.exam)}</td>
                <td>${escapeHtml(r.pitfall)}</td>
                <td>${escapeHtml(r.memory)}</td>
                <td class="muted">${escapeHtml(r.source)}</td>
              </tr>
            `).join("")}</tbody>
          </table>
        </div>
      `).join("")}
    </section>
  `).join("");
  $("outlineBody").querySelectorAll("button[data-map]").forEach(btn => btn.addEventListener("click", () => {
    showView("mindmaps");
    const card = document.querySelector(`[data-map-card="${CSS.escape(btn.dataset.map)}"]`);
    if (card) card.scrollIntoView({behavior: "smooth", block: "start"});
  }));
}

function renderMapTree(map) {
  return `<div class="map-tree"><ul>${map.chapters.map(ch => `
    <li><span class="map-node">${escapeHtml(ch.title)}</span>
      <ul>${(ch.terms || []).slice(0, 16).map(t => `<li><span class="map-node term" data-term="${escapeHtml(t)}">${escapeHtml(t)}</span></li>`).join("")}</ul>
    </li>
  `).join("")}</ul></div>`;
}

function renderMindmaps() {
  $("mindmapBody").innerHTML = DB.mindmaps.map(course => `
    <article class="panel map-card" data-map-card="${escapeHtml(course.file)}">
      <h3>${escapeHtml(course.title)}</h3>
      <p class="muted">${escapeHtml(course.file)} · ${course.maps.length} 张蜘蛛网导图${course.maps.length > 1 ? "，已按章节拆分" : ""}</p>
      ${course.maps.slice(0, 1).map(map => `
        <div class="chapter">
          <div class="result-title">
            <strong>${escapeHtml(map.title)} <span class="badge">${map.node_count} 节点</span></strong>
            <span class="map-actions">
              <a class="btn secondary" href="${escapeHtml(map.svg)}" target="_blank">打开大图</a>
              <a class="btn secondary" href="${escapeHtml(map.svg)}" download>下载 SVG</a>
              <a class="btn secondary" href="${escapeHtml(map.markdown)}" download>下载大纲</a>
            </span>
          </div>
          <div class="map-preview"><img src="${escapeHtml(map.svg)}" alt="${escapeHtml(map.title)} 思维导图"></div>
          <div class="map-tree compact-tree">${renderMapTree(map)}</div>
        </div>
      `).join("")}
      ${course.maps.length > 1 ? `
        <details class="submaps">
          <summary>展开章节小图和大纲下载（${course.maps.length - 1} 张）</summary>
          ${course.maps.slice(1).map(map => `
            <div class="submap-row">
              <strong>${escapeHtml(map.title)}</strong>
              <span class="map-actions">
                <a href="${escapeHtml(map.svg)}" target="_blank">打开</a>
                <a href="${escapeHtml(map.svg)}" download>下载 SVG</a>
                <a href="${escapeHtml(map.markdown)}" download>下载大纲</a>
              </span>
            </div>
          `).join("")}
        </details>` : ""}
    </article>
  `).join("");
  $("mindmapBody").querySelectorAll(".map-node.term").forEach(node => node.addEventListener("click", () => {
    $("searchInput").value = node.dataset.term;
    runSearch(node.dataset.term);
    showView("search");
  }));
}

function pickQuestions(bank, type, topic, count) {
  const q = norm(topic || "");
  const all = bank.filter(x => x.type === type);
  const related = q ? all.filter(x => norm([x.topic, x.stem, x.source].join(" ")).includes(q)) : [];
  const base = related.length >= Math.min(count, 3) ? related : all;
  return shuffle(base).slice(0, count);
}

function generateQuiz(bank, mode, topic = "") {
  const spec = mode === "beginner"
    ? [["mcq", 10], ["tf", 10], ["fill", 10], ["short", 3]]
    : [["mcq", 20], ["fill", 10], ["short", 5]];
  const labels = {mcq: "单选题", tf: "判断题", fill: "填空题", short: "简答题"};
  const chosen = spec.flatMap(([type, count]) => pickQuestions(bank, type, topic, count));
  const title = mode === "beginner" ? "基础练习随机题" : "标准综合随机题";
  const targetTitle = mode === "beginner" ? $("beginnerTitle") : $("quizTitle");
  const targetMeta = mode === "beginner" ? $("beginnerMeta") : $("quizMeta");
  const targetBody = mode === "beginner" ? $("beginnerBody") : $("quizBody");
  lastQuizState[mode] = {chosen, spec, topic, title};
  targetTitle.textContent = topic ? `${title}：${topic}` : title;
  targetMeta.textContent = spec.map(([type, count]) => `${count} 道${labels[type]}`).join("、") + "。再次点击会重新随机生成。";
  targetBody.innerHTML = spec.map(([type]) => {
    const items = chosen.filter(q => q.type === type);
    return `<section class="quiz-section"><h3>${labels[type]}</h3>${items.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>`;
  }).join("") + `<section class="panel answers"><h3>答案与解析</h3>${chosen.map((q, i) => renderAnswer(q, i + 1)).join("")}</section>`;
}

function generateStandardQuiz(topic = "", activate = true) {
  generateQuiz(DB.quizBank, "standard", topic);
  if (activate) {
    showView("quiz");
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

function generateBeginnerQuiz(topic = "", activate = true) {
  generateQuiz(DB.beginnerQuizBank, "beginner", topic);
  if (activate) {
    showView("beginner");
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

function renderQuestion(q, idx) {
  if (q.type === "mcq") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options">${q.options.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></div>`;
  }
  if (q.type === "tf") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options"><li>A. 正确</li><li>B. 错误</li></ul></div>`;
  }
  if (q.type === "fill") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="blank-answer"></span> <span class="badge">${escapeHtml(q.difficulty)}</span></div></div>`;
  }
  return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="muted">答题区：</p><p style="height:54px;border-bottom:1px solid var(--line)"></p></div>`;
}

function renderAnswer(q, idx) {
  return `<div class="answer-item"><strong>${idx}. [${escapeHtml(q.id)}] ${escapeHtml(q.answer)}</strong><p>${escapeHtml(q.explanation)}</p><span class="muted">来源：${escapeHtml(q.source)} · ${escapeHtml(q.topic)}</span></div>`;
}

function renderReview() {
  const r = DB.review;
  const guideBlock = DB.guides ? `
    <section class="review-section">
      <h2>考试版 Markdown 讲义下载</h2>
      <p class="muted">每份课件一份讲义，按“问题动机 -> 学习路线 -> 章节详解 -> 考法提醒 -> 自测”整理。Markdown 内置小页边距打印样式。</p>
      <p><a class="btn secondary" href="${escapeHtml(DB.guides.index)}" download>下载总目录</a></p>
      <div class="guide-grid">
        ${(DB.guides.items || []).map(g => `
          <div class="guide-card">
            <strong>${escapeHtml(g.title)}</strong>
            <span class="muted">${escapeHtml(g.file)} · ${g.chapter_count} 章块</span>
            <br><a class="btn secondary" href="${escapeHtml(g.href)}" download>下载讲义 .md</a>
          </div>
        `).join("")}
      </div>
    </section>` : "";
  $("reviewBody").innerHTML = guideBlock + (r.sections || []).map(sec => `
    <section class="review-section">
      <h2>${escapeHtml(sec.title)}</h2>
      <ul>${(sec.points || []).map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>
      ${(sec.details || []).length ? `<h3>详细知识点</h3><ul>${sec.details.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.must_memorize || []).length ? `<h3>必背结论</h3><ul>${sec.must_memorize.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.formulas || []).map(f => `<div class="formula">${escapeHtml(f)}</div>`).join("")}
      ${(sec.exam_focus || []).length ? `<p><strong>常考点：</strong>${escapeHtml(sec.exam_focus.join("；"))}</p>` : ""}
    </section>
  `).join("");
}

function init() {
  document.querySelectorAll("nav button").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));
  $("searchInput").addEventListener("input", e => runSearch(e.target.value));
  $("makeStandardQuiz").addEventListener("click", () => generateStandardQuiz($("searchInput").value.trim()));
  $("makeBeginnerQuiz").addEventListener("click", () => generateBeginnerQuiz($("searchInput").value.trim()));
  $("quizAll").addEventListener("click", () => generateStandardQuiz(""));
  $("beginnerAll").addEventListener("click", () => generateBeginnerQuiz(""));
  $("quizExportQuestions").addEventListener("click", () => exportQuizMarkdown("standard", "questions"));
  $("quizExportAnswers").addEventListener("click", () => exportQuizMarkdown("standard", "answers"));
  $("beginnerExportQuestions").addEventListener("click", () => exportQuizMarkdown("beginner", "questions"));
  $("beginnerExportAnswers").addEventListener("click", () => exportQuizMarkdown("beginner", "answers"));
  renderStats();
  renderTermCloud();
  renderOutline();
  renderMindmaps();
  renderReview();
  runSearch("");
  generateBeginnerQuiz("", false);
}

init();
"""


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>深度学习期末复习资料库</title>
  <link rel="stylesheet" href="assets/course_site.css">
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <h1>深度学习期末复习资料库</h1>
        <p>检索 · 教材目录 · 思维导图 · 基础练习 · 标准组卷</p>
      </div>
      <nav aria-label="主导航">
        <button class="active" data-view="search">搜索</button>
        <button data-view="outline">教材目录</button>
        <button data-view="mindmaps">思维导图</button>
        <button data-view="beginner">基础练习</button>
        <button data-view="quiz">标准组卷</button>
        <button data-view="review">综合资料</button>
      </nav>
    </div>
  </header>
  <main>
    <section id="view-search" class="view active">
      <div class="hero-grid">
        <div class="panel">
          <h2>从一个词条开始复习</h2>
          <p class="muted">搜索会回到课件原文位置，也能继续跳到基础题、标准题、目录和导图。</p>
          <div class="toolbar">
            <input id="searchInput" type="search" placeholder="输入词条：如 感受野、BatchNorm、LSTM、CrossEntropyLoss">
            <button id="makeBeginnerQuiz" class="btn secondary">基础练习</button>
            <button id="makeStandardQuiz" class="btn">标准组卷</button>
          </div>
        </div>
        <div id="stats" class="stats"></div>
      </div>
      <div class="layout">
        <div>
          <div class="panel" style="margin-bottom:14px">
            <div class="result-title"><h2 style="margin:0">搜索结果</h2><span id="resultCount" class="meta"></span></div>
          </div>
          <div id="results" class="panel"></div>
        </div>
        <aside>
          <div id="explainBox" class="panel"></div>
          <div class="panel" style="margin-top:14px">
            <h3>高频词条</h3>
            <div id="termCloud" class="term-list"></div>
          </div>
        </aside>
      </div>
    </section>

    <section id="view-outline" class="view">
      <div class="course-layout">
        <aside class="panel side-index">
          <h3>课件目录</h3>
          <div id="courseIndex"></div>
        </aside>
        <div id="outlineBody"></div>
      </div>
    </section>

    <section id="view-mindmaps" class="view">
      <div class="panel" style="margin-bottom:14px">
        <h2>课件思维导图</h2>
        <p class="muted">每份课件先给整份总览蜘蛛网；可打开 SVG 大图，也可下载 Markdown 大纲后导入 Markmap、XMind 等工具继续美化。</p>
      </div>
      <div id="mindmapBody" class="mindmap-grid"></div>
    </section>

    <section id="view-beginner" class="view">
      <div class="panel">
        <div class="quiz-head">
          <div>
            <h2 id="beginnerTitle">基础练习</h2>
            <p id="beginnerMeta" class="muted">适合先熟悉课程脉络。</p>
          </div>
          <div class="quiz-actions">
            <button id="beginnerExportQuestions" class="btn secondary">导出题目.md</button>
            <button id="beginnerExportAnswers" class="btn secondary">导出答案.md</button>
            <button id="beginnerAll" class="btn">重新生成基础题</button>
          </div>
        </div>
        <div id="beginnerBody"></div>
      </div>
    </section>

    <section id="view-quiz" class="view">
      <div class="panel">
        <div class="quiz-head">
          <div>
            <h2 id="quizTitle">标准综合随机题</h2>
            <p id="quizMeta" class="muted">保留原标准题库，适合考前检测。</p>
          </div>
          <div class="quiz-actions">
            <button id="quizExportQuestions" class="btn secondary">导出题目.md</button>
            <button id="quizExportAnswers" class="btn secondary">导出答案.md</button>
            <button id="quizAll" class="btn">生成标准题</button>
          </div>
        </div>
        <div id="quizBody"></div>
      </div>
    </section>

    <section id="view-review" class="view">
      <div class="panel">
        <h2>综合复习资料</h2>
        <p class="muted">这一页保留章节化讲义；从零开始复习建议优先看“教材目录”。</p>
        <div id="reviewBody"></div>
      </div>
    </section>
  </main>
  <script src="assets/course_data.js"></script>
  <script src="assets/course_site.js"></script>
</body>
</html>
"""


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    DATABASE.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    MINDMAP_OUT.mkdir(parents=True, exist_ok=True)
    GUIDE_OUT.mkdir(parents=True, exist_ok=True)

    standard_quiz = load_standard_quiz()
    manifest, chunks, source_cache = extract_sources()
    terms = build_term_index(chunks)
    outline = build_course_outline(manifest, chunks, concept_lookup_from_quiz(standard_quiz))
    review = load_review_or_fallback(outline)
    mindmaps = build_mindmaps(outline)
    guides = build_study_guides(outline)
    beginner_quiz = build_beginner_quiz(outline)

    for img in (ROOT / "image").glob("*.png"):
        if img.name.startswith("截屏"):
            continue
        shutil.copy2(img, IMAGE_OUT / img.name)

    write_json(DATABASE / "source_manifest.json", manifest)
    write_json(DATABASE / "source_cache.json", source_cache)
    write_json(DATABASE / "course_chunks.json", chunks)
    write_json(DATABASE / "course_terms.json", terms)
    write_json(DATABASE / "course_outline.json", outline)
    write_json(DATABASE / "review_material.json", review)
    write_json(DATABASE / "quiz_bank.json", standard_quiz)
    write_json(DATABASE / "beginner_quiz_bank.json", beginner_quiz)
    write_json(DATABASE / "mindmaps.json", mindmaps)
    write_json(DATABASE / "study_guides.json", guides)

    bundle = {
        "manifest": manifest,
        "chunks": chunks,
        "terms": terms,
        "outline": outline,
        "review": review,
        "mindmaps": mindmaps,
        "guides": guides,
        "quizBank": standard_quiz,
        "beginnerQuizBank": beginner_quiz,
    }
    (ASSETS / "course_data.js").write_text("window.COURSE_DB = " + json.dumps(bundle, ensure_ascii=False) + ";\n", encoding="utf-8")
    (ASSETS / "course_site.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (ASSETS / "course_site.js").write_text(JS.strip() + "\n", encoding="utf-8")
    (SITE / "index.html").write_text(HTML, encoding="utf-8")

    print(
        f"sources={len(manifest)} chunks={len(chunks)} terms={len(terms)} "
        f"standard_quiz={len(standard_quiz)} beginner_quiz={len(beginner_quiz)} "
        f"mindmaps={sum(len(m['maps']) for m in mindmaps)} guides={len(guides['items'])}"
    )
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
