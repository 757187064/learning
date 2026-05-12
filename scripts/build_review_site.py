#!/usr/bin/env python3
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "深度学习复习网站"
DATABASE = OUT / "database"
SITE = OUT / "site"
ASSETS = SITE / "assets"
IMAGE_OUT = SITE / "image"


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
    "PackedSequence", "1D卷积", "TCN", "IMDB情感分类",
    "Seq2Seq", "Encoder-Decoder", "Teacher Forcing", "注意力机制", "Attention",
    "Query", "Key", "Value", "Q", "K", "V", "上下文向量", "Context Vector",
    "Scaled Dot-Product Attention", "Source Mask", "Multi-Head Attention",
    "Self-Attention", "Cross-Attention", "Target Mask", "Subsequent Mask",
    "位置编码", "Positional Encoding", "Transformer", "RoPE",
]


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


def heading_level(line: str):
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not match:
        return None
    title = strip_html(match.group(2)).strip(" #")
    if not title:
        return None
    return len(match.group(1)), title


def compact(text: str, max_len: int = 900) -> str:
    text = re.sub(r"\s+", " ", strip_html(text))
    return text[:max_len].rstrip() + ("..." if len(text) > max_len else "")


def extract_notebooks():
    chunks = []
    manifest = []
    for nb_path in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])
        manifest.append({
            "file": nb_path.name,
            "size_bytes": nb_path.stat().st_size,
            "cell_count": len(cells),
            "markdown_cells": sum(1 for c in cells if c.get("cell_type") == "markdown"),
            "code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
        })
        path_stack = []
        for i, cell in enumerate(cells):
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue

            for raw_line in source.splitlines():
                parsed = heading_level(raw_line)
                if parsed:
                    level, title = parsed
                    path_stack = path_stack[: level - 1]
                    path_stack.append(title)

            clean = strip_html(source)
            if not clean:
                continue
            title = path_stack[-1] if path_stack else nb_path.stem
            terms = [term for term in KEY_TERMS if re.search(re.escape(term), clean, flags=re.I)]
            chunks.append({
                "id": f"{nb_path.stem}::cell-{i+1}",
                "file": nb_path.name,
                "cell_index": i + 1,
                "type": cell.get("cell_type", "unknown"),
                "title": title,
                "heading_path": path_stack[-4:],
                "text": clean,
                "summary": compact(clean, 680),
                "terms": terms,
            })
    return manifest, chunks


def build_term_index(chunks):
    term_data = {}
    for term in KEY_TERMS:
        hits = []
        for chunk in chunks:
            text = chunk["text"]
            count = len(re.findall(re.escape(term), text, flags=re.I))
            if count:
                hits.append({
                    "chunk_id": chunk["id"],
                    "file": chunk["file"],
                    "cell_index": chunk["cell_index"],
                    "title": chunk["title"],
                    "count": count,
                    "summary": chunk["summary"],
                })
        if hits:
            related = Counter()
            for hit in hits:
                chunk = next(c for c in chunks if c["id"] == hit["chunk_id"])
                related.update(t for t in chunk["terms"] if t != term)
            term_data[term] = {
                "term": term,
                "total_count": sum(h["count"] for h in hits),
                "files": sorted({h["file"] for h in hits}),
                "hits": hits[:80],
                "related": [t for t, _ in related.most_common(8)],
            }
    return term_data


def review_material():
    return {
        "title": "深度学习课件综合复习资料",
        "updated_from": "当前课件目录内全部 Jupyter 课件",
        "sections": [
            {
                "title": "1. 神经网络训练流程",
                "points": [
                    "训练目标是让模型参数逐步降低损失函数。典型流程是：准备数据、前向传播、计算损失、反向传播求梯度、优化器更新参数、在验证集上评估。",
                    "梯度下降是一阶迭代优化方法。学习率控制每次参数移动的步长；学习率过大容易震荡或发散，过小会收敛缓慢。",
                    "Epoch 表示完整遍历训练集一次；Batch 表示一次参数更新使用的一小批样本。小批量训练在稳定性与效率之间折中。",
                    "PyTorch 默认会累积梯度，所以每轮反向传播前需要清零梯度。训练时使用 model.train()，验证/预测时使用 model.eval() 与 torch.no_grad()。",
                    "计算图记录前向计算依赖，自动微分通过链式法则从损失反向计算各参数梯度。动态计算图的特点是边运行边构建，适合 Python 控制流。",
                ],
                "formulas": ["MSE = mean((y_hat - y)^2)", "theta = theta - eta * grad(theta)"],
                "exam_focus": ["训练循环四步/五步顺序", "为什么要清零梯度", "train/eval 模式差异", "计算图和链式法则"],
            },
            {
                "title": "2. 分类问题、Logistic、Softmax 与 MLP",
                "points": [
                    "分类任务的标签空间是离散类别。二分类常用一个 logit 经 Sigmoid 得到正类概率；多分类常用多个 logits 经 Softmax 得到概率分布。",
                    "Logit 是概率的对数几率 z=log(p/(1-p))，Sigmoid 是它的反函数。p=0.5 对应 z=0，z 越大正类概率越高。",
                    "BCEWithLogitsLoss 把 Sigmoid 与二元交叉熵合在一起，数值稳定性优于先 Sigmoid 再 BCELoss。",
                    "CrossEntropyLoss 内部包含 LogSoftmax 和负对数似然，输入应该是原始 logits，不应先手动 Softmax。",
                    "MLP 的每层通常由仿射变换加激活函数构成。仅堆叠仿射变换仍等价于一个线性变换，激活函数提供非线性表达能力。",
                ],
                "formulas": [
                    "sigmoid(z)=1/(1+e^(-z))",
                    "softmax(z_i)=e^(z_i)/sum_j e^(z_j)",
                    "BCE = -[y log(p)+(1-y)log(1-p)]",
                ],
                "exam_focus": ["Logit 与概率互转", "BCEWithLogitsLoss 和 CrossEntropyLoss 的输入", "Softmax 概率和温度/尺度效应", "激活函数的本质作用"],
            },
            {
                "title": "3. CNN 基础：卷积、填充、池化、感受野",
                "points": [
                    "CNN 针对图像利用局部连接、参数共享和平移不变性，避免 MLP 展平图像造成空间结构丢失和参数量膨胀。",
                    "深度学习框架中的卷积通常实际执行互相关运算，即卷积核不翻转。卷积核参数不是人工固定规则，而是在训练中学习得到。",
                    "输出尺寸由输入尺寸、卷积核大小、步长、填充共同决定。无填充 Valid 输出最小；Same 通过填充尽量保持尺寸；Full 输出可大于输入。",
                    "池化通过局部聚合降低空间尺寸，并增强局部平移不变性。最大池化强调强响应，平均池化保留局部平均信息。",
                    "感受野表示某层神经元能对应到原始输入的区域。层数、卷积核、池化和步长会逐步扩大感受野，多层小卷积常能以较少参数扩大有效视野。",
                ],
                "formulas": ["out = floor((n + 2p - k)/s) + 1", "RF_prev = (RF_current - 1) * stride + kernel"],
                "exam_focus": ["CNN vs MLP", "Valid/Same/Full 区别", "卷积输出尺寸计算", "感受野递推", "局部连接和参数共享"],
            },
            {
                "title": "4. 图像表示、通道与 CNN 架构",
                "points": [
                    "图像可表示为像素矩阵。灰度图通常是单通道，RGB 图像是三通道。PyTorch 常用 NCHW，TensorFlow/图像显示常见 NHWC。",
                    "多通道卷积会在每个输入通道上分别卷积后求和，输出通道数等于卷积核组数。",
                    "归一化和标准化能改善优化过程。训练集统计量必须只从训练集计算，避免验证/测试信息泄漏。",
                    "典型 CNN 块常由 Conv、BatchNorm、激活函数、Pooling 或 Dropout 组成。LeNet 展示早期卷积分类结构，AlexNet 引入更深网络、ReLU、Dropout 等现代实践。",
                    "自适应池化可以把不同尺寸的特征图统一到固定输出尺寸，便于连接分类器。",
                ],
                "formulas": ["Conv2d 参数量 = out_channels * (in_channels * k_h * k_w + bias项)"],
                "exam_focus": ["NCHW/NHWC 维度含义", "多通道卷积维度变化", "标准化统计量来源", "经典网络结构特点"],
            },
            {
                "title": "5. 训练技巧：梯度、优化器、正则化和调参",
                "points": [
                    "梯度消失会让靠近输入层的参数几乎不更新；梯度爆炸会造成损失 NaN 或训练不稳定。链式法则中的连乘是核心原因。",
                    "梯度爆炸常用梯度裁剪解决，包括按元素裁剪 Value Clipping、按范数缩放 Norm Clipping，RNN/LSTM 训练中特别常见。",
                    "Xavier 初始化适合 tanh/sigmoid 等，Kaiming 初始化适合 ReLU 系列。BatchNorm 通过批内均值方差归一化稳定中间分布，训练和推理行为不同。",
                    "Momentum 在梯度方向上引入惯性，Nesterov 先前瞻再修正；Adagrad 对稀疏特征友好但学习率会持续衰减；RMSProp 缓解 Adagrad 衰减过快；Adam 结合动量与自适应学习率。",
                    "过拟合处理包括早停、Dropout、数据增强、权重衰减、合理减小模型容量。学习率调度器可在训练停滞时降低学习率。",
                    "超参数调试建议：先看初始损失，再尝试小样本过拟合，确认训练链路无误后做学习率范围测试、粗细粒度搜索和消融实验。",
                ],
                "formulas": ["ReduceLROnPlateau: 指标停滞 patience 轮后 lr = lr * factor"],
                "exam_focus": ["梯度消失/爆炸原因与解决", "BatchNorm 训练/推理差异", "优化器对比", "Dropout train/eval 差异", "调参六步"],
            },
            {
                "title": "6. CNN 完整实践与现代架构",
                "points": [
                    "ImageFolder 要求数据按类别文件夹组织，文件夹名自动映射为类别索引，常用于自定义图像分类任务。",
                    "完整图像分类流程包括数据下载/读取、变换、DataLoader、模型定义、损失函数与优化器、训练验证、指标评估和模型保存。",
                    "类别不平衡可以从采样和损失两个角度处理：WeightedRandomSampler 改变抽样概率，BCE/CE 的 pos_weight 或 weight 改变损失贡献。",
                    "VGG 使用小卷积核堆叠加深网络；GoogLeNet/Inception 并行使用不同尺度卷积；ResNet 用残差连接缓解深层网络退化和梯度传播困难。",
                    "迁移学习通常加载预训练模型，冻结或微调特征提取层，并替换最后分类头以适配新任务。",
                ],
                "formulas": ["二分类 pos_weight 常取 负样本数 / 正样本数"],
                "exam_focus": ["ImageFolder 目录结构", "类别不平衡处理", "VGG/GoogLeNet/ResNet 特点", "迁移学习步骤"],
            },
            {
                "title": "7. RNN、GRU、LSTM 与序列建模",
                "points": [
                    "序列数据具有顺序依赖和可变长度。MLP 缺少显式时间建模，CNN 感受野有限，RNN 通过隐藏状态在时间步之间传递信息。",
                    "RNN 隐藏状态常写作 h_t=tanh(W_xh x_t + W_hh h_{t-1}+b)，目标是用隐藏状态概括当前及历史信息。",
                    "nn.RNNCell 适合手写时间步循环，nn.RNN 直接处理整个序列。batch_first=True 时输入形状为 batch、seq_len、feature。",
                    "双向 RNN 同时利用过去和未来上下文；堆叠 RNN 增强表达能力但也增加参数和训练难度。双向 RNN 中 output 最后一步不等同于 final_hidden 的完整双向表示。",
                    "GRU 用更新门和重置门控制记忆更新，结构比 LSTM 简洁。LSTM 用细胞状态、遗忘门、输入门、输出门缓解长程依赖问题。",
                    "变长序列可用 Padding 对齐，但填充值可能影响计算；Packing 让 RNN 跳过填充部分，PackedSequence 记录有效 token 的组织方式。",
                    "1D 卷积适合局部模式提取和并行计算；RNN 适合顺序依赖建模；TCN 通过因果卷积和扩张卷积扩大感受野。",
                ],
                "formulas": ["RNN: h_t=tanh(W_xh x_t + W_hh h_{t-1}+b)"],
                "exam_focus": ["RNN 输入输出形状", "双向/堆叠参数变化", "GRU 和 LSTM 门控作用", "Padding vs Packing", "1D CNN vs RNN 选择"],
                "images": ["image/gru.png", "image/gru公式.png", "image/lstm.png"],
            },
            {
                "title": "8. Seq2Seq、注意力机制与 Transformer 基础",
                "points": [
                    "Seq2Seq 解决的是“一个序列映射到另一个序列”的任务，典型例子包括机器翻译、文本摘要、对话生成等。",
                    "经典 Encoder-Decoder 架构先由 Encoder 把源序列编码，再由 Decoder 逐步生成目标序列；Teacher Forcing 是训练时常见技巧。",
                    "传统 Encoder-Decoder 的核心瓶颈是把整段源序列压缩成一个固定长度上下文向量，长序列时信息容易丢失。",
                    "注意力机制让 Decoder 在生成每个目标位置时，动态查看源序列不同位置，而不是始终依赖同一个固定上下文向量。",
                    "Q/K/V 是注意力机制的核心表示：Query 表示“当前要找什么”，Key 表示“各位置能匹配什么”，Value 表示“各位置携带什么信息”。",
                    "Scaled Dot-Product Attention 先计算 Q 和 K 的相似度，再经 Softmax 得到权重，最后对 V 加权求和形成上下文向量。",
                    "除以 sqrt(d_k) 是为了抑制高维点积过大，避免 Softmax 过度尖锐，导致梯度不稳定。",
                    "Source Mask 用于屏蔽源序列中的 Padding 位置；Target Mask 或 Subsequent Mask 用于防止 Decoder 偷看未来词。",
                    "Multi-Head Attention 通过多个头并行关注不同子空间和不同关系，再拼接汇总，提高表达能力。",
                    "Self-Attention 让序列内部位置彼此交互；Cross-Attention 则常见于 Decoder 查询 Encoder 输出。",
                    "Transformer 的关键突破之一是用 Self-Attention 替代循环结构，并结合位置编码补回顺序信息。",
                    "Sinusoidal Positional Encoding 用不同频率的 sin/cos 为每个位置编码；RoPE 则通过旋转方式注入相对位置信息。",
                ],
                "formulas": [
                    "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
                    "score(q, k) = q · k",
                    "PE(pos, 2i) = sin(pos / 10000^(2i/d_model))",
                    "PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))",
                ],
                "exam_focus": [
                    "Seq2Seq 与序列分类/图像分类的区别",
                    "Encoder-Decoder 瓶颈问题",
                    "Q/K/V 各自含义与来源",
                    "为什么除以 sqrt(d_k)",
                    "Source Mask 与 Target Mask 的作用",
                    "Self-Attention、Cross-Attention、Multi-Head Attention 的区别",
                    "为什么 Transformer 需要位置编码",
                ],
            },
        ],
    }


def q_mc(id_, topic, stem, options, answer, explanation, source, difficulty="中"):
    return {"id": id_, "type": "mcq", "topic": topic, "stem": stem, "options": options, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_fill(id_, topic, stem, answer, explanation, source, difficulty="中"):
    return {"id": id_, "type": "fill", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def q_short(id_, topic, stem, answer, explanation, source, difficulty="中"):
    return {"id": id_, "type": "short", "topic": topic, "stem": stem, "answer": answer, "explanation": explanation, "source": source, "difficulty": difficulty}


def quiz_bank():
    mcq = [
        q_mc("M01", "训练流程 梯度下降", "关于梯度下降训练流程，下列说法错误的是", ["A. 前向传播用于得到模型预测值", "B. 损失函数用于衡量预测与真实标签差异", "C. 反向传播的核心目标是计算参数梯度", "D. 参数更新应在清零梯度之前完成"], "D", "A、B、C均符合标准训练流程；D错误，通常每轮先清零梯度，再前向、计算损失、反向传播、优化器更新，否则旧梯度会参与本轮计算。", "多层感知机1_学生分发版.ipynb", "易"),
        q_mc("M02", "学习率", "关于学习率的说法错误的是", ["A. 学习率控制参数沿梯度方向更新的步长", "B. 学习率过大可能导致震荡或发散", "C. 学习率越小训练一定越快且效果越好", "D. 学习率可以配合调度器动态调整"], "C", "C错误：学习率过小会导致收敛很慢，甚至看起来训练停滞。A、B、D都是课件中的实践要点。", "多层感知机1_学生分发版.ipynb", "易"),
        q_mc("M03", "自动微分 计算图", "关于计算图和自动微分，下列说法错误的是", ["A. 计算图记录前向计算中的依赖关系", "B. 反向传播依赖链式法则计算梯度", "C. PyTorch动态图可以根据实际运行路径构建", "D. 计算图只用于保存输入数据，不能用于求梯度"], "D", "D错误：计算图的关键用途就是支持自动求导。A、B、C分别对应依赖记录、链式法则和动态图特点。", "多层感知机1_学生分发版.ipynb"),
        q_mc("M04", "Logistic Sigmoid", "关于Logit与Sigmoid，下列说法错误的是", ["A. logit(p)=log(p/(1-p))", "B. Sigmoid可以把任意实数映射到(0,1)", "C. p=0.5时logit为0", "D. z越大，Sigmoid(z)越接近0"], "D", "D错误：z越大，Sigmoid(z)越接近1。A、B、C是Logit/Sigmoid的基本定义和性质。", "多层感知机2-学生分发版-终版2026.ipynb", "易"),
        q_mc("M05", "Softmax 交叉熵", "关于PyTorch多分类训练，下列说法错误的是", ["A. CrossEntropyLoss通常接收原始logits", "B. CrossEntropyLoss内部包含LogSoftmax和NLLLoss思想", "C. 训练前应总是手动对logits做Softmax再传入CrossEntropyLoss", "D. 预测阶段可以用torch.softmax查看各类别概率"], "C", "C错误：CrossEntropyLoss应输入原始logits，提前Softmax会改变数值稳定性和预期输入。A、B、D正确。", "多层感知机2-学生分发版-终版2026.ipynb"),
        q_mc("M06", "BCEWithLogitsLoss", "关于二分类损失函数，下列说法错误的是", ["A. BCEWithLogitsLoss合并了Sigmoid和BCE", "B. BCEWithLogitsLoss通常比Sigmoid后接BCELoss更数值稳定", "C. BCEWithLogitsLoss的输入通常是概率值而不是logit", "D. 二元交叉熵会强烈惩罚高置信度错误预测"], "C", "C错误：BCEWithLogitsLoss的输入是原始logit，函数内部处理Sigmoid相关计算。A、B、D正确。", "多层感知机2-学生分发版-终版2026.ipynb"),
        q_mc("M07", "MLP 激活函数", "关于MLP和激活函数，下列说法错误的是", ["A. 仿射变换包含加权求和与偏置", "B. 激活函数提供非线性映射能力", "C. 没有激活函数时多层线性层堆叠仍等价于线性变换", "D. 只要增加全连接层数，即使无激活函数也能拟合任意非线性边界"], "D", "D错误：没有非线性激活，多层仿射变换仍可合并成一个仿射变换，不能表达复杂非线性。A、B、C正确。", "多层感知机2-学生分发版-终版2026.ipynb", "易"),
        q_mc("C01", "CNN MLP", "关于全连接网络处理图像的缺陷与CNN特点，下列说法错误的是", ["A. MLP展平图像会破坏空间结构", "B. 全连接处理高分辨率图像容易参数量巨大", "C. CNN通过局部连接和参数共享减少参数", "D. CNN必须采用全层全连接结构才能保留平移不变性"], "D", "D错误：CNN核心是局部连接、权值共享与池化等机制，不是全层全连接。A、B、C正确。", "卷积神经网络1-学生分发版-终版2026.ipynb", "易"),
        q_mc("C02", "卷积 互相关", "关于深度学习中的卷积操作，下列说法错误的是", ["A. 实际框架中常用互相关运算，不翻转卷积核", "B. 卷积核参数通常通过训练学习得到", "C. 卷积核在图像上滑动提取局部模式", "D. 互相关不能表达特征匹配，因此CNN必须使用严格数学卷积"], "D", "D错误：CNN中互相关已能完成局部模板匹配，并且省去卷积核翻转。A、B、C正确。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_mc("C03", "卷积尺寸", "输入尺寸10×10，卷积核3×3，步长1，无填充，输出尺寸为", ["A. 12×12", "B. 10×10", "C. 8×8", "D. 7×7"], "C", "按公式out=floor((n+2p-k)/s)+1=(10+0-3)/1+1=8，因此输出为8×8。", "卷积神经网络1-学生分发版-终版2026.ipynb", "易"),
        q_mc("C04", "Valid Same Full", "关于Valid、Same、Full卷积，下列说法错误的是", ["A. Valid卷积通常不填充，输出尺寸较小", "B. Same卷积通过合适填充尽量保持输入输出空间尺寸一致", "C. Full卷积会进行更充分的边界填充，输出可大于输入", "D. Valid卷积会大量填充边界，因此输出最大"], "D", "D错误：Valid不进行填充或最少填充，输出通常最小；Full才会让输出更大。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_mc("C05", "感受野", "关于感受野，下列说法错误的是", ["A. 感受野表示某层神经元对应到原始输入的区域大小", "B. 更深层神经元通常拥有更大的原始输入感受野", "C. 增大步长在递推公式中会扩大回溯得到的前层感受野", "D. 堆叠多层小卷积核无法扩大感受野"], "D", "D错误：多层小卷积核堆叠可以逐步扩大感受野，同时常比单层大卷积参数更少。A、B、C正确。", "卷积神经网络1-课堂练习-终版2026.ipynb"),
        q_mc("C06", "池化", "关于池化操作，下列说法错误的是", ["A. 池化可以降低特征图空间尺寸", "B. 最大池化保留局部区域中响应最强的值", "C. 池化有助于获得一定局部平移不变性", "D. 池化层的参数量通常大于卷积层"], "D", "D错误：常见池化操作没有需要学习的权重参数。A、B、C正确。", "卷积神经网络1-学生分发版-终版2026.ipynb", "易"),
        q_mc("C07", "NCHW NHWC", "关于图像张量格式，下列说法错误的是", ["A. NCHW通常表示批量、通道、高、宽", "B. NHWC通常表示批量、高、宽、通道", "C. PyTorch卷积常使用NCHW格式", "D. NCHW与NHWC只是名字不同，内存维度顺序完全相同"], "D", "D错误：二者维度顺序不同，代码中需要显式permute/transpose转换。A、B、C正确。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_mc("C08", "标准化 数据泄漏", "关于图像标准化，下列说法错误的是", ["A. 标准化常用均值和标准差", "B. 统计量应只从训练集计算", "C. 使用验证集和测试集共同计算统计量可能造成数据泄漏", "D. 标准化一定会改变标签类别含义"], "D", "D错误：标准化改变输入数值尺度，不改变标签类别含义。B、C是避免数据泄漏的关键。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_mc("C09", "类别不平衡", "关于类别不平衡处理，下列说法错误的是", ["A. WeightedRandomSampler可以提高少数类被采样概率", "B. pos_weight可用于二分类中提高正类损失权重", "C. 多分类CrossEntropyLoss可使用weight设置类别权重", "D. 类别不平衡只能通过删除多数类样本解决"], "D", "D错误：采样、损失加权、数据增强等都能处理类别不平衡，不只能删除样本。A、B、C正确。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_mc("T01", "梯度消失 梯度爆炸", "关于梯度消失与梯度爆炸，下列说法错误的是", ["A. 梯度消失会让靠近输入层的参数更新很慢", "B. 梯度爆炸可能导致损失出现NaN", "C. 二者都与反向传播链式法则中的连乘有关", "D. 梯度爆炸的唯一原因是batch size过小"], "D", "D错误：梯度爆炸可由权重尺度、学习率过高、长序列连乘等多因素导致，并非只有batch size。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T02", "梯度裁剪", "关于梯度裁剪，下列说法错误的是", ["A. Value Clipping按元素限制梯度取值范围", "B. Norm Clipping按整体范数缩放梯度", "C. 梯度裁剪常用于缓解梯度爆炸", "D. 梯度裁剪会完全解决所有过拟合问题"], "D", "D错误：梯度裁剪主要处理梯度爆炸或训练不稳定，不是过拟合的完整解决方案。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T03", "BatchNorm", "关于BatchNorm，下列说法错误的是", ["A. 训练时通常使用当前batch统计量", "B. 推理时通常使用训练过程中累计的移动平均统计量", "C. BN可稳定中间层分布并加速训练", "D. BN在训练和推理时行为完全相同"], "D", "D错误：BatchNorm训练和推理模式行为不同，因此需要正确切换model.train()/model.eval()。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T04", "初始化", "关于参数初始化，下列说法错误的是", ["A. Xavier初始化常与tanh/sigmoid等激活配合", "B. Kaiming初始化常与ReLU系列激活配合", "C. 合理初始化可缓解梯度消失或爆炸", "D. 初始化方式不影响深层网络训练稳定性"], "D", "D错误：初始化会显著影响激活和梯度尺度，是深层网络训练稳定性的关键因素。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T05", "优化器", "关于优化器，下列说法错误的是", ["A. Momentum通过速度项积累历史梯度方向", "B. Adagrad会为频繁更新的参数降低有效学习率", "C. Adam结合了动量和自适应学习率思想", "D. 所有优化器在任何任务上效果完全相同"], "D", "D错误：不同优化器有不同假设与动态，实际效果依任务、学习率和超参数而变化。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T06", "Dropout", "关于Dropout，下列说法错误的是", ["A. 训练时随机失活部分神经元", "B. 可作为正则化手段缓解过拟合", "C. 推理时通常不再随机丢弃神经元", "D. Dropout在train和eval模式下表现完全一样"], "D", "D错误：Dropout训练时随机失活，评估/推理时关闭随机失活。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T07", "学习率调度", "关于ReduceLROnPlateau调度器，下列说法错误的是", ["A. 它可以根据验证损失等指标是否停滞来降低学习率", "B. patience表示容忍多少轮没有改善", "C. factor表示学习率缩放倍数", "D. 它必须在每个batch之前调用且不需要监测指标"], "D", "D错误：ReduceLROnPlateau通常在一个epoch验证后用监测指标调用，例如scheduler.step(val_loss)。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_mc("T08", "调参 消融实验", "关于超参数实践与消融实验，下列说法错误的是", ["A. 小样本过拟合可以检查训练链路是否可学习", "B. 消融实验应尽量一次只改变一个变量", "C. 只看最终指标可能忽略训练动态和成本", "D. 粗粒度搜索前无需确认损失是否能下降"], "D", "D错误：调参前应先确认基本训练链路可用、损失能下降，否则搜索会浪费时间。", "卷积神经网络2-学生分发版-终版2026(1).ipynb", "难"),
        q_mc("A01", "ImageFolder", "关于ImageFolder，下列说法错误的是", ["A. 它适合按文件夹组织的图像分类数据集", "B. 子文件夹名称可自动作为类别名", "C. 它会将类别映射为整数索引", "D. 所有图片必须放在同一个文件夹且不能有类别子目录"], "D", "D错误：ImageFolder正是依赖类别子目录组织数据。A、B、C正确。", "卷积神经网络3-学生分发版-终版2026.ipynb"),
        q_mc("A02", "VGG GoogLeNet ResNet", "关于现代CNN结构，下列说法错误的是", ["A. VGG偏向使用小卷积核堆叠构建深网络", "B. GoogLeNet/Inception利用多分支提取不同尺度特征", "C. ResNet通过残差连接改善深层网络训练", "D. ResNet的核心思想是完全删除跳跃连接"], "D", "D错误：ResNet的标志正是残差/跳跃连接。A、B、C正确。", "卷积神经网络3-学生分发版-终版2026.ipynb"),
        q_mc("A03", "迁移学习", "关于迁移学习，下列说法错误的是", ["A. 可以加载在大数据集上预训练的模型", "B. 常见做法是替换最后分类头以适配新类别数", "C. 可以冻结部分特征提取层再训练分类器", "D. 迁移学习要求新任务类别必须与原任务完全相同"], "D", "D错误：迁移学习正是利用通用特征迁移到相关但不完全相同的新任务。", "卷积神经网络3-学生分发版-终版2026.ipynb"),
        q_mc("R01", "RNN 序列", "关于序列数据和RNN，下列说法错误的是", ["A. 序列数据的数据点之间存在顺序关系", "B. RNN通过隐藏状态传递历史信息", "C. MLP天然适合任意长度序列且无需额外处理", "D. RNN可用于文本、时间序列等任务"], "C", "C错误：MLP输入维度固定，缺少显式时间状态机制，不能天然处理任意变长序列。", "循环神经网络-学生分发版-终版2026.ipynb", "易"),
        q_mc("R02", "RNN Shape", "batch_first=True时，PyTorch RNN常见输入形状是", ["A. (seq_len, batch, feature)", "B. (batch, seq_len, feature)", "C. (feature, batch, seq_len)", "D. (batch, feature, seq_len)"], "B", "batch_first=True表示batch维度在最前，因此输入是(batch, seq_len, feature)。", "循环神经网络-学生分发版-终版2026.ipynb", "易"),
        q_mc("R03", "双向RNN", "关于双向RNN，下列说法错误的是", ["A. 双向RNN同时包含前向和后向序列处理", "B. 双向RNN输出特征维度通常会因方向拼接而翻倍", "C. 对双向RNN，output[:, -1]总是等价于完整final_hidden", "D. 双向结构可利用左右上下文信息"], "C", "C错误：课件强调双向RNN中output最后一步不等同于完整final_hidden，因为后向方向的最后状态对应序列开头侧。", "循环神经网络-学生分发版-终版2026.ipynb", "难"),
        q_mc("R04", "GRU LSTM", "关于GRU和LSTM，下列说法错误的是", ["A. GRU使用门控机制控制信息保留与更新", "B. LSTM包含细胞状态以帮助长期记忆", "C. LSTM常包含遗忘门、输入门和输出门", "D. GRU和LSTM都完全没有参数，因此不会过拟合"], "D", "D错误：GRU/LSTM都有可学习参数，也可能过拟合。A、B、C是门控循环网络核心。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_mc("R05", "Padding Packing", "关于Padding与Packing，下列说法错误的是", ["A. Padding把变长序列补齐到同一长度", "B. 填充值若直接参与RNN计算可能影响隐藏状态", "C. Packing可以告知RNN跳过填充部分", "D. Packing的目的就是增加更多无效填充计算"], "D", "D错误：Packing的目的正是减少/避免填充部分参与有效计算。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_mc("R06", "1D卷积 TCN", "关于1D卷积和RNN用于序列建模，下列说法错误的是", ["A. 1D卷积可并行提取局部时序模式", "B. RNN按时间步递推隐藏状态", "C. TCN可通过扩张卷积扩大感受野", "D. 1D卷积必须逐时间步串行计算，无法并行"], "D", "D错误：卷积相对RNN的重要优势之一就是更容易并行计算。", "循环神经网络-学生分发版-终版2026.ipynb"),
    ]
    fill = [
        q_fill("F01", "梯度下降", "梯度下降中，控制每次参数更新步长的超参数叫做____。", "学习率", "学习率越大参数移动越激进，过大可能发散，过小会收敛慢。", "多层感知机1_学生分发版.ipynb", "易"),
        q_fill("F02", "PyTorch训练", "PyTorch中每轮反向传播前通常需要调用优化器的____来避免梯度累积。", "zero_grad()", "PyTorch默认梯度累积，不清零会把上一次梯度叠加到本轮。", "多层感知机1_学生分发版.ipynb", "易"),
        q_fill("F03", "Sigmoid", "概率p的logit定义为____。", "log(p/(1-p))", "Logit是几率p/(1-p)的自然对数，Sigmoid是其反函数。", "多层感知机2-学生分发版-终版2026.ipynb"),
        q_fill("F04", "Softmax", "Softmax输出的所有类别概率之和为____。", "1", "Softmax把多个logits归一化为概率分布，因此各类别概率和为1。", "多层感知机2-学生分发版-终版2026.ipynb", "易"),
        q_fill("F05", "MLP", "没有激活函数时，多层仿射变换堆叠仍等价于一个____变换。", "仿射/线性", "仿射变换的复合仍是仿射变换，无法产生非线性决策边界。", "多层感知机2-学生分发版-终版2026.ipynb"),
        q_fill("F06", "卷积尺寸", "卷积输出尺寸常用公式为 out=floor((n+2p-k)/s)+____。", "1", "n为输入尺寸，p为填充，k为核大小，s为步长。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_fill("F07", "CNN", "CNN的三个核心特点常概括为局部连接、____和一定的平移不变性。", "权值共享/参数共享", "同一卷积核在空间位置上共享参数，是CNN参数效率的关键。", "卷积神经网络1-学生分发版-终版2026.ipynb", "易"),
        q_fill("F08", "感受野", "感受野递推公式中，RF_prev=(RF_current-1)*stride+____。", "kernel/kernel size/卷积核大小", "回溯到前一层时，当前感受野间隔由步长放大，再加本层核大小。", "卷积神经网络1-课堂练习-终版2026.ipynb"),
        q_fill("F09", "BatchNorm", "BatchNorm在推理阶段通常使用训练过程中累计的____统计量。", "移动平均/运行均值和方差", "训练时用batch统计量，推理时用running mean/var，模式切换很关键。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_fill("F10", "Dropout", "Dropout在____模式下随机失活神经元，在评估模式下关闭随机失活。", "训练/train", "这也是为什么验证和预测前要调用model.eval()。", "卷积神经网络2-学生分发版-终版2026.ipynb", "易"),
        q_fill("F11", "梯度裁剪", "按整体梯度范数进行缩放的梯度裁剪方法叫____ Clipping。", "Norm", "Norm Clipping在整体范数超过阈值时等比例缩放梯度。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_fill("F12", "ImageFolder", "ImageFolder通过数据根目录下的____名称自动确定类别。", "子文件夹/类别文件夹", "每个类别一个子文件夹，文件夹名映射为类别标签。", "卷积神经网络3-学生分发版-终版2026.ipynb"),
        q_fill("F13", "ResNet", "ResNet缓解深层网络训练困难的关键结构是____连接。", "残差/跳跃/skip", "残差连接让网络学习F(x)+x，改善梯度和信息传播。", "卷积神经网络3-学生分发版-终版2026.ipynb"),
        q_fill("F14", "RNN", "RNN通过时间步之间传递____状态来保留历史信息。", "隐藏", "隐藏状态h_t概括当前输入和过去信息。", "循环神经网络-学生分发版-终版2026.ipynb", "易"),
        q_fill("F15", "RNN Shape", "batch_first=True时，RNN输入形状为(batch, ____, feature)。", "seq_len/序列长度", "batch_first只改变batch维度位置，序列长度仍是中间维。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_fill("F16", "LSTM", "LSTM相比普通RNN的核心长期记忆通道是____状态。", "细胞/cell", "细胞状态配合门控机制帮助长期信息保留。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_fill("F17", "Packing", "PackedSequence常用于让RNN跳过____序列中的填充部分。", "变长", "变长序列padding后会有无效位置，packing记录真实长度以避免无效计算。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_fill("F18", "类别不平衡", "二分类中pos_weight常可按____样本数/正样本数估计。", "负", "pos_weight提高正类样本的损失贡献，常按负正样本数量比设置。", "卷积神经网络1-学生分发版-终版2026.ipynb", "难"),
    ]
    short = [
        q_short("S01", "训练流程", "简述一个PyTorch监督学习训练循环的核心步骤。", "通常包括：把模型置为训练模式；遍历batch；清零梯度；前向传播得到预测；计算损失；反向传播计算梯度；优化器更新参数；记录训练损失。验证阶段切换到eval并使用no_grad。", "该题考查完整流程，不要求写代码但需要顺序正确，尤其是zero_grad、backward、step和train/eval切换。", "多层感知机1_学生分发版.ipynb"),
        q_short("S02", "损失函数", "为什么PyTorch中多分类通常直接把logits传给CrossEntropyLoss？", "CrossEntropyLoss内部已经完成LogSoftmax和负对数似然相关计算，直接输入logits数值更稳定；如果先手动Softmax，会让输入不符合预期并可能损害梯度与稳定性。", "关键点是“原始logits”和“内部包含softmax思想但实现更稳定”。", "多层感知机2-学生分发版-终版2026.ipynb"),
        q_short("S03", "CNN", "说明CNN相对MLP处理图像的主要优势。", "CNN保留局部空间结构，通过局部连接减少连接范围，通过参数共享减少参数量，并借助卷积/池化对局部平移具有更强鲁棒性。MLP展平图像会破坏行列邻接关系且参数量巨大。", "该题来源于CNN引入部分，需同时答出空间结构、参数效率和平移不变性。", "卷积神经网络1-学生分发版-终版2026.ipynb", "易"),
        q_short("S04", "感受野", "解释感受野的含义，并说明为什么多层小卷积核可以扩大感受野。", "感受野是某层神经元能够对应到原始输入图像上的区域。每经过一层卷积或池化，下一层的一个位置都汇聚了前一层一定邻域的信息；层层堆叠后，高层神经元间接覆盖更大的原始区域。小卷积核多层堆叠还能在扩大感受野的同时减少参数并增加非线性。", "难点在于不要只说“层数越深越大”，还要说明逐层聚合和参数效率。", "卷积神经网络1-课堂练习-终版2026.ipynb", "难"),
        q_short("S05", "梯度问题", "比较梯度消失和梯度爆炸，并各给出一种常见缓解方法。", "梯度消失是反向传播时梯度逐层变小，浅层参数难以更新；可用合适初始化、ReLU类激活、BatchNorm或残差结构缓解。梯度爆炸是梯度逐层变大导致训练不稳定或NaN；可用梯度裁剪、降低学习率、合理初始化等缓解。", "解析重点是二者方向相反但都来自链式法则连乘，并能对应解决方案。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_short("S06", "BatchNorm Dropout", "说明BatchNorm和Dropout在train/eval模式下为什么需要不同处理。", "BatchNorm训练时使用当前batch均值方差并更新运行统计量，推理时使用训练得到的运行均值方差；Dropout训练时随机失活以正则化，推理时关闭随机失活以使用完整网络。因此训练、验证、部署必须正确切换模式。", "这题常用于检查对model.train()和model.eval()实际影响的理解。", "卷积神经网络2-学生分发版-终版2026.ipynb"),
        q_short("S07", "类别不平衡", "类别不平衡时，采样加权和损失加权分别解决什么问题？", "采样加权如WeightedRandomSampler改变每个样本进入batch的概率，让少数类更常被看到；损失加权如pos_weight或class weight改变不同类别错误的损失贡献，让模型对少数类错误付出更高代价。二者可以单独或配合使用。", "需要区分“数据进入训练的频率”和“错误产生的惩罚大小”。", "卷积神经网络1-学生分发版-终版2026.ipynb"),
        q_short("S08", "现代CNN", "概括VGG、GoogLeNet和ResNet的核心设计差异。", "VGG强调小卷积核堆叠形成深网络；GoogLeNet/Inception在同一层并行使用多尺度分支以捕获不同感受野特征；ResNet通过残差/跳跃连接改善深层网络的优化和梯度传播。", "这题要求按网络分别给关键词，不需要展开全部结构细节。", "卷积神经网络3-学生分发版-终版2026.ipynb"),
        q_short("S09", "RNN", "为什么普通RNN容易遇到长期依赖问题？", "普通RNN在反向传播中需要沿时间步反复乘以状态转移相关梯度。若这些乘积的范数长期小于1，梯度会消失；大于1则可能爆炸。长距离信息很难稳定传回早期时间步，因此难以学习长期依赖。", "核心是时间维度上的链式法则连乘，不只是“序列太长”。", "循环神经网络-学生分发版-终版2026.ipynb", "难"),
        q_short("S10", "GRU LSTM", "简述GRU和LSTM如何缓解普通RNN的长期依赖问题。", "GRU通过更新门和重置门控制旧信息保留、新信息写入和历史信息使用；LSTM通过细胞状态提供较稳定的信息通路，并用遗忘门、输入门、输出门控制信息流。门控机制让模型学习何时记住、遗忘或输出信息。", "答题重点是“门控控制信息流”，而不是简单说“更复杂”。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_short("S11", "Padding Packing", "说明Padding和Packing在变长序列处理中的区别。", "Padding把不同长度序列补到同一长度，便于组成batch，但填充值可能参与计算并浪费算力；Packing记录每条序列真实长度，使RNN只处理有效时间步，跳过填充部分，常与pack_padded_sequence或pack_sequence配合。", "关键是Padding解决形状统一，Packing解决无效填充计算和隐藏状态污染。", "循环神经网络-学生分发版-终版2026.ipynb"),
        q_short("S12", "1D卷积", "什么情况下可以优先考虑1D卷积或TCN而不是RNN？", "当任务更依赖局部时序模式、需要高并行效率，或可以通过扩张卷积获得足够感受野时，可考虑1D卷积/TCN。RNN更适合强调顺序递推状态的场景，但并行性较弱。", "该题考查模型选型，不是要求绝对优劣，而是结合局部模式、长程依赖和并行效率。", "循环神经网络-学生分发版-终版2026.ipynb"),
    ]
    return mcq + fill + short


def supplemental_concepts():
    return [
        ("训练/前向传播", "多层感知机1_学生分发版.ipynb", "前向传播", "前向传播把输入依次送过模型各层，得到预测或logits，是计算损失之前的步骤。", "前向传播会自动修改模型参数，不需要反向传播和优化器。", "考试常把前向传播、反向传播、参数更新顺序打乱，先找是否缺少zero_grad、backward或step。", "易"),
        ("训练/损失函数", "多层感知机1_学生分发版.ipynb", "损失函数", "损失函数衡量预测值与真实标签之间的差异，是反向传播求梯度的标量目标。", "损失函数只用于测试阶段，训练阶段不需要计算损失。", "命题常问MSE、BCE、CE分别适配什么任务，以及损失越小是否代表训练目标更好。", "易"),
        ("训练/反向传播", "多层感知机1_学生分发版.ipynb", "反向传播", "反向传播基于链式法则计算损失对各参数的梯度。", "反向传播的主要作用是把验证集准确率直接写入模型参数。", "遇到计算图题时，抓住从loss沿依赖关系反向求导这一点。", "中"),
        ("训练/优化器step", "多层感知机1_学生分发版.ipynb", "optimizer.step()", "optimizer.step()根据当前梯度和优化算法更新参数。", "optimizer.step()用于清空梯度，因此应替代zero_grad。", "代码题常考zero_grad、loss.backward、optimizer.step三者的顺序。", "易"),
        ("训练/梯度累积", "多层感知机1_学生分发版.ipynb", "梯度累积", "PyTorch默认把梯度累积到参数的grad属性中，不清零会叠加旧梯度。", "PyTorch每次backward后都会自动清空旧梯度。", "如果题目问训练异常、梯度越来越大，先检查是否忘记zero_grad。", "中"),
        ("训练/model.train", "多层感知机1_学生分发版.ipynb", "model.train()", "model.train()让Dropout、BatchNorm等模块进入训练行为。", "model.train()会自动执行一轮训练并更新参数。", "train/eval是模块行为开关，不是训练循环本身。", "易"),
        ("训练/model.eval", "多层感知机1_学生分发版.ipynb", "model.eval()", "model.eval()让Dropout关闭随机失活、BatchNorm使用运行统计量。", "model.eval()会禁止所有梯度计算，因此不需要torch.no_grad。", "eval改变层行为，no_grad控制是否记录梯度，两者职责不同。", "中"),
        ("训练/torch.no_grad", "多层感知机1_学生分发版.ipynb", "torch.no_grad()", "torch.no_grad()在验证或推理时关闭梯度记录，节省显存并加速。", "torch.no_grad()会把模型永久切换成评估模式。", "考试常把no_grad与eval混淆：一个管梯度记录，一个管层行为。", "中"),
        ("训练/Epoch", "多层感知机1_学生分发版.ipynb", "Epoch", "一个Epoch表示完整遍历训练集一次。", "一个Epoch表示只处理一个样本或一个batch。", "题目若出现batch、iteration、epoch，注意三者粒度不同。", "易"),
        ("训练/Batch", "多层感知机1_学生分发版.ipynb", "Batch", "Batch是一小批样本，一次迭代通常基于一个batch计算梯度并更新参数。", "Batch越大模型一定泛化越好。", "Batch大小影响显存、梯度噪声和训练稳定性，不是越大越好。", "中"),
        ("数据/DataLoader", "多层感知机1_学生分发版.ipynb", "DataLoader", "DataLoader负责按batch迭代数据，可配合shuffle实现训练集随机打乱。", "DataLoader本身会定义神经网络结构。", "训练集常shuffle，验证/测试通常不需要shuffle。", "易"),
        ("数据/random_split", "多层感知机1_学生分发版.ipynb", "random_split", "random_split可把数据集划分为训练集、验证集等子集。", "random_split会自动保证类别分布完全一致。", "如果题目强调类别比例，应考虑stratify或分层划分，而不是默认随机划分。", "中"),
        ("计算图/动态计算图", "多层感知机1_学生分发版.ipynb", "动态计算图", "PyTorch动态图按实际执行路径构建，适合包含Python控制流的模型。", "动态计算图必须在训练前完整静态声明，运行中不能变化。", "考点是define-by-run：运行到哪里，图就按实际操作建到哪里。", "中"),
        ("自动微分/requires_grad", "多层感知机1_学生分发版.ipynb", "requires_grad", "requires_grad=True表示需要跟踪该张量相关操作以便求梯度。", "requires_grad=True会让张量自动成为优化器。", "需要优化的参数必须能产生梯度，但还要交给优化器管理。", "中"),
        ("自动微分/叶子张量", "多层感知机1_学生分发版.ipynb", "叶子张量", "用户直接创建且需要梯度的参数通常是叶子张量，梯度保存在其grad中。", "所有中间张量默认都会长期保存grad。", "命题会问为什么某些中间变量grad为空，默认只保留叶子张量梯度。", "难"),
        ("分类/回归 vs 分类", "多层感知机2-学生分发版-终版2026.ipynb", "分类", "分类任务输出离散类别，回归任务输出连续数值。", "分类任务的标签空间必须是连续实数区间。", "先判断任务类型，再匹配输出层和损失函数。", "易"),
        ("分类/二分类", "多层感知机2-学生分发版-终版2026.ipynb", "二分类", "二分类常用一个logit表示正类倾向，再通过Sigmoid得到概率。", "二分类必须输出两个经过Softmax的概率，不能用单logit。", "单logit+BCEWithLogitsLoss是常见二分类配置。", "中"),
        ("分类/多分类", "多层感知机2-学生分发版-终版2026.ipynb", "多分类", "多分类通常输出C个logits，每个logit对应一个类别分数。", "多分类CrossEntropyLoss要求目标标签一定是one-hot且不能是类别索引。", "PyTorch中类别索引是最常见、效率较高的CE目标格式。", "中"),
        ("Logistic/Odds", "多层感知机2-学生分发版-终版2026.ipynb", "Odds", "Odds表示p/(1-p)，描述事件发生概率与不发生概率之比。", "Odds与概率p完全相同，没有任何变换。", "概率、odds、logit互转是填空和计算题高频点。", "中"),
        ("Logistic/Logit", "多层感知机2-学生分发版-终版2026.ipynb", "Logit", "Logit是log(p/(1-p))，把(0,1)概率映射到全体实数。", "Logit只能取0到1之间的数。", "p=0.5时logit=0；p越接近1，logit越大。", "易"),
        ("Logistic/Sigmoid", "多层感知机2-学生分发版-终版2026.ipynb", "Sigmoid", "Sigmoid把实数logit映射到(0,1)，可解释为二分类正类概率。", "Sigmoid输出可以小于0或大于1。", "题目若问概率输出区间，Sigmoid是(0,1)。", "易"),
        ("损失/BCE", "多层感知机2-学生分发版-终版2026.ipynb", "BCE", "二元交叉熵衡量二分类概率预测与0/1标签的差异。", "BCE专门用于多分类互斥C类任务且输入必须是C个logits。", "二分类用BCE，多分类互斥用CE，别混。", "中"),
        ("损失/BCEWithLogitsLoss", "多层感知机2-学生分发版-终版2026.ipynb", "BCEWithLogitsLoss", "BCEWithLogitsLoss把Sigmoid和BCE合并，输入是未归一化logits。", "BCEWithLogitsLoss要求先手动Sigmoid得到概率再输入。", "官方文档也强调输入是logits，考试常以此设陷阱。", "中"),
        ("损失/CrossEntropyLoss", "多层感知机2-学生分发版-终版2026.ipynb", "CrossEntropyLoss", "CrossEntropyLoss用于多分类时通常接收未归一化logits和类别索引目标。", "CrossEntropyLoss输入必须已经经过Softmax且总和为1。", "官方文档明确输入是logits；预测看概率时才softmax。", "中"),
        ("Softmax/概率分布", "多层感知机2-学生分发版-终版2026.ipynb", "Softmax", "Softmax把多个logits转换为和为1的概率分布。", "Softmax会把所有类别概率独立地映射到(0,1)，但总和不受限制。", "注意Softmax不是逐元素Sigmoid，类别之间通过分母耦合。", "中"),
        ("Softmax/尺度效应", "多层感知机2-课堂练习答案-终版2026.ipynb", "尺度效应", "放大logits会让Softmax分布更尖锐，缩小logits会让分布更平滑。", "所有logits乘以10后，Softmax一定变成完全均匀分布。", "课堂练习已考过，常用数值表格变式。", "中"),
        ("MLP/仿射变换", "多层感知机2-学生分发版-终版2026.ipynb", "仿射变换", "仿射变换是线性加权求和再加偏置。", "仿射变换本身就能提供任意复杂非线性边界。", "看到线性层堆叠无激活，结论仍是线性/仿射。", "易"),
        ("MLP/激活函数", "多层感知机2-学生分发版-终版2026.ipynb", "激活函数", "激活函数提供非线性，使网络能够拟合非线性关系。", "激活函数的作用只是减少参数量，与非线性无关。", "例题同款高频：无激活的多层网络表达力不会变成非线性。", "易"),
        ("MLP/ReLU", "多层感知机2-学生分发版-终版2026.ipynb", "ReLU", "ReLU常写作max(0,x)，正半轴梯度稳定，计算简单。", "ReLU会把所有正数都压缩到0到1之间。", "ReLU不是概率函数，不要和Sigmoid混淆。", "易"),
        ("CNN/局部连接", "卷积神经网络1-学生分发版-终版2026.ipynb", "局部连接", "CNN每个输出位置只连接输入的局部区域，保留空间邻域结构。", "CNN每个神经元必须连接整张图像所有像素。", "CS231n也把局部连接作为Conv层核心假设之一。", "易"),
        ("CNN/参数共享", "卷积神经网络1-学生分发版-终版2026.ipynb", "参数共享", "同一卷积核在不同空间位置共享权重，显著减少参数量。", "参数共享意味着每个空间位置都学习一套完全不同的卷积核。", "看到“减少参数量、检测同一特征不同位置”就是参数共享。", "易"),
        ("CNN/平移不变性", "卷积神经网络1-学生分发版-终版2026.ipynb", "平移不变性", "卷积的参数共享和池化等操作让模型对局部平移更鲁棒。", "平移不变性表示图像任意移动后所有像素值完全不变。", "表述要严谨：通常是一定程度的鲁棒性，不是数学绝对不变。", "中"),
        ("卷积/互相关", "卷积神经网络1-学生分发版-终版2026.ipynb", "互相关", "深度学习框架中的卷积层通常执行互相关，不翻转卷积核。", "CNN必须严格翻转卷积核，否则无法学习特征。", "考点是工程实现与数学卷积的区别。", "中"),
        ("卷积/卷积核", "卷积神经网络1-学生分发版-终版2026.ipynb", "卷积核", "卷积核是可学习参数，用于提取局部模式。", "卷积核在训练中固定不变，只能由人工手动设计。", "题目常问卷积核是不是学习得到，答案是训练更新。", "易"),
        ("卷积/步长", "卷积神经网络1-学生分发版-终版2026.ipynb", "步长", "步长控制卷积核滑动间隔，步长越大输出空间尺寸通常越小。", "步长越大输出尺寸一定越大。", "尺寸计算题先写公式，再代n、p、k、s。", "易"),
        ("卷积/填充", "卷积神经网络1-学生分发版-终版2026.ipynb", "填充", "填充在边界补像素，用于控制输出尺寸并保留边缘信息。", "零填充不能和BatchNorm配合使用。", "你给的例题已覆盖：零填充工程上很常见。", "易"),
        ("卷积/Valid", "卷积神经网络1-学生分发版-终版2026.ipynb", "Valid卷积", "Valid卷积通常不填充，输出空间尺寸小于输入。", "Valid卷积通过大量填充得到最大输出尺寸。", "Valid、Same、Full的相反描述是常见陷阱。", "易"),
        ("卷积/Same", "卷积神经网络1-学生分发版-终版2026.ipynb", "Same卷积", "Same卷积通过合适填充使输出空间尺寸尽量与输入一致。", "Same卷积完全不使用填充且输出最小。", "stride=1且奇数核时常取p=(k-1)/2。", "中"),
        ("卷积/Full", "卷积神经网络1-学生分发版-终版2026.ipynb", "Full卷积", "Full卷积进行更充分填充，输出尺寸可大于输入。", "Full卷积与Valid卷积完全相同，输出最小。", "三类卷积排序常考：Full最大，Valid最小。", "中"),
        ("卷积/输出尺寸公式", "卷积神经网络1-学生分发版-终版2026.ipynb", "输出尺寸公式", "卷积输出尺寸为floor((n+2p-k)/s)+1。", "卷积输出尺寸只由输入尺寸决定，与核大小、步长、填充无关。", "考试遇到具体数字必须检查结果是否为整数或取floor。", "中"),
        ("池化/最大池化", "卷积神经网络1-学生分发版-终版2026.ipynb", "最大池化", "最大池化取局部窗口最大响应，突出最强特征。", "最大池化需要学习大量卷积核参数。", "池化通常无可学习参数，常用于下采样。", "易"),
        ("池化/平均池化", "卷积神经网络1-学生分发版-终版2026.ipynb", "平均池化", "平均池化取局部窗口平均值，保留局部平均信息。", "平均池化会把通道数固定变成类别数。", "区分空间下采样与分类层输出类别。", "易"),
        ("感受野/定义", "卷积神经网络1-课堂练习-终版2026.ipynb", "感受野", "感受野是某层神经元映射回原始输入的区域范围。", "感受野表示模型参数总量，与输入空间区域无关。", "定义题要写“原始输入区域”，不要只写“看见的东西”。", "易"),
        ("感受野/递推", "卷积神经网络1-课堂练习-终版2026.ipynb", "感受野递推", "回溯公式可写为RF_prev=(RF_current-1)*stride+kernel。", "递推中步长越大，回溯得到的感受野越小。", "你给的例题同款陷阱：步长会放大前一层覆盖范围。", "中"),
        ("图像/RGB", "卷积神经网络1-学生分发版-终版2026.ipynb", "RGB", "RGB图像通常有红、绿、蓝三个通道。", "RGB图像只有一个灰度通道。", "通道数会影响第一层卷积参数量。", "易"),
        ("图像/NCHW", "卷积神经网络1-学生分发版-终版2026.ipynb", "NCHW", "NCHW表示batch、channel、height、width，是PyTorch常用图像格式。", "NCHW表示batch、height、width、channel。", "维度顺序题常用permute代码考查。", "易"),
        ("图像/NHWC", "卷积神经网络1-学生分发版-终版2026.ipynb", "NHWC", "NHWC表示batch、height、width、channel，常见于TensorFlow或图像显示场景。", "NHWC与NCHW完全等价且无需维度转换。", "格式转换错误会导致卷积层通道维不匹配。", "中"),
        ("预处理/归一化", "卷积神经网络1-学生分发版-终版2026.ipynb", "归一化", "归一化常把数据缩放到固定范围，如[0,1]。", "归一化会自动解决所有过拟合问题。", "归一化改善数值尺度，但不是正则化万能药。", "易"),
        ("预处理/标准化", "卷积神经网络1-学生分发版-终版2026.ipynb", "标准化", "标准化通常减去均值再除以标准差。", "标准化统计量应从训练、验证、测试全集共同计算。", "只用训练集统计量，避免数据泄漏。", "易"),
        ("预处理/数据泄漏", "卷积神经网络1-学生分发版-终版2026.ipynb", "数据泄漏", "数据泄漏指训练过程使用了验证或测试阶段才应知道的信息。", "用测试集均值标准化训练集不会影响评估公平性。", "统计量、调参、早停都要警惕测试集信息泄漏。", "中"),
        ("卷积/多通道卷积", "卷积神经网络1-学生分发版-终版2026.ipynb", "多通道卷积", "多通道卷积在输入各通道计算后汇总，每个输出通道对应一组卷积核。", "多通道卷积会把每个输入通道完全独立输出，不能跨通道融合。", "参数量常考out_channels*in_channels*k_h*k_w。", "中"),
        ("卷积/参数量", "卷积神经网络1-学生分发版-终版2026.ipynb", "卷积参数量", "Conv2d参数量通常为out_channels*(in_channels*k_h*k_w+bias项)。", "卷积参数量还要乘以输出特征图的高和宽。", "权值共享意味着参数量不随空间位置线性增长。", "中"),
        ("架构/卷积块", "卷积神经网络1-学生分发版-终版2026.ipynb", "卷积块", "典型卷积块可由Conv、BatchNorm、激活和池化等组成。", "卷积块只能由全连接层组成，不能包含卷积层。", "问结构顺序时注意Conv-BN-ReLU是常见组合。", "易"),
        ("架构/LeNet", "卷积神经网络1-学生分发版-终版2026.ipynb", "LeNet", "LeNet是早期经典CNN，用卷积、池化和全连接完成图像分类。", "LeNet的核心是Transformer自注意力结构。", "经典网络题多考年代、基本结构和作用，不要求背全部参数。", "易"),
        ("架构/AlexNet", "卷积神经网络1-学生分发版-终版2026.ipynb", "AlexNet", "AlexNet推动深度CNN在ImageNet上成功，常与ReLU、Dropout、更深卷积结构相关。", "AlexNet完全不使用卷积层，只使用RNN处理图像。", "和LeNet对比：更深、更大、现代训练技巧更多。", "中"),
        ("架构/自适应池化", "卷积神经网络1-学生分发版-终版2026.ipynb", "自适应池化", "自适应池化把不同输入空间尺寸映射到指定输出尺寸。", "自适应池化要求输入图像尺寸必须完全一致，否则无法工作。", "考点是连接分类器前统一特征尺寸。", "中"),
        ("不平衡/WeightedRandomSampler", "卷积神经网络1-学生分发版-终版2026.ipynb", "WeightedRandomSampler", "WeightedRandomSampler通过采样权重改变不同样本进入batch的概率。", "WeightedRandomSampler会自动修改损失函数公式。", "区分采样加权和损失加权。", "中"),
        ("不平衡/pos_weight", "卷积神经网络1-学生分发版-终版2026.ipynb", "pos_weight", "pos_weight在二分类BCEWithLogitsLoss中提高正样本损失权重。", "pos_weight用于指定每个输入像素的卷积核大小。", "常按负样本数/正样本数估算。", "中"),
        ("不平衡/class weight", "卷积神经网络1-学生分发版-终版2026.ipynb", "class weight", "多分类CrossEntropyLoss可用weight为不同类别设置损失权重。", "class weight只能用于二分类，不能用于多分类。", "少数类权重通常更大，但要结合验证指标调试。", "中"),
        ("梯度/梯度消失", "卷积神经网络2-学生分发版-终版2026.ipynb", "梯度消失", "梯度消失使浅层参数更新很小，深层网络难以学习早期特征。", "梯度消失表现为梯度无限变大并导致NaN。", "和梯度爆炸相反，答题要分清方向和后果。", "易"),
        ("梯度/梯度爆炸", "卷积神经网络2-学生分发版-终版2026.ipynb", "梯度爆炸", "梯度爆炸使梯度过大，可能导致训练震荡、溢出或NaN。", "梯度爆炸表示所有梯度都变成0。", "RNN长序列和高学习率都可能触发。", "易"),
        ("梯度/Value Clipping", "卷积神经网络2-学生分发版-终版2026.ipynb", "Value Clipping", "Value Clipping逐元素限制梯度值范围。", "Value Clipping按整体L2范数等比例缩放所有梯度。", "和Norm Clipping区分：一个逐元素，一个看整体范数。", "中"),
        ("梯度/Norm Clipping", "卷积神经网络2-学生分发版-终版2026.ipynb", "Norm Clipping", "Norm Clipping当梯度范数超过阈值时整体缩放梯度。", "Norm Clipping会随机删除训练样本以防过拟合。", "RNN/LSTM训练常用clip_grad_norm_。", "中"),
        ("初始化/Xavier", "卷积神经网络2-学生分发版-终版2026.ipynb", "Xavier初始化", "Xavier初始化常用于tanh或sigmoid等激活，目标是保持信号尺度。", "Xavier初始化只适用于文件读取，与网络参数无关。", "答题时可写Glorot初始化。", "中"),
        ("初始化/Kaiming", "卷积神经网络2-学生分发版-终版2026.ipynb", "Kaiming初始化", "Kaiming初始化常与ReLU系列激活配合。", "Kaiming初始化要求网络不能使用任何非线性激活。", "看到ReLU优先联想到He/Kaiming。", "中"),
        ("归一化/BatchNorm", "卷积神经网络2-学生分发版-终版2026.ipynb", "BatchNorm", "BatchNorm在训练时使用batch统计量并维护运行均值方差，推理时使用运行统计量。", "BatchNorm训练和推理行为完全一致，不受model.train/eval影响。", "官方文档和课件都强调running mean/var与模式切换。", "中"),
        ("正则/Dropout", "卷积神经网络2-学生分发版-终版2026.ipynb", "Dropout", "Dropout训练时按概率随机置零激活，评估时通常等价于恒等映射。", "Dropout评估时仍随机丢弃一半神经元。", "官方文档强调训练时缩放，评估时identity。", "易"),
        ("正则/Dropout2d", "卷积神经网络2-学生分发版-终版2026.ipynb", "Dropout2d", "Dropout2d在CNN中特征图通道级随机置零，适合空间相关较强的特征图。", "Dropout2d只会随机删除单个标量，永远不会按通道处理。", "CNN正则化题可区分Dropout和Dropout2d。", "中"),
        ("正则/权重衰减", "卷积神经网络2-学生分发版-终版2026.ipynb", "权重衰减", "权重衰减常等价于L2正则化倾向，惩罚过大的权重。", "权重衰减用于把学习率每轮固定加倍。", "Adam/SGD优化器中weight_decay参数常考。", "中"),
        ("正则/早停", "卷积神经网络2-学生分发版-终版2026.ipynb", "早停", "早停根据验证集表现停止训练，防止继续过拟合。", "早停应根据测试集反复调参直到最高分。", "验证集用于早停，测试集只做最终评估。", "中"),
        ("增强/数据增强", "卷积神经网络2-学生分发版-终版2026.ipynb", "数据增强", "数据增强通过随机翻转、裁剪、旋转等扩大训练分布。", "数据增强应同样随机施加在测试集以提高公平性。", "训练集增强，验证/测试通常使用确定性预处理。", "中"),
        ("优化/Momentum", "卷积神经网络2-学生分发版-终版2026.ipynb", "Momentum", "Momentum引入历史梯度方向的速度项，帮助加速并抑制震荡。", "Momentum会完全忽略历史梯度，只看当前batch。", "动量题要写“历史方向/惯性”。", "中"),
        ("优化/Nesterov", "卷积神经网络2-学生分发版-终版2026.ipynb", "Nesterov", "Nesterov动量先按当前速度前瞻，再在前瞻位置计算修正梯度。", "Nesterov与普通SGD完全相同，没有前瞻思想。", "考点是look-ahead，不必推导全部公式。", "难"),
        ("优化/Adagrad", "卷积神经网络2-学生分发版-终版2026.ipynb", "Adagrad", "Adagrad累积平方梯度，使频繁更新参数的有效学习率下降。", "Adagrad的学习率会对所有参数持续增大。", "缺点是学习率可能衰减过快。", "中"),
        ("优化/RMSProp", "卷积神经网络2-学生分发版-终版2026.ipynb", "RMSProp", "RMSProp用指数滑动平均缓解Adagrad学习率持续衰减过快。", "RMSProp完全不使用梯度平方信息。", "和Adagrad对比是常见命题方式。", "中"),
        ("优化/Adam", "卷积神经网络2-学生分发版-终版2026.ipynb", "Adam", "Adam结合一阶矩动量和二阶矩自适应学习率。", "Adam不能与权重衰减或学习率调度配合。", "Adam不是免调参万能解，学习率仍需关注。", "中"),
        ("调度/ReduceLROnPlateau", "卷积神经网络2-学生分发版-终版2026.ipynb", "ReduceLROnPlateau", "ReduceLROnPlateau在监测指标停滞若干轮后降低学习率。", "ReduceLROnPlateau无需传入验证指标即可判断平台期。", "常在每轮验证后scheduler.step(val_loss)。", "中"),
        ("调参/小样本过拟合", "卷积神经网络2-学生分发版-终版2026.ipynb", "小样本过拟合", "小样本过拟合用于检查模型和训练链路是否具备学习能力。", "如果小样本都无法过拟合，说明模型一定泛化很好。", "这是排查代码/数据/损失配置错误的实用步骤。", "难"),
        ("调参/消融实验", "卷积神经网络2-学生分发版-终版2026(1).ipynb", "消融实验", "消融实验一次改变一个因素，观察该因素对结果和训练动态的影响。", "消融实验应该同时改动多个变量以节约时间。", "公平对比、控制变量、看动态和最终指标都是答题点。", "难"),
        ("实践/ImageFolder", "卷积神经网络3-学生分发版-终版2026.ipynb", "ImageFolder", "ImageFolder按类别子文件夹读取图像，并把文件夹名映射为类别。", "ImageFolder要求所有图像必须放在根目录，不能有类别文件夹。", "目录结构题高频：root/class_x/img.png。", "易"),
        ("实践/Transform", "卷积神经网络3-学生分发版-终版2026.ipynb", "Transform", "Transform用于Resize、ToTensor、Normalize、数据增强等输入变换。", "Transform会自动改变模型最后一层类别数。", "训练和验证transform常不同：训练可随机增强，验证要稳定。", "中"),
        ("实践/训练验证测试", "卷积神经网络3-学生分发版-终版2026.ipynb", "训练验证测试", "训练集用于学习参数，验证集用于调参与早停，测试集用于最终评估。", "测试集可以在训练过程中反复用于选择超参数。", "三者职责划分是考试常规简答点。", "易"),
        ("实践/模型保存", "卷积神经网络3-学生分发版-终版2026.ipynb", "模型保存", "常保存state_dict或checkpoint，部署预测时加载参数并切换eval。", "保存模型后预测时必须保持train模式以启用Dropout。", "checkpoint可包含模型、优化器、epoch等以恢复训练。", "中"),
        ("现代/VGG", "卷积神经网络3-学生分发版-终版2026.ipynb", "VGG", "VGG使用多层小卷积核堆叠构建深层网络。", "VGG的核心是用循环隐藏状态处理文本序列。", "VGG题常考3x3小卷积堆叠。", "易"),
        ("现代/GoogLeNet", "卷积神经网络3-学生分发版-终版2026.ipynb", "GoogLeNet", "GoogLeNet/Inception用多分支并行卷积捕获不同尺度特征。", "GoogLeNet完全依赖单一路径3x3卷积，不能并行分支。", "关键词是Inception、多尺度、多分支。", "中"),
        ("现代/ResNet", "卷积神经网络3-学生分发版-终版2026.ipynb", "ResNet", "ResNet通过残差/跳跃连接缓解深层网络退化和梯度传播困难。", "ResNet的核心创新是删除所有跳跃连接。", "残差连接题常问F(x)+x的意义。", "中"),
        ("迁移/迁移学习", "卷积神经网络3-学生分发版-终版2026.ipynb", "迁移学习", "迁移学习利用预训练模型的通用特征，替换或微调分类头适配新任务。", "迁移学习要求新任务与原任务类别完全相同。", "步骤题：加载预训练、冻结/微调、替换head、训练评估。", "中"),
        ("序列/序列数据", "循环神经网络-学生分发版-终版2026.ipynb", "序列数据", "序列数据有顺序关系，样本长度可能不同。", "序列数据中元素顺序可以任意打乱而不影响含义。", "文本、时间序列、音频都是常见例子。", "易"),
        ("RNN/隐藏状态", "循环神经网络-学生分发版-终版2026.ipynb", "隐藏状态", "隐藏状态在时间步之间传递，用于概括历史信息。", "隐藏状态只存在于CNN池化层，与RNN无关。", "RNN核心目标是获得能表示序列的状态。", "易"),
        ("RNN/RNNCell", "循环神经网络-学生分发版-终版2026.ipynb", "RNNCell", "RNNCell处理单个时间步，适合手写循环和自定义逻辑。", "RNNCell会自动处理完整序列且不能逐步调用。", "和nn.RNN区别：Cell单步，Layer整段。", "中"),
        ("RNN/nn.RNN", "循环神经网络-学生分发版-终版2026.ipynb", "nn.RNN", "nn.RNN可直接处理完整序列并返回output和hidden。", "nn.RNN只能处理单个时间步，不能处理序列张量。", "Shape题要区分batch_first。", "中"),
        ("RNN/batch_first", "循环神经网络-学生分发版-终版2026.ipynb", "batch_first", "batch_first=True时输入常为(batch, seq_len, feature)。", "batch_first=True时输入一定是(seq_len, batch, feature)。", "这是RNN形状选择题高频点。", "易"),
        ("RNN/堆叠RNN", "循环神经网络-学生分发版-终版2026.ipynb", "堆叠RNN", "堆叠RNN把多层循环层纵向叠加，增强表达能力但增加训练难度。", "堆叠RNN层数增加会让参数量变为0。", "参数量和hidden形状会随层数变化。", "中"),
        ("RNN/双向RNN", "循环神经网络-学生分发版-终版2026.ipynb", "双向RNN", "双向RNN同时使用前向和后向序列信息，输出维度常因方向拼接翻倍。", "双向RNN只能看到过去信息，不能利用未来上下文。", "课件强调output最后步不等同完整final_hidden。", "难"),
        ("RNN/长期依赖", "循环神经网络-学生分发版-终版2026.ipynb", "长期依赖", "长期依赖困难来自时间维度反向传播中的梯度连乘。", "长期依赖问题只和数据文件大小有关，与梯度传播无关。", "答题要写链式法则、范数连乘、消失/爆炸。", "难"),
        ("GRU/更新门", "循环神经网络-学生分发版-终版2026.ipynb", "更新门", "GRU更新门控制保留旧状态和写入新候选状态的比例。", "更新门用于把所有历史信息强制清零。", "门控题要说控制信息保留/更新。", "中"),
        ("GRU/重置门", "循环神经网络-学生分发版-终版2026.ipynb", "重置门", "GRU重置门控制计算候选状态时使用多少过去信息。", "重置门专门用于调整学习率。", "更新门与重置门功能要分清。", "中"),
        ("LSTM/细胞状态", "循环神经网络-学生分发版-终版2026.ipynb", "细胞状态", "LSTM细胞状态是较稳定的长期记忆通道。", "LSTM没有细胞状态，只有普通RNN隐藏状态。", "LSTM核心优势常围绕cell state展开。", "中"),
        ("LSTM/遗忘门", "循环神经网络-学生分发版-终版2026.ipynb", "遗忘门", "遗忘门控制旧细胞状态中哪些信息被保留或遗忘。", "遗忘门决定卷积核在图像上滑动的步长。", "门控名称和作用是填空题高频。", "中"),
        ("LSTM/输入门", "循环神经网络-学生分发版-终版2026.ipynb", "输入门", "输入门控制新信息写入细胞状态的程度。", "输入门只负责把输出概率归一化为1。", "和输出门、遗忘门区别记忆。", "中"),
        ("LSTM/输出门", "循环神经网络-学生分发版-终版2026.ipynb", "输出门", "输出门控制细胞状态中哪些信息暴露为隐藏状态输出。", "输出门用于设置DataLoader是否shuffle。", "LSTM三门作用常作为简答题。", "中"),
        ("变长/Padding", "循环神经网络-学生分发版-终版2026.ipynb", "Padding", "Padding把不同长度序列补齐到同一长度以便组成batch。", "Padding能自动告诉RNN哪些位置是无效填充，因此无需长度信息。", "填充值可能污染隐藏状态，所以常配合mask或packing。", "中"),
        ("变长/Packing", "循环神经网络-学生分发版-终版2026.ipynb", "Packing", "Packing记录真实长度，使RNN跳过填充位置处理有效时间步。", "Packing的目的就是增加无效padding计算量。", "pack_padded_sequence常要求长度信息，注意排序参数。", "中"),
        ("变长/PackedSequence", "循环神经网络-学生分发版-终版2026.ipynb", "PackedSequence", "PackedSequence用data和batch_sizes等结构表示压缩后的变长序列。", "PackedSequence只是普通列表，不能送入RNN。", "考试会问packed.data和batch_sizes含义。", "难"),
        ("序列/1D卷积", "循环神经网络-学生分发版-终版2026.ipynb", "1D卷积", "1D卷积沿时间或序列维滑动，适合提取局部时序模式并行计算。", "1D卷积只能处理二维图像，不能用于序列。", "和RNN比较：并行强、局部模式强、状态递推弱。", "中"),
        ("序列/TCN", "循环神经网络-学生分发版-终版2026.ipynb", "TCN", "TCN常用因果卷积和扩张卷积建模序列并扩大感受野。", "TCN必须使用循环连接，不能并行计算。", "关键词：causal、dilated、receptive field。", "难"),
        ("序列/IMDB情感分类", "循环神经网络-学生分发版-终版2026.ipynb", "IMDB情感分类", "IMDB情感分类是文本二分类任务，常用Embedding、RNN/LSTM和BCEWithLogitsLoss。", "IMDB情感分类是三通道图像分割任务，必须用2D卷积输出像素类别。", "综合案例题常考数据加载、词表、padding/packing、二分类损失。", "中"),
        ("Seq2Seq/定义", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Seq2Seq", "Seq2Seq用于把一个序列映射成另一个序列，常见于翻译、摘要和对话生成。", "Seq2Seq只能处理固定长度向量到单个类别的映射，不能生成序列。", "题目常通过“序列到类别”和“序列到序列”对比来设陷阱。", "易"),
        ("Seq2Seq/Encoder-Decoder", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Encoder-Decoder", "Encoder-Decoder结构先编码源序列，再由Decoder逐步生成目标序列。", "Encoder-Decoder中编码器和解码器都只负责图像卷积，不处理文本序列。", "回答时建议写清编码、压缩表示、逐步解码三个动作。", "中"),
        ("Seq2Seq/Teacher Forcing", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Teacher Forcing", "Teacher Forcing训练时把真实前一个目标词送给Decoder，帮助稳定和加速训练。", "Teacher Forcing表示推理阶段直接把正确答案整句输入模型，因此测试最稳定。", "高频陷阱是把训练阶段和推理阶段混为一谈。", "中"),
        ("Seq2Seq/瓶颈问题", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "上下文向量瓶颈", "传统Encoder-Decoder把整段源序列压缩为一个固定长度上下文向量，长序列时容易丢失信息。", "固定长度上下文向量会让模型自动获得无限记忆，因此长句效果更好。", "这题常作为“为什么需要注意力机制”的铺垫。", "中"),
        ("Attention/作用", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "注意力机制", "注意力机制让模型在生成当前目标位置时动态关注源序列中更相关的位置。", "注意力机制要求每个目标词都使用完全相同的源侧权重分布。", "答题时可用“不同词需要关注源序列不同部分”这句解释直觉。", "易"),
        ("Attention/QKV", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Q/K/V", "Query表示当前要查询什么，Key表示各位置可供匹配什么，Value表示各位置携带什么信息。", "Q、K、V三个向量在注意力里没有分工，本质上完全相同且不可区分。", "Q/K/V角色分工和来源是考试高频定义题。", "中"),
        ("Attention/Query", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Query", "在Encoder-Decoder注意力中，Query通常来自Decoder当前状态，表示“现在想找什么信息”。", "Query通常来自源序列所有隐藏状态，与Decoder当前状态无关。", "如果题目考来源，先判断是Self-Attention还是Cross-Attention。", "中"),
        ("Attention/Key", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Key", "Key常作为被匹配对象，用来和Query计算相似度。", "Key在注意力里只负责最终加权求和，不参与匹配分数计算。", "Key和Value有联系但职责不同：一个用于匹配，一个用于提供内容。", "中"),
        ("Attention/Value", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Value", "Value是被注意力权重加权求和的内容载体。", "Value只用于生成mask，不参与上下文向量形成。", "看到“加权求和得到上下文向量”时就要想到Value。", "中"),
        ("Attention/Scaled Dot-Product", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Scaled Dot-Product Attention", "Scaled Dot-Product Attention先算QK^T相似度，再除以sqrt(d_k)，Softmax后对V加权求和。", "Scaled Dot-Product Attention只需要Value，不需要Query和Key。", "完整公式是考试中的核心必背公式。", "难"),
        ("Attention/sqrt(d_k)", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "sqrt(d_k)", "除以sqrt(d_k)是为了防止高维点积过大，导致Softmax过度尖锐和梯度不稳定。", "除以sqrt(d_k)的目的是让序列长度自动变短。", "一旦题目提到“方差变大”“Softmax极端化”，基本就在考这个点。", "难"),
        ("Attention/上下文向量", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "上下文向量", "上下文向量是对Value按注意力权重加权求和后的结果。", "上下文向量与注意力权重无关，它始终等于Query本身。", "回答时最好写出“Softmax权重 × Value”的汇总过程。", "中"),
        ("Attention/Source Mask", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Source Mask", "Source Mask用于屏蔽源序列中的Padding位置，避免模型关注无意义填充符。", "Source Mask用于让Decoder看到未来目标词，从而加快推理。", "这题常和Target Mask混考。", "中"),
        ("Attention/Multi-Head Attention", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Multi-Head Attention", "多头注意力用多个头并行在不同子空间中学习不同关注模式，再拼接整合。", "多头注意力的每个头必须看到完全相同的投影，因此不能学到不同模式。", "关键表述是“并行多个头”和“不同表示子空间”。", "中"),
        ("Transformer/Self-Attention", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Self-Attention", "Self-Attention让同一序列内部各位置彼此交互，每个位置都能聚合整段序列的信息。", "Self-Attention只能处理两个不同序列之间的关系，不能处理单序列内部依赖。", "看到“序列内部位置互相关注”就是Self-Attention。", "中"),
        ("Transformer/Cross-Attention", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Cross-Attention", "Cross-Attention通常由Decoder查询Encoder输出，用于结合源序列信息。", "Cross-Attention的Query、Key、Value一定都来自同一个Decoder序列。", "和Self-Attention区别在于Q与K/V来源不同。", "中"),
        ("Transformer/Target Mask", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Target Mask", "Target Mask或Subsequent Mask用于阻止Decoder在当前位置看到未来目标词。", "Target Mask的作用是屏蔽源序列中的Padding位置，与目标侧自回归无关。", "一句话记忆：Source Mask挡PAD，Target Mask挡未来。", "中"),
        ("Transformer/位置编码", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "位置编码", "位置编码为Transformer补充顺序信息，因为纯注意力本身不天然编码位置先后。", "Transformer天然具备严格的位置顺序感知，因此不需要任何位置编码。", "为什么需要位置编码是复习资料和简答题高频。", "中"),
        ("Transformer/Sinusoidal Positional Encoding", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Sinusoidal Positional Encoding", "正弦位置编码使用不同频率的sin/cos函数为不同位置和维度编码。", "正弦位置编码要求每个位置都学习独立可训练参数，否则无法工作。", "常见考法是问奇偶维分别用什么函数。", "难"),
        ("Transformer/RoPE", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "RoPE", "RoPE通过旋转方式把位置信息注入表示，现代大模型中很常见。", "RoPE的本质是删除所有位置信息，只保留词向量内容。", "若题目问现代大模型位置编码趋势，RoPE是关键词。", "中"),
        ("Transformer/Attention Is All You Need", "注意力机制及Transformer1-学生分发版-终版2026.ipynb", "Transformer", "Transformer用注意力机制作为核心计算模块，显著减少对循环结构的依赖。", "Transformer必须依赖RNN隐藏状态逐时间步递推，否则无法处理序列。", "考试常把Transformer与RNN逐步递推做对比。", "中"),
    ]


def extend_quiz_bank(base):
    questions = list(base)
    letters = ["A", "B", "C", "D"]
    concepts = supplemental_concepts()

    for idx, (topic, source, key, truth, false_stmt, exam_tip, difficulty) in enumerate(concepts, 1):
        false_pos = idx % 4
        true_options = [
            truth,
            exam_tip,
            f"{key}属于“{topic}”相关考点，答题时要结合任务类型、输入输出形状或训练/推理阶段判断。",
        ]
        raw_options = true_options[:]
        raw_options.insert(false_pos, false_stmt)
        options = [f"{letters[i]}. {raw_options[i]}" for i in range(4)]
        answer = letters[false_pos]
        questions.append(q_mc(
            f"XMC{idx:03d}",
            topic,
            f"关于{key}，下列说法错误的是",
            options,
            answer,
            f"{answer}错误：{false_stmt}。正确理解是：{truth} {exam_tip}",
            source,
            difficulty,
        ))

        false_pos2 = (idx + 2) % 4
        raw_options2 = [
            f"可以把“{key}”与{topic.split('/')[0]}章节中的相邻概念联系起来复习。",
            f"做题时应优先检查题干中的任务类型、数据形状、训练阶段或公式条件。",
            f"若题干把{key}说成与课件定义相反，通常就是错误项。",
        ]
        raw_options2.insert(false_pos2, f"只要题干出现{key}，就可以忽略课件中的公式、输入输出形状和训练阶段。")
        options2 = [f"{letters[i]}. {raw_options2[i]}" for i in range(4)]
        answer2 = letters[false_pos2]
        questions.append(q_mc(
            f"YMC{idx:03d}",
            topic,
            f"围绕{key}的考试判断，下列说法错误的是",
            options2,
            answer2,
            f"{answer2}错误：考试题恰恰常通过公式条件、输入输出形状、训练/推理阶段制造陷阱。{exam_tip}",
            source,
            "难" if difficulty == "难" else "中",
        ))

        questions.append(q_fill(
            f"XF{idx:03d}",
            topic,
            f"{truth} 这句话描述的核心词是____。",
            key,
            f"答案是{key}。{exam_tip}",
            source,
            difficulty,
        ))

        questions.append(q_short(
            f"XS{idx:03d}",
            topic,
            f"围绕“{key}”，写出考试作答时必须抓住的两个要点。",
            f"要点一：{truth} 要点二：{exam_tip} 同时要避免这样的错误表述：{false_stmt}",
            f"本题考查概念定义、适用条件和易错陷阱。答题时先给定义，再说明它在训练流程、模型结构、损失函数或序列处理中的作用。",
            source,
            difficulty,
        ))

    seen = set()
    unique = []
    for q in questions:
        if q["id"] not in seen:
            unique.append(q)
            seen.add(q["id"])
    return unique


def enrich_review_material(review):
    additions = {
        "1.": {
            "details": [
                "训练循环类题目要从“数据进入模型”这条线索追踪：DataLoader给出batch，模型前向得到预测，损失函数把预测和标签合成标量loss，loss.backward沿计算图写入grad，optimizer.step真正修改参数。",
                "验证循环不更新参数，所以通常不需要zero_grad/backward/step；但需要model.eval()保证Dropout/BatchNorm行为正确，并用torch.no_grad()避免构图。",
                "若题干出现loss不下降、显存增长、梯度异常，常见排查顺序是学习率、梯度清零、标签/损失匹配、输入标准化、模型是否处于正确模式。",
            ],
            "must_memorize": [
                "标准训练顺序：zero_grad -> forward -> loss -> backward -> step。",
                "验证/测试顺序：eval -> no_grad -> forward -> metric，不更新参数。",
                "梯度在PyTorch中默认累积，不会自动清零。",
            ],
            "formula_explainer": [
                "theta = theta - eta * grad(theta) 中，eta是学习率，决定每次更新幅度；grad(theta)来自反向传播。",
                "MSE对大误差惩罚更明显，常用于回归；分类题一般不会用MSE作为标准答案。",
            ],
            "answer_template": [
                "若考训练流程，建议按“数据输入 -> 前向传播 -> 损失计算 -> 反向传播 -> 参数更新 -> 验证评估”六步作答。",
                "若考train/eval差异，先写Dropout和BatchNorm行为变化，再补no_grad是否记录梯度。",
            ],
            "question_patterns": [
                "给一段训练代码，判断哪一步顺序错误。",
                "解释为什么验证阶段要eval和no_grad。",
                "区分epoch、iteration、batch、sample。",
            ],
            "traps": [
                "把model.eval()误认为会关闭梯度计算。",
                "把zero_grad写在step之后导致本轮梯度混乱。",
                "验证集参与训练参数更新，造成评估不可信。",
            ],
        },
        "2.": {
            "details": [
                "二分类最容易考“单logit+BCEWithLogitsLoss”和“双logit+CrossEntropyLoss”的区别。两者都能做二分类，但输入形状、标签形式和损失函数不同。",
                "CrossEntropyLoss考试重点是输入raw logits，目标常用类别索引；预测展示概率时才额外softmax。BCEWithLogitsLoss输入同样是logits，内部包含Sigmoid相关的稳定计算。",
                "Softmax计算题一般先算指数，再除以指数和；尺度放大让最大项更占优，尺度缩小让分布更平滑。",
                "MLP非线性表达能力来自激活函数，不来自单纯堆叠线性层。无激活函数时，多层线性/仿射层仍能合并成一层。",
            ],
            "must_memorize": [
                "二分类常配：单logit + BCEWithLogitsLoss。",
                "多分类常配：C个logits + CrossEntropyLoss。",
                "Sigmoid输出和Softmax输出都是概率，但适用任务不同。",
            ],
            "formula_explainer": [
                "logit(p)=log(p/(1-p))：把概率映射到实数轴，p=0.5时logit=0。",
                "softmax(z_i)=e^(z_i)/sum_j e^(z_j)：哪个logit大，哪个概率更大；整体和为1。",
                "BCE本质是在惩罚错误概率分配，预测越自信但越错，损失越大。",
            ],
            "answer_template": [
                "若考损失函数匹配，先写任务类型，再写输出层形式，最后写损失函数名称。",
                "若考MLP非线性，标准表述是“仿射变换提供线性映射，激活函数提供非线性表达能力”。",
            ],
            "question_patterns": [
                "给logits和标签，选择合适损失函数。",
                "计算logit、sigmoid或softmax概率。",
                "判断没有激活函数的多层网络能否拟合非线性边界。",
            ],
            "traps": [
                "先Softmax再传CrossEntropyLoss。",
                "先Sigmoid再传BCEWithLogitsLoss。",
                "把Sigmoid和Softmax都说成互不影响的逐元素概率函数。",
            ],
        },
        "3.": {
            "details": [
                "CNN基础题通常围绕三件事：为什么不用MLP直接展平图像、卷积输出尺寸如何算、感受野如何递推。答题时要把“空间结构、参数共享、局部连接”连起来。",
                "卷积尺寸公式out=floor((n+2p-k)/s)+1必须熟练。Same/Valid/Full不是背名字，而是看是否填充以及输出尺寸变化。",
                "感受野递推题要从后往前或从前往后保持一致。你给的例题中“步长变大会缩小感受野”就是典型反向陷阱，正确是步长放大回溯覆盖范围。",
                "池化层通常无可学习参数，主要改变空间尺寸和局部鲁棒性。不要把池化说成增加参数量的层。",
            ],
            "must_memorize": [
                "CNN三关键词：局部连接、参数共享、局部平移鲁棒性。",
                "Valid通常最小，Same尽量保持尺寸，Full通常最大。",
                "感受野递推里，步长增大会扩大回溯到前层的覆盖范围。",
            ],
            "formula_explainer": [
                "out=floor((n+2p-k)/s)+1：n是输入尺寸，p是padding，k是卷积核大小，s是步长。",
                "RF_prev=(RF_current-1)*stride+kernel：先把当前感受野间隔按步长拉开，再补上本层核宽。",
            ],
            "answer_template": [
                "若考CNN优于MLP，按“保留空间结构 -> 减少参数 -> 便于提取局部特征”三点写。",
                "若考卷积尺寸，先列公式，再代入参数，最后说明padding是否改变边界信息。",
            ],
            "question_patterns": [
                "给输入尺寸、核、步长、填充，算输出尺寸。",
                "判断Valid/Same/Full哪个输出最大或最小。",
                "根据多层卷积/池化表递推感受野。",
            ],
            "traps": [
                "忽略padding或stride。",
                "把深度学习里的互相关说成必须翻转卷积核。",
                "把池化当成有大量可学习参数的层。",
            ],
        },
        "4.": {
            "details": [
                "图像维度题要先识别框架约定：PyTorch卷积常用NCHW，图像显示和部分框架常用NHWC。维度错位会直接导致通道数错误。",
                "多通道卷积不是每个通道独立给一个最终结果，而是每个输出通道有一组覆盖所有输入通道的核，跨通道求和得到输出特征图。",
                "标准化统计量只来自训练集。考试若说用全数据集、验证集或测试集统计量优化训练，就要警惕数据泄漏。",
                "参数量题要记住卷积核空间参数乘输入通道数再乘输出通道数；由于参数共享，不再乘输出特征图的H和W。",
            ],
            "must_memorize": [
                "PyTorch图像张量常用NCHW。",
                "RGB是3通道，灰度通常是1通道。",
                "卷积参数量不乘输出特征图的空间位置数。",
            ],
            "formula_explainer": [
                "Conv2d参数量 = out_channels * (in_channels * k_h * k_w + bias项)。若不使用bias，就去掉偏置项。",
                "标准化可写作 x'=(x-mu)/sigma，mu和sigma应来自训练集。",
            ],
            "answer_template": [
                "若考NCHW/NHWC，建议按每个字母对应含义逐一写出，避免漏掉通道维。",
                "若考数据泄漏，标准句式是“训练阶段不应利用验证集或测试集的统计信息”。",
            ],
            "question_patterns": [
                "NCHW/NHWC维度转换。",
                "计算Conv2d参数量。",
                "判断标准化统计量是否产生数据泄漏。",
            ],
            "traps": [
                "把H和W误认为通道维。",
                "卷积参数量错误乘以输出空间位置数。",
                "用测试集参与预处理统计量拟合。",
            ],
        },
        "5.": {
            "details": [
                "梯度消失/爆炸都来自链式法则连乘，但方向不同：消失是梯度趋近0，爆炸是梯度过大。题目常把二者症状互换。",
                "BatchNorm和Dropout都是train/eval差异高频点。BN训练时用batch统计量并维护running统计量，推理时用running统计量；Dropout训练时随机置零并缩放，推理时通常是恒等映射。",
                "优化器比较题重点不是背公式，而是抓住机制：Momentum看历史方向，Adagrad累积平方梯度导致学习率衰减，RMSProp用滑动平均缓解，Adam结合一阶/二阶矩。",
                "调参题先问训练链路是否正常，再谈搜索。小样本不能过拟合通常说明代码、数据、损失或学习率有问题。",
            ],
            "must_memorize": [
                "梯度爆炸常见对策：梯度裁剪、减小学习率、合理初始化。",
                "梯度消失常见对策：ReLU系激活、合理初始化、BatchNorm、残差结构。",
                "BatchNorm和Dropout都依赖train/eval模式切换。",
            ],
            "formula_explainer": [
                "ReduceLROnPlateau核心逻辑是：若验证指标连续若干轮不改善，则lr乘factor。",
                "权重衰减本质上是偏向较小权重的正则项，不是简单“让模型更浅”。",
            ],
            "answer_template": [
                "若考优化器比较，按“核心机制 -> 优点 -> 典型缺点”三句写最稳。",
                "若考梯度问题，先写原因，再写症状，最后写至少一种解决方法。",
            ],
            "question_patterns": [
                "判断梯度问题症状并选择解决方法。",
                "比较BatchNorm、Dropout在训练/推理阶段的行为。",
                "给优化器描述选名称。",
                "设计超参数搜索或消融实验流程。",
            ],
            "traps": [
                "把梯度裁剪说成解决过拟合的主要方法。",
                "认为Adam完全不需要调学习率。",
                "消融实验一次改多个变量。",
            ],
        },
        "6.": {
            "details": [
                "完整实践题会把ImageFolder目录结构、Transform、DataLoader、模型、损失函数、优化器、评估指标串起来考。任何一环出错都会影响训练。",
                "类别不平衡题要区分采样层面和损失层面：WeightedRandomSampler改变样本出现频率，pos_weight/class weight改变错误代价。",
                "现代CNN结构题常考关键词：VGG是小卷积核堆叠，GoogLeNet/Inception是多尺度分支，ResNet是残差/跳跃连接。",
                "迁移学习题要写清“加载预训练、冻结或微调、替换分类头、用新任务数据训练”。",
            ],
            "must_memorize": [
                "ImageFolder按子文件夹名确定类别。",
                "类别不平衡至少有两类方案：采样加权、损失加权。",
                "VGG=小卷积堆叠，GoogLeNet=多分支，ResNet=残差连接。",
            ],
            "formula_explainer": [
                "二分类常见pos_weight近似为负样本数/正样本数，用于提高正类错误的损失权重。",
                "迁移学习中替换最后分类头，本质是把预训练特征抽取器接到新任务标签空间上。",
            ],
            "answer_template": [
                "若考完整流程，可按“数据读取 -> 预处理 -> 建模 -> 训练验证 -> 指标评估 -> 模型保存”作答。",
                "若考现代CNN结构比较，直接给每个模型一个核心关键词最有效。",
            ],
            "question_patterns": [
                "根据文件夹树判断ImageFolder类别。",
                "给不平衡样本数计算pos_weight。",
                "匹配VGG/GoogLeNet/ResNet核心设计。",
                "说明迁移学习步骤。",
            ],
            "traps": [
                "把ImageFolder所有图片放根目录。",
                "把pos_weight用于多分类类别索引CE的每类权重。",
                "认为迁移学习类别必须完全一致。",
            ],
        },
        "7.": {
            "details": [
                "RNN形状题先看batch_first。batch_first=True是(batch, seq_len, feature)，默认常见形式是(seq_len, batch, feature)。hidden还要乘num_layers和num_directions。",
                "双向RNN常考output最后一步与final_hidden的差别。前向最后状态和后向最后状态对应的时间位置不同，不能简单说output[:, -1]就是完整序列表示。",
                "GRU/LSTM门控题要按功能记：GRU更新门控制保留/更新，重置门控制候选状态看多少历史；LSTM遗忘门、输入门、输出门围绕细胞状态控制信息流。",
                "变长序列题要把Padding和Packing分开：Padding解决batch形状统一，Packing让RNN跳过填充部分，减少无效计算和隐藏状态污染。",
                "1D卷积/TCN与RNN的选型题要从局部模式、并行效率、长程依赖和感受野角度回答。",
            ],
            "must_memorize": [
                "RNN核心是隐藏状态沿时间传递。",
                "双向RNN常导致输出或隐藏状态维度翻倍。",
                "Padding管对齐，Packing管跳过无效填充。",
            ],
            "formula_explainer": [
                "h_t=tanh(W_xh x_t + W_hh h_(t-1) + b)：当前状态由当前输入和上一时刻状态共同决定。",
                "长期依赖问题的本质是时间维度上的梯度连乘，不是“序列长就一定学不好”的口号化描述。",
            ],
            "answer_template": [
                "若考RNN与GRU/LSTM比较，先写普通RNN难点，再写门控机制如何控制信息保留与遗忘。",
                "若考Padding和Packing，标准结构是“Padding解决形状统一，Packing解决无效计算和状态污染”。",
            ],
            "question_patterns": [
                "给RNN参数判断input/output/hidden形状。",
                "解释普通RNN长期依赖困难。",
                "比较GRU和LSTM门控。",
                "说明Padding、Packing、PackedSequence的关系。",
            ],
            "traps": [
                "忽略双向导致hidden维度翻倍。",
                "Padding后不传长度信息却认为RNN自动忽略填充值。",
                "把1D卷积说成只能处理图像。",
            ],
        },
        "8.": {
            "details": [
                "Seq2Seq 题目第一步先判断输出是不是“另一个序列”。如果输出是一个类别，那通常更接近序列分类；如果输出是一段词序列，那才是 Seq2Seq 场景。",
                "Encoder-Decoder 的经典短板是固定长度上下文向量瓶颈。题目若问“为什么注意力机制有效”，标准答案不是“因为参数更多”，而是“因为生成每一步都能动态查看不同源位置”。",
                "Q/K/V 的答题方式要稳定：Q 表示当前查询需求，K 表示被匹配索引，V 表示最终被聚合的内容。Self-Attention 与 Cross-Attention 的差别，本质是 Q 与 K/V 是否来自同一序列。",
                "Scaled Dot-Product Attention 的公式通常既考整体形式，也考每一步含义：先匹配分数，再缩放，再 Softmax，再对 V 加权求和。",
                "除以 sqrt(d_k) 的问题常被包装成“为什么不直接对点积做 Softmax”。关键回答是：高维下点积方差变大，Softmax容易极端化，梯度会变差。",
                "Mask 类题一定要分源侧和目标侧：Source Mask 屏蔽 PAD，Target Mask 屏蔽未来。两者都在 Softmax 前把不该看的位置压到极小。",
                "多头注意力不是简单复制一个头，而是让不同头在不同投影子空间中学习不同关系，比如局部依赖、长距离依赖、语义对齐等。",
                "Transformer 之所以还需要位置编码，是因为注意力本身对序列顺序不敏感。位置编码是在“没有循环”的前提下补回位置信号。",
                "正弦位置编码常见考法包括：偶数维用 sin、奇数维用 cos；不同维度对应不同频率；位置差异可以通过三角函数模式表达。",
                "RoPE 更偏现代模型考点，适合作为拓展题或选择题背景项。答题不必推导旋转矩阵，但要知道它是在表示空间中注入相对位置信息。",
            ],
            "must_memorize": [
                "Seq2Seq = 序列到序列，不是序列到单类别。",
                "注意力机制解决的是固定上下文向量瓶颈，不是简单“让模型更深”。",
                "Source Mask 挡 PAD，Target Mask 挡未来。",
                "Self-Attention 是同序列内部交互，Cross-Attention 是目标侧查询源侧。",
                "Transformer 需要位置编码，因为纯注意力不自带顺序信息。",
            ],
            "formula_explainer": [
                "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V：QK^T 得到匹配分数，除以 sqrt(d_k) 做缩放，softmax 得到权重，再乘 V 汇总信息。",
                "PE(pos, 2i)=sin(pos/10000^(2i/d_model)) 与 PE(pos, 2i+1)=cos(pos/10000^(2i/d_model))：偶数维和奇数维使用不同三角函数，频率随维度变化。",
                "上下文向量不是单个位置的隐藏状态复制，而是所有 Value 在注意力权重下的加权和。",
            ],
            "answer_template": [
                "若考“为什么需要注意力机制”，推荐答法：先写 Encoder-Decoder 固定上下文瓶颈，再写不同目标位置应关注不同源位置，最后写注意力可动态分配权重。",
                "若考 Q/K/V，可直接答：Q 提出查询，K 提供匹配依据，V 提供被聚合的信息内容。",
                "若考位置编码，可答：Transformer 去掉循环后缺少天然顺序感，因此需通过位置编码显式注入位置信息。",
            ],
            "question_patterns": [
                "判断 Seq2Seq、序列分类、图像分类三类任务的区别。",
                "解释 Encoder-Decoder 为何会有瓶颈。",
                "给 Q/K/V 或注意力公式，让你判断每一项含义。",
                "解释为什么要除以 sqrt(d_k)。",
                "比较 Source Mask、Target Mask、Self-Attention、Cross-Attention。",
                "说明为什么 Transformer 需要位置编码。",
            ],
            "traps": [
                "把 Teacher Forcing 误写成推理阶段也使用真实标签。",
                "把 Source Mask 和 Target Mask 的作用对调。",
                "把 Self-Attention 说成只能处理不同序列之间的关系。",
                "把位置编码说成只是在输入后随便加一个编号，不涉及顺序建模。",
                "把除以 sqrt(d_k) 解释成缩短序列长度或减少参数量。",
            ],
        },
    }
    review["references"] = [
        {"title": "PyTorch CrossEntropyLoss 文档", "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"},
        {"title": "PyTorch BCEWithLogitsLoss 文档", "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html"},
        {"title": "PyTorch Dropout 文档", "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.dropout.Dropout.html"},
        {"title": "PyTorch RNN/PackedSequence 文档", "url": "https://docs.pytorch.org/docs/stable/nn.html#recurrent-layers"},
        {"title": "Stanford CS231n Convolutional Networks", "url": "https://cs231n.github.io/convolutional-networks/"},
    ]
    for section in review["sections"]:
        prefix = section["title"].split()[0]
        if prefix in additions:
            section.update(additions[prefix])
    return review


CSS = r"""
:root {
  color-scheme: light;
  --ink: #17211b;
  --muted: #5b6960;
  --line: #dfe7e1;
  --soft: #f5f7f2;
  --panel: #ffffff;
  --accent: #176b5b;
  --accent-2: #b23b2a;
  --gold: #946b15;
  --code: #263238;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  color: var(--ink);
  background: #fbfcf8;
  line-height: 1.58;
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  border-bottom: 1px solid var(--line);
  background: rgba(251, 252, 248, .94);
  backdrop-filter: blur(10px);
}
.bar {
  max-width: 1180px;
  margin: 0 auto;
  padding: 14px 20px 10px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 14px;
  align-items: end;
}
h1 {
  margin: 0;
  font-size: 22px;
  letter-spacing: 0;
}
.sub {
  margin: 2px 0 0;
  color: var(--muted);
  font-size: 13px;
}
nav {
  display: flex;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}
nav button {
  appearance: none;
  border: 0;
  border-right: 1px solid var(--line);
  background: #fff;
  padding: 9px 14px;
  min-width: 86px;
  font-size: 14px;
  cursor: pointer;
}
nav button:last-child { border-right: 0; }
nav button.active {
  background: var(--accent);
  color: #fff;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 20px;
}
.view { display: none; }
.view.active { display: block; }
.toolbar {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  align-items: center;
  margin-bottom: 14px;
}
input[type="search"] {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 13px;
  font-size: 16px;
  background: #fff;
}
.btn {
  border: 1px solid var(--accent);
  background: var(--accent);
  color: #fff;
  border-radius: 8px;
  padding: 11px 14px;
  min-height: 42px;
  font-size: 14px;
  cursor: pointer;
  white-space: nowrap;
}
.btn.secondary {
  background: #fff;
  color: var(--accent);
}
select {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 8px;
  padding: 10px 12px;
  min-height: 42px;
  font-size: 14px;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 16px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.panel h2, .panel h3 {
  margin: 0 0 10px;
  letter-spacing: 0;
}
.panel h2 { font-size: 19px; }
.panel h3 { font-size: 16px; }
.meta {
  color: var(--muted);
  font-size: 13px;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.chip {
  border: 1px solid var(--line);
  background: var(--soft);
  color: var(--ink);
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 13px;
  cursor: pointer;
}
.result {
  border-top: 1px solid var(--line);
  padding: 13px 0;
}
.result:first-child { border-top: 0; padding-top: 0; }
.result-title {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: flex-start;
  font-weight: 700;
}
.path { margin-top: 4px; color: var(--gold); font-size: 13px; }
.snippet { margin: 8px 0 0; }
mark {
  background: #ffe08a;
  padding: 0 2px;
  border-radius: 3px;
}
.source {
  display: inline-block;
  color: var(--muted);
  font-size: 12px;
  margin-top: 8px;
}
.term-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
  gap: 8px;
}
.term-btn {
  text-align: left;
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 8px;
  padding: 8px 10px;
  cursor: pointer;
}
.term-btn strong { display: block; }
.term-btn span { color: var(--muted); font-size: 12px; }
.quiz-head {
  display: flex;
  gap: 10px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.quiz-section {
  margin: 18px 0;
}
.q {
  border-top: 1px solid var(--line);
  padding: 13px 0;
}
.q:first-child { border-top: 0; }
.q-stem { font-weight: 700; }
.options { margin: 8px 0 0; padding-left: 0; list-style: none; }
.options li { margin: 5px 0; }
.blank-answer {
  display: inline-block;
  min-width: 160px;
  border-bottom: 1px solid #9aa69d;
}
.answers {
  margin-top: 18px;
  background: var(--soft);
}
.answer-item {
  border-top: 1px solid var(--line);
  padding: 12px 0;
}
.answer-item:first-child { border-top: 0; }
.badge {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  color: var(--muted);
  margin-left: 6px;
}
.review-section {
  padding: 20px 0;
  border-top: 1px solid var(--line);
}
.review-section:first-child { border-top: 0; padding-top: 0; }
.review-section h2 { margin: 0 0 10px; font-size: 21px; }
.review-section li { margin: 7px 0; }
.formula {
  font-family: "SFMono-Regular", Consolas, monospace;
  background: #eef3ef;
  color: var(--code);
  padding: 8px 10px;
  border-radius: 7px;
  margin: 6px 0;
  overflow-wrap: anywhere;
}
.review-images {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
.review-images img {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.stat {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.stat strong {
  display: block;
  font-size: 22px;
}
@media (max-width: 860px) {
  .bar, .toolbar, .layout, .quiz-head {
    grid-template-columns: 1fr;
  }
  nav { width: 100%; }
  nav button { flex: 1; min-width: 0; }
  .toolbar { align-items: stretch; }
}
"""


JS = r"""
const DB = window.COURSE_DB;
const state = { query: "", selectedTerm: "" };

const $ = (id) => document.getElementById(id);
const norm = (s) => (s || "").toString().toLowerCase();
const escapeHtml = (s) => (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function highlight(text, query) {
  const safe = escapeHtml(text || "");
  const q = (query || "").trim();
  if (!q) return safe;
  const parts = q.split(/\s+/).filter(Boolean).slice(0, 6).map(x => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!parts.length) return safe;
  return safe.replace(new RegExp(`(${parts.join("|")})`, "gi"), "<mark>$1</mark>");
}

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach(b => b.classList.toggle("active", b.dataset.view === name));
}

function scoreChunk(chunk, query) {
  const q = norm(query).split(/\s+/).filter(Boolean);
  if (!q.length) return 0;
  const title = norm(chunk.title + " " + chunk.heading_path.join(" "));
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

function renderStats() {
  const distinctRandomPapers = Math.min(
    Math.floor(DB.quizBank.filter(q => q.type === "mcq").length / 20),
    Math.floor(DB.quizBank.filter(q => q.type === "fill").length / 10),
    Math.floor(DB.quizBank.filter(q => q.type === "short").length / 5),
  );
  $("stats").innerHTML = `
    <div class="stat"><strong>${DB.manifest.length}</strong><span>课件文件</span></div>
    <div class="stat"><strong>${DB.chunks.length}</strong><span>可搜索片段</span></div>
    <div class="stat"><strong>${Object.keys(DB.terms).length}</strong><span>索引词条</span></div>
    <div class="stat"><strong>${DB.quizBank.length}</strong><span>结构化题目</span></div>
    <div class="stat"><strong>${distinctRandomPapers}</strong><span>可支撑不重复整卷数</span></div>
  `;
}

function renderTermCloud() {
  const terms = Object.values(DB.terms).sort((a, b) => b.total_count - a.total_count).slice(0, 40);
  $("termCloud").innerHTML = terms.map(t => `
    <button class="term-btn" data-term="${escapeHtml(t.term)}">
      <strong>${escapeHtml(t.term)}</strong>
      <span>${t.total_count} 次 · ${t.files.length} 文件</span>
    </button>
  `).join("");
  $("termCloud").querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    $("searchInput").value = btn.dataset.term;
    runSearch(btn.dataset.term);
  }));
}

function explainQuery(query, results) {
  const exact = DB.terms[query];
  const fuzzy = exact || Object.values(DB.terms).find(t => norm(t.term).includes(norm(query)) || norm(query).includes(norm(t.term)));
  if (!query.trim()) {
    $("explainBox").innerHTML = `<h2>搜索说明</h2><p>输入词条后，会显示它在所有课件中的出现位置、所属章节和相关知识点。点击词条或结果可以生成相关复习题。</p>`;
    return;
  }
  if (fuzzy) {
    $("explainBox").innerHTML = `
      <h2>${escapeHtml(fuzzy.term)}</h2>
      <p>该词条在 <strong>${fuzzy.files.length}</strong> 个课件文件中出现，共索引到 <strong>${fuzzy.total_count}</strong> 次。下面列出最相关的位置和相邻知识点。</p>
      <div class="chips">${(fuzzy.related || []).map(r => `<button class="chip" data-term="${escapeHtml(r)}">${escapeHtml(r)}</button>`).join("")}</div>
      <button class="btn secondary" id="quizThisTerm">生成这个词条的复习题</button>
    `;
    $("explainBox").querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => {
      $("searchInput").value = chip.dataset.term;
      runSearch(chip.dataset.term);
    }));
    $("quizThisTerm").addEventListener("click", () => generateQuiz(fuzzy.term));
  } else {
    $("explainBox").innerHTML = `<h2>${escapeHtml(query)}</h2><p>没有找到精确词条索引，但全文搜索得到 <strong>${results.length}</strong> 个相关片段。可以根据这些片段生成综合复习题。</p><button class="btn secondary" id="quizThisTerm">生成相关复习题</button>`;
    $("quizThisTerm").addEventListener("click", () => generateQuiz(query));
  }
}

function runSearch(query) {
  state.query = query;
  const scored = DB.chunks.map(c => [scoreChunk(c, query), c]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 80);
  const results = scored.map(([, c]) => c);
  explainQuery(query, results);
  $("resultCount").textContent = query.trim() ? `${results.length} 个结果` : "等待输入";
  $("results").innerHTML = results.length ? results.map(c => `
    <article class="result">
      <div class="result-title">
        <span>${escapeHtml(c.title || c.file)}</span>
        <button class="chip result-quiz" data-topic="${escapeHtml((c.terms && c.terms[0]) || query)}">出题</button>
      </div>
      <div class="path">${escapeHtml((c.heading_path || []).join(" / "))}</div>
      <p class="snippet">${highlight(c.summary, query)}</p>
      <span class="source">${escapeHtml(c.file)} · cell ${c.cell_index} · ${escapeHtml(c.type)}</span>
      <div class="chips">${(c.terms || []).slice(0, 8).map(t => `<button class="chip" data-term="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join("")}</div>
    </article>
  `).join("") : `<div class="panel"><p class="meta">输入课件中的词条，例如“感受野”“BatchNorm”“LSTM”“CrossEntropyLoss”。</p></div>`;
  $("results").querySelectorAll(".chip[data-term]").forEach(chip => chip.addEventListener("click", () => {
    $("searchInput").value = chip.dataset.term;
    runSearch(chip.dataset.term);
  }));
  $("results").querySelectorAll(".result-quiz").forEach(btn => btn.addEventListener("click", () => generateQuiz(btn.dataset.topic)));
}

function pickQuestions(type, topic, count) {
  const q = norm(topic || "");
  const all = DB.quizBank.filter(x => x.type === type);
  const related = all.filter(x => norm([x.topic, x.stem, x.source].join(" ")).includes(q));
  const fallback = all.filter(x => !related.includes(x));
  const shuffle = (arr) => {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  };
  let selected = [];
  if (q) {
    const desiredRelated = Math.min(count, Math.max(Math.ceil(count * 0.7), Math.min(related.length, 3)));
    selected = shuffle(related).slice(0, desiredRelated);
    if (selected.length < count) {
      selected = selected.concat(shuffle(fallback).slice(0, count - selected.length));
    }
  } else {
    const hardQuota = type === "mcq" ? Math.min(3, count) : 0;
    const hard = shuffle(all.filter(x => x.difficulty === "难")).slice(0, hardQuota);
    const rest = shuffle(all.filter(x => !hard.includes(x))).slice(0, count - hard.length);
    selected = hard.concat(rest);
  }
  return selected.slice(0, count);
}

function generateQuiz(topic = "") {
  const mcq = pickQuestions("mcq", topic, 20);
  const fill = pickQuestions("fill", topic, 10);
  const short = pickQuestions("short", topic, 5);
  const title = topic ? `复习题：${topic}` : "综合随机复习题";
  const chosen = [...mcq, ...fill, ...short];
  $("quizTitle").textContent = title;
  $("quizMeta").textContent = `本次随机抽取 ${mcq.length} 道单选、${fill.length} 道填空、${short.length} 道简答。再次点击会重新随机生成。`;
  $("quizBody").innerHTML = `
    <section class="quiz-section"><h3>一、单选题</h3>${mcq.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>
    <section class="quiz-section"><h3>二、填空题</h3>${fill.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>
    <section class="quiz-section"><h3>三、简答题</h3>${short.map((q, i) => renderQuestion(q, i + 1)).join("")}</section>
    <section class="panel answers"><h3>答案与解析</h3>${chosen.map((q, i) => renderAnswer(q, i + 1)).join("")}</section>
  `;
  showView("quiz");
  window.scrollTo({top: 0, behavior: "smooth"});
}

function renderQuestion(q, idx) {
  if (q.type === "mcq") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><ul class="options">${q.options.map(o => `<li>${escapeHtml(o)}</li>`).join("")}</ul></div>`;
  }
  if (q.type === "fill") {
    return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="blank-answer"></span> <span class="badge">${escapeHtml(q.difficulty)}</span></div></div>`;
  }
  return `<div class="q"><div class="q-stem">${idx}. ${escapeHtml(q.stem)} <span class="badge">${escapeHtml(q.difficulty)}</span></div><p class="meta">答题区：</p><p style="height:54px;border-bottom:1px solid var(--line)"></p></div>`;
}

function renderAnswer(q, idx) {
  return `<div class="answer-item"><strong>${idx}. [${escapeHtml(q.id)}] ${escapeHtml(q.answer)}</strong><p>${escapeHtml(q.explanation)}</p><span class="source">来源：${escapeHtml(q.source)} · ${escapeHtml(q.topic)}</span></div>`;
}

function renderReview() {
  const r = DB.review;
  $("reviewBody").innerHTML = r.sections.map(sec => `
    <section class="review-section">
      <h2>${escapeHtml(sec.title)}</h2>
      <ul>${sec.points.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>
      ${sec.details ? `<h3>详细知识点</h3><ul>${sec.details.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${sec.must_memorize ? `<h3>必背结论</h3><ul>${sec.must_memorize.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      ${(sec.formulas || []).map(f => `<div class="formula">${escapeHtml(f)}</div>`).join("")}
      ${sec.formula_explainer ? `<h3>公式拆解</h3><ul>${sec.formula_explainer.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>` : ""}
      <p><strong>常考点：</strong>${escapeHtml((sec.exam_focus || []).join("；"))}</p>
      ${sec.question_patterns ? `<p><strong>命题方式：</strong>${escapeHtml(sec.question_patterns.join("；"))}</p>` : ""}
      ${sec.answer_template ? `<p><strong>答题模板：</strong>${escapeHtml(sec.answer_template.join("；"))}</p>` : ""}
      ${sec.traps ? `<p><strong>易错陷阱：</strong>${escapeHtml(sec.traps.join("；"))}</p>` : ""}
      ${sec.images ? `<div class="review-images">${sec.images.map(src => `<img src="${escapeHtml(src)}" alt="${escapeHtml(sec.title)}">`).join("")}</div>` : ""}
    </section>
  `).join("") + (r.references ? `<section class="review-section"><h2>参考资料</h2><ul>${r.references.map(ref => `<li><a href="${escapeHtml(ref.url)}">${escapeHtml(ref.title)}</a></li>`).join("")}</ul></section>` : "");
}

function init() {
  document.querySelectorAll("nav button").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.view)));
  $("searchInput").addEventListener("input", (e) => runSearch(e.target.value));
  $("makeQuiz").addEventListener("click", () => generateQuiz($("searchInput").value.trim()));
  $("makeFullQuiz").addEventListener("click", () => generateQuiz(""));
  $("quizAll").addEventListener("click", () => generateQuiz(""));
  renderStats();
  renderTermCloud();
  renderReview();
  runSearch("");
}

init();
"""


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>深度学习课件复习网站</title>
  <link rel="stylesheet" href="assets/course_site.css">
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>深度学习课件复习网站</h1>
        <p class="sub">搜索定位 · 自动组卷 · 综合复习资料</p>
      </div>
      <nav aria-label="主导航">
        <button class="active" data-view="search">搜索</button>
        <button data-view="quiz">复习题</button>
        <button data-view="review">复习资料</button>
      </nav>
    </div>
  </header>
  <main>
    <section id="view-search" class="view active">
      <div id="stats" class="stats"></div>
      <div class="toolbar">
        <input id="searchInput" type="search" placeholder="输入词条：如 感受野、BatchNorm、LSTM、CrossEntropyLoss">
        <button id="makeQuiz" class="btn secondary">按当前词条出题</button>
        <button id="makeFullQuiz" class="btn">综合出题</button>
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

    <section id="view-quiz" class="view">
      <div class="panel">
        <div class="quiz-head">
          <div>
            <h2 id="quizTitle">综合复习题</h2>
            <p id="quizMeta" class="meta">点击右侧按钮生成一份题。</p>
          </div>
          <button id="quizAll" class="btn">生成综合题</button>
        </div>
        <div id="quizBody"></div>
      </div>
    </section>

    <section id="view-review" class="view">
      <div class="panel">
        <h2>综合复习资料</h2>
        <p class="meta">这页按当前课件合并整理。之后新增课件时，重新构建即可更新搜索数据库和本页基础索引。</p>
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

    manifest, chunks = extract_notebooks()
    terms = build_term_index(chunks)
    review = enrich_review_material(review_material())
    quizzes = extend_quiz_bank(quiz_bank())

    for img in (ROOT / "image").glob("*.png"):
        shutil.copy2(img, IMAGE_OUT / img.name)

    write_json(DATABASE / "source_manifest.json", manifest)
    write_json(DATABASE / "course_chunks.json", chunks)
    write_json(DATABASE / "course_terms.json", terms)
    write_json(DATABASE / "review_material.json", review)
    write_json(DATABASE / "quiz_bank.json", quizzes)

    bundle = {
        "manifest": manifest,
        "chunks": chunks,
        "terms": terms,
        "review": review,
        "quizBank": quizzes,
    }
    (ASSETS / "course_data.js").write_text("window.COURSE_DB = " + json.dumps(bundle, ensure_ascii=False) + ";\n", encoding="utf-8")
    (ASSETS / "course_site.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (ASSETS / "course_site.js").write_text(JS.strip() + "\n", encoding="utf-8")
    (SITE / "index.html").write_text(HTML, encoding="utf-8")

    print(f"notebooks={len(manifest)} chunks={len(chunks)} terms={len(terms)} quiz={len(quizzes)}")
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
