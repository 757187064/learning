from collections import Counter


FORBIDDEN_QUIZ_PHRASES = [
    "只需要记名称",
    "不需要结合模型或训练流程",
    "与本章其它知识点没有关系",
    "最应该先掌握",
    "某同学复习",
    "只记住了名词",
    "阅读材料并回答问题",
    "本章中的作用",
    "容易混淆的点",
    "复习这一部分",
    "放回数据、模型结构",
    "会考定义、作用",
]


SOURCES = {
    "guide": "深度学习导学_20260609.md",
    "mlp1": "多层感知机1_学生分发版.ipynb",
    "mlp2": "多层感知机2-学生分发版-终版2026.ipynb",
    "cnn1": "卷积神经网络1-学生分发版-终版2026.ipynb",
    "cnn2": "卷积神经网络2-学生分发版-终版2026.ipynb",
    "cnn3": "卷积神经网络3-学生分发版-终版2026.ipynb",
    "rnn": "循环神经网络-学生分发版-终版2026.ipynb",
    "tf1": "注意力机制及Transformer1-学生分发版-终版2026.ipynb",
    "tf2": "注意力机制及Transformer2-学生分发版-终版2026.ipynb",
    "tf3": "注意力机制及Transformer3-学生分发版-终版2026(3).ipynb",
    "graph1": "图学习课件_第一部分_20260602.pptx",
    "graph2": "图学习课件_第二部分_20260602.pptx",
    "gnn": "图神经网络课件_20260609.pptx",
}


def rotate_options(correct, distractors, seed):
    choices = [correct] + distractors[:3]
    shift = seed % 4
    ordered = choices[-shift:] + choices[:-shift] if shift else choices
    labels = "ABCD"
    options = [f"{label}. {text}" for label, text in zip(labels, ordered)]
    answer = labels[ordered.index(correct)]
    return options, answer


def mcq(qid, topic, stem, correct, distractors, explanation, source, difficulty="中", seed=0, beginner=False):
    options, answer = rotate_options(correct, distractors, seed)
    return {
        "id": qid,
        "type": "mcq",
        "topic": topic,
        "stem": stem,
        "options": options,
        "answer": answer,
        "explanation": f"{answer}正确：{explanation}",
        "source": source,
        "difficulty": difficulty,
        "beginner": beginner,
    }


def short(qid, topic, stem, answer, explanation, source, difficulty="中", beginner=False):
    return {
        "id": qid,
        "type": "short",
        "topic": topic,
        "stem": stem,
        "answer": answer,
        "explanation": explanation,
        "source": source,
        "difficulty": difficulty,
        "beginner": beginner,
    }


def material(qid, topic, material_text, subquestions, answer, explanation, source, difficulty="中", beginner=False):
    return {
        "id": qid,
        "type": "material",
        "topic": topic,
        "stem": f"材料题 {qid}：根据下面关于“{topic}”的场景材料回答问题。",
        "material": material_text,
        "subquestions": subquestions,
        "answer": answer,
        "explanation": explanation,
        "source": source,
        "difficulty": difficulty,
        "beginner": beginner,
    }


def option_text(option):
    return option[3:].strip() if len(option) > 3 and option[1:3] == ". " else option.strip()


def relabel_mcq(q, target_label):
    labels = "ABCD"
    current_idx = labels.index(q["answer"])
    texts = [option_text(option) for option in q["options"]]
    correct = texts[current_idx]
    distractors = [text for idx, text in enumerate(texts) if idx != current_idx]
    target_idx = labels.index(target_label)
    ordered = [None, None, None, None]
    ordered[target_idx] = correct
    for idx in range(4):
        if ordered[idx] is None:
            ordered[idx] = distractors.pop(0)
    old_answer = q["answer"]
    q["options"] = [f"{label}. {text}" for label, text in zip(labels, ordered)]
    q["answer"] = target_label
    q["explanation"] = q["explanation"].replace(f"{old_answer}正确：", f"{target_label}正确：", 1)


def balance_mcq_answers(bank):
    labels = "ABCD"
    mcqs = [q for q in bank if q["type"] == "mcq"]
    for idx, q in enumerate(mcqs):
        relabel_mcq(q, labels[idx % 4])


MCQ_SPECS = [
    # 导学、神经元、MLP
    ("神经元与感知机", "关于 TLU 与感知机的关系，下列说法正确的是",
     "二者都以加权求和为核心，感知机进一步把这种结构用于线性分类边界。",
     ["TLU 不包含权重，只比较输入个数。", "单层感知机可以直接解决 XOR 的非线性可分问题。", "ADALINE 和感知机训练时完全一样，都只使用阶跃后的二值输出。"],
     "TLU、感知机都围绕加权求和和阈值/偏置判断；XOR 需要多层或非线性结构。", SOURCES["guide"], "易", True),
    ("神经元与感知机", "若二维平面中两类样本不能用一条直线分开，单层感知机通常会遇到什么问题？",
     "模型表达能力不足，即使继续训练也无法表示非线性决策边界。",
     ["只要把学习率调大就一定能收敛到正确分类。", "只要增加 epoch，单层感知机就能自动变成多层网络。", "把偏置去掉后反而能表达任意非线性边界。"],
     "单层感知机对应线性边界，训练不能弥补结构表达能力不足。", SOURCES["guide"], "中", True),
    ("神经元与感知机", "ADALINE 相比普通感知机，更适合引出损失函数和梯度下降，原因是",
     "它在连续线性输出上衡量误差，使参数更新可以和误差最小化联系起来。",
     ["它完全不使用权重参数。", "它只适用于图像卷积任务。", "它训练时不需要真实标签。"],
     "ADALINE 的连续误差视角自然连接损失函数、梯度和参数更新。", SOURCES["guide"], "中", False),
    ("MLP 与激活函数", "多层线性层中间不加激活函数时，下列判断正确的是",
     "多层线性变换仍可合并为一个线性变换，表达能力不会变成非线性。",
     ["层数越多就一定能表达 XOR。", "不加激活函数时模型更容易表达复杂曲线边界。", "线性层堆叠后会自动产生 ReLU 效果。"],
     "线性变换的复合仍是线性变换，激活函数负责引入非线性。", SOURCES["mlp2"], "易", True),
    ("MLP 与分类损失", "二分类模型最后一层输出一个 logit，训练时最稳妥的 PyTorch 损失通常是",
     "BCEWithLogitsLoss，因为它把 Sigmoid 和 BCE 合并，数值更稳定。",
     ["CrossEntropyLoss，并且输入必须先手动 Softmax。", "MSELoss，因为分类任务都要回归到 0.5。", "NLLLoss，并且标签必须是 one-hot 浮点矩阵。"],
     "BCEWithLogitsLoss 适合二分类或多标签 logits；CrossEntropyLoss 多用于单标签多分类 logits。", SOURCES["mlp2"], "中", True),
    ("MLP 与分类损失", "多分类任务使用 CrossEntropyLoss 时，模型输出和标签通常应满足",
     "输出是每类 logits，标签是类别编号，不需要先手动 Softmax。",
     ["输出必须是 Sigmoid 后的独立概率，标签必须全为浮点数。", "输出必须先转 one-hot，标签必须是概率分布。", "CrossEntropyLoss 只适用于二分类，不能处理多分类。"],
     "CrossEntropyLoss 内部包含 LogSoftmax/NLLLoss 类计算，直接接收 logits。", SOURCES["mlp2"], "中", True),
    ("训练流程与自动微分", "一个标准 PyTorch 训练 batch 的顺序最合理的是",
     "清零梯度、前向传播、计算损失、反向传播、优化器更新。",
     ["前向传播、优化器更新、清零梯度、计算损失、反向传播。", "计算损失、清零梯度、优化器更新、前向传播、反向传播。", "优化器更新、反向传播、前向传播、计算损失、清零梯度。"],
     "旧梯度需要先清零，loss.backward() 计算梯度，optimizer.step() 才更新参数。", SOURCES["mlp1"], "易", True),
    ("训练流程与自动微分", "验证模型性能时通常使用 model.eval() 和 torch.no_grad()，主要目的是",
     "切换训练态层的行为并避免记录梯度，减少验证阶段干扰和开销。",
     ["让优化器继续更新参数。", "强制 Dropout 丢弃更多神经元。", "让 BatchNorm 使用当前 batch 重新训练参数。"],
     "eval 影响 Dropout/BatchNorm 等层行为，no_grad 避免构建反向图。", SOURCES["mlp1"], "中", True),
    ("训练流程与自动微分", "如果在 PyTorch 训练循环中长期忘记 optimizer.zero_grad()，最可能出现的问题是",
     "梯度会跨 batch 累积，导致参数更新方向和步幅异常。",
     ["模型会完全停止前向传播。", "损失函数会自动变成准确率。", "验证集会自动参与训练。"],
     "PyTorch 默认累积梯度，不清零会把多次 backward 的梯度叠加。", SOURCES["mlp1"], "中", False),
    ("数据处理与训练", "StandardScaler 用在训练集和验证集时，正确做法是",
     "只在训练集 fit，再用同一个 scaler transform 训练集和验证集。",
     ["分别在训练集和验证集 fit，以便各自更接近标准正态。", "只在验证集 fit，因为验证集代表真实泛化。", "先把训练集和验证集合并 fit，再重新拆开。"],
     "预处理参数不能从验证集泄漏到训练流程，否则评估会偏乐观。", SOURCES["mlp2"], "中", False),
    ("张量与 Batch", "关于 Batch、Epoch、Iteration 的关系，下列说法正确的是",
     "一个 epoch 表示完整遍历训练集一次，通常包含多个 batch/iteration。",
     ["一个 batch 就等于完整训练集。", "一个 iteration 必须包含所有 epoch。", "Epoch 表示一次 optimizer.zero_grad()，与数据量无关。"],
     "Batch 是一批样本；iteration 通常对应一次 batch 更新；epoch 是全训练集一遍。", SOURCES["mlp1"], "易", True),
    ("优化器与学习率", "学习率设置过大时，训练中常见现象是",
     "损失震荡、难以收敛，甚至出现数值发散。",
     ["收敛一定更快且泛化一定更好。", "反向传播将不再产生梯度。", "BatchNorm 会被自动关闭。"],
     "学习率控制更新步长，过大可能越过较优区域。", SOURCES["mlp1"], "易", True),
    ("优化器与学习率", "RMSProp 相比 Adagrad 的一个重要改进是",
     "用梯度平方的指数滑动平均调整尺度，缓解学习率过快衰减。",
     ["完全不使用梯度信息。", "只适用于卷积核，不适用于全连接层。", "把所有参数的学习率固定成 0。"],
     "RMSProp 保留近期梯度平方统计，避免 Adagrad 累积和持续增大导致学习率过小。", SOURCES["cnn2"], "中", False),
    ("正则化与泛化", "权重衰减主要通过什么方式缓解过拟合？",
     "惩罚过大的权重，使模型倾向于更平滑、更简单的参数解。",
     ["随机删除训练样本标签。", "在推理时强制改变类别数。", "把卷积核大小固定为 1x1。"],
     "权重衰减通常表现为 L2 正则项，限制参数过大。", SOURCES["cnn2"], "中", False),
    ("初始化与训练稳定性", "Kaiming 初始化通常更适合与哪类激活函数搭配？",
     "ReLU 及其变体，因为它考虑了这类激活的方差传播特点。",
     ["Softmax 输出层。", "任意不可导阶跃函数。", "只用于 RNN 的隐藏状态初始化。"],
     "Kaiming/He 初始化常用于 ReLU 网络，Xavier 更常用于 tanh/sigmoid 等场景。", SOURCES["cnn2"], "中", False),
    # CNN
    ("CNN 基础", "全连接网络直接处理图像时，最核心的缺陷是",
     "展平会破坏空间结构，且参数量随像素数快速膨胀。",
     ["它无法接收浮点数输入。", "它一定比 CNN 参数更少。", "它会自动产生平移不变性。"],
     "CNN 出现的动机就是保留局部空间关系并通过参数共享减少参数。", SOURCES["cnn1"], "易", True),
    ("CNN 卷积计算", "输入尺寸为 10x10，卷积核 3x3，步长 1，无填充，输出空间尺寸是",
     "8x8。",
     ["10x10。", "7x7。", "12x12。"],
     "输出边长为 (10 + 2*0 - 3) / 1 + 1 = 8。", SOURCES["cnn1"], "易", True),
    ("CNN 卷积计算", "若希望 5x5 卷积核、步长 1 时输出尺寸尽量与输入相同，通常需要",
     "在边界做 padding，常见对称填充大小为 2。",
     ["把 stride 改成 5。", "删除所有边界像素。", "把卷积核参数固定不训练。"],
     "Same 卷积通过合适填充保持空间尺寸，5x5 核对应半径 2。", SOURCES["cnn1"], "中", True),
    ("CNN 卷积计算", "深度学习框架中的 Conv2d 通常实际执行的是",
     "互相关运算，卷积核不做数学意义上的翻转。",
     ["严格卷积且每次手动翻转卷积核。", "只做全连接矩阵乘法，不看局部区域。", "只做最大池化，不含可学习参数。"],
     "卷积核会被训练学习，互相关足以完成局部模板匹配。", SOURCES["cnn1"], "中", False),
    ("CNN 通道", "RGB 图像输入 Conv2d 时，卷积核的输入通道数应与什么一致？",
     "输入图像通道数，一般 RGB 为 3。",
     ["输出类别数。", "batch size。", "训练 epoch 数。"],
     "Conv2d 的 in_channels 必须匹配输入张量的通道维。", SOURCES["cnn1"], "易", True),
    ("CNN 数据格式", "PyTorch 中图像张量常用的维度顺序是",
     "NCHW，即 batch、channel、height、width。",
     ["NHWC 是 PyTorch 卷积层唯一接受的格式。", "CHWN，即 channel、height、width、batch。", "HWNC，即 height、width、batch、channel。"],
     "PyTorch Conv2d 默认使用 NCHW；不同框架可能使用不同格式。", SOURCES["cnn1"], "易", True),
    ("CNN 池化", "最大池化层的主要作用更接近于",
     "保留局部显著响应、降低空间尺寸，并增强一定平移不变性。",
     ["增加大量可学习参数。", "把类别标签转换为 one-hot。", "替代损失函数计算交叉熵。"],
     "池化通常没有可学习参数，属于降采样/汇聚操作。", SOURCES["cnn1"], "易", True),
    ("CNN 感受野", "关于感受野，下列说法正确的是",
     "层数加深或卷积/池化堆叠后，高层单元通常对应更大的原始输入区域。",
     ["感受野只会随网络加深而缩小。", "感受野与卷积核大小和步长都无关。", "感受野只存在于全连接层。"],
     "感受野描述高层特征对应原图范围，多层小卷积也能逐步扩大感受野。", SOURCES["cnn1"], "中", False),
    ("CNN 架构", "VGG 网络设计最典型的特点是",
     "大量堆叠 3x3 小卷积核构建更深网络。",
     ["使用复杂多分支 Inception 模块。", "完全不用卷积层，只用循环结构。", "只依赖图邻接矩阵传播消息。"],
     "VGG 的代表性设计是小卷积核、深层堆叠。", SOURCES["cnn3"], "易", True),
    ("CNN 架构", "GoogLeNet/Inception 模块的核心思想是",
     "在同一层并行使用不同尺度卷积/池化分支，再拼接多尺度特征。",
     ["只使用单一路径 3x3 卷积。", "把所有像素展开后直接输入 LSTM。", "取消所有非线性激活函数。"],
     "Inception 通过多分支结构捕捉不同尺度信息。", SOURCES["cnn3"], "中", False),
    ("CNN 架构", "ResNet 中残差连接的主要作用是",
     "提供跨层信息和梯度通路，缓解深层网络退化与训练困难。",
     ["让所有层参数完全共享。", "把图像通道数强制改为 1。", "使模型只能学习线性分类边界。"],
     "残差块学习 F(x)+x，有助于训练更深网络。", SOURCES["cnn3"], "中", True),
    ("CNN 训练技巧", "BatchNorm 在训练和推理阶段的关键差异是",
     "训练时使用当前 batch 统计并更新累计统计，推理时通常使用累计均值方差。",
     ["推理时随机丢弃神经元。", "训练时完全不计算均值方差。", "推理时必须重新反向传播。"],
     "BatchNorm 的模式差异是训练/评估切换的重要考点。", SOURCES["cnn2"], "中", True),
    ("CNN 训练技巧", "Dropout 的正确理解是",
     "训练时随机失活部分神经元，推理时关闭随机失活。",
     ["推理时随机失活更多神经元。", "它的主要作用是扩大卷积核尺寸。", "它会替代 optimizer.step()。"],
     "Dropout 是正则化方法，训练和推理行为不同。", SOURCES["cnn2"], "易", True),
    ("CNN 数据增强", "图像数据增强的主要目的通常是",
     "让模型见到合理变化，提高泛化能力。",
     ["让验证集标签参与训练。", "减少所有训练样本数量。", "保证训练准确率必然达到 100%。"],
     "随机裁剪、翻转、颜色扰动等增强有助于缓解过拟合。", SOURCES["cnn2"], "易", True),
    ("CNN 类别不平衡", "WeightedRandomSampler 更适合处理什么问题？",
     "训练集中不同类别样本数量差异明显，采样时希望提高少数类被抽到的概率。",
     ["卷积核参数无法更新。", "输入图像通道顺序错误。", "Transformer 缺少位置编码。"],
     "加权采样是处理类别不平衡的一种数据层面方法。", SOURCES["cnn2"], "中", False),
    ("CNN 迁移学习", "迁移学习中常见的做法是",
     "使用预训练网络作为特征提取基础，再替换或微调分类头。",
     ["从验证集生成训练标签。", "删除所有卷积层，只保留随机输出。", "把 batch size 当成类别数。"],
     "预训练模型已有通用视觉特征，微调可减少数据和训练成本。", SOURCES["cnn3"], "中", False),
    # RNN
    ("RNN 基础", "RNN 能处理序列数据的关键在于",
     "隐藏状态在时间步之间传递，当前计算能利用过去信息。",
     ["每个时间步完全独立。", "只使用二维卷积核处理 token。", "输出层不需要任何参数。"],
     "隐藏状态是 RNN 连接前后时间步的核心。", SOURCES["rnn"], "易", True),
    ("RNN 基础", "普通 RNN 在长序列上常见的训练困难是",
     "梯度消失或梯度爆炸，使较早时间步信息难以有效学习。",
     ["无法接收离散 token。", "不能使用任何损失函数。", "每个时间步都会自动拥有无限记忆。"],
     "长时间反向传播会产生梯度连乘问题。", SOURCES["rnn"], "中", True),
    ("RNN GRU", "GRU 中更新门的直观作用是",
     "控制历史信息保留多少、新信息写入多少。",
     ["控制卷积核是否翻转。", "决定 batch size 的大小。", "替代词嵌入矩阵。"],
     "GRU 用门控在保留旧状态和接受新输入之间折中。", SOURCES["rnn"], "中", True),
    ("RNN GRU", "GRU 与 LSTM 的一个常见区别是",
     "GRU 通常没有独立细胞状态，结构比 LSTM 更简洁。",
     ["GRU 有输入门、遗忘门、输出门和细胞状态四套结构。", "LSTM 没有门控。", "GRU 只能处理图像，不能处理序列。"],
     "GRU 常见门是更新门、重置门；LSTM 有细胞状态和三类门。", SOURCES["rnn"], "中", True),
    ("RNN LSTM", "LSTM 中遗忘门的作用是",
     "决定上一时刻细胞状态中哪些信息继续保留。",
     ["决定卷积输出尺寸。", "把所有 token 替换为 padding。", "计算 CrossEntropyLoss。"],
     "遗忘门是 LSTM 控制长期记忆的重要组成。", SOURCES["rnn"], "中", False),
    ("RNN LSTM", "LSTM 的细胞状态 c_t 更接近于",
     "一条相对稳定的长期信息通道，由门控结构控制读写和遗忘。",
     ["模型固定权重参数。", "当前 batch 的类别标签。", "卷积核的空间尺寸。"],
     "细胞状态是 LSTM 区别于简单 RNN/GRU 的关键。", SOURCES["rnn"], "中", False),
    ("RNN 变长序列", "处理变长序列组成 batch 时，padding 的问题是",
     "补齐位置不是真实时间步，需要 mask、packing 或长度信息避免干扰。",
     ["padding 位置一定代表真实词。", "padding 会自动提升少数类采样概率。", "padding 只能用于 CNN，不能用于 RNN。"],
     "变长序列训练要区分真实 token 和补齐 token。", SOURCES["rnn"], "中", True),
    ("RNN 变长序列", "pack_padded_sequence 的目的主要是",
     "让 RNN 跳过 padding 位置，只处理有效时间步。",
     ["把所有序列变成图邻接矩阵。", "在输出层自动 Softmax。", "让每个序列长度都变为 1。"],
     "Packing 是 PyTorch 处理变长序列的常见工具。", SOURCES["rnn"], "中", False),
    ("RNN 双向结构", "双向 RNN 的优势是",
     "当前位置表示可以同时利用前文和后文上下文。",
     ["它适合严格实时预测未来输入。", "它删除了隐藏状态。", "它只能用于无监督聚类。"],
     "双向结构拼接正向和反向信息，但不适合只能看过去的在线场景。", SOURCES["rnn"], "中", False),
    ("RNN 堆叠结构", "堆叠 RNN 指的是",
     "把多层循环网络上下堆叠，上一层的输出序列作为下一层输入。",
     ["把同一个时间步复制多份作为 batch。", "把所有隐藏状态直接丢弃。", "把序列长度固定为卷积核大小。"],
     "堆叠增加的是层深，不是简单增加时间步。", SOURCES["rnn"], "中", False),
    ("RNN 1D卷积", "1D 卷积用于序列时更擅长",
     "并行提取局部窗口模式，例如局部 n-gram 特征。",
     ["像 LSTM 一样天然保存任意长度记忆。", "替代所有 token embedding。", "只用于二维彩色图像。"],
     "1D 卷积沿序列维滑动，适合局部模式，不等价于循环记忆。", SOURCES["rnn"], "中", False),
    ("RNN 输出", "在 many-to-one 序列分类中，常见做法是",
     "使用最后有效时间步隐藏状态或池化后的序列表示进行分类。",
     ["对每个 padding 位置单独分类后取最大标签。", "只使用第一个 batch 的标签训练全部样本。", "把类别数作为序列长度输入 RNN。"],
     "序列分类要从时间维得到一个整体表示，变长序列要注意最后有效位置。", SOURCES["rnn"], "中", False),
    ("RNN Teacher Forcing", "Seq2Seq 训练中的 Teacher Forcing 指的是",
     "训练解码器时把真实上一时刻目标 token 作为下一步输入的一种策略。",
     ["推理时直接读取未来真实答案。", "把 encoder 的所有参数冻结为 0。", "用图邻接矩阵替代目标序列。"],
     "Teacher Forcing 能稳定训练，但训练推理输入分布可能不同。", SOURCES["tf1"], "中", False),
    ("RNN 梯度裁剪", "RNN 训练中使用梯度裁剪，主要是为了",
     "限制过大的梯度范数，缓解梯度爆炸。",
     ["让梯度消失更严重。", "使隐藏状态不再传递。", "自动完成词表分词。"],
     "梯度裁剪不是解决过拟合的首要方法，而是训练稳定性技巧。", SOURCES["rnn"], "中", False),
    ("RNN 情感分类", "IMDB 情感分类中，Embedding 层的作用通常是",
     "把离散词编号映射为可训练的稠密向量表示。",
     ["直接输出最终正负标签。", "删除序列顺序。", "把所有句子转换成图像像素。"],
     "神经网络不能直接理解词编号大小，需要 embedding 表示语义和可学习特征。", SOURCES["rnn"], "易", True),
    # Transformer
    ("Attention 机制", "缩放点积注意力的核心计算顺序是",
     "Q 与 K 点积得到分数，除以 sqrt(d_k)，Softmax 后对 V 加权求和。",
     ["V 与 K 点积后直接作为标签。", "Q、K、V 都不参与权重计算。", "先池化图像再计算邻接矩阵。"],
     "Q/K 决定权重，V 提供被加权汇总的信息。", SOURCES["tf1"], "易", True),
    ("Attention 机制", "注意力公式中除以 sqrt(d_k) 的主要原因是",
     "避免点积随维度增大而过大，导致 Softmax 过于尖锐、梯度不稳定。",
     ["增加一个可学习参数。", "把序列长度缩短为 1。", "强制所有权重完全相等。"],
     "缩放项是稳定数值的固定因子，不是模型学出来的参数。", SOURCES["tf1"], "中", False),
    ("Attention 机制", "Self-Attention 中 Q、K、V 的来源通常是",
     "同一输入序列经过不同线性变换得到。",
     ["三个完全无关的数据集。", "卷积层的三个池化输出。", "优化器保存的三个学习率。"],
     "Self-Attention 是序列内部位置之间相互建模。", SOURCES["tf1"], "中", True),
    ("Transformer Mask", "Target Mask 在自回归解码器中的作用是",
     "遮住未来 token，防止当前位置看到还不该知道的答案。",
     ["只用于屏蔽源序列中的 padding。", "让模型忽略所有历史 token。", "把注意力头数变成 1。"],
     "目标序列生成必须保持因果性，不能偷看未来。", SOURCES["tf1"], "中", True),
    ("Transformer Mask", "Source Mask 更常用于",
     "屏蔽源序列中的 padding 位置，避免注意力关注无效补位。",
     ["阻止解码器看未来 token。", "替代位置编码。", "改变词表大小。"],
     "Source Mask 和 Target/Subsequent Mask 的目的不同。", SOURCES["tf1"], "中", False),
    ("Transformer 位置编码", "Transformer 需要位置编码的原因是",
     "自注意力本身不含天然顺序信息，需要额外注入 token 位置。",
     ["RNN 已经提供隐藏状态，所以 Transformer 不需要输入。", "卷积核会自动记录每个词的绝对位置。", "Softmax 会自动产生词序。"],
     "没有位置编码，序列重排可能难以区分。", SOURCES["tf1"], "易", True),
    ("Transformer 多头注意力", "多头注意力的主要价值是",
     "让模型在不同子空间并行关注不同关系，再拼接融合。",
     ["把一个注意力头复制成相同输出，不增加表达角度。", "只为了减少所有参数到 0。", "用于替代所有残差连接。"],
     "不同头可学习不同类型的依赖关系。", SOURCES["tf1"], "中", True),
    ("Transformer 结构", "Transformer 编码器层通常包含",
     "多头自注意力、前馈网络、残差连接和归一化。",
     ["卷积核翻转、池化、反卷积和图采样。", "遗忘门、输入门、输出门和细胞状态。", "判别器、生成器和对抗损失。"],
     "编码器不包含 LSTM 门结构，也不是 GAN。", SOURCES["tf2"], "易", True),
    ("Transformer 结构", "残差连接在 Transformer 中的一个重要作用是",
     "保留输入信息并改善深层网络中的梯度传播。",
     ["让注意力权重总和不为 1。", "删除位置编码。", "把 token 变成图节点。"],
     "残差连接有助于深层堆叠训练稳定。", SOURCES["tf2"], "中", False),
    ("Transformer 结构", "LayerNorm 与 BatchNorm 的常见区别之一是",
     "LayerNorm 通常沿特征维对单个样本归一化，更适合变长序列场景。",
     ["LayerNorm 必须依赖整个 batch 的均值方差。", "BatchNorm 只能用于文本，不能用于图像。", "二者都只在推理时工作。"],
     "Transformer 中常用 LayerNorm，避免过度依赖 batch 统计。", SOURCES["tf2"], "中", False),
    ("Transformer 结构", "前馈网络 FFN 在 Transformer 层中的作用更接近",
     "对每个位置的表示做非线性变换，提升特征表达能力。",
     ["负责在时间步之间递归传递隐藏状态。", "直接计算图的度矩阵。", "只用于图片上采样。"],
     "FFN 通常逐位置共享，用线性层和激活函数增强表达。", SOURCES["tf2"], "中", False),
    ("Transformer RoPE", "RoPE 属于哪类思想？",
     "位置编码思想，通过旋转方式把位置信息注入表示。",
     ["损失函数选择方法。", "卷积池化方法。", "优化器动量算法。"],
     "RoPE 是现代 Transformer 常见的位置表达方式。", SOURCES["tf3"], "易", False),
    ("Seq2Seq 与 Attention", "传统 Seq2Seq 引入 Attention，主要是为了",
     "让解码器在生成时动态关注编码器不同时间步输出，缓解固定上下文瓶颈。",
     ["让模型完全不需要编码器。", "把所有目标词一次性随机生成。", "用池化替代词嵌入。"],
     "Attention 使解码器不只依赖一个固定长度向量。", SOURCES["tf1"], "中", False),
    ("BERT 与 GPT", "BERT 和 GPT 的预训练目标常见区别是",
     "BERT 偏双向理解和 MLM，GPT 偏自回归预测下一个 token。",
     ["BERT 只能做图像卷积，GPT 只能做邻接矩阵分解。", "二者都必须使用同一个 Target Mask 训练。", "GPT 训练时可以直接看到未来 token。"],
     "BERT/GPT 的 mask 和目标不同，理解与生成侧重点不同。", SOURCES["tf3"], "中", False),
    ("Transformer 代码配置", "若 MultiHeadAttention 的 embed_dim=128、num_heads=8，则每个头的维度通常是",
     "16。",
     ["8。", "64。", "136。"],
     "每头维度通常为 embed_dim / num_heads = 128 / 8 = 16。", SOURCES["tf2"], "易", True),
    # 图学习、GNN
    ("图学习基础", "图学习中，邻接矩阵 A 通常表示",
     "节点之间是否有边或边的权重。",
     ["每个节点的类别标签。", "每个样本的 batch 编号。", "卷积核中每个参数的梯度。"],
     "邻接矩阵是图结构连接关系的矩阵表示。", SOURCES["graph1"], "易", True),
    ("图学习基础", "节点分类任务的预测对象是",
     "图中每个节点的类别或属性。",
     ["整张图唯一的标签。", "两个节点之间是否连边。", "每个卷积核的大小。"],
     "图任务要区分节点级、边级和图级输出。", SOURCES["graph1"], "易", True),
    ("图学习基础", "分子性质预测若输出整个分子的一个性质值，通常属于",
     "图级任务。",
     ["节点分类任务。", "边方向预测任务。", "卷积输出尺寸计算任务。"],
     "输出对应整张分子图，不是每个原子节点。", SOURCES["graph1"], "中", False),
    ("图表示学习", "DeepWalk 的基本思路是",
     "用随机游走生成节点序列，再借鉴 Skip-Gram 学习节点表示。",
     ["直接使用 CNN 在规则网格上滑动。", "只根据节点编号大小排序分类。", "用 Teacher Forcing 生成目标句子。"],
     "DeepWalk 把图上的游走序列类比为文本句子。", SOURCES["graph1"], "中", False),
    ("图表示学习", "node2vec 相比 DeepWalk 的改进重点是",
     "通过参数控制随机游走偏向 BFS 或 DFS 风格，调整局部/全局探索。",
     ["取消随机游走，只保留全连接层。", "把所有节点都当作 padding。", "用扩散模型替代邻接矩阵。"],
     "node2vec 的 p、q 控制游走策略。", SOURCES["graph1"], "中", False),
    ("GNN 消息传递", "GNN 消息传递的核心流程可以概括为",
     "节点接收邻居消息，聚合后更新自身表示。",
     ["每个节点只看自己，不接收任何边上的信息。", "先把图像展平，再做 CrossEntropyLoss。", "把所有邻接关系删除后再训练。"],
     "多数空间 GNN 都可用 send/aggregate/update 思路解释。", SOURCES["gnn"], "易", True),
    ("GCN", "GCN 中加入自环的目的通常是",
     "让节点聚合邻居信息时也保留自身特征。",
     ["让所有边权变为 0。", "删除度矩阵。", "把节点分类改成图像分类。"],
     "A+I 可以让节点自身也参与消息聚合。", SOURCES["gnn"], "中", True),
    ("GCN", "GCN 公式中的度矩阵归一化主要是为了",
     "校正不同度数节点聚合时的信息尺度，使训练更稳定。",
     ["把所有节点标签变成 one-hot。", "让模型不再使用邻接矩阵。", "替代所有激活函数。"],
     "度大的节点直接求和会带来尺度差异，归一化能缓解。", SOURCES["gnn"], "中", False),
    ("GAT", "GAT 相比普通 GCN 的一个核心差异是",
     "GAT 学习不同邻居的重要性权重，再进行加权聚合。",
     ["GAT 完全不使用邻居信息。", "GAT 只能处理规则图像网格。", "GAT 不包含任何可学习参数。"],
     "图注意力让模型区分邻居贡献，而不是固定平均。", SOURCES["gnn"], "中", True),
    ("GraphSAGE", "GraphSAGE 适合大图或新节点场景的重要原因是",
     "它通过邻居采样和聚合函数学习归纳式表示。",
     ["它必须一次性读取全图所有邻居。", "它只适用于无边图。", "它把节点全部转换为图片像素。"],
     "GraphSAGE 的 sampling/aggregation 有助于扩展到大图和未见节点。", SOURCES["gnn"], "中", False),
    ("图学习应用", "金融欺诈检测使用图神经网络时，边通常可以表示",
     "账号之间的交易、关联或交互关系。",
     ["卷积核的宽度。", "训练 epoch 编号。", "Softmax 的类别概率总和。"],
     "图学习应用的关键是把对象及其关系建模出来。", SOURCES["gnn"], "易", False),
    ("图学习应用", "材料或分子建图时，节点和边较常见的含义是",
     "节点表示原子，边表示化学键或空间邻近相互作用。",
     ["节点表示 batch，边表示学习率。", "节点表示卷积核，边表示 padding。", "节点表示 token 位置，边表示 Teacher Forcing。"],
     "分子/晶体结构天然适合用图表达局部关系。", SOURCES["graph2"], "易", True),
    # NLP/LLM/生成式/PINN/材料
    ("NLP 任务", "词向量或 embedding 的作用是",
     "把离散词或 token 映射为可训练的连续向量表示。",
     ["把所有词替换为同一个编号。", "直接删除序列顺序和语义。", "只用于计算图像通道数。"],
     "神经网络需要连续向量输入，embedding 是 NLP 的基础表示。", SOURCES["graph2"], "易", True),
    ("语言模型", "自回归语言模型生成文本时，一般遵循",
     "根据已有 token 逐步预测下一个 token。",
     ["一次性读取未来所有答案后再生成当前 token。", "只用邻接矩阵预测图节点标签。", "完全不依赖上下文。"],
     "GPT 类模型典型目标是 next-token prediction。", SOURCES["graph2"], "易", True),
    ("LLM Prompt", "Prompt 在大语言模型中的作用更接近",
     "提供任务指令、上下文或示例，引导模型生成目标输出。",
     ["永久修改模型参数。", "替代所有预训练数据。", "把 Transformer 改成 CNN。"],
     "Prompt 不等于参数更新，但会明显影响模型输出方向。", SOURCES["graph2"], "易", True),
    ("LLM MoE", "MoE 架构中的路由器主要负责",
     "为输入选择部分专家参与计算，以提高容量和计算效率。",
     ["让每个专家每次都完整运行。", "把所有 token 变成 padding。", "只计算卷积输出尺寸。"],
     "MoE 通常是稀疏激活，不是所有专家全量参与。", SOURCES["graph2"], "中", False),
    ("LLM 推理模型", "DeepSeek R1 一类推理模型的复习重点更接近",
     "理解其通过后训练/强化学习等方式增强复杂推理表现。",
     ["背诵某个网页界面按钮。", "把图像像素直接当作邻接矩阵。", "认为它与 Transformer 和语言模型完全无关。"],
     "课程层面更可能考概念和演进脉络，而不是工程细节。", SOURCES["graph2"], "易", False),
    ("生成式模型", "判别模型与生成模型的核心区别是",
     "判别模型偏判断标签，生成模型偏学习数据分布并产生新样本。",
     ["生成模型只能做分类，不能产生样本。", "判别模型必须先从噪声生成图片。", "二者训练目标完全相同，没有区别。"],
     "生成式算法的关键词是数据分布、采样、生成过程。", SOURCES["gnn"], "易", True),
    ("GAN", "GAN 的训练过程可以概括为",
     "生成器试图生成逼真样本，判别器试图区分真实与生成样本，二者对抗优化。",
     ["只有一个网络负责同时做所有任务。", "生成器输入必须是真实标签而不是噪声。", "判别器的目标是让所有输入都判为真实。"],
     "GAN 的核心是 generator 与 discriminator 的博弈。", SOURCES["gnn"], "中", True),
    ("VAE", "VAE 的基本生成思路是",
     "把输入编码到潜在分布，再从潜变量采样并解码生成样本。",
     ["只做最大池化，不涉及潜变量。", "完全不需要编码器或解码器。", "只能处理图的邻接矩阵归一化。"],
     "VAE 用潜在变量建模复杂数据分布。", SOURCES["gnn"], "中", False),
    ("Diffusion", "扩散模型的直观流程是",
     "前向逐步加噪，反向学习逐步去噪生成样本。",
     ["先分类再删除所有噪声参数。", "只训练一个判别器判断真假。", "通过邻居聚合更新节点标签。"],
     "Diffusion 的关键词是加噪过程和反向去噪过程。", SOURCES["gnn"], "易", True),
    ("Flow-based model", "Flow-based model 的关键特点是",
     "通过可逆变换连接简单分布和复杂数据分布。",
     ["依赖不可逆随机删除所有输入。", "只使用 LSTM 遗忘门。", "只能做二分类逻辑回归。"],
     "Flow 强调 invertible transform，可进行密度建模和采样。", SOURCES["gnn"], "中", False),
    ("PINN", "PINN 与普通监督神经网络相比，关键变化是",
     "把物理方程残差、边界条件或初始条件写进损失函数。",
     ["只把网络层数加深，不使用物理规律。", "训练时完全不需要损失函数。", "只能用于文本生成。"],
     "PINN 的核心是 physics-informed loss，不是换一个激活函数。", SOURCES["gnn"], "中", True),
    ("PINN", "PINN 更适合下列哪类场景？",
     "观测数据有限但已知物理方程或边界条件的科学计算问题。",
     ["只有随机标签且无任何规律的问题。", "只需判断图片猫狗且不含物理约束的问题。", "词表编码错误导致 token 缺失的问题。"],
     "物理先验能在数据不足时约束模型输出。", SOURCES["gnn"], "中", False),
    ("AI for material science", "材料科学中使用 GNN 的自然原因是",
     "原子、键、晶格邻近关系可以表示为节点、边和图结构。",
     ["材料数据一定没有结构关系。", "GNN 只能处理社交网络，不能处理分子。", "材料性质预测只需要随机噪声。"],
     "材料结构和分子图天然适合图建模。", SOURCES["gnn"], "易", True),
    ("AI for material science", "生成式模型在材料发现中的一个用途是",
     "生成候选晶体或分子结构，再筛选其性质。",
     ["把训练标签全部删除。", "替代所有实验验证且无需筛选。", "计算 RNN 的隐藏状态维度。"],
     "生成模型可辅助候选设计，但仍需性质评估和验证。", SOURCES["gnn"], "中", False),
    ("AI for material science", "材料性质预测中加入物理先验的意义是",
     "让模型输出不仅拟合数据，还尽量符合已知物理规律。",
     ["让模型忽略结构和组成。", "使所有样本标签相同。", "把图任务强行改为填空题。"],
     "物理约束有助于提高科学问题建模的可靠性。", SOURCES["gnn"], "中", False),
]


SHORT_SPECS = [
    ("神经元与感知机", "说明 TLU、感知机、ADALINE 的递进关系。",
     "TLU 强调加权求和和阈值输出；感知机把这种结构用于线性分类，形成决策边界；ADALINE 在连续线性输出上计算误差，引出损失函数、梯度下降和参数更新。三者从逻辑判断逐步过渡到可训练模型。",
     "要写出递进关系，而不是三个孤立定义。", SOURCES["guide"], "中", True),
    ("神经元与感知机", "为什么单层感知机不能直接解决 XOR？",
     "XOR 的正负样本在线性空间中不能用一条直线分开，而单层感知机只能表示线性决策边界。要解决 XOR，需要多层结构或非线性变换先构造中间特征。",
     "关键是线性可分性，不是训练次数不够。", SOURCES["guide"], "中", True),
    ("训练流程与自动微分", "写出一个 batch 的 PyTorch 训练流程，并说明 backward 与 step 的区别。",
     "流程为清零梯度、前向传播、计算损失、loss.backward()、optimizer.step()。backward 根据计算图计算并累积梯度；step 根据已有梯度更新参数。",
     "backward 不直接改参数，step 才改参数。", SOURCES["mlp1"], "中", True),
    ("训练流程与自动微分", "解释为什么验证阶段常用 model.eval() 与 torch.no_grad()。",
     "model.eval() 切换 Dropout、BatchNorm 等层的评估行为；torch.no_grad() 不记录梯度，节省显存和计算，避免验证阶段影响训练图。",
     "二者作用不同，一个管层行为，一个管梯度记录。", SOURCES["mlp1"], "中", True),
    ("MLP 与激活函数", "为什么 MLP 不能只堆叠线性层？",
     "多个线性层复合后仍等价于一个线性变换，不能表达复杂非线性边界。激活函数引入非线性，使隐藏层能构造更复杂的特征表示。",
     "必须写出“线性复合仍线性”。", SOURCES["mlp2"], "中", True),
    ("MLP 与分类损失", "比较 BCEWithLogitsLoss 与 CrossEntropyLoss 的适用场景。",
     "BCEWithLogitsLoss 常用于二分类或多标签任务，输入每个标签的 logit；CrossEntropyLoss 常用于单标签多分类，输入每类 logits，标签通常是类别编号。",
     "不要在 BCEWithLogitsLoss 前手动 Sigmoid，也不要在 CrossEntropyLoss 前手动 Softmax。", SOURCES["mlp2"], "中", True),
    ("数据处理与训练", "为什么 StandardScaler 只能在训练集 fit？",
     "如果在验证集或全数据上 fit，会把验证集统计信息泄漏进训练流程，使评估结果偏乐观。正确做法是训练集 fit，然后用同一个 scaler transform 训练和验证数据。",
     "这是数据泄漏问题。", SOURCES["mlp2"], "中", False),
    ("优化器与学习率", "比较 Momentum、RMSProp、Adam 的基本思想。",
     "Momentum 使用历史梯度方向形成惯性；RMSProp 用梯度平方的指数滑动平均调整不同参数的步长；Adam 同时结合一阶动量和二阶矩估计，是常用自适应优化器。",
     "不要把它们都写成普通 SGD。", SOURCES["cnn2"], "中", False),
    ("CNN 基础", "为什么 CNN 比 MLP 更适合处理图像？",
     "图像有局部空间结构，展平输入会破坏位置关系且参数量巨大。CNN 用局部连接保留邻域关系，用参数共享减少参数，用池化降低空间尺寸并增强一定平移不变性。",
     "答案要包含空间结构、局部连接、参数共享。", SOURCES["cnn1"], "中", True),
    ("CNN 卷积计算", "写出二维卷积输出尺寸公式，并说明各符号含义。",
     "输出尺寸通常为 (输入尺寸 + 2P - K) / S + 1，其中 P 是 padding，K 是卷积核大小，S 是 stride。若结果不是整数，实际框架会按配置处理或报错。",
     "要说明 padding、kernel、stride 的作用。", SOURCES["cnn1"], "中", True),
    ("CNN 池化", "说明池化层的作用，并指出它和卷积层的区别。",
     "池化对局部区域做最大值或平均值汇聚，降低空间尺寸并保留显著信息；卷积层有可学习卷积核并提取特征，池化通常没有可学习参数。",
     "不要说池化用来增加参数。", SOURCES["cnn1"], "易", True),
    ("CNN 训练技巧", "BatchNorm 与 Dropout 在训练/推理阶段有什么差异？",
     "BatchNorm 训练时使用当前 batch 统计并更新累计统计，推理时通常用累计均值方差；Dropout 训练时随机失活神经元，推理时关闭随机失活。",
     "这两个都是训练/推理差异高频考点。", SOURCES["cnn2"], "中", True),
    ("CNN 架构", "比较 VGG、GoogLeNet、ResNet 的核心设计。",
     "VGG 以多层 3x3 小卷积堆叠为特点；GoogLeNet/Inception 用多分支提取多尺度特征；ResNet 用残差连接缓解深层网络退化和梯度传播问题。",
     "要写出三者各自的结构关键词。", SOURCES["cnn3"], "中", False),
    ("CNN 迁移学习", "迁移学习用于图像分类时通常怎么做？",
     "常用预训练 CNN 作为特征提取器，替换最后分类层以匹配新任务类别数，再选择冻结前层或整体微调。",
     "要说明为什么预训练特征有用。", SOURCES["cnn3"], "中", False),
    ("RNN 基础", "RNN 中隐藏状态的作用是什么？",
     "隐藏状态保存当前时间步对历史信息的压缩表示，并传递给下一时间步，使模型能根据上下文处理序列。",
     "隐藏状态不是固定参数，而是随输入序列变化。", SOURCES["rnn"], "易", True),
    ("RNN 门控", "比较 GRU 与 LSTM 的结构差异。",
     "GRU 通常有更新门和重置门，没有独立细胞状态，结构较简洁；LSTM 有输入门、遗忘门、输出门和细胞状态，长期记忆通道更明确。",
     "别把 GRU 说成也有 LSTM 的三门一状态。", SOURCES["rnn"], "中", True),
    ("RNN 变长序列", "为什么变长序列需要 padding、mask 或 packing？",
     "组成 batch 时序列长度需对齐，因此会 padding；但 padding 位置不是真实信息，需要 mask 或 packing 避免模型把补位当有效时间步。",
     "答案要写出 padding 是无效补位。", SOURCES["rnn"], "中", True),
    ("RNN 梯度问题", "RNN 为什么容易出现梯度消失或梯度爆炸？如何缓解？",
     "RNN 反向传播要沿时间步展开，梯度会多次连乘，可能变得很小或很大。可用 GRU/LSTM 门控、梯度裁剪、合适初始化和归一化等方法缓解。",
     "梯度裁剪主要对应梯度爆炸。", SOURCES["rnn"], "中", False),
    ("Attention 机制", "解释 Q、K、V 在注意力机制中的作用。",
     "Query 表示当前位置要查找什么，Key 表示候选位置可匹配的索引，Value 表示真正被加权汇总的信息。Q 与 K 计算相关性，Softmax 得到权重，再加权求和 V。",
     "不要把 V 用来计算匹配分数。", SOURCES["tf1"], "中", True),
    ("Attention 机制", "为什么缩放点积注意力要除以 sqrt(d_k)？",
     "维度较大时 QK 点积数值可能过大，使 Softmax 过于尖锐、梯度不稳定。除以 sqrt(d_k) 可以稳定分数尺度。",
     "sqrt(d_k) 不是可学习参数。", SOURCES["tf1"], "中", True),
    ("Transformer Mask", "比较 Source Mask 与 Target Mask。",
     "Source Mask 通常屏蔽源序列 padding，防止模型关注无效补位；Target Mask 用于自回归解码，遮住未来 token，保证生成时不能偷看答案。",
     "二者目的不同。", SOURCES["tf1"], "中", True),
    ("Transformer 结构", "Transformer 编码器层由哪些主要模块组成？",
     "典型编码器层包括多头自注意力、前馈网络、残差连接和 LayerNorm。位置编码在进入层前为序列补充顺序信息。",
     "不要把 LSTM 门控写进 Transformer 编码器。", SOURCES["tf2"], "中", True),
    ("Seq2Seq 与 Attention", "为什么传统 Encoder-Decoder 结构要引入 Attention？",
     "固定长度上下文向量难以承载长序列全部信息。Attention 让解码器每一步动态关注编码器不同位置输出，缓解信息瓶颈。",
     "重点是动态关注，而不是删除编码器。", SOURCES["tf1"], "中", False),
    ("BERT 与 GPT", "简述 BERT 与 GPT 预训练目标的差异。",
     "BERT 通常通过 MLM 等任务学习双向上下文表示，偏理解；GPT 使用自回归 next-token prediction，生成时不能看未来，偏生成。",
     "要结合 mask 和方向性说明。", SOURCES["tf3"], "中", False),
    ("图学习基础", "图学习为什么不能只看单个样本特征？",
     "图数据的关键信息包含对象之间的关系。节点特征和边结构共同决定任务表现，例如社交关系、分子键、交易网络都需要关系建模。",
     "图学习的核心是特征 + 结构。", SOURCES["graph1"], "易", True),
    ("图任务类型", "区分节点分类、链路预测和图分类。",
     "节点分类预测每个节点标签；链路预测判断两个节点之间是否存在边或关系；图分类/回归预测整张图的属性，例如分子性质。",
     "先看输出对象是谁。", SOURCES["graph1"], "中", True),
    ("GCN", "解释 GCN 公式中邻接矩阵、自环和度归一化的作用。",
     "邻接矩阵决定哪些邻居参与聚合；自环让节点保留自身信息；度归一化校正不同度数节点的聚合尺度，使训练更稳定。",
     "不要把 GCN 理解成规则图片上的方形卷积核。", SOURCES["gnn"], "中", True),
    ("GAT 与 GraphSAGE", "比较 GCN、GAT、GraphSAGE 的聚合思想。",
     "GCN 常按归一化邻接矩阵聚合；GAT 为不同邻居学习注意力权重；GraphSAGE 通过邻居采样和聚合函数学习归纳式表示。",
     "共同点是邻居消息聚合，差异是聚合权重和采样策略。", SOURCES["gnn"], "中", False),
    ("生成式模型", "比较 GAN、VAE、Diffusion 的基本生成思想。",
     "GAN 通过生成器和判别器对抗训练；VAE 学习潜在分布并采样解码；Diffusion 先逐步加噪，再学习反向去噪生成样本。",
     "要写生成思想，不要只列名字。", SOURCES["gnn"], "中", True),
    ("LLM", "Prompt、MoE、推理型模型分别强调什么？",
     "Prompt 用任务说明和上下文引导输出；MoE 通过路由选择部分专家提高容量和效率；推理型模型强调通过后训练等方式增强复杂推理能力。",
     "这部分偏概念和演进脉络。", SOURCES["graph2"], "易", False),
    ("PINN", "PINN 如何把物理规律加入神经网络训练？",
     "PINN 在数据损失之外加入物理方程残差、边界条件或初始条件等物理损失，使模型输出既拟合数据又满足先验规律。",
     "关键是物理约束进入损失函数。", SOURCES["gnn"], "中", True),
    ("AI for material science", "为什么材料科学适合结合 GNN、生成式模型或 PINN？",
     "材料有原子、键、晶格和物理规律。GNN 可建模结构关系，生成式模型可产生候选结构，PINN 或物理约束可提高小数据科学问题的可靠性。",
     "不要把材料科学当成孤立应用名词。", SOURCES["gnn"], "中", False),
    ("LLM", "大语言模型中的 Prompt 为什么会影响输出？",
     "Prompt 提供任务目标、上下文、约束和示例，相当于把同一个模型引导到不同的条件生成模式。回答时要说明它不改变模型参数，但会改变输入条件和生成方向。",
     "不要把 Prompt 写成重新训练模型。", SOURCES["graph2"], "易", True),
    ("LLM", "MoE 结构为什么能在参数规模和计算成本之间取得折中？",
     "MoE 设置多个专家网络，但每个样本通常只激活其中一部分专家。这样总参数容量变大，而单次前向计算不必使用全部专家。",
     "关键是路由选择部分专家，不是每次全专家一起计算。", SOURCES["graph2"], "中", False),
    ("PINN", "PINN 与普通监督学习在训练目标上有什么区别？",
     "普通监督学习主要拟合输入与标签；PINN 在数据误差之外加入物理方程残差、边界条件或初始条件，使输出受到已知物理规律约束。",
     "区别要落在损失函数，而不是只说应用领域不同。", SOURCES["gnn"], "中", True),
    ("PINN", "为什么 PINN 常用于数据较少但物理规律明确的问题？",
     "当观测数据少时，物理方程可以提供额外约束，减少模型只依赖样本拟合的风险，让预测在未观测区域也更符合规律。",
     "不要把 PINN 理解成只靠更深网络提升效果。", SOURCES["gnn"], "中", False),
    ("AI for material science", "材料性质预测为什么常需要结构信息而不只是元素列表？",
     "相同或相近组成的材料可能因晶体结构、键连接和局部环境不同而性质差异很大。结构信息能帮助模型理解原子之间的关系和相互作用。",
     "材料科学不是普通表格分类，结构关系很关键。", SOURCES["gnn"], "中", True),
    ("AI for material science", "生成式模型用于材料发现时，为什么还需要筛选和物理约束？",
     "生成模型可以提出候选结构，但候选不一定稳定、可合成或满足目标性质。后续需要性质预测、稳定性判断和物理约束筛掉不可用结果。",
     "生成新样本不等于直接得到可用材料。", SOURCES["gnn"], "中", False),
]


MATERIAL_SPECS = [
    ("训练代码", "下面是一段训练代码：\n\n```python\nfor x, y in loader:\n    pred = model(x)\n    loss = loss_fn(pred, y)\n    loss.backward()\n    optimizer.step()\n```\n训练若干 batch 后，loss 大幅震荡，梯度也越来越异常。",
     ["指出这段代码缺少哪一步。", "说明缺少该步骤会导致什么问题。", "给出正确训练顺序。"],
     "缺少 optimizer.zero_grad()。PyTorch 默认累积梯度，不清零会让不同 batch 的梯度叠加，导致更新方向和步幅异常。正确顺序是 zero_grad、forward、loss、backward、step。",
     "材料题要从代码顺序定位训练流程错误。", SOURCES["mlp1"], "中", True),
    ("分类损失配置", "二分类任务中，模型最后一层是 `nn.Linear(hidden, 1)`，输出未经过 Sigmoid 的 logit。代码写成：\n\n```python\nprob = torch.sigmoid(model(x))\nloss = nn.BCEWithLogitsLoss()(prob, y.float().view(-1, 1))\n```",
     ["指出损失函数使用中的问题。", "写出更合适的写法。", "说明这种写法为什么更稳定。"],
     "问题是 BCEWithLogitsLoss 前又手动 Sigmoid，造成重复处理。更合适：`logits = model(x)`，`loss = BCEWithLogitsLoss()(logits, y.float().view(-1,1))`。该损失内部合并 Sigmoid 与 BCE，数值更稳定。",
     "关键是识别 logits 与概率的区别。", SOURCES["mlp2"], "中", True),
    ("卷积尺寸", "输入单通道图像大小为 28x28，使用 `Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)`。",
     ["计算输出特征图的空间尺寸。", "如果希望空间尺寸保持 28x28，应如何设置 padding？", "out_channels=16 表示什么？"],
     "无填充输出边长为 (28+0-5)/1+1=24，所以输出空间尺寸为 24x24。若 stride=1 且 kernel=5，要保持 28x28，padding 通常设为 2。out_channels=16 表示学习 16 个卷积核，输出 16 个通道。",
     "资料题把公式、padding 和通道含义放在一起考。", SOURCES["cnn1"], "中", True),
    ("CNN 结构判断", "某图像分类模型把 224x224x3 图像先展平成长度 150528 的向量，再接一个很大的全连接层；另一模型用多层卷积、池化和最后分类头。两者都用于同一个分类任务，但前者没有显式利用相邻像素之间的局部结构。",
     ["比较两种模型在空间结构利用上的差异。", "说明 CNN 参数共享的意义。", "指出池化层的主要作用。"],
     "展平全连接会破坏图像二维空间邻域关系且参数量巨大；CNN 用卷积局部连接保留空间结构，用参数共享减少参数量；池化用于降采样、保留显著响应并增强一定平移不变性。",
     "材料考 CNN 出现动机，而不是死背结构名。", SOURCES["cnn1"], "中", True),
    ("训练/推理模式", "某模型含 BatchNorm 和 Dropout。验证集评估时忘记调用 `model.eval()`，结果同一批验证图片多次测试输出不稳定，并且显存占用比预期更高。训练代码没有更新参数，但仍然保留了计算图。",
     ["解释输出不稳定的原因。", "BatchNorm 和 Dropout 在 eval 模式下有什么变化？", "验证阶段为什么常配合 `torch.no_grad()`？"],
     "Dropout 在训练模式会随机失活，BatchNorm 在训练模式会用当前 batch 统计，因此输出可能不稳定。eval 后 Dropout 关闭随机失活，BatchNorm 使用累计统计。no_grad 避免记录梯度，节省资源。",
     "这是训练/推理阶段差异的典型资料题。", SOURCES["cnn2"], "中", True),
    ("RNN 变长序列", "一个 batch 中句子长度分别为 8、5、3。为了组成张量，短句被补 `<pad>` 到长度 8，随后直接把最后时间步输出用于分类。短句的最后几个时间步实际上是补位符，不携带原句真实语义。",
     ["这种做法可能有什么问题？", "可以用哪些方法避免 padding 干扰？", "如果用最后隐藏状态，应该注意什么？"],
     "最后时间步可能对应 `<pad>`，不是句子真实结尾，分类会受无效补位干扰。可以使用 mask、pack_padded_sequence，或根据真实长度取最后有效时间步。使用隐藏状态时也要确保对应真实序列信息。",
     "材料考 padding/packing 和有效时间步。", SOURCES["rnn"], "中", True),
    ("GRU/LSTM 结构", "课堂讨论中有人认为：GRU 和 LSTM 都有输入门、遗忘门、输出门和独立细胞状态，因此二者只是名字不同。已知二者都用于缓解普通 RNN 的长序列依赖问题，但门控数量和状态设计并不相同。",
     ["判断这句话是否正确。", "写出 GRU 的主要门控。", "写出 LSTM 的主要结构。"],
     "不正确。GRU 通常有更新门和重置门，没有独立细胞状态；LSTM 有输入门、遗忘门、输出门和细胞状态。二者都用于缓解长序列依赖，但结构复杂度不同。",
     "材料考 GRU/LSTM 对比。", SOURCES["rnn"], "易", True),
    ("Attention 计算", "在一个 self-attention 层中，输入序列先线性变换得到 Q、K、V。模型计算 `scores = Q @ K.T` 后直接 Softmax，没有除以 `sqrt(d_k)`。",
     ["这一步缺少什么缩放？", "为什么需要缩放？", "Softmax 后的权重用于加权哪一个矩阵？"],
     "缺少除以 sqrt(d_k)。维度大时点积可能过大，使 Softmax 过于尖锐、梯度不稳定。Softmax 得到的注意力权重用于对 V 加权求和。",
     "材料考注意力公式各项含义。", SOURCES["tf1"], "中", True),
    ("Transformer Mask", "机器翻译解码器训练时，目标序列为 `<bos> 我 喜欢 机器 学习`。若 self-attention 不加后续位置遮罩，当前位置可以看到右侧未来词。这样训练时模型获得了推理阶段不可能拥有的信息。",
     ["应使用哪类 mask？", "它解决什么问题？", "它和 source padding mask 有何区别？"],
     "应使用 Target/Subsequent Mask，防止解码器当前位置看到未来 token，保证自回归训练和推理一致。Source padding mask 主要屏蔽源序列补位，目的不同。",
     "材料考 mask 的类型和用途。", SOURCES["tf1"], "中", True),
    ("Transformer 结构配置", "某 Transformer 层设置 `embed_dim=256, num_heads=8`，并在注意力子层和 FFN 子层外都使用残差连接与 LayerNorm。",
     ["每个注意力头的维度通常是多少？", "残差连接有什么作用？", "LayerNorm 为什么适合序列模型？"],
     "每头维度通常是 256/8=32。残差连接保留输入并改善梯度传播；LayerNorm 通常按特征维对单个样本归一化，不强依赖 batch 统计，适合变长序列。",
     "材料题综合考多头、残差、归一化。", SOURCES["tf2"], "中", False),
    ("图结构任务", "一个分子被表示为图：节点是原子，边是化学键，每个节点带有元素种类、价电子等特征。模型不是预测单个原子类别，而是输出整个分子的毒性类别，用于筛选候选化合物，需要综合整张图的信息。",
     ["这是节点级、边级还是图级任务？", "节点和边分别表示什么？", "为什么 GNN 适合这类数据？"],
     "这是图级任务，因为输出是整张分子的标签。节点表示原子，边表示化学键或相互作用。GNN 能沿边聚合邻居信息，把局部结构和整体性质联系起来。",
     "材料考图任务类型和建图含义。", SOURCES["graph1"], "中", True),
    ("GCN 邻接关系", "某无向图中节点 0 与节点 1、2 相连，节点 1 还连接多个其它节点。GCN 更新节点 0 表示时，会从邻居聚合信息，并在邻接矩阵中加入自环。不同节点度数不同，直接求和会导致尺度差异。",
     ["为什么要加入自环？", "度归一化有什么作用？", "多层 GCN 为什么能利用更远邻居？"],
     "自环让节点保留自身特征；度归一化校正不同度数节点的聚合尺度；多层 GCN 逐层传播信息，一层看一跳邻居，多层可接收更远邻居信息。",
     "材料考 GCN 公式背后的物理含义。", SOURCES["gnn"], "中", False),
    ("GAT 与邻居重要性", "在欺诈检测图中，一个账号连接到普通用户、商户和高风险账号。不同邻居的风险提示价值不一样，模型希望高风险邻居对账号风险判断贡献更大，而普通交易关系贡献较小，因此需要区分邻居权重。",
     ["GCN 固定归一化聚合可能有什么局限？", "GAT 如何改进？", "这种任务中边表示什么？"],
     "固定聚合不区分不同邻居的重要性；GAT 学习注意力权重，对关键邻居赋予更高权重。边可表示交易、转账、设备共享或其他关联关系。",
     "材料考图注意力的应用动机。", SOURCES["gnn"], "中", False),
    ("GAN 训练", "一个 GAN 用随机噪声输入生成器生成手写数字图像，判别器接收真实 MNIST 图像和生成图像并判断真假。训练过程中，生成器希望图像更像真实样本，判别器希望更准确区分真假样本。",
     ["生成器和判别器的目标分别是什么？", "为什么说 GAN 是对抗训练？", "训练完成后通常使用哪个网络生成样本？"],
     "生成器目标是生成足以欺骗判别器的样本；判别器目标是区分真实和生成样本。二者目标相互竞争，所以是对抗训练。生成样本时通常使用生成器。",
     "材料考 GAN 两个网络的角色。", SOURCES["gnn"], "易", True),
    ("Diffusion 过程", "扩散模型训练中，前向过程把真实图像逐步加入高斯噪声，若步数足够多，图像会接近纯噪声。反向网络学习从带噪图像恢复更干净的图像，生成时从噪声逐步去噪得到样本，不依赖判别器对抗。",
     ["前向过程和反向过程分别做什么？", "模型主要学习哪个方向的过程？", "它和 GAN 的训练结构有何不同？"],
     "前向过程逐步加噪，反向过程逐步去噪生成样本。模型主要学习反向去噪。它不像 GAN 那样依赖生成器和判别器对抗，而是学习噪声恢复过程。",
     "材料考扩散模型直觉。", SOURCES["gnn"], "中", False),
    ("PINN 损失", "求解一个受微分方程约束的物理问题时，网络输出既要接近少量观测数据，也要满足方程残差和边界条件。研究者希望模型在观测点之外也符合已知物理规律，而不是只做普通曲线拟合。",
     ["PINN 的总损失通常包含哪几类项？", "物理损失起什么作用？", "为什么它适合数据较少的科学问题？"],
     "总损失可包含数据误差、方程残差、边界/初始条件误差。物理损失把已知规律加入优化目标，使模型在数据较少时也受物理先验约束。",
     "材料考 PINN 的核心不是网络更深，而是损失含物理约束。", SOURCES["gnn"], "中", False),
    ("材料科学应用", "某材料生成模型输入随机噪声和空间群、元素组成等条件，生成候选晶体结构，再用性质预测模型筛选稳定性。候选材料还需要结合物理约束、结构关系和实验可行性判断，不能只看生成结果是否新颖。",
     ["这里生成式模型的作用是什么？", "为什么还需要性质预测或筛选？", "如果把晶体表示为图，节点和边可表示什么？"],
     "生成式模型用于提出候选结构；性质预测/筛选用于判断候选是否满足目标性能和稳定性。若表示为图，节点可表示原子，边可表示键或空间邻近相互作用。",
     "材料题考生成式模型与材料科学任务结合。", SOURCES["gnn"], "中", False),
]


def build_curated_quiz_banks():
    standard = []
    beginner = []
    for idx, spec in enumerate(MCQ_SPECS, 1):
        topic, stem, correct, distractors, explanation, source, difficulty, is_beginner = spec
        q = mcq(f"MCQ{idx:04d}", topic, stem, correct, distractors, explanation, source, difficulty, idx, is_beginner)
        standard.append({k: v for k, v in q.items() if k != "beginner"})
        if is_beginner or difficulty == "易":
            b = mcq(f"BMCQ{idx:04d}", topic, stem, correct, distractors, explanation, source, "易", idx + 1, True)
            beginner.append({k: v for k, v in b.items() if k != "beginner"})

    for idx, spec in enumerate(SHORT_SPECS, 1):
        topic, stem, answer, explanation, source, difficulty, is_beginner = spec
        q = short(f"SHORT{idx:04d}", topic, stem, answer, explanation, source, difficulty, is_beginner)
        standard.append({k: v for k, v in q.items() if k != "beginner"})
        if is_beginner or difficulty == "易":
            b = short(f"BSHORT{idx:04d}", topic, stem, answer, explanation, source, "易", True)
            beginner.append({k: v for k, v in b.items() if k != "beginner"})

    for idx, spec in enumerate(MATERIAL_SPECS, 1):
        topic, mat, subquestions, answer, explanation, source, difficulty, is_beginner = spec
        q = material(f"MAT{idx:04d}", topic, mat, subquestions, answer, explanation, source, difficulty, is_beginner)
        standard.append({k: v for k, v in q.items() if k != "beginner"})
        if is_beginner or difficulty == "易":
            b = material(f"BMAT{idx:04d}", topic, mat, subquestions[:2], answer, explanation, source, "易", True)
            beginner.append({k: v for k, v in b.items() if k != "beginner"})

    balance_mcq_answers(standard)
    balance_mcq_answers(beginner)
    validate_quiz_bank(standard, "标准题库")
    validate_quiz_bank(beginner, "基础题库")
    return standard, beginner


def validate_quiz_bank(bank, name):
    joined_items = [" ".join(str(v) for v in q.values()) for q in bank]
    hits = [(phrase, i) for i, text in enumerate(joined_items) for phrase in FORBIDDEN_QUIZ_PHRASES if phrase in text]
    if hits:
        phrase, idx = hits[0]
        raise ValueError(f"{name}含禁用模板句：{phrase} @ {bank[idx].get('id')}")

    stems = [q.get("stem", "") for q in bank]
    repeated_stems = [stem for stem, count in Counter(stems).items() if count > 1]
    if repeated_stems:
        raise ValueError(f"{name}存在重复题干：{repeated_stems[0]}")

    mcqs = [q for q in bank if q.get("type") == "mcq"]
    option_sets = [tuple(q.get("options", [])) for q in mcqs]
    repeated_options = [opts for opts, count in Counter(option_sets).items() if count > 1]
    if repeated_options:
        raise ValueError(f"{name}存在重复选项组：{repeated_options[0]}")

    answer_counts = Counter(q.get("answer") for q in mcqs)
    if mcqs:
        if set(answer_counts) != set("ABCD"):
            raise ValueError(f"{name}单选答案未覆盖 A/B/C/D：{dict(answer_counts)}")
        max_count = max(answer_counts.values())
        min_count = min(answer_counts.values())
        if max_count - min_count > max(4, len(mcqs) * 0.12):
            raise ValueError(f"{name}单选答案分布不均衡：{dict(answer_counts)}")

    for q in mcqs:
        if len(q.get("options", [])) != 4:
            raise ValueError(f"{name}单选选项数不是4：{q.get('id')}")
        answer = q.get("answer")
        if answer not in "ABCD":
            raise ValueError(f"{name}单选答案非法：{q.get('id')}")
        texts = [opt[3:].strip() if len(opt) > 3 and opt[1:3] == ". " else opt for opt in q["options"]]
        if len(set(texts)) != 4:
            raise ValueError(f"{name}单选选项重复：{q.get('id')}")

    for q in bank:
        if q.get("type") == "material":
            if len(q.get("material", "")) < 80 or len(q.get("subquestions", [])) < 2:
                raise ValueError(f"{name}资料题材料或小问不足：{q.get('id')}")
            if not q.get("answer") or not q.get("explanation"):
                raise ValueError(f"{name}资料题缺答案或解析：{q.get('id')}")
