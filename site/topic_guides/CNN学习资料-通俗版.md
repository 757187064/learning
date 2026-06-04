# CNN 学习资料：从图像直觉到考试复盘

## 资料来源说明

本讲义依据以下课程材料重新整理，目标是帮助基础不牢的同学把 CNN 的核心逻辑串起来：

- `卷积神经网络1-学生分发版-终版2026.ipynb`：CNN 基础、卷积与互相关、图像通道、NCHW/NHWC、归一化、LeNet/AlexNet、损失函数匹配、基础 CNN 训练流程。
- `卷积神经网络2-学生分发版-终版2026.ipynb`：梯度消失/爆炸、BatchNorm、初始化、优化器、学习率调度、过拟合处理、超参数调试。
- `卷积神经网络2-学生分发版-终版2026(1).ipynb`：训练技巧相关补充版本，重点与第 2 份课件互相校验。
- `卷积神经网络3-学生分发版-终版2026.ipynb`：ImageFolder 实战、RPS 图像分类、VGG、GoogLeNet/Inception、ResNet、迁移学习。
- `卷积神经网络3-学生分发版-终版2026(1).ipynb`：CNN 高级主题补充版本，重点与第 3 份课件互相校验。
- 参考缓存：`database/source_cache.json`、`database/course_chunks.json`，用于核对课程片段与来源覆盖。

这份材料不是逐页摘抄课件，而是按“为什么要用 CNN”到“怎么训练和复盘”的顺序重写。你可以把它当成一条学习路线：先弄懂图像为什么不能随便展平，再弄懂卷积怎么保留局部结构，最后掌握现代 CNN 和迁移学习怎么落到 PyTorch 代码里。

## 全局学习路线图

| 阶段 | 要解决的问题 | 学到什么 | 对应代码关键词 |
|---|---|---|---|
| 1. MLP 为什么吃力 | 图像展平成向量后会丢掉什么 | 空间结构、参数量、平移不变性 | `nn.Linear`, `Flatten` |
| 2. 卷积为什么有效 | 卷积核如何在局部区域找特征 | 局部连接、参数共享、互相关 | `nn.Conv2d` |
| 3. 尺寸怎么算 | 卷积后特征图为什么变大或变小 | kernel、stride、padding、output size | `kernel_size`, `stride`, `padding` |
| 4. 通道怎么理解 | RGB、多通道卷积、NCHW 是什么 | 输入通道、输出通道、滤波器组 | `[N, C, H, W]` |
| 5. 池化和感受野 | 网络如何从局部看到整体 | MaxPool、AvgPool、receptive field | `nn.MaxPool2d`, `AdaptiveAvgPool2d` |
| 6. 典型 CNN | 经典网络为什么这样设计 | LeNet、AlexNet、VGG、GoogLeNet、ResNet | `features`, `classifier`, `fc` |
| 7. 训练技巧 | 为什么训练会不稳定或过拟合 | BatchNorm、初始化、优化器、调度器、正则化 | `BatchNorm2d`, `Adam`, `weight_decay` |
| 8. 数据实战 | 图像文件夹如何变成训练数据 | ImageFolder、数据增强、DataLoader | `ImageFolder`, `transforms.Compose` |
| 9. 迁移学习 | 数据少时怎样借用大模型 | 冻结、替换分类头、微调 | `requires_grad`, `model.fc` |
| 10. 考前复盘 | 考题通常卡在哪里 | 公式、维度、损失函数、训练流程 | 手算 + 代码解释 |

## 贯穿全章的直觉解释

学 CNN 时不要把它看成一堆层名。它解决的是“图像里的局部模式如何被逐步组合成整体判断”这个问题。第一层卷积像是在找很小的线条、边缘、颜色变化；中间层把这些小模式组合成纹理、角点、局部形状；更深的层再组合成物体部件和类别线索。池化和步长让特征图逐渐变小，但每个位置能回看的原图范围逐渐变大。最后分类器不是直接看原始像素，而是看前面卷积层整理出来的高级特征。

所以 CNN 的学习顺序可以压缩成一句正常的话：先保留图像的局部结构，再用共享卷积核到处找相同模式，接着通过下采样扩大视野，最后把特征汇总成类别判断。

## 1. 为什么 MLP 不适合直接处理图像

MLP 的做法通常是：把一张图像从二维或三维数组展平成一条长向量，然后送进全连接层。比如一张 `32×32` 灰度图会变成长度为 `1024` 的向量，彩色图像 `224×224×3` 会变成长度为 `150528` 的向量。

这个做法有三个硬伤。

第一，空间结构被打散。图像里相邻像素通常有意义：一条边缘、一块纹理、一个角点，都是局部像素共同形成的。展平后，第 100 个位置和第 101 个位置是否相邻，要靠原始图像布局才能知道，MLP 本身并不知道“上下左右”。

第二，参数量太大。全连接层的参数量是：

$$
\text{参数量} = \text{输入维度} \times \text{输出维度} + \text{偏置}
$$

如果输入是 `32×32=1024`，输出想得到 `28×28=784` 个神经元，那么只是一层就大约有：

$$
1024 \times 784 = 802816
$$

个权重。相比之下，一个 `5×5` 卷积核只有 25 个空间权重；多通道时再乘输入通道数和输出通道数，仍然比大规模全连接更省。

第三，MLP 不会自然利用“同一个特征可以出现在不同位置”。一只眼睛在图片左上角和右下角，都仍然是眼睛。全连接层会把不同位置当作不同输入维度分别学习；CNN 用同一个卷积核滑过整张图，相当于同一个检测器到处查看。

所以 CNN 的核心不是“多了一种层”，而是把图像任务里的两个事实写进了网络结构：

- 局部像素关系重要，所以用局部连接。
- 同类局部模式可能出现在任意位置，所以用参数共享。

## 2. 卷积：局部连接和参数共享

### 2.1 卷积核到底在做什么

把卷积核想成一个小窗口，比如 `3×3`。它每次只看图像的一小块区域，把窗口内的像素和卷积核权重逐个相乘，再全部加起来，得到输出特征图上的一个数。然后窗口向右或向下移动，重复这个过程。

对一个 `3×3` 灰度小区域：

$$
\begin{bmatrix}
x_{11} & x_{12} & x_{13}\\
x_{21} & x_{22} & x_{23}\\
x_{31} & x_{32} & x_{33}
\end{bmatrix}
$$

和一个卷积核：

$$
\begin{bmatrix}
w_{11} & w_{12} & w_{13}\\
w_{21} & w_{22} & w_{23}\\
w_{31} & w_{32} & w_{33}
\end{bmatrix}
$$

输出值就是：

$$
y = \sum_{i=1}^{3}\sum_{j=1}^{3} x_{ij}w_{ij} + b
$$

每一项的含义：

- $x_{ij}$：当前窗口里的像素值或上一层特征值。
- $w_{ij}$：卷积核在对应位置的权重，是训练学出来的。
- $b$：偏置项，可选；如果后面接 BatchNorm，卷积层常设 `bias=False`。
- $\sum$：把局部区域的信息压缩成一个响应值。

如果这个卷积核学成“横线检测器”，看到横线就输出大；如果学成“边缘检测器”，看到边缘就输出大。卷积核不是人工固定的滤镜，而是通过反向传播自动学出来的。

### 2.2 CNN 里常说的“卷积”其实多是互相关

严格数学卷积会翻转卷积核。离散一维卷积可以写成：

$$
(f * g)[n] = \sum_{m=0}^{M-1} f[n-m]g[m]
$$

互相关不翻转卷积核：

$$
(f \star g)[n] = \sum_{m=0}^{M-1} f[n+m]g[m]
$$

二维互相关写成：

$$
(F \star G)[i,j]
=
\sum_{m=0}^{K_1-1}\sum_{n=0}^{K_2-1}
F[i+m,j+n]G[m,n]
$$

PyTorch 的 `nn.Conv2d` 实际执行的是这种不翻转核的互相关。为什么仍然叫卷积？因为卷积核是学习出来的，翻不翻转只会影响最后学到的权重排列，不影响模型表达能力。考试里要知道：深度学习框架里的 `Conv2d` 通常按互相关实现。

## 3. 卷积尺寸公式：每个符号都要会解释

一维或二维单边尺寸的通用公式是：

$$
O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1
$$

逐项拆开：

- $I$：输入尺寸，例如输入高 `H=32`。
- $P$：padding，在边缘补几圈。`P=1` 表示上下左右各补 1。
- $K$：kernel size，卷积核大小，例如 `3×3` 的 `K=3`。
- $S$：stride，步长。`S=1` 每次移动一格，`S=2` 每次移动两格。
- $\lfloor \cdot \rfloor$：向下取整。遇到不能整除时，框架通常丢掉最后放不下的部分。
- `+1`：窗口从起点放下去就已经产生第一个输出位置。

二维图像高和宽分别算：

$$
H_{out} = \left\lfloor \frac{H_{in}+2P_h-K_h}{S_h} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in}+2P_w-K_w}{S_w} \right\rfloor + 1
$$

例子：输入 `32×32`，卷积核 `3×3`，padding=1，stride=1。

$$
O = \left\lfloor \frac{32+2\times1-3}{1}\right\rfloor + 1 = 32
$$

所以 `padding=1` 的 `3×3` 卷积可以保持空间尺寸不变。

常见三种形式：

| 类型 | 做法 | 输出尺寸直觉 | 常见用途 |
|---|---|---|---|
| Valid | 不填充 | 变小：$M-K+1$ | 早期简单网络、严格不看边界外 |
| Same | 适当填充 | 尽量保持输入大小 | 现代 CNN 常用 |
| Full | 大量填充 | 变大：$M+K-1$ | 信号处理中更常见 |

代码对应关系：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)
```

这段代码表示：输入是 RGB 三通道；用 64 组卷积核提取 64 个输出通道；每个空间卷积核是 `3×3`；每次移动 1 格；边缘补 1 圈，因此高宽通常保持不变。

## 4. 通道、NCHW 和归一化

### 4.1 图像在网络里是什么形状

灰度图像有 1 个通道，RGB 彩色图像有 3 个通道。像素值通常是 0 到 255。RGB 转灰度常用加权公式：

$$
Gray = 0.2126R + 0.7152G + 0.0722B
$$

绿色权重大，是因为人眼对绿色更敏感。

PyTorch 里图像批次通常是：

$$
[N, C, H, W]
$$

也就是 NCHW：

- $N$：batch size，一次送进模型的图片数。
- $C$：channel，通道数。
- $H$：height，高。
- $W$：width，宽。

TensorFlow 常用 NHWC：`[N, H, W, C]`。PIL 或普通 NumPy 图像常见 HWC：`[H, W, C]`。维度顺序错了，模型不会“自动理解”，轻则报错，重则训练结果很奇怪。

代码对应：

```python
# NumPy: NCHW -> NHWC
nhwc = np.transpose(nchw, (0, 2, 3, 1))

# PyTorch Tensor: NCHW -> NHWC
tensor_nhwc = tensor_nchw.permute(0, 2, 3, 1)
```

### 4.2 多通道卷积怎么计算

如果输入是 RGB 图像，形状是 `[3, H, W]`，一个输出通道需要的滤波器不是一个 `K×K`，而是 `[3, K, K]`。它分别看 R/G/B 三个通道，对应位置相乘求和，再把三个通道的结果加在一起，得到输出特征图上的一个数。

单个输出通道的一个位置可以写成：

$$
Output[0,y,x]
=
\sum_{c=0}^{C_{in}-1}
\sum_{i=0}^{K-1}
\sum_{j=0}^{K-1}
Input[c,y+i,x+j] \times Filter[c,i,j]
$$

如果要输出 64 个通道，就需要 64 组这样的滤波器。PyTorch 卷积权重形状是：

$$
[C_{out}, C_{in}, K_h, K_w]
$$

这件事很容易考。记住：

- `in_channels` 必须等于输入张量的通道数。
- `out_channels` 等于卷积核组数，也等于输出特征图通道数。
- 一个输出通道由一整组跨输入通道的卷积核产生。

### 4.3 归一化和标准化

图像像素直接是 `[0,255]`，数值范围偏大。常见预处理有三层。

第一层：缩放到 `[0,1]`。

$$
x' = \frac{x}{255}
$$

`ToTensor()` 会把 PIL 图像或 NumPy 图像转成 Tensor，并把 0 到 255 缩放到 0 到 1。

第二层：Z-score 标准化。

$$
x' = \frac{x-\mu}{\sigma}
$$

- $\mu$：训练集均值。
- $\sigma$：训练集标准差。
- RGB 图像通常每个通道单独算一组均值和标准差。

关键原则：如果是从头训练，用训练集统计量；验证集、测试集、新数据都复用训练集统计量，不能用测试集参与计算。否则就是信息泄露。

第三层：迁移学习使用预训练模型时，要用预训练时的统计量。例如 ImageNet 常用：

```python
Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

这时不要重新计算自己数据集的均值方差来替代它。因为预训练模型原来就是在 ImageNet 这种输入分布下学到权重的。

## 5. 池化和感受野

### 5.1 池化在干什么

池化也是一个小窗口，但它通常没有可学习参数。最大池化取窗口最大值，平均池化取窗口平均值。

| 池化类型 | 操作 | 适合理解 |
|---|---|---|
| Max Pooling | 取局部最大值 | “这块区域有没有强特征” |
| Average Pooling | 取局部平均 | “这块区域整体活跃程度如何” |
| Global Average Pooling | 每个通道全图平均 | “这个语义检测器整体有多活跃” |

池化输出尺寸公式：

$$
O = \left\lfloor \frac{M-K}{S} \right\rfloor + 1
$$

如果 `MaxPool2d(kernel_size=2, stride=2)` 作用在 `28×28` 上：

$$
O = \left\lfloor \frac{28-2}{2} \right\rfloor + 1 = 14
$$

所以常见的 `2×2`、步长 2 池化会把高宽减半。

### 5.2 感受野：深层神经元看到原图多大范围

感受野指某一层某个位置能“追溯”到原始输入图像的多大区域。浅层看边缘、角点这类局部细节；深层因为经过多次卷积和池化，会汇聚更大范围的信息，更适合识别部件和物体。

课件给出的从后往前递推公式：

$$
RF_{l-1} = (RF_l - 1)\times s_l + k_l
$$

逐项解释：

- $RF_l$：第 $l$ 层一个神经元在上一层特征图上的感受野边长。
- $s_l$：第 $l$ 层操作的步长。
- $k_l$：第 $l$ 层卷积核或池化窗口大小。
- $(RF_l-1)\times s_l$：下一层覆盖多个位置，这些位置在上一层之间相隔 stride。
- `+ k_l`：每个位置本身还要展开成一个窗口。

直觉例子：两个连续 `3×3, stride=1` 卷积，第二层一个点不是只看原图 `3×3`，而是看原图 `5×5`。堆叠小卷积核能逐渐扩大视野，这也是 VGG 喜欢连续堆 `3×3` 卷积的重要原因。

## 6. 典型 CNN 架构

### 6.1 一个标准卷积块

基础写法：

```python
nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
```

现代写法常加 BatchNorm：

```python
nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(out_ch),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2)
)
```

整体 CNN 通常分成两部分：

- `features`：卷积块堆叠，负责从像素提取特征。
- `classifier` 或 `fc`：把特征映射成类别 logits。

注意输出层和损失函数要匹配。多分类用 `CrossEntropyLoss` 时，模型最后一层直接输出 logits，不要手动加 Softmax。

### 6.2 LeNet-5

LeNet-5 用于手写数字识别，是早期 CNN 代表。

```text
输入: 1@32×32
C1: Conv 5×5, 1→6        32×32 → 28×28
S2: Pool 2×2             28×28 → 14×14
C3: Conv 5×5, 6→16       14×14 → 10×10
S4: Pool 2×2             10×10 → 5×5
C5: Conv 5×5, 16→120     5×5 → 1×1
F6: Linear 120→84
输出: Linear 84→10
```

代码骨架：

```python
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

### 6.3 AlexNet

AlexNet 的重要性在于它把 CNN 推向大规模图像分类：更深的网络、ReLU、Dropout、GPU 训练。

三段结构：

| 部分 | 作用 | 典型内容 |
|---|---|---|
| `features` | 提取特征 | 5 个卷积块，含 ReLU 和 Pooling |
| `avgpool` | 固定特征尺寸 | `AdaptiveAvgPool2d((6,6))` |
| `classifier` | 分类 | 大全连接层 + Dropout |

AlexNet 分类器前的固定输入是 `256×6×6=9216`。这里自适应池化很关键：无论前面特征图尺寸略有变化，都能变成固定 `6×6`，方便接全连接层。

### 6.4 VGG

VGG 的思想非常整齐：用很多 `3×3` 小卷积核堆叠。两个 `3×3` 卷积叠起来感受野接近 `5×5`，三个叠起来接近 `7×7`，同时中间多了非线性激活，表达能力更强。

VGG-16 的问题是参数量巨大，尤其全连接层占大头。课件中标准 VGG-16 约 138M 参数，其中全连接层占绝大多数。

### 6.5 GoogLeNet / Inception

GoogLeNet 的核心是 Inception 模块：同一层里并行做多种尺度的处理，再在通道维度拼接。

典型 Inception 分支：

```text
分支1: 1×1 Conv
分支2: 1×1 Conv 降维 → 3×3 Conv
分支3: 1×1 Conv 降维 → 5×5 Conv
分支4: Pooling → 1×1 Conv
最后: torch.cat([...], dim=1)
```

`1×1` 卷积是关键。它不融合邻域空间，只在同一像素位置上混合通道：

$$
\text{参数量}_{1\times1} = C_{in}\times C_{out}
$$

普通 `3×3` 卷积参数量是：

$$
\text{参数量}_{3\times3} = C_{in}\times C_{out}\times 9
$$

所以先用 `1×1` 把通道降下来，再做 `3×3` 或 `5×5`，计算会省很多。GoogLeNet 还用全局平均池化减少全连接参数，并用辅助分类器缓解深层网络训练时的梯度问题。

### 6.6 ResNet

ResNet 解决的是“网络退化”：网络加深后，训练准确率反而下降。问题不只是过拟合，因为训练集上也变差，说明深网络优化困难。

残差块不直接学习 $H(x)$，而是学习：

$$
F(x)=H(x)-x
$$

输出写成：

$$
y = F(x) + x
$$

如果某些层暂时学不到有用变换，可以让 $F(x)$ 接近 0，那么输出接近输入 $x$。这给深层网络提供了更容易优化的路径，也让梯度能沿着加法分支更顺畅地回传。

基础残差块代码结构：

```python
class BasicBlock(nn.Module):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)
```

当 stride 改变或通道数改变时，`x` 和 `F(x)` 形状不同，不能直接相加，需要 `downsample`，通常是 `1×1 Conv + BN`。

## 7. 模型对比表

| 模型 | 年份 | 大致参数量 | 结构关键词 | 优点 | 缺点或注意点 |
|---|---:|---:|---|---|---|
| LeNet-5 | 1998 | 约 60K | Conv + Pool + FC | 结构清楚，适合入门 | 处理小图像，能力有限 |
| AlexNet | 2012 | 约 60M | ReLU、Dropout、GPU、大卷积核 | 开启大规模 CNN 时代 | 全连接层重，参数较多 |
| VGG-16 | 2014 | 约 138M | 连续 `3×3` 卷积 | 简洁、迁移效果好 | 参数和计算量大 |
| GoogLeNet | 2014 | 约 6.8M | Inception、`1×1` 降维、GAP | 参数少，设计精巧 | 结构分支多，手写较复杂 |
| ResNet-18/34 | 2015 | 约 11M/21M | BasicBlock、残差连接 | 易训练，迁移常用 | 极小数据上仍要防过拟合 |
| ResNet-50+ | 2015 | 约 25M 起 | Bottleneck、残差连接 | 深而高效，通用强 | 微调要控制学习率 |

考前抓主线：VGG 说明小卷积核堆叠有效；GoogLeNet 说明结构设计能省参数；ResNet 说明残差连接能训练很深的网络。

## 8. 训练技巧：让 CNN 真的训得动

### 8.1 BatchNorm

BatchNorm 对卷积输出 `[N, C, H, W]` 按通道计算统计量。对第 $c$ 个通道：

$$
\mu_c = \frac{1}{NHW}\sum_{n,h,w}x_{n,c,h,w}
$$

$$
\sigma_c^2 = \frac{1}{NHW}\sum_{n,h,w}(x_{n,c,h,w}-\mu_c)^2
$$

$$
\hat{x}_{n,c,h,w} =
\frac{x_{n,c,h,w}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}
$$

$$
y_{n,c,h,w} = \gamma_c\hat{x}_{n,c,h,w}+\beta_c
$$

逐项解释：

- $\mu_c,\sigma_c^2$：当前通道在 batch 和空间位置上的均值、方差。
- $\epsilon$：防止除以 0。
- $\gamma_c,\beta_c$：可学习缩放和平移参数，每个通道一对。

训练时 BN 用当前 batch 的统计量，并更新 running mean/var。推理时用训练期间累计的 running mean/var。因此测试前必须 `model.eval()`，否则 BatchNorm 和 Dropout 行为都会不对。

常见卷积块：

```python
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
nn.BatchNorm2d(out_ch)
nn.ReLU(inplace=True)
```

卷积层设 `bias=False` 的原因：BN 已经有 $\beta$ 负责平移，卷积 bias 通常冗余。

### 8.2 初始化

初始化的目标是让前向激活和反向梯度不要层层变小或变大。

| 初始化 | 适合激活 | 公式直觉 | PyTorch |
|---|---|---|---|
| Xavier/Glorot | Tanh、Sigmoid | 同时考虑输入和输出维度 | `xavier_uniform_`, `xavier_normal_` |
| Kaiming/He | ReLU、LeakyReLU | ReLU 会截断一部分激活，所以方差补偿更强 | `kaiming_uniform_`, `kaiming_normal_` |

Kaiming normal 常见标准差：

$$
\sigma = \sqrt{\frac{2}{n_{in}}}
$$

在 CNN 里，如果大量使用 ReLU，优先想到 Kaiming 初始化。

### 8.3 优化器和学习率

SGD 直接沿梯度方向走，但在狭长山谷里容易左右震荡。Momentum 引入速度：

$$
v = \gamma v - \alpha \nabla_\theta J(\theta)
$$

$$
\theta = \theta + v
$$

- $v$：历史方向的积累。
- $\gamma$：动量系数，常见 0.9。
- $\alpha$：学习率。

Nesterov 的区别是先到预估位置看梯度，减少冲过头。

Adam 结合动量和自适应学习率：

$$
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
c_t = \beta_2c_{t-1}+(1-\beta_2)g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta\frac{m_t}{\sqrt{c_t+\epsilon}}
$$

常见经验：

- 快速做实验：Adam 是好起点。
- 大规模视觉任务、追求最终精度：SGD + Momentum + 学习率调度很常见。
- 验证集停滞时降学习率：`ReduceLROnPlateau` 实用。

### 8.4 过拟合处理

过拟合的典型表现：训练准确率继续升，验证准确率不升甚至下降。

| 方法 | 代码位置 | 作用 |
|---|---|---|
| 数据增强 | `transforms` | 增加训练样本变化 |
| Dropout | 分类器或大 FC 层 | 随机失活，减少依赖特定神经元 |
| Weight Decay | 优化器参数 | 惩罚大权重，限制复杂度 |
| Early Stopping | 训练流程 | 验证集不改善就停止 |
| 更小模型 | 模型设计 | 降低记忆能力 |

代码对应：

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

```python
self.classifier = nn.Sequential(
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, num_classes)
)
```

卷积层常用 BN，Dropout 更多放在全连接层；卷积层如要用，可考虑 `Dropout2d`，它按通道丢弃而不是按单个像素丢弃。

## 9. 数据增强、ImageFolder 和训练流程

### 9.1 ImageFolder 的文件夹格式

`ImageFolder` 要求每个类别一个子文件夹：

```text
dataset_root/
├── paper/
│   ├── img1.png
│   └── ...
├── rock/
│   └── ...
└── scissors/
    └── ...
```

它会自动把文件夹名映射成类别编号。例如 RPS 数据集通常得到：

```python
dataset.classes
# ['paper', 'rock', 'scissors']
```

代码：

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root="./rps", transform=train_transform)
```

训练集可以随机翻转、旋转、颜色扰动；验证集和测试集不要随机增强，只做确定性的 resize、tensor、normalize。否则每次评估输入都变，会影响判断。

### 9.2 完整训练流程

标准流程可以写成 9 步：

1. 准备数据：下载或整理文件夹。
2. 划分训练/验证/测试集，保存索引，避免数据泄露。
3. 定义 transform：训练集可增强，验证/测试集保持确定。
4. 创建 Dataset 和 DataLoader。
5. 定义模型：卷积特征提取器 + 分类器。
6. 选择损失函数：多分类常用 `CrossEntropyLoss`。
7. 选择优化器和调度器。
8. 训练循环：`train()`、清梯度、前向、算损失、反向、更新。
9. 验证和测试：`eval()`、`torch.no_grad()`、统计 accuracy、precision、recall、F1、混淆矩阵。

训练循环核心代码：

```python
for epoch in range(n_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            val_loss = criterion(logits, labels)
```

几个关键点：

- `optimizer.zero_grad()` 必须在每个 batch 更新前调用，否则梯度会累积。
- 多分类的 `labels` 应是类别索引，类型通常是 `long`，形状常见 `[N]`。
- `CrossEntropyLoss` 接收 logits，不接收 softmax 后的概率。
- 验证和测试阶段不用反向传播，所以放在 `torch.no_grad()` 里。

### 9.3 类别不平衡

如果数据类别不平衡，准确率可能骗人。比如 99% 都是正常样本，模型全预测正常也有 99% 准确率，但少数类召回率为 0。

多分类可以用两类办法：

```python
# 方法1：损失函数类别权重
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

```python
# 方法2：采样时平衡
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

二分类里 `BCEWithLogitsLoss` 的 `pos_weight` 是给正样本加权；多分类里 `CrossEntropyLoss` 的 `weight` 是给每个类别加权。不要把这两个参数混用。

## 10. 实验和案例：从线条分类到 RPS

课件里有两个很适合复习的案例。

第一个是线条分类：生成 `5×5` 小图，类别 0 是水平/垂直线，类别 1 是对角线。这个案例非常适合理解 CNN 为什么能识别局部模式。输入形状是 NCHW：

```python
images = np.array(images)[:, np.newaxis, :, :]
# [N, 1, 5, 5]
```

一个小模型可以是：

```python
self.features = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3),  # 1@5×5 -> 4@3×3
    nn.ReLU(),
    nn.Conv2d(4, 8, kernel_size=3),  # 4@3×3 -> 8@1×1
    nn.ReLU()
)
```

这个例子要会手算尺寸：`5 -> 3 -> 1`，因为都没有 padding，kernel 都是 3。

第二个是 RPS 石头剪刀布分类。数据用 `ImageFolder` 读取，训练集做增强，模型用多个 `Conv + BN + ReLU + Pool` 卷积块，最后用 Global Average Pooling 和较小分类器。训练时可以用 Adam、学习率调度和早停，评估时看分类报告和混淆矩阵。

这个案例的复习重点不是背数据集数字，而是掌握流程：

- 文件夹名就是类别名。
- 训练增强和验证预处理分开。
- BN 要配合 `model.train()` / `model.eval()`。
- GAP 可以减少 `Flatten + 大全连接` 带来的参数爆炸。
- 如果某一类预测差，要看混淆矩阵、类别样本质量、增强是否合理。

## 11. ResNet 和迁移学习

### 11.1 迁移学习为什么有效

大型模型在 ImageNet 上学到的底层特征，如边缘、颜色、纹理、形状，很多任务都能复用。小数据集从零训练 CNN 容易过拟合；迁移学习可以把预训练模型当作特征提取器，只训练新的分类层。

基本步骤：

1. 加载预训练模型。
2. 冻结原有参数。
3. 替换最后分类层。
4. 用新数据训练分类层。
5. 如果效果不够，再解冻后面一两组层，用较小学习率微调。

ResNet18 示例：

```python
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
```

注意：替换后的 `model.fc` 默认是可训练的，因为它是新建层。

### 11.2 不同模型替换哪一层

| 模型 | 输入尺寸 | 替换层 | 替换代码 |
|---|---:|---|---|
| AlexNet | 224×224 | `model.classifier[6]` | `nn.Linear(4096, num_classes)` |
| VGG16/19 | 224×224 | `model.classifier[6]` | `nn.Linear(4096, num_classes)` |
| ResNet-18/34 | 224×224 | `model.fc` | `nn.Linear(512, num_classes)` |
| ResNet-50/101/152 | 224×224 | `model.fc` | `nn.Linear(2048, num_classes)` |
| InceptionV3 | 299×299 | `model.fc` 和 `model.AuxLogits.fc` | 主分类器和辅助分类器都要处理 |

### 11.3 冻结、特征提取和微调

如果所有卷积层都冻结，那么同一张图片经过特征提取器得到的向量每次都一样。可以先把特征提取出来保存成特征数据集，只训练分类头，这会更快。

如果冻结效果不够好，可以 fine-tuning：

```python
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3}
])
```

这里的学习率设计很重要：预训练层用小学习率，保护已经学到的通用特征；新分类层用大学习率，让它快速适应新类别。

迁移学习常见误区：

| 错误做法 | 为什么错 | 更合适的做法 |
|---|---|---|
| 用很大学习率训练整个预训练模型 | 容易破坏已有特征 | 先冻结，只训练分类头 |
| 使用自己数据集 mean/std 替代 ImageNet 参数 | 输入分布和预训练不一致 | 用 ImageNet mean/std |
| 只看训练准确率 | 小数据很容易记忆 | 重点看验证/测试表现 |
| InceptionV3 只替换主分类器 | 辅助分类器训练时也会输出 | 同时处理 `AuxLogits.fc` |

## 12. 易混点集中整理

| 易混点 | 正确认法 |
|---|---|
| `Conv2d` 是严格数学卷积吗 | 框架里通常是互相关，不翻转卷积核 |
| `out_channels` 是什么 | 卷积核组数，也是输出特征图通道数 |
| 一个 RGB 输入、一个输出通道需要几个核 | 需要 3 个 `K×K` 核组成一组，对 3 个输入通道分别作用后求和 |
| `padding=1` 一定保持尺寸吗 | 对 `3×3, stride=1` 通常保持；其他 kernel/stride 要套公式 |
| `CrossEntropyLoss` 前要不要 Softmax | 不要，它内部包含 log-softmax 相关计算 |
| 二分类一定用 Sigmoid + BCELoss 吗 | 更推荐 `BCEWithLogitsLoss`，模型输出 logits |
| BatchNorm 和输入标准化一样吗 | 不一样。输入标准化是数据预处理，BN 是网络中间层按 batch 统计激活 |
| 训练和推理 BN 是否一样 | 不一样。训练用 batch 统计，推理用 running mean/var |
| Dropout 在验证时是否生效 | `model.eval()` 后不按训练方式随机丢弃 |
| 迁移学习能否随便改输入标准化 | 不能，预训练模型要匹配原训练分布 |
| ResNet 的残差连接什么时候要 downsample | stride 改变或通道数改变，导致两条分支形状不一致时 |

## 13. 关键概念速查表

| 概念 | 快速解释 | 代码/公式 |
|---|---|---|
| 局部连接 | 每个输出只看输入局部区域 | `kernel_size` |
| 参数共享 | 同一个卷积核滑过整张图 | `Conv2d` 权重复用 |
| 互相关 | 不翻转卷积核的滑窗点积 | `nn.Conv2d` 实际行为 |
| 输出尺寸 | 卷积后高宽 | $\lfloor(I+2P-K)/S\rfloor+1$ |
| 输入通道 | 输入图像或特征图通道数 | `in_channels` |
| 输出通道 | 学多少组滤波器 | `out_channels` |
| NCHW | PyTorch 图像批次格式 | `[batch, channel, height, width]` |
| 标准化 | 用均值方差调整输入分布 | `(x-mean)/std` |
| MaxPool | 保留局部最强响应 | `nn.MaxPool2d(2)` |
| 感受野 | 深层单元能看到原图范围 | $RF_{l-1}=(RF_l-1)s_l+k_l$ |
| GAP | 每个通道求全图平均 | `AdaptiveAvgPool2d((1,1))` |
| BN | 标准化中间激活 | `nn.BatchNorm2d(C)` |
| Kaiming 初始化 | ReLU 网络常用初始化 | `kaiming_normal_` |
| Weight Decay | L2 正则化 | `weight_decay=1e-4` |
| ImageFolder | 按文件夹读图像分类数据 | `ImageFolder(root, transform)` |
| 冻结 | 参数不更新 | `requires_grad=False` |
| 微调 | 解冻部分预训练层小步训练 | 分层学习率 |

## 14. 自测题

### 题目 1：卷积尺寸

输入特征图大小是 `[N, 3, 64, 64]`，经过：

```python
nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
```

输出形状是多少？写出计算过程。

### 题目 2：多通道参数量

一个卷积层 `nn.Conv2d(3, 32, kernel_size=3, padding=1)`，如果有 bias，参数量是多少？如果接了 BatchNorm 并把卷积设为 `bias=False`，卷积层参数量又是多少？

### 题目 3：损失函数匹配

三分类图像任务，模型最后输出形状 `[batch_size, 3]`。标签是类别编号 `0/1/2`。应该用什么损失函数？最后一层要不要加 Softmax？标签 dtype 应该是什么？

### 题目 4：BN 行为

为什么训练时调用 `model.train()`，验证时调用 `model.eval()` 对含 BatchNorm 和 Dropout 的 CNN 很重要？

### 题目 5：迁移学习

你用 ResNet18 做 5 分类任务，数据只有 800 张。请写出迁移学习的基本操作：冻结哪些层、替换哪一层、标准化用什么参数、学习率大概怎么设。

### 题目 6：判断过拟合还是欠拟合

训练 30 个 epoch 后，训练准确率 99%，验证准确率 72%，验证损失从第 10 个 epoch 后开始上升。请判断问题，并给出至少 4 个处理办法。

### 题目 7：ResNet 残差连接

残差块输出 `F(x)+x`。如果 `F(x)` 的输出形状是 `[N, 128, 28, 28]`，而 `x` 是 `[N, 64, 56, 56]`，为什么不能直接相加？通常怎么处理？

### 参考答案

1. 输出高宽：
   $$
   \left\lfloor\frac{64+2\times2-5}{2}\right\rfloor+1
   =
   \left\lfloor\frac{63}{2}\right\rfloor+1
   =31+1=32
   $$
   输出形状是 `[N, 16, 32, 32]`。

2. 有 bias：权重 `32×3×3×3=864`，bias `32`，总计 `896`。如果 `bias=False`，卷积层参数量是 `864`。注意 BN 自己还有 `gamma/beta`，共 `2×32=64` 个可学习参数。

3. 用 `nn.CrossEntropyLoss()`。最后一层不加 Softmax，直接输出 logits。标签 dtype 应该是 `torch.long`，通常形状 `[batch_size]`。

4. `train()` 下 Dropout 会随机失活，BatchNorm 用当前 batch 统计量并更新 running 统计；`eval()` 下 Dropout 停止随机失活，BatchNorm 用训练累积的 running mean/var。验证时如果还处于训练模式，指标会不稳定且不可信。

5. 用 `models.resnet18(weights=...)` 加载预训练权重；先冻结全部参数；替换 `model.fc = nn.Linear(512, 5)`；预处理用 ImageNet 的 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`；先只训练 `fc`，学习率可从 `1e-3` 附近试；若效果不够，解冻 `layer4` 或 `layer3+layer4`，预训练层用 `1e-4` 或更小，新分类头仍可用较大学习率。

6. 这是过拟合。可用数据增强、增大 weight decay、加入或增大 Dropout、早停、减小模型、收集更多数据、降低训练轮数、检查类别泄露或训练/验证分布差异。

7. 不能直接相加，因为通道数和空间尺寸都不同。通常用 downsample 分支，例如 `1×1 Conv` 把通道从 64 变到 128，同时 `stride=2` 把 `56×56` 降到 `28×28`，再接 BN，使 identity 分支形状和 `F(x)` 匹配。

## 15. 最后总复习

把 CNN 串成一条线：

图像不是普通表格。图像的局部结构很重要，同一个局部特征可以出现在很多位置。MLP 展平图像后会丢空间关系，还会产生大量参数。CNN 用卷积核局部查看图像，并让同一个卷积核滑过全图，因此能保留空间结构、减少参数、复用特征检测器。

卷积的计算本质是局部窗口和卷积核做点积。深度学习框架中的卷积通常按互相关实现，不翻转卷积核。卷积输出尺寸由输入大小、padding、kernel size、stride 共同决定：

$$
O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1
$$

多通道卷积里，一个输出通道需要一组跨全部输入通道的卷积核；输出通道数等于滤波器组数。PyTorch 图像输入一般是 `[N,C,H,W]`。训练前要做缩放和标准化，迁移学习要使用预训练模型对应的标准化参数。

池化减少空间尺寸，增大后续层感受野，并带来一定局部平移不敏感性。深层 CNN 通过卷积和池化让感受野从局部逐渐扩大。GAP 则把每个语义通道的全图响应压成一个数，常用于替代大规模全连接层。

经典模型按思想记：LeNet 是入门 CNN；AlexNet 把 CNN 推向大规模图像分类；VGG 证明小卷积核堆叠有效但参数多；GoogLeNet 用 Inception、`1×1` 降维和 GAP 提高参数效率；ResNet 用残差连接解决深层网络退化，让很深的 CNN 可训练。

训练 CNN 时，真正要盯住的是数据、模型、损失和验证曲线。BatchNorm 稳定中间激活，Kaiming 初始化适合 ReLU，Adam 适合快速起步，SGD+Momentum 常用于大规模视觉精调。过拟合时用数据增强、Dropout、Weight Decay、早停和更合适的模型容量。类别不平衡时不要只看 accuracy，要考虑采样器、类别权重、召回率和混淆矩阵。

迁移学习是小数据图像任务的常用方案：加载 ImageNet 预训练模型，冻结特征提取层，替换分类头，先训练新分类头；需要更好效果时，再用小学习率微调后面几层。ResNet18/50 是常见选择，替换层通常是 `model.fc`。

考前最后检查自己能不能完成这些动作：

- 手算 `Conv2d` 和 `MaxPool2d` 输出尺寸。
- 说清 `in_channels`、`out_channels`、卷积核权重形状。
- 分清 NCHW、NHWC、HWC。
- 解释 `CrossEntropyLoss` 为什么不需要手动 Softmax。
- 解释 BatchNorm 训练和推理行为差异。
- 看训练/验证曲线判断过拟合、欠拟合、学习率过大或过小。
- 写出 ImageFolder + transforms + DataLoader 的基本流程。
- 写出 ResNet 迁移学习替换 `fc`、冻结、微调的核心代码。

如果这些都能做到，CNN 这一章就不是零散知识点，而是一套从图像结构、模型设计到训练实践的完整方法。
