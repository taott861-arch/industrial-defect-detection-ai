# 钢铁表面缺陷识别项目学习总结

本文结合当前 inspectiontesting 项目代码，梳理从图片到训练、评价、解释和网页展示的知识链。当前主路线是 ResNet18，保留 CNN baseline；网页和 `gradcam_utils.py` 已实现。内容依据 2026-09-30 检查到的代码，不沿用早期“网页尚未实现”的描述。

## 1. 图像分类：先明确模型要回答什么

项目输入是一张钢铁表面图片，输出是六种缺陷之一。这是**有监督、单标签、多类别图像分类**：图片是输入，目录类别是监督标签，每条样本对应一个类别编号。

`utils/dataset.py` 定义的顺序为：

| 编号 | 名称 | 中文含义 |
| --- | --- | --- |
| 0 | crazing | 龟裂 |
| 1 | inclusion | 夹杂 |
| 2 | patches | 斑块 |
| 3 | pitted_surface | 点蚀表面 |
| 4 | rolled-in_scale | 轧入氧化皮 |
| 5 | scratches | 划痕 |

训练时，类别名称转换成整数标签；预测时，再把最大输出对应的整数转换成名称。两端顺序必须一致。模型不会自动理解 Python 字符串或读取文件名前缀来完成网页预测；文件名和目录仅在准备训练标签时起作用。

当前没有正常图像类别，也没有未知类别分支，所以对任意输入取 `argmax` 都会得到这六类中的一个。系统名称中的“检测”指应用功能；实际算法没有输出目标框或像素掩码。

## 2. 数据准备：把文件组织变成监督信号

`utils/prepare_data.py` 的 `prepare_dataset()` 根据 `Cr、In、Pa、PS、RS、Sc` 前缀，将原始 BMP 复制到六个类别目录。它只做分类归档，不做图像缩放、归一化、数据增强或训练测试划分。

当前六类各 300 张，共 1,800 张；`processed_data/` 根目录的 `test.bmp` 不在类别目录内，不属于 Dataset 的训练样本。原始数据文件数、处理后目录中图片数、Dataset 的长度是三个不同概念，需要分别确认。

当前整理脚本末尾还有引用未定义 `dataset` 的统计片段，直接执行会在归档后报错。因此“文件复制完成”和“整个脚本正常结束”也需要区分。已有 `test_dataset.py` 能实例化 Dataset 并统计标签，适合用于检查整理结果。

本项目体现的一个基本经验是：**在关注模型之前，先核对每类样本、标签映射与读取范围。** 输出层有六个节点，不代表数据一定包含六类；数据数量必须独立验证。

## 3. Dataset：定义如何读取一条样本

`DefectDataset` 继承 `torch.utils.data.Dataset`，实现三个方法：

| 方法 | 项目中的实现 | 要理解的职责 |
| --- | --- | --- |
| `__init__` | 扫描六个目录，将路径和整数标签存入列表 | 建立可索引的样本清单 |
| `__len__` | 返回路径列表长度 | 告诉训练框架样本总数 |
| `__getitem__` | 读图、RGB 转换、transform，返回 `(image,label)` | 定义一条样本的读取规则 |

读取图片时的核心代码是：

```python
image = Image.open(img_path).convert("RGB")
if self.transform:
    image = self.transform(image)
return image, label
```

原始灰度图转换为三个 RGB 通道，满足 CNN 和预训练 ResNet 的三通道输入要求；三个通道相同并不等于新增了真实颜色信息。

Dataset 初始化主要保存路径，不会立即把所有图片读到内存。访问 `dataset[index]` 时才实际解码对应图片。代码当前接受 `.bmp`、`.jpg`、`.png`，不递归扫描更深层目录，也没有显式排序文件名。

这里的 transform 由调用者传入：CNN 训练用 `ToTensor()`，ResNet 训练用 Resize + ToTensor。Dataset 本身不绑定某个网络结构。

## 4. DataLoader：把样本组织成模型能处理的批次

Dataset 回答“第 i 个样本是什么”，DataLoader 回答“这一批取哪些样本、怎样堆叠、是否打乱顺序”。

`train_resnet.py` 使用 batch size 32、`shuffle=True`。对当前 1,800 张图片，80% 训练子集是 1,440 张，每轮 45 个 batch；剩余 360 张构成留出子集。通常训练 batch 的形状是：

```text
images: [32, 3, 224, 224]，float32
labels: [32]，整数类别索引
```

`test_resnet.py` 的 `shuffle=False` 表示按测试子集顺序读取。最后不足 32 张时仍保留，因为代码没有设置 `drop_last=True`。CNN 路线图像 batch 是 `[B,3,200,200]`。

`random_split()` 返回引用原 Dataset 索引的 `Subset`，不是把图片移动到两个文件夹。ResNet 三个相关入口用相同种子 42 生成划分，要求输入样本顺序也一致。当前没有保存清单，目录顺序变化可能让相同索引对应不同图像；CNN 路线则尚未固定共享划分。

## 5. CNN：从局部纹理到分类分数

`models/cnn.py` 定义 `DefectCNN`，先用卷积提取特征，再用全连接层分类：

```text
[B,3,200,200]
    ↓ Conv 3→16 + ReLU + MaxPool
[B,16,100,100]
    ↓ Conv 16→32 + ReLU + MaxPool
[B,32,50,50]
    ↓ Conv 32→64 + ReLU + MaxPool
[B,64,25,25]
    ↓ Flatten
[B,40000]
    ↓ Linear 40000→128 + ReLU
[B,128]
    ↓ Linear 128→6
[B,6]
```

卷积核在空间上共享权重，学习局部边缘和纹理模式；ReLU 引入非线性；池化缩小空间尺寸；全连接层汇总特征并给每类打分。

该模型第一层全连接固定接收 `64×25×25` 个值，因此不能随意改变输入尺寸。项目 CNN 数据是 `200×200`；如果直接改成 ResNet 的 `224×224`，三次池化后是 `28×28`，全连接层维度就不匹配。

CNN baseline 从随机初始化开始学习，没有导入外部预训练模型。`test_model.py` 用随机 Tensor 检查 `[1,6]` 输出，只能证明结构可前向计算，不能证明网络认识缺陷。

## 6. ResNet：用残差连接构建更深的网络

`models/resnet.py` 调用 TorchVision 的 ResNet18，并将最后的 `fc` 替换为 `Linear(512,6)`。包装类的 `forward()` 将输入交给内部 `self.model`。

对本项目 `224×224` 输入，主要形状如下：

| 阶段 | 输出形状 | 含义 |
| --- | --- | --- |
| 输入 | `[B,3,224,224]` | RGB batch |
| 7×7 卷积、BN、ReLU | `[B,64,112,112]` | 初始纹理提取 |
| 最大池化 | `[B,64,56,56]` | 空间下采样 |
| layer1 | `[B,64,56,56]` | 2 个 BasicBlock |
| layer2 | `[B,128,28,28]` | 2 个 BasicBlock |
| layer3 | `[B,256,14,14]` | 2 个 BasicBlock |
| layer4 | `[B,512,7,7]` | 2 个 BasicBlock，CAM 使用最后一个块 |
| 自适应平均池化、展平 | `[B,512]` | 每通道汇总空间信息 |
| 六类 fc | `[B,6]` | 分类 logits |

BasicBlock 的主分支包含两个卷积，通过捷径与输入相加：

\[
y=\operatorname{ReLU}(F(x)+S(x))
\]

形状相同时 `S(x)=x`；需要下采样或改变通道时，捷径用投影对齐。残差连接让网络学习对输入的修正，为梯度提供更直接的传播路径，使深层网络更易优化。

ResNet18 的主路径包含开头 1 个卷积、8 个残差块中的 16 个卷积以及最后 1 个全连接层，这构成常说的“18 层”。池化、激活、BN 和捷径投影不全部计入这个命名。

## 7. 迁移学习：继承预训练参数，再学习缺陷任务

模型构造代码为：

```python
self.model = models.resnet18(weights="DEFAULT")
in_features = self.model.fc.in_features
self.model.fc = nn.Linear(in_features, num_classes)
```

第一步获得 ImageNet 预训练模型；第二步将原先 1,000 类分类头替换为六类分类头。新分类头随机初始化，骨干特征层继承预训练参数。

`train_resnet.py` 将全部 `model.parameters()` 交给 Adam，且没有冻结层，所以本项目是**全网络微调**。这与“固定骨干、只训练新分类头”不同。

预训练特征可能帮助小样本任务更快学习，但自然图像与钢铁灰度纹理存在差异，是否比 CNN 更好需要在相同、独立的数据划分上比较。不能仅凭网络名称断言准确率提高。

当前 ResNet 预处理是 `Resize((224,224)) + ToTensor()`，没有 ImageNet Normalize。训练、普通预测、Grad-CAM 目前保持同样约定。若之后比较标准化方案，需要作为新的实验同步调整训练和推理，不能仅在网页端更改输入处理。

## 8. Loss：用交叉熵衡量预测与标签的差异

两个训练脚本都使用 `nn.CrossEntropyLoss()`。输入为 `[B,6]` 原始分数和 `[B]` 整数标签。

对真实类别为 y 的单张图片，设输出 logits 为 z：

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad L=-\log p_y
\]

真实类别概率低，Loss 大；真实类别概率高，Loss 小。PyTorch 的交叉熵内部完成对应的稳定计算，所以训练时直接传 logits，不需要先执行 Softmax，也不需要手工生成 one-hot 标签。

`loss.item()` 把标量 Tensor 取为 Python 数值，供日志累计。当前脚本对各 batch 的平均 Loss 再取平均；若最后一批更小，这与按样本数加权的平均稍有不同。当前 ResNet 训练数 1,440 恰好能被 batch size 32 整除。

Loss 不等于准确率：当正确类别已经排第一，继续提高它的概率仍可能降低 Loss，但准确率不会变化。训练 Loss 持续降低也不能单独证明泛化能力提高。

## 9. Optimizer 与 Backpropagation：怎样真正学到参数

项目使用 `torch.optim.Adam(model.parameters(), lr=0.001)`。可以先用“参数减去学习率乘梯度”理解基本更新方向；Adam 进一步维护梯度的一阶、二阶统计，调整各参数的更新尺度。

当前训练循环的关键代码为：

```python
outputs = model(images)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这五步分别对应：

1. **前向传播**：按当前参数产生分类分数。
2. **计算损失**：将分数与真实类别比较。
3. **清空梯度**：PyTorch 默认累加梯度，先清除上一批残留。
4. **反向传播**：按链式法则，计算 Loss 对卷积、全连接等参数的导数。
5. **优化器更新**：根据梯度和 Adam 状态更新参数。

`backward()` 计算梯度，`step()` 改变参数，二者职责不同。一个 epoch 表示遍历训练集一次；按当前 ResNet 设置，每轮 45 次参数更新，10 轮共 450 次。这是依据代码推算的运行规模，不是本次重新训练的记录。

当前没有学习率调度、早停、数据增强或差分学习率。是否需要这些方法属于后续算法实验；本次工程整理没有添加它们。

## 10. train、eval、no_grad 和设备管理

`model.train()` 和 `model.eval()` 控制模型层的运行行为。ResNet 中 BatchNorm 训练时使用批次统计并更新运行统计，评估时使用保存的统计；自建 CNN 没有 BatchNorm 或 Dropout，模式切换对现有数值路径的影响较小。

`torch.no_grad()` 控制自动微分是否记录计算图，不会自动切换模型模式。普通预测通常同时需要 `eval()` 与 `no_grad()`。

| 场景 | 项目中的做法 | 是否更新参数 |
| --- | --- | --- |
| ResNet 训练 | `train()`、Loss、backward、Adam.step | 是 |
| ResNet 评价／普通 predict | `eval()`、`no_grad()` | 否 |
| CAM 计算 | `eval()`，允许梯度，针对类别分数反向传播 | 否，没有 optimizer.step |

`train_resnet.py`、`test_resnet.py`、`predict.py` 和 `gradcam_utils.py` 根据 CUDA 是否可用选择设备，并移动模型和输入。当前 `gradcam.py`、`confusion_matrix.py` 使用 CPU；CNN 训练／测试也是 CPU。

设备约定要求参与同一运算的模型和输入匹配，不能只把权重移到 GPU 而留下 CPU 图片。`map_location` 控制参数读取的位置，`.to(device)` 控制模型或 Tensor 的实际设备。

## 11. 模型保存与加载：保存的是学到的数值

训练用：

```python
torch.save(model.state_dict(), "models/defect_resnet18.pth")
```

推理先构造相同结构，再用 `torch.load()` 与 `load_state_dict()` 恢复参数，并设置 `eval()`。

`state_dict` 是“参数／缓冲区名称 → Tensor”的映射，包含卷积、全连接参数及 BatchNorm 的统计量，不包含完整 Python 前向函数。ResNet 包装后的键带 `model.` 前缀，如 `model.fc.weight`，六类头的权重形状为 `[6,512]`。

因此不能只拿到 `.pth` 就随便搭一个网络：模型结构、层名、输出维度必须对应，类别顺序和预处理也要保持一致。

当前两条路线的保存策略不同：

- `train.py`：训练结束保存最后一轮 CNN 参数。
- `train_resnet.py`：当平均训练 Loss 创新低时保存 ResNet 参数；未使用已创建的 `test_loader` 做验证。

所以代码里的“最佳模型”只表示训练 Loss 最低，不能解释为测试表现最佳。文件也没有保存优化器状态、epoch、类别映射、数据划分或预处理配置，不能直接精确续训或追溯历史实验。

此外，`DefectResNet()` 始终先请求 ImageNet 预训练初始化，推理时随后才恢复本地缺陷参数。这解释了为什么本地任务权重存在时，缺少预训练缓存仍可能发生联网下载。

## 12. 评价指标：不仅看总体正确率

`test_resnet.py` 逐批计算预测，取六类 logits 的最大索引，并统计：

\[
\mathrm{Accuracy}=\frac{\text{预测正确数量}}{\text{测试总数}}
\]

脚本最后乘 100 显示百分比。Accuracy 适合观察总体表现，但可能掩盖某类容易漏判的问题。

`confusion_matrix.py` 的 `classification_report()` 提供每类 Precision、Recall、F1 和 support。对某一类：TP 为正确识别成该类，FP 为其他类被误判成该类，FN 为该类被判为其他类。

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}
\]

Precision 关注“被判为该类的结果有多可靠”，Recall 关注“真实该类有多少被识别出来”，F1 综合二者。support 是某类实际样本数；macro average 为各类等权平均，weighted average 按 support 加权。没有对应真实样本时，不能据此可靠评估该类召回表现。

项目评估还应注意样本独立性：ResNet 已固定分割种子，但依赖相同的 Dataset 索引顺序；CNN 训练与测试仍独立随机划分。使用同一可靠划分，才有依据比较两条模型路线。本文不提供未经完整评估获得的准确率数字。

## 13. 混淆矩阵：观察模型具体错在哪里

`confusion_matrix.py` 收集 `all_labels` 和 `all_preds`，调用 scikit-learn 生成矩阵，再由 Seaborn 画热力图。

矩阵 `C[i,j]` 表示“真实为 i，预测为 j”的数量。行是真实类别，列是预测类别；对角线表示正确预测，非对角线表示分类混淆。

例如，真实夹杂被判成划痕，会进入行 1、列 5。这个例子仅用于解释矩阵位置，不代表当前模型确实发生了该错误。

如果某类大量流向另一个类别，可以继续检查原图、标签、特征相似性与 CAM；矩阵本身不自动说明错误原因。当前显示的是计数，不是归一化百分比，且仅 `plt.show()` 展示，没有保存图文件。

代码没有显式固定 `labels=[0,1,2,3,4,5]`。如果某类不出现在真实或预测集合中，矩阵可能缩小，与硬编码六类刻度不对应。明确矩阵轴的类别映射，也是正确解读指标的一部分。

## 14. 单张图片预测：从像素到概率

当前 `predict.py` 提供 `predict(image_path)`，步骤为：

```text
文件路径
    ↓ Image.open(...).convert("RGB")
PIL RGB 图片
    ↓ Resize((224,224))
224×224 图片
    ↓ ToTensor()
[3,224,224] float32，像素从 0～255 缩放到 0～1
    ↓ unsqueeze(0)，再 to(device)
[1,3,224,224]
    ↓ ResNet18
[1,6] logits
    ↓ Softmax(dim=1)、max(dim=1)
类别索引与最大概率
    ↓ classes[index]
英文类别名与 Python float 置信度
```

batch 维度不能省略，因为模型以 `[N,C,H,W]` 组织输入；单张图片就是 N=1。模型输出未归一化的 logits，Softmax 把它们变为和为 1 的六类概率。

取 logits 的最大值与取 Softmax 后概率的最大值会得到相同类别，因为 Softmax 保持分数顺序。单图函数需要展示概率，因此进行了 Softmax；批量准确率脚本只关心类别，可以直接对 logits 取最大索引。

最大概率称为置信度，但不等于模型在测试集上的 Accuracy，也不保证这个概率已被校准。网页对未知输入没有拒识能力，仍会选择六类中的一个。

## 15. Grad-CAM：解释某个类别分数关注了哪里

`gradcam_utils.py` 在 `generate_gradcam()` 中先得到预测类别，再选择：

```python
target_layers = [model.model.layer4[-1]]
targets = [ClassifierOutputTarget(prediction.item())]
```

外层 `model` 是 `DefectResNet`，内部 `.model` 才是 TorchVision ResNet。目标层是 layer4 的最后一个 BasicBlock，当前输入尺寸下产生 512 个 `7×7` 特征图。

CAM 计算过程为：前向记录特征 → 对预测类别 logit 求梯度 → 对每个通道的梯度做空间平均 → 用平均梯度加权该通道 → 相加并保留正贡献 → 放大和归一化。

设第 k 个通道特征为 `A^k`，目标类别分数为 `z^c`：

\[
\alpha_k^c=\frac{1}{Z}\sum_i\sum_j\frac{\partial z^c}{\partial A^k_{ij}},
\qquad
L^c=\operatorname{ReLU}\left(\sum_k\alpha_k^c A^k\right)
\]

`grayscale_cam[0]` 取的是 batch 中第 1 张图片的热力图，不是类别 0。工具函数随后执行：

```python
rgb_img = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
```

模型输入是 CHW Tensor，显示底图是 HWC NumPy 数组；二者不能混淆。CAM 库内部使用 OpenCV 生成颜色映射，因此即使项目没有直接写 `import cv2`，OpenCV 仍在真实调用链里。

当前首次分类和 CAM 的再次计算都允许梯度记录。首次仅用于取类别，可以在后续优化时单独关闭梯度；CAM 部分必须保留。这里的反向传播目标是类别分数，而训练的反向目标是监督 Loss；两者目的不同，CAM 不更新模型参数。

热力图是对模型判断的解释，不是缺陷分割标注。由 `7×7` 放大到 `224×224` 不会创造精细定位精度；模型预测错误时，CAM 解释的仍是这个错误类别。

## 16. Streamlit 部署：把推理函数连接到用户交互

`app.py` 的职责是接收用户输入并呈现结果。实际顺序如下：

1. `st.set_page_config()`、`st.title()` 和 `st.write()` 设置页面。
2. `st.file_uploader()` 接收 BMP、JPG、PNG 文件。
3. `Image.open(uploaded_file).convert("RGB")` 从上传对象读取图片。
4. `st.image()` 预览原图。
5. 点击 `st.button("开始检测")` 后，将图片写入临时 BMP。
6. 调用 `generate_gradcam(image_path)`，获得类别、置信度和 CAM 图片。
7. 用 `st.success()` 显示类别，`st.info()` 以两位小数百分比显示置信度，再用 `st.image()` 显示热力图。

启动方式为：

```powershell
python -m streamlit run app.py
```

浏览器中的交互触发 Python 脚本重运行，深度学习推理在运行 Streamlit 的 Python 进程及其 CPU／GPU 上执行，不是在浏览器 JavaScript 中执行。页面不会调用训练入口，也不会将上传图片纳入 Dataset。

当前 `generate_gradcam()` 每次都重新加载模型；app 没有缓存，也没有使用 `st.session_state` 保存检测结果。结果位于按钮分支内，后续其他交互引起重运行时不一定保留该分支输出。

临时文件使用 `delete=False`，当前没有删除步骤；输入解码和模型加载也没有友好的异常处理。这些是部署工程知识，与改变 ResNet 的算法不同。未来缓存共享模型时，还应注意 Grad-CAM 的 hooks 和梯度操作，避免多会话同时修改共享状态。

## 17. 把知识点连接起来

| 学习问题 | 在项目中查看的位置 | 当前可以观察到的事实 |
| --- | --- | --- |
| 标签从哪里来？ | `prepare_data.py`、`dataset.py` | 文件前缀转目录，目录顺序转整数标签 |
| 图片怎样成为 batch？ | Dataset、训练脚本的 transforms／DataLoader | PIL → CHW Tensor → NCHW batch |
| CNN 怎样抽取特征？ | `models/cnn.py` | 三组卷积池化与固定维度全连接 |
| 为什么叫迁移学习？ | `models/resnet.py`、`train_resnet.py` | 继承预训练骨干，替换分类头，全网络微调 |
| 模型如何学习？ | 两个训练入口 | logits → CrossEntropyLoss → backward → Adam.step |
| 学习结果存在哪里？ | `torch.save(model.state_dict(),...)` | 固定路径 `.pth` 参数文件 |
| 怎样判断预测好坏？ | 测试与混淆矩阵脚本 | Accuracy、每类指标和分类错误分布 |
| 怎样解释一次判断？ | 两种 CAM 入口 | 对目标类别求梯度，加权最后残差块的特征 |
| 怎样把模型给用户使用？ | `app.py`、`gradcam_utils.py` | 上传、临时文件、函数调用与展示 |

复现实验时还需要记住：代码能运行、单张图预测正确、独立测试表现可靠，以及网页能稳定服务用户，是四种不同层次的验证。当前项目已贯通主要功能；数据划分记录、验证策略、资源管理和错误处理仍是进一步工程化的学习内容。
