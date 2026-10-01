# inspectiontesting 工程结构与代码质量检查报告

检查日期：2026-09-30。范围：项目根目录、`models/`、`utils/` 中全部 15 个 Python 文件，以及数据目录、现有权重、文档和实际依赖。检查以当前代码为准，不沿用早期技术说明中的过时结论。

本次按“先分析，再生成交付文件”的顺序进行：阅读代码和统计数据 → 检查导入、依赖与关键运行路径 → 新增 README、requirements、检查报告和学习总结。没有移动文件，没有改写 Python 源码、数据或模型参数，也没有重新训练。

## 1. 当前状态与重要结论

项目已经形成“数据处理 → 模型训练 → 模型评价 → 单图推理／解释 → Streamlit 展示”的基本结构。`app.py` 调用 `gradcam_utils.generate_gradcam()`，后者完成 ResNet 推理及 Grad-CAM；`predict.py` 现在同样使用 ResNet。

当前数据已完整：原始 BMP 1,800 张，六个已整理类别目录各 300 张；Dataset 使用 1,800 张，根目录独立 `test.bmp` 不参加 Dataset。原先 `Ps`／`Rs` 大小写问题已在现有代码中改为 `PS`／`RS`，不再作为未修复问题报告。

ResNet 训练、评估、混淆矩阵已经统一使用 `torch.Generator().manual_seed(42)`；CNN 路线仍是独立随机划分。现阶段最直接的运行错误是 `utils/prepare_data.py:68` 引用了未定义的 `dataset`。其余主要问题为配置散落、导入副作用、网页重复加载模型、临时文件未清理及实验记录不足。

## 2. 按职责检查当前目录

### 2.1 数据处理层

| 文件／目录 | 职责 | 判断 |
| --- | --- | --- |
| `data/NEU-CLS/` | 原始图片存储；标签由文件名前缀编码 | 位置合理，属于数据资源，不是 Python 模块 |
| `data/processed_data/` | 六个类别目录及独立示例图片 | 类别组织清晰；训练子集和测试子集并未物理分目录 |
| `utils/prepare_data.py` | 按前缀创建目录、复制 BMP | 数据准备脚本；末尾附带的统计代码有确定错误 |
| `utils/dataset.py` | `DefectDataset`：保存路径和标签、读取 RGB 图片、应用 transform | 可复用的数据访问组件 |
| `test_dataset.py` | 打印数量、样本形状和类别计数 | 数据层的人工检查入口，不是严格的自动化单元测试 |

`utils/` 目前主要承担数据访问，不是完全无关的杂项集合。小型项目使用当前结构可以接受；暂不需要为了分层而大量增加目录。

### 2.2 模型层

| 文件／目录 | 职责 | 判断 |
| --- | --- | --- |
| `models/cnn.py` | 定义 `DefectCNN` | 模型结构的位置合理 |
| `models/resnet.py` | 定义 `DefectResNet`，加载预训练 ResNet18 并替换分类头 | 结构与初始化策略目前绑定在一起 |
| `models/defect_cnn.pth` | CNN 参数字典 | 属于模型产物，不属于源代码 |
| `models/defect_resnet18.pth` | ResNet 参数字典 | 同上，当前被多个入口按固定路径加载 |
| `test_model.py` | 用随机 Tensor 检查 CNN 输出形状 | 模型层的人工检查入口；没有加载学习后的参数 |

结构源码和权重放在同一目录对当前规模可理解，但后续若增加多个实验，宜把产物独立管理。此建议不代表本次移动权重。

### 2.3 训练层与评价层

| 文件 | 职责 | 当前主要行为 |
| --- | --- | --- |
| `train.py` | CNN 训练 | 未固定划分种子；CPU；保存最后一轮参数 |
| `train_resnet.py` | ResNet 全网络微调 | 固定种子 42；自动选择设备；按最低训练 Loss 保存 |
| `test.py` | CNN 整体准确率 | 新做随机划分并加载 CNN 权重 |
| `test_resnet.py` | ResNet 整体准确率 | 固定划分种子 42，模型及 batch 使用同一设备 |
| `confusion_matrix.py` | ResNet 分类报告与混淆矩阵 | 固定划分种子 42；当前在 CPU 评估，Matplotlib 展示 |

评价逻辑属于与训练配套的独立职责，不应被混称为模型结构。各入口直接依赖 Dataset 和模型，并通过 `.pth` 文件衔接。

### 2.4 推理层与可解释分析层

| 文件 | 职责 | 调用边界 |
| --- | --- | --- |
| `predict.py` | `predict(image_path) → (label, confidence)` | 模型在模块导入时构造并加载；main guard 仅保护最后的示例调用 |
| `gradcam_utils.py` | `generate_gradcam(image_path) → (label, confidence, visualization)` | 每次函数调用重新构造模型，加载权重并生成 CAM |
| `gradcam.py` | 单张图片的离线 CAM 演示 | 独立实现处理链，最后 `plt.show()`；没有调用公共工具函数 |

目前网页使用的是 `gradcam_utils.py`，不是导入并执行 `gradcam.py`。`gradcam.py` 的 `targets=None` 会由当前 CAM 库默认选择预测类别；工具函数则显式使用 `ClassifierOutputTarget` 指定预测类别，两者意图一致。

### 2.5 网页应用层及其他目录

`app.py` 负责页面配置、图片上传、预览、检测按钮、临时文件适配及结果展示。它已把主要深度学习逻辑交给工具函数，是合理的初步分层。

`docs/` 存放背景与解释资料；`.venv/`、`__pycache__/` 是环境与缓存；`.git/` 是版本控制元数据。这些不属于训练或推理业务层。

### 2.6 位置调整建议：仅建议，不执行移动

| 当前位置 | 将来可考虑的位置 | 原因与迁移注意事项 |
| --- | --- | --- |
| 根目录 `gradcam_utils.py` | 小规模时 `utils/gradcam_utils.py`；功能扩大后 `inference/gradcam.py` | 其职责是推理／解释服务；二选一即可，迁移需同步 app 导入 |
| 根目录 `predict.py` | 保留命令入口，公共函数将来放 `inference/predictor.py` | 保持用户命令不变，分离可复用逻辑与运行入口 |
| 根目录训练、评价脚本 | 当前保持；规模扩大后考虑 `scripts/` | 目前入口少，集中放根目录易于学习；搬迁会影响命令与模块路径 |
| `models/*.pth` | 后续可放 `artifacts/checkpoints/` | 源码与训练产物分离；需要同步所有读取路径 |
| `test_dataset.py`、`test_model.py` | 加断言后放 `tests/`，或保留为检查脚本 | 当前主要靠打印检查，不能仅移动后就宣称有自动化测试 |
| `data/` | 保持；可在将来增加 `splits/` 文件清单 | 清楚记录训练、验证、测试样本身份，比改目录名更重要 |

本次没有创建上述目录、没有修改 import，也没有移动任何文件。

## 3. 当前调用关系

```text
原始图片 → prepare_data.py → 类别目录
                                 ↓
                        DefectDataset + transforms
                                 ↓
                       random_split + DataLoader
                                 ↓
               train.py / train_resnet.py + 对应模型
                                 ↓
                         模型 state_dict 权重
                                 ↓
          test.py / test_resnet.py / confusion_matrix.py

单图路径 → predict.predict() → ResNet → 类别与置信度

浏览器上传 → app.py → RGB 图片 → 临时 BMP 文件
                                 ↓
                gradcam_utils.generate_gradcam()
                                 ↓
                   ResNet 分类 + Grad-CAM
                                 ↓
               类别、置信度、热力图 → app.py 展示
```

训练脚本不会被网页调用。Grad-CAM 使用反向梯度解释结果，但不执行优化器更新；没有把用户上传图像继续加入训练的逻辑。

## 4. 重复代码检查

| 重复内容 | 位置 | 现状与建议 |
| --- | --- | --- |
| 六类名称和顺序 | Dataset、predict、gradcam、gradcam_utils、混淆矩阵，共 5 处 | 当前名称顺序一致；将来可共享只读类别常量，必须保持编号不变 |
| ResNet 预处理 | train_resnet、test_resnet、混淆矩阵、predict、gradcam、gradcam_utils，共 6 处 | 均为 `Resize((224,224)) + ToTensor()`；可集中构造，但不能顺便添加 Normalize／增强 |
| ResNet 构造、权重加载、设备设置、eval | 测试、预测和解释入口 | 可提取加载函数；训练的预训练初始化与推理的权重恢复需明确区分 |
| 分类后处理 | predict 与 gradcam_utils | 均执行 Softmax／最大类别映射；公共函数可减少偏差 |
| CAM 特征层、叠图和缩放 | gradcam 与 gradcam_utils | 可让离线演示复用工具函数，但 CPU／CUDA策略及返回结果应保持约定 |
| 数据划分和 DataLoader | 训练、测试、混淆矩阵 | 可共享划分方法，但应先保存当前样本名单再重构 |
| CNN／ResNet 训练循环 | 两个训练入口 | 教学项目保留两份对照有价值；无需为了消除重复引入复杂训练框架 |
| 类别计数 | test_dataset 与 prepare_data 文件末尾 | prepare_data 中的复制片段缺少 Dataset 实例，是实际错误来源 |

重复代码不自动意味着需要本次重构。对于预处理、类别顺序、随机数使用等敏感逻辑，抽取公共组件必须有等价性验证，不能以整理代码为由改变模型输入分布或训练顺序。

## 5. 硬编码路径与配置文件建议

### 5.1 已发现的固定路径和参数

| 内容 | 当前形式 | 涉及入口 |
| --- | --- | --- |
| 处理后数据目录 | `data/processed_data` | 训练、测试、混淆矩阵、数据检查 |
| ResNet 权重 | `models/defect_resnet18.pth` | ResNet 训练、评估、预测、两种 CAM 入口 |
| CNN 权重 | `models/defect_cnn.pth` | CNN 训练和评估 |
| 示例图片 | `data/processed_data/crazing/Cr_1.bmp` | predict 的 main 示例、gradcam |
| 输入尺寸 | ResNet 224；CNN 与其全连接头匹配的 200 | 多处 transform 与模型结构 |
| 训练超参数 | batch 32、epoch 10、lr 0.001 | 两个训练入口 |
| 分割比例／随机种子 | 0.8／0.2，ResNet 为 42 | ResNet 训练、测试、混淆矩阵 |
| Grad-CAM 层 | `model.model.layer4[-1]` | 两种 CAM 入口 |

`prepare_data.py` 使用 `__file__` 定位项目根目录，是现有相对路径问题的一个正面例子；大多数其他入口依赖当前工作目录。把 `data_path` 写成变量仍然是固定配置，并没有使其与运行目录解耦。

### 5.2 是否需要增加 `config.py`

**有必要作为后续小步重构，但本次只提出设计，不创建未被使用的配置文件。** 对当前规模，一个无副作用的 Python 配置模块比新增配置框架更合适。

可考虑集中以下值：

| 配置项 | 保持现有行为时的取值／约束 |
| --- | --- |
| `PROJECT_ROOT` | 以配置文件所在位置推导项目根目录 |
| `RAW_DATA_DIR`、`PROCESSED_DATA_DIR` | 指向现有 data 子目录 |
| `CNN_WEIGHTS_PATH`、`RESNET_WEIGHTS_PATH` | 指向现有权重，不改变保存目标 |
| `CLASS_NAMES`、`NUM_CLASSES` | 保持现有六类顺序与 6 类输出 |
| `RESNET_INPUT_SIZE`、`CNN_INPUT_SIZE` | 分别保持 224 和 200；不强迫两个模型共用尺寸 |
| `BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE` | 保持 32、10、0.001 |
| `RESNET_SPLIT_SEED`、`TRAIN_RATIO` | 保持 42、0.8，不擅自给 CNN 增加种子 |
| `EXAMPLE_IMAGE_PATH` | 保持现有示例路径 |

配置模块不应在导入时设置随机种子、创建模型、选择并初始化设备、读写数据或加载权重。更改这些动作的先后顺序可能改变随机数状态，影响训练结果。

也不建议本次增加 YAML 等配置依赖：当前任务没有使用对应库，简单常量已足够。真正接入配置前，应核对数据样本顺序、每一步 transform、模型构造顺序与保存策略完全不变。

## 6. 潜在运行问题与质量风险

严重度说明：P1 为明确运行错误或影响评价可信度的问题；P2 为维护、可靠性或性能问题。下列均为检查结果及建议，尚未修改源码。

### P1-01：数据整理入口引用未定义变量

**位置：** `utils/prepare_data.py:64–73`，首次错误发生在第 68 行的 `len(dataset)`。

文件没有定义或导入 `dataset`。直接执行时，先运行 `prepare_dataset()` 复制图片，再进入统计片段报 `NameError`；导入该模块时 main guard 虽然会跳过复制，末尾统计仍会执行并报错。

本次通过 `runpy.run_path(..., run_name="__review_read_only__")` 跳过 main 分支，复现了 `name 'dataset' is not defined`，没有运行归档函数。后续可移除误放的诊断片段，或将完整统计集中在已有检查入口；本次不执行该修复。

### P1-02：CNN 测试样本可能与训练样本重叠

**位置：** `train.py:43`、`test.py:36` 的 `random_split()`。

两处都没有共享索引或固定生成器，单独运行时会重新随机划分，测试样本可能曾用于训练。当前 CNN 准确率不能直接作为严格独立测试集的性能。后续应保存划分及样本身份；这会改变实验条件，不在本次文档整理中执行。

### P2-01：ResNet 固定种子仍依赖样本顺序和权重来源

**位置：** `utils/dataset.py` 的 `class_dir.iterdir()`；ResNet 三个入口的种子 42。

固定生成器已经解决了相同输入索引序列下划分不一致的问题；不能继续描述为“ResNet 每次无种子随机测试”。但 Dataset 未显式排序，不同机器、文件增删或目录遍历顺序变化仍可能让相同索引指向不同图片。现有权重也没有记录训练名单。

后续宜保存具体文件清单。不要在现有权重评估前直接加 `sorted()` 并认为划分仍相同：排序本身会改变索引与样本的对应关系。`torch.manual_seed(42)` 也不保证跨设备、驱动、后端版本训练逐位一致。

### P2-02：“最佳模型”按训练 Loss 选择，没有验证循环

**位置：** `train_resnet.py:84` 创建但未消费的 `test_loader`；第 183 行比较 `avg_loss < best_loss`。

保存条件是平均训练损失降低，不是验证 Loss 或验证准确率最佳；无法据此判断泛化是否提升。后续若加入验证集，应另行设计训练／验证／最终测试协议，而非反复用最终测试集选模型。本次保留当前保存逻辑。

### P2-03：网页每次点击都重新加载模型

**位置：** `gradcam_utils.py:35` 的函数体内部构造模型和加载参数。

这会重复占用初始化和权重读取时间；预训练缓存缺失时还可能触发下载。可在后续用独立加载函数和 Streamlit 资源缓存优化。不过 Grad-CAM 会注册 hooks、清零模型梯度并执行反向计算，共享模型时必须考虑多会话并发和 hook 生命周期，不能直接缓存一个长期共用的 CAM 对象。

### P2-04：上传图片产生的临时文件未删除

**位置：** `app.py:67–78` 的 `NamedTemporaryFile(delete=False)`。

退出 `with` 只关闭句柄，不删除该文件。后续没有 `unlink` 或 `finally` 清理，多次使用会留下临时 BMP。将来可在推理结束后清理，或让推理函数直接接受 PIL／内存图片；后一方案需验证 RGB 转换和预处理结果一致。

### P2-05：输入与资源错误没有友好处理

**位置：** app 的 `Image.open()`、工具函数的权重加载、Dataset 的目录遍历。

扩展名过滤不保证文件内容可解码。损坏图片、缺失权重、类别目录缺失或显存不足会直接成为异常。Dataset 对空目录无类别完整性检查；总样本数为零还可能使 DataLoader 或准确率计算失败。

当前已整理数据数量正常，因此这属于异常输入／资源条件下的风险，不是本次发现现有目录缺类。

### P2-06：模块导入有执行副作用

`predict.py` 在模块顶层加载模型，main guard 只保护最后的示例调用。多数训练、评估和 `gradcam.py` 也在顶层直接执行。将这些文件作为函数库导入，可能触发耗时初始化、训练、评估或图窗。

后续可给普通命令行脚本增加入口函数并把示例执行放进 main guard。Streamlit 的 `app.py` 本来按脚本重运行，不需要机械套用同一种重构方式。

### P2-07：Grad-CAM 初次分类额外保留计算图

`gradcam_utils.py` 和 `gradcam.py` 先执行一次普通分类，随后 CAM 库再执行一次前向与反向。首次分类没有使用 `no_grad()`，会保留不必要的梯度图。

可只对第一次用于选类别和概率的分类使用无梯度上下文；不能将整个 `generate_gradcam()` 包在 `no_grad()` 或 `inference_mode()` 中，否则 CAM 的梯度路径会失效。本次不调整该逻辑。

### P2-08：预处理需要明确版本约定

当前所有 ResNet 路径都使用 `Resize((224,224)) + ToTensor()`，训练和推理一致，这是需要保留的行为。与 ImageNet 标准 Normalize 的差异属于后续算法实验议题，不在本次补依赖或整理文档时修正。

直接把任意长宽比图片拉伸到正方形也可能改变缺陷形态。不能为了工程化擅自改为裁剪、补边或新尺寸，因为这会改变预测输入。

### P2-09：评价输出缺少明确元数据

`confusion_matrix()` 未指定完整六类 `labels`，`classification_report()` 未指定名称；缺类测试子集下矩阵可能缩小，而坐标轴仍写六类。当前六类目录完整，问题不一定在本次划分中触发，仍应在通用评估中处理。

图窗与文本报告没有保存为文件；权重仅含参数，没有版本、预处理、类别映射、样本划分或优化器状态。后续宜保存实验元数据，不能根据参数文件名推断历史训练条件。

### P2-10：GitHub 展示资料与早期文档不一致

已有 `docs/project_explanation.md` 仍写着 `app.py` 为空、Dataset 只有 1,200 张、`predict.py` 为 CNN，这些已过时。`project_background.md` 也仍把网页列为计划。本次保留原文，在新 README 明确标记它们是历史阶段资料，当前状态以新报告为准。

此外，当前没有 `.gitignore`，Git 状态中可见 `.venv/`、缓存、数据和权重均为未跟踪内容。未来发布时应明确哪些资源随仓库分发、哪些单独获取，并避免误提交环境与缓存；本次不增加忽略规则，也不执行提交或发布。

## 7. 实际依赖审计

本次以源码 AST 导入、已安装包元数据和 Grad-CAM 实际调用链交叉检查，没有直接复制整个 `pip freeze`。

| PyPI 包 | 源码中的导入／用途 | requirements 处理 |
| --- | --- | --- |
| `torch` | Tensor、Dataset、DataLoader、模型、梯度、优化器 | 保留 2.5.1；本机为 +cu121 构建 |
| `torchvision` | ResNet18、transforms | 保留 0.20.1，与 torch 配套 |
| `numpy` | CAM 原图数组与叠加输入 | 保留 2.5.2 |
| `Pillow` | `PIL.Image` 图片读取和保存 | 保留 12.3.0 |
| `scikit-learn` | `sklearn.metrics` 混淆矩阵和分类报告 | 保留 1.9.1 |
| `matplotlib` | 图窗和 CAM 库图像工具 | 保留 3.11.2 |
| `seaborn` | 混淆矩阵 heatmap | 保留 0.13.2 |
| `grad-cam` | `pytorch_grad_cam` | 保留 1.5.7 |
| `streamlit` | 网页控件与显示 | 保留 1.64.0 |
| `opencv-python` | 本项目无直接 `cv2` import；已检查 CAM 内部确实使用 OpenCV | 不额外加入 GUI 发行包 |
| `opencv-python-headless` | CAM 使用 `cv2.applyColorMap`、`cvtColor` 等；grad-cam 1.5.7 元数据也依赖此包 | 保留并约束为 5.0.0.93 |

本地同时装有两个 OpenCV 发行包，`pip check` 仍显示无损坏依赖，但这不说明共享的 `cv2` 文件不会相互覆盖。官方打包说明要求同一环境选择一种发行包；新清单只选 headless。本次未卸载现有包。[OpenCV 打包说明](https://pypi.org/project/opencv-python-headless/)

`pathlib`、`shutil`、`collections`、`tempfile` 属于 Python 标准库，不写入 requirements；也不加入未使用的 TensorFlow、torchaudio、Flask、FastAPI、pytest 或配置解析库。Grad-CAM／Streamlit 自身需要的 SciPy、ttach、pandas 等由包依赖解析，不因为环境中已安装就全部列成项目直接依赖。

锁定观察到的顶层版本可减少无意升级，但 requirements 不是完整的传递依赖哈希锁文件。NumPy 2.5.2 元数据要求 Python 至少 3.12，因此 README 以 Python 3.12 为安装基线。没有做全新环境在线安装，不承诺所有操作系统与解释器版本均兼容；本次也没有修改已安装环境。

## 8. 本次验证与范围限制

| 检查 | 结果 | 覆盖范围 |
| --- | --- | --- |
| 全部 15 个 Python 文件 AST 解析 | 通过 | 语法和导入扫描，不等于全部运行路径正确 |
| 实际依赖导入 | 通过 | torch、torchvision、NumPy、PIL、sklearn、Matplotlib、Seaborn、cv2、Streamlit、GradCAM |
| `python -m pip check` | 无损坏依赖 | 不能发现 OpenCV 共享命名空间问题 |
| Dataset 样本与标签 | 1,800 张、六类各 300 张，首样本 `[3,200,200]` | 读取规则和数据数量 |
| 数据整理模块的非 main 执行 | 复现 `NameError: dataset is not defined` | 跳过归档函数，没有重新复制数据 |
| 实际 `generate_gradcam()` 单图调用 | 返回 `crazing`、有效概率及 `[224,224,3]` uint8 图 | 已使用现有权重完成推理和 CAM，不是全数据评估 |
| Streamlit AppTest 初始页面 | 页面异常数 0，标题和上传控件均出现 | 未选择文件、未点击检测，不代表端到端上传已验证 |

运行检查还观察到两类信息：当前 `torch.load()` 产生默认加载方式的 FutureWarning；检查进程退出时，第三方临时目录清理出现 Windows `PermissionError`。后者发生在退出清理阶段，AppTest 已报告页面异常数为 0；本次没有将其误记为网页业务异常，也没有扩大权限清理用户临时目录。

未运行任何训练入口、未改变随机种子或划分实现、未执行完整准确率评估、未启动浏览器服务器、未做新环境安装。生成文件后核对了 19 个原有文件的 SHA-256：15 个 Python 文件、2 份权重和 2 份既有文档均保持不变。新增文档的本地链接、代码块和知识点覆盖检查通过，依赖清单中的版本约束均与当前已安装版本匹配。

## 9. 后续工程整理的实施顺序建议

1. 先处理数据整理入口的未定义变量，并给归档结果添加明确检查。这不需要修改分类模型。
2. 保存当前权重来源和样本清单，再讨论统一配置、路径和公共函数，避免整理过程改变实验条件。
3. 分离普通推理、模型加载、CAM 计算和网页文件适配；保持类别顺序、预处理和网络结构不变。
4. 增加临时文件清理、合理异常展示，并在考虑 CAM 并发后设计模型缓存。
5. 将数据划分、验证协议、Normalize、数据增强等可能改变训练结果的事项作为独立实验处理。

本次实际交付仅为 `README.md`、`requirements.txt`、`docs/project_review.md` 和 `docs/learning_summary.md`。目录调整、`config.py`、功能修复及训练策略变更均未实施。
