import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataset import DefectDataset
from models.resnet import DefectResNet


# =====================
# 参数
# =====================

batch_size = 32


# =====================
# 数据处理
# =====================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


dataset = DefectDataset(
    "data/processed_data",
    transform=transform
)


# =====================
# 固定测试集
# =====================

train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size


generator = torch.Generator().manual_seed(42)


_, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# =====================
# 加载模型
# =====================

model = DefectResNet(
    num_classes=6
)


model.load_state_dict(
    torch.load(
        "models/defect_resnet18.pth",
        map_location="cpu"
    )
)


model.eval()



# =====================
# 预测
# =====================

all_labels = []
all_preds = []


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)


        _, preds = torch.max(
            outputs,
            1
        )


        all_labels.extend(
            labels.numpy()
        )

        all_preds.extend(
            preds.numpy()
        )



# =====================
# 混淆矩阵
# =====================

cm = confusion_matrix(
    all_labels,
    all_preds
)


print("混淆矩阵:")
print(cm)



# =====================
# 分类报告
# =====================

print(
    classification_report(
        all_labels,
        all_preds
    )
)



# =====================
# 可视化
# =====================

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


plt.figure(figsize=(8,6))


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)


plt.xlabel("Prediction")

plt.ylabel("Truth")

plt.title(
    "NEU Surface Defect Confusion Matrix"
)


plt.show()