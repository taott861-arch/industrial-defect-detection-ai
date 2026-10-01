import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import DefectDataset
from models.cnn import DefectCNN


# =====================
# 1. 参数
# =====================

batch_size = 32


# =====================
# 2. 数据
# =====================

transform = transforms.Compose([
    transforms.ToTensor()
])


dataset = DefectDataset(
    "data/processed_data",
    transform=transform
)


# 测试集
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size


_, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# =====================
# 3. 加载模型
# =====================

model = DefectCNN()


model.load_state_dict(
    torch.load(
        "models/defect_cnn.pth"
    )
)


model.eval()


# =====================
# 4. 测试
# =====================

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()



accuracy = correct / total


print("================")
print("测试完成")
print(f"测试图片数量: {total}")
print(f"正确数量: {correct}")
print(f"准确率: {accuracy*100:.2f}%")
print("================")