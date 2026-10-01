import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.dataset import DefectDataset
from models.resnet import DefectResNet


# ==========================
# 固定随机种子
# ==========================

torch.manual_seed(42)


# ==========================
# 参数
# ==========================

data_path = "data/processed_data"

batch_size = 32
epochs = 10
learning_rate = 0.001


# ==========================
# 数据预处理
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ==========================
# 数据集
# ==========================

dataset = DefectDataset(
    data_path,
    transform=transform
)


print("总数据量:", len(dataset))


# ==========================
# 固定训练测试划分
# ==========================

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size


generator = torch.Generator().manual_seed(42)


train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)


print("训练集:", len(train_dataset))
print("测试集:", len(test_dataset))


# ==========================
# DataLoader
# ==========================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)



# ==========================
# 模型
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = DefectResNet(
    num_classes=6
)


model = model.to(device)


# ==========================
# 损失函数和优化器
# ==========================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)



# ==========================
# 训练
# ==========================

print("开始训练ResNet18...")


best_loss = float("inf")


for epoch in range(epochs):

    model.train()

    total_loss = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        # 前向传播
        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        # 清零梯度
        optimizer.zero_grad()


        # 反向传播
        loss.backward()


        # 更新参数
        optimizer.step()


        total_loss += loss.item()



    avg_loss = total_loss / len(train_loader)


    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}"
    )


    # 保存最佳模型

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "models/defect_resnet18.pth"
        )

        print("保存最佳模型")



print("训练完成")