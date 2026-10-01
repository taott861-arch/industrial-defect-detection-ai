import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.dataset import DefectDataset
from models.cnn import DefectCNN


# =========================
# 1. 参数设置
# =========================

batch_size = 32
epochs = 10
learning_rate = 0.001


# =========================
# 2. 数据处理
# =========================

transform = transforms.Compose([
    transforms.ToTensor()
])


dataset = DefectDataset(
    "data/processed_data",
    transform=transform
)


print("总数据量:", len(dataset))


# 划分训练集和测试集

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)


print("训练集:", len(train_dataset))
print("测试集:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# =========================
# 3. 创建模型
# =========================

model = DefectCNN()


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# =========================
# 4. 训练
# =========================

print("开始训练...")


for epoch in range(epochs):

    model.train()

    total_loss = 0


    for images, labels in train_loader:


        # 前向传播
        outputs = model(images)


        # 计算loss
        loss = criterion(
            outputs,
            labels
        )


        # 梯度清零
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



# =========================
# 5. 保存模型
# =========================

print("训练结束，开始保存模型...")


torch.save(
    model.state_dict(),
    "models/defect_cnn.pth"
)


print("模型保存成功！")
print("保存位置: models/defect_cnn.pth")