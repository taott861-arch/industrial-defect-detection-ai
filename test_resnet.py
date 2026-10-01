import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.dataset import DefectDataset
from models.resnet import DefectResNet


# =====================
# 参数
# =====================

data_path = "data/processed_data"

batch_size = 32


# =====================
# 数据处理
# =====================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# =====================
# 数据集
# =====================

dataset = DefectDataset(
    data_path,
    transform=transform
)


# =====================
# 固定测试集
# =====================

train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size


generator = torch.Generator().manual_seed(42)


train_dataset, test_dataset = random_split(
    dataset,
    [train_size,test_size],
    generator=generator
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)



# =====================
# 模型
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = DefectResNet(
    num_classes=6
)


model.load_state_dict(
    torch.load(
        "models/defect_resnet18.pth",
        map_location=device
    )
)


model.to(device)

model.eval()



# =====================
# 测试
# =====================

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()



accuracy = 100 * correct / total


print(
    f"测试集准确率: {accuracy:.2f}%"
)