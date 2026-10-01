from utils.dataset import DefectDataset
from torchvision import transforms


# 图片转换方法
transform = transforms.Compose([
    transforms.ToTensor()
])


# 创建数据集
dataset = DefectDataset(
    "data/processed_data",
    transform=transform
)


# 查看数据数量
print("数据数量:")
print(len(dataset))


# 读取第一张图片
image, label = dataset[0]


# 查看转换后的Tensor
print("图片Tensor:")
print(image.shape)


# 查看标签
print("标签:")
print(label)
from collections import Counter

labels = []

for i in range(len(dataset)):
    _, label = dataset[i]
    labels.append(label)

print("类别数量统计:")
print(Counter(labels))