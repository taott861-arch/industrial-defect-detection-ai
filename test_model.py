import torch

from models.cnn import DefectCNN


# 创建模型
model = DefectCNN()


# 设置测试模式
model.eval()


# 随机生成一张图片Tensor
# 模拟一张200×200 RGB图片
image = torch.randn(1, 3, 200, 200)


# 前向传播
output = model(image)


print("模型输出:")
print(output)


print("输出尺寸:")
print(output.shape)