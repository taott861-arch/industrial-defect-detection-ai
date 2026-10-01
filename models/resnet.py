import torch.nn as nn
from torchvision import models


class DefectResNet(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()


        # 加载预训练ResNet18
        self.model = models.resnet18(
            weights="DEFAULT"
        )


        # 修改最后分类层
        in_features = self.model.fc.in_features


        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)