import torch
import torch.nn as nn


class DefectCNN(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()


        self.features = nn.Sequential(

            # 第一层卷积
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 第二层卷积
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 第三层卷积
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 25 * 25,
                128
            ),

            nn.ReLU(),


            nn.Linear(
                128,
                num_classes
            )

        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x