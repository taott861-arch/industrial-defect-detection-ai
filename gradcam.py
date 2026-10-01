import torch
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from torchvision import transforms


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet import DefectResNet



# =====================
# 参数
# =====================

image_path = (
    "data/processed_data/"
    "crazing/"
    "Cr_1.bmp"
)


model_path = (
    "models/"
    "defect_resnet18.pth"
)



classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]



# =====================
# 图像处理
# =====================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



img = Image.open(
    image_path
).convert("RGB")


input_tensor = transform(img)
input_tensor = input_tensor.unsqueeze(0)



# =====================
# 模型
# =====================

model = DefectResNet(
    num_classes=6
)


model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)


model.eval()



# =====================
# 预测
# =====================

output = model(
    input_tensor
)


prediction = torch.argmax(
    output,
    dim=1
)


print(
    "预测类别:",
    classes[prediction.item()]
)



# =====================
# GradCAM
# =====================


target_layers = [
    model.model.layer4[-1]
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)


grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=None
)[0]



# =====================
# 生成热力图
# =====================


rgb_img = np.array(
    img.resize((224,224))
) / 255.0


visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)



plt.figure(figsize=(8,4))


plt.subplot(1,2,1)

plt.imshow(rgb_img)

plt.title("Original")

plt.axis("off")



plt.subplot(1,2,2)

plt.imshow(visualization)

plt.title(
    "Grad-CAM"
)

plt.axis("off")



plt.show()