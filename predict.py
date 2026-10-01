import torch
from torchvision import transforms
from PIL import Image


from models.resnet import DefectResNet


# =====================
# 类别名称
# =====================

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# =====================
# 加载模型
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
# 图片预处理
# =====================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



# =====================
# 预测函数
# =====================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)



    with torch.no_grad():

        output = model(
            image_tensor
        )


        probability = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probability,
            1
        )


    label = classes[
        prediction.item()
    ]


    confidence = confidence.item()


    return label, confidence



# =====================
# 测试
# =====================

if __name__ == "__main__":

    result, confidence = predict(
        "data/processed_data/crazing/Cr_1.bmp"
    )


    print("预测类别:", result)

    print(
        "置信度:",
        confidence
    )