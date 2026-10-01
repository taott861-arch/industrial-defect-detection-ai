import torch
import numpy as np

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from models.resnet import DefectResNet



classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]



transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



def generate_gradcam(image_path):


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    # 加载模型

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



    # 图片

    img = Image.open(
        image_path
    ).convert("RGB")



    input_tensor = transform(img)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)



    # 预测类别

    output = model(
        input_tensor
    )


    prediction = torch.argmax(
        output,
        dim=1
    )



    label = classes[
        prediction.item()
    ]



    confidence = torch.softmax(
        output,
        dim=1
    )[0][prediction.item()].item()



    # Grad-CAM

    target_layers = [
        model.model.layer4[-1]
    ]


    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )


    targets = [
        ClassifierOutputTarget(
            prediction.item()
        )
    ]



    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]



    rgb_img = np.array(
        img.resize((224,224))
    ) / 255.0



    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )


    return (
        label,
        confidence,
        visualization
    )