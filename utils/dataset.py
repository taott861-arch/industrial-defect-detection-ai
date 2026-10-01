from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset


class DefectDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = Path(data_dir)

        self.transform = transform


        # 类别名称
        self.classes = [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches"
        ]


        # 保存所有图片路径
        self.images = []


        # 保存对应标签
        self.labels = []


        # 遍历类别
        for label, class_name in enumerate(self.classes):

            class_dir = self.data_dir / class_name


            for img_path in class_dir.iterdir():

                if img_path.suffix.lower() in [
                    ".bmp",
                    ".jpg",
                    ".png"
                ]:

                    self.images.append(img_path)

                    self.labels.append(label)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, index):

        img_path = self.images[index]

        label = self.labels[index]


        # 读取图片
        image = Image.open(img_path).convert("RGB")


        # 图像预处理
        if self.transform:

            image = self.transform(image)


        return image, label