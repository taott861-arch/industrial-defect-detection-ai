from pathlib import Path
import shutil


# 项目路径
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 原始数据路径
RAW_DATA_DIR = PROJECT_DIR / "data" / "NEU-CLS"

# 整理后的数据路径
OUTPUT_DIR = PROJECT_DIR / "data" / "processed_data"


# 类别对应关系
CLASS_MAP = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches",
}


def prepare_dataset():

    # 创建目标文件夹
    for class_name in CLASS_MAP.values():
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)


    # 遍历所有图片
    for image_path in RAW_DATA_DIR.iterdir():

        if image_path.suffix.lower() != ".bmp":
            continue

        filename = image_path.name


        # 判断类别
        for prefix, class_name in CLASS_MAP.items():

            if filename.startswith(prefix):

                target = OUTPUT_DIR / class_name / filename

                shutil.copy(
                    image_path,
                    target
                )

                print(
                    f"{filename} -> {class_name}"
                )

                break


if __name__ == "__main__":
    prepare_dataset()
from collections import Counter

labels = []

for i in range(len(dataset)):
    _, label = dataset[i]
    labels.append(label)

print("类别数量统计:")
print(Counter(labels))