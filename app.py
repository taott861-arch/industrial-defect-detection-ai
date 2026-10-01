import streamlit as st
from PIL import Image
import tempfile


from gradcam_utils import generate_gradcam


# =====================
# 页面设置
# =====================

st.set_page_config(
    page_title="工业缺陷识别系统",
    layout="centered"
)



st.title(
    "🔍 钢板表面缺陷识别系统"
)


st.write(
    "基于 ResNet18 深度学习模型"
)



# =====================
# 上传图片
# =====================

uploaded_file = st.file_uploader(
    "上传缺陷图片",
    type=[
        "bmp",
        "jpg",
        "png"
    ]
)



if uploaded_file is not None:


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="上传图片",
        width=300
    )



    if st.button("开始检测"):


        # 临时保存图片

        with tempfile.NamedTemporaryFile(
            suffix=".bmp",
            delete=False
        ) as tmp:


            image.save(
                tmp.name
            )


            image_path = tmp.name



        label, confidence, cam_image = generate_gradcam(
            image_path
        )


        st.success(
            f"预测结果: {label}"
        )


        st.info(
            f"置信度: {confidence:.2%}"
        )
        
        st.image(
            cam_image,
            caption="Grad-CAM模型关注区域",
            width=500
)