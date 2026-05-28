import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# ========================
# 載入模型
# ========================
model_path = "model.keras"


@st.cache_resource
def load_my_model():
    return load_model(model_path)


model = load_my_model()

# ========================
# 類別名稱
# ========================
class_names = [
    "NCCU Main Library",
    "NCCU Dah Hsian Library",
    "NCCU Social Sciences Library",
    "NCCU Commerce Library",
    "NCCU Law Library",
    "NCCU Research Center and Innovation Incubation Center",
    "NCCU College of Communication Library",
    "NCCU Art Culture Center",
]

# ========================
# Google Maps 對應
# ========================
location_map = {
    "NCCU Main Library": "https://maps.app.goo.gl/NCCUMainLibExample",
    "NCCU Dah Hsian Library": "https://maps.app.goo.gl/NCCUDahHsianExample",
    "NCCU Social Sciences Library": "https://maps.app.goo.gl/NCCUSocSciExample",
    "NCCU Commerce Library": "https://maps.app.goo.gl/NCCUCommerceExample",
    "NCCU Law Library": "https://maps.app.goo.gl/NCCULawExample",
    "NCCU Research Center and Innovation Incubation Center": "https://maps.app.goo.gl/NCCUResearchExample",
    "NCCU College of Communication Library": "https://maps.app.goo.gl/NCCUCommExample",
    "NCCU Art Culture Center": "https://maps.app.goo.gl/NCCUArtExample",
}


# ========================
# 預測函式
# ========================
def predict_image(image):
    # 1. 強制轉換為 RGB（移除手機照片可能帶有的透明通道 Alpha Channel）
    img = image.convert("RGB")

    # 2. 縮放到與訓練時相同的 224x224
    img = img.resize((224, 224))

    # 3. 轉換成 numpy 陣列，並嚴格對齊 Colab 的 /255.0 前處理
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 4. 模型預測
    prediction = model.predict(img_array)
    max_prob = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    return predicted_class, max_prob, prediction


# ========================
# UI 設計
# ========================
st.set_page_config(page_title="NCCU Library Finder", page_icon="📍")

st.title("📍 NCCU Library Finder")
st.caption("Upload a photo to identify the location 📚")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# ========================
# 主流程
# ========================
if uploaded_file:
    image = Image.open(uploaded_file)

    # 1. 自動修正手機拍照的旋轉角度（EXIF 問題）
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)

    # 2. 自動限制最大解析度，防止大容量手機照片讓雲端伺服器記憶體爆掉
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # 顯示預覽圖
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 預測
    predicted_class, confidence, prediction = predict_image(image)

    st.subheader("🔍 Prediction Result")

    # 閾值判斷（建議微調降到 0.60，避免現場拍照角度不同時被直接拒絕）
    if confidence < 0.60:
        st.error(
            "Unable to identify this image confidently. Please upload another photo."
        )
    else:
        st.success(f"**{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        map_url = location_map.get(
            predicted_class, "https://www.google.com/maps"
        )
        st.markdown(f"[👉 Click here to navigate to {predicted_class}]({map_url})")

    # 展開查看所有機率（這在期末報告展示時是完美的加分項！）
    with st.expander("📊 Show all probabilities"):
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: **{prediction[0][i] * 100:.2f}%**")
