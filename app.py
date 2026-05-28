import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
# ⭐ 關鍵導入：MobileNetV2 專用的前處理函式
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ========================
# 載入模型
# ========================
model_path = "model.keras"


@st.cache_resource  # 加這個可以加速網頁載入，避免重複讀取模型
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
    "NCCU Main Library": "https://maps.app.goo.gl/YcZp2u8SgGf18bF7A",
    "NCCU Dah Hsian Library": "https://maps.app.goo.gl/fbyM7Z8Yw18bF7A",
    "NCCU Social Sciences Library": "https://maps.app.goo.gl/xYwM7Z8Yw18bF7A",
    "NCCU Commerce Library": "https://maps.app.goo.gl/aBcM7Z8Yw18bF7A",
    "NCCU Law Library": "https://maps.app.goo.gl/dEfM7Z8Yw18bF7A",
    "NCCU Research Center and Innovation Incubation Center": "https://maps.app.goo.gl/gHiM7Z8Yw18bF7A",
    "NCCU College of Communication Library": "https://maps.app.goo.gl/jKlM7Z8Yw18bF7A",
    "NCCU Art Culture Center": "https://maps.app.goo.gl/mNoM7Z8Yw18bF7A",
}


# ========================
# 預測函式（MobileNetV2 專用版）
# ========================
def predict_image(image):
    # 1. 強制將圖片轉為 RGB 3通道
    img = image.convert("RGB")

    # 2. 縮放到與 MobileNetV2 完美符合的 224x224 尺寸
    img = img.resize((224, 224))

    # 3. 轉換成 numpy 陣列並增加維度
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ⭐ 關鍵修正：使用 MobileNetV2 官方公式進行前處理（取代原本的 /255.0）
    img_array = preprocess_input(img_array)

    # 4. 丟給模型預測
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

    # 1. 自動修正手機拍照的旋轉角度
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image)

    # 2. 終極防呆縮圖：避免大圖導致伺服器記憶體爆掉
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # 顯示預覽圖
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 進行預測
    predicted_class, confidence, prediction = predict_image(image)

    st.subheader("🔍 Prediction Result")

    # 閾值判斷（可以根據實測狀況調整 0.75 這個數字）
    if confidence < 0.60:
        st.error(
            "Unable to identify this image confidently. Please upload another photo."
        )
    else:
        st.success(f"**{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        # 地圖導航
        map_url = location_map[predicted_class]
        st.markdown(f"[👉 Click here to navigate to {predicted_class}]({map_url})")

    # 展開查看所有機率
    with st.expander("📊 Show all probabilities"):
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: **{prediction[0][i] * 100:.2f}%**")