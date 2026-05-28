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

# 📋 這裡先放你原本的順序（等等我們要根據實測來大洗牌！）
class_names = [
    "NCCU Main Library",
    "NCCU Dah Hsian Library",
    "NCCU Social Sciences Library",
    "NCCU Commerce Library",
    "NCCU Law Library",
    "NCCU Research Center and Innovation Incubation Center",
    "NCCU College of Communication Library",
    "NCCU Art Culture Center"
]

location_map = {
    "NCCU Main Library": "https://maps.app.goo.gl/NCCUMainLibExample",
    "NCCU Dah Hsian Library": "https://maps.app.goo.gl/NCCUDahHsianExample",
    "NCCU Social Sciences Library": "https://maps.app.goo.gl/NCCUSocSciExample",
    "NCCU Commerce Library": "https://maps.app.goo.gl/NCCUCommerceExample",
    "NCCU Law Library": "https://maps.app.goo.gl/NCCULawExample",
    "NCCU Research Center and Innovation Incubation Center": "https://maps.app.goo.gl/NCCUResearchExample",
    "NCCU College of Communication Library": "https://maps.app.goo.gl/NCCUCommExample",
    "NCCU Art Culture Center": "https://maps.app.goo.gl/NCCUArtExample"
}

def predict_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    max_prob = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    
    return predicted_class, max_prob, prediction, predicted_index

# ========================
# UI 設計
# ========================
st.set_page_config(page_title="NCCU Library Finder", page_icon="📍")
st.title("📍 NCCU Library Finder (Debug Mode)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image)
    
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    predicted_class, confidence, prediction, predicted_index = predict_image(image)
    
    # 🔍 核心除錯面板：直接看模型吐出什麼數字
    st.warning(f"🚨 【除錯訊息】模型對這張照片預測出的數字索引是： **【 {predicted_index} 】**")
    
    st.subheader("🔍 Prediction Result")
    st.success(f"目前對應的名字：{predicted_class} (信心度: {confidence*100:.2f}%)")
    
    with st.expander("📊 查看所有數字的機率分配"):
        for i in range(len(class_names)):
            st.write(f"索引【 {i} 】 ({class_names[i]}): **{prediction[0][i]*100:.2f}%**")
