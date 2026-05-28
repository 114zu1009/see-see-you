!pip install bing-image-downloader


from google.colab import drive
drive.mount('/content/drive')


dataset_path = "/content/drive/MyDrive/AI導論 final pics."


# Install Streamlit if not already installed
try:
    import streamlit
except ImportError:
    !pip install streamlit

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# ========================
# 載入模型
# ========================
# Assuming model.keras is in the dataset_path directory
model_path = os.path.join(dataset_path, "model.keras")
model = load_model(model_path)

# ========================
# 類別名稱（請確認順序跟訓練一致）
# ========================
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

# ========================
# Google Maps 對應
# ========================
location_map = {
    "NCCU Main Library": "https://www.google.com/maps/search/?api=1&query=NCCU+Main+Library",
    "NCCU Dah Hsian Library": "https://www.google.com/maps/search/?api=1&query=NCCU+Dah+Hsian+Library",
    "NCCU Social Sciences Library": "https://www.google.com/maps/search/?api=1&query=NCCU+Social+Sciences+Library",
    "NCCU Commerce Library": "https://www.google.com/maps/search/?api=1&query=NCCU+Commerce+Library",
    "NCCU Law Library": "https://www.google.com/maps/search/?api=1&query=NCCU+Law+Library",
    "NCCU Research Center and Innovation Incubation Center": "https://www.google.com/maps/search/?api=1&query=NCCU+Research+Center+and+Innovation+Incubation+Center",
    "NCCU College of Communication Library": "https://www.google.com/maps/search/?api=1&query=NCCU+College+of+Communication+Library",
    "NCCU Art Culture Center": "https://www.google.com/maps/search/?api=1&query=NCCU+Art+Culture+Center"
}

# ========================
# 預測函式（核心）
# ========================
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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

    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_class, confidence, prediction = predict_image(image)

    st.subheader("🔍 Prediction Result")

    # 閾值判斷
    if confidence < 0.75:
        st.error("Unable to identify this image. Please upload another photo.")
    else:
        st.success(f"{predicted_class}")
        st.write(f"Confidence: {confidence:.4f}")

        # 地圖
        map_url = location_map[predicted_class]
        st.markdown(f"[👉 Navigate to {predicted_class}]({map_url})")

    # 所有機率（加分）
    with st.expander("📊 Show all probabilities"):
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[0][i]:.4f}")



model.save('/content/drive/MyDrive/AI導論 final pics./model.keras')
