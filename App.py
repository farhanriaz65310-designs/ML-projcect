import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1️⃣ Page Configuration
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to identify the tumor type.")

# 2️⃣ Load the Model
@st.cache_resource
def load_my_model():
    try:
        # ✅ Model should be in same folder as app.py
        model_path = "my_model.h5"

        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.info("Please make sure 'my_model.h5' is in the same folder as app.py")
            return None

        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(e)  # See exact error in terminal/console
        return None

model = load_my_model()

# 3️⃣ Class Labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 4️⃣ File Uploader
uploaded_file = st.file_uploader(
    "Choose an MRI Image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if model is None:
        st.error("⚠️ Model not loaded.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        with st.spinner("🔄 Analyzing MRI scan..."):
            # Resize image
            img = image.resize((299, 299))
            img_array = np.array(img)

            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,) * 3, axis=-1)

            # Handle RGBA images
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

            # Normalize
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array, verbose=0)
            score = predictions[0]

            # Result
            predicted_class = class_names[np.argmax(score)]
            confidence = np.max(score) * 100

            st.success(
                f"**Result: {predicted_class.upper().replace('NOTUMOR','NO TUMOR')}**"
            )
            st.info(f"**Confidence: {confidence:.2f}%**")

            # Probability breakdown chart
            st.write("### 📊 Probability Breakdown")
            chart_data = {
                class_names[i].replace('notumor', 'No Tumor').title(): float(score[i])
                for i in range(len(class_names))
            }
            st.bar_chart(chart_data)
