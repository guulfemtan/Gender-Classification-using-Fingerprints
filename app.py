import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Fingerprint Gender Classifier")
st.write("Upload a fingerprint image to predict the gender (Male/Female)")

model = load_model("fingerprint_gender_cnn.keras")

uploaded_file = st.file_uploader("Choose a fingerprint image", type=["jpg","png","bmp"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((64,64))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)
    
    prediction = model.predict(img_array)
    gender = "Male" if prediction[0][0] < 0.5 else "Female"
    
    st.image(img, caption=f"Predicted Gender: {gender}", use_column_width=True)
