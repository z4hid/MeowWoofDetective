import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# Define class names
class_names = ['cats', 'dogs']

# Load the trained model and compile it
model = keras.models.load_model("models/cnn.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values
    return image

# Set page config
st.set_page_config(
    page_title="Meow Woof Detective",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS style
st.markdown(
    """
    <style>
    .stApp {
        color: white;
        background-color: #15202B;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput input {
        color: white;
        background-color: #1F2937;
        border: 1px solid #3E4C59;
    }
    .stButton button {
        color: white;
        background-color: #1F2937;
        border: 1px solid #3E4C59;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #374151;
    }
    .stImage {
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
    footer {
        font-size: 12px;
        color: #999999;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define header
st.title("Meow Woof Detective")
st.subheader("Upload an image and let the model classify it as Meow (cat) or Woof (dog).")

# Define image upload and prediction logic
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    if prediction > 0.5:
        st.write("Prediction: Woof (dog)")
    else:
        st.write("Prediction: Meow (cat)")

    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

# Add footer
st.markdown(
    """
    <footer>
    Styled with ❤️ by Zahid Hasan Shuvo
    </footer>
    """,
    unsafe_allow_html=True
)
