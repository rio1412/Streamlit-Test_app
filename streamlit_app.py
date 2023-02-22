import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model

st.title('High-Quality Image Super-Resolution')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    
    # ここに高画質化処理のコードを記述します

def build_srgan():
    """Builds the SRGAN model."""
    srgan = load_model('Streamlit-Test_app/blob/main/srgan.h5.py')
    return srgan

def upscale_image(image):
    """Upscales the image using the SRGAN model."""
    srgan = build_srgan()
    lr_shape = (image.shape[1], image.shape[0])
    sr_shape = (lr_shape[0] * 4, lr_shape[1] * 4)
    lr_image = cv2.resize(image, lr_shape, interpolation=cv2.INTER_CUBIC)
    lr_image = np.expand_dims(lr_image, axis=0)
    sr_image = srgan.predict(lr_image)
    sr_image = np.squeeze(sr_image, axis=0)
    sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)
    sr_image = cv2.resize(sr_image, sr_shape, interpolation=cv2.INTER_CUBIC)
    return sr_image

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Original Image', use_column_width=True)
    
    high_res_image = upscale_image(image)
    st.image(high_res_image, caption='High-Quality Image', use_column_width=True)
