import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

hub_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
sr_model = hub.load(hub_url)

def upscale_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    output = sr_model(img)[0]
    output = tf.squeeze(output)
    output = tf.clip_by_value(output, 0, 1)
    output = tf.image.convert_image_dtype(output, dtype=tf.uint8)
    output = np.array(output)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

# Streamlitのインターフェースを作成する
st.title('画像を高画質化するアプリ')
uploaded_file = st.file_uploader('画像をアップロードしてください', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # アップロードされた画像を表示する
    input_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(input_image, caption='アップロードされた画像', use_column_width=True)
    # 画像を高画質化するボタンを作成する
    if st.button('画像を高画質化する'):
        with st.spinner('画像を高画質化する...'):
            # Upscale the image
            output_image = upscale_image(uploaded_file)
        # Show the upscaled image
        st.image(output_image, caption='高画質化された画像', use_column_width=True)
