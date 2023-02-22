import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model

st.title('High-Quality Image Super-Resolution')

import streamlit as st
import tensorflow as tf
from PIL import Image

# モデルを読み込む
model = tf.keras.models.load_model('model.h5')

# 画像を高画質化する関数を定義する
def enhance_image(image):
    # 画像を読み込む
    img = Image.open(image).convert('RGB')
    # 画像をリサイズする
    img = img.resize((256, 256))
    # 画像をTensorに変換する
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.expand_dims(x, axis=0)
    # 画像を高画質化する
    enhanced = model.predict(x)
    # 画像をPIL Imageに変換する
    enhanced = tf.squeeze(enhanced, axis=0)
    enhanced = tf.keras.preprocessing.image.array_to_img(enhanced)
    # 画像を保存する
    enhanced.save('enhanced_image.jpg')
    return enhanced

# Streamlitのインターフェースを作成する
st.title('画像を高画質化するアプリ')
uploaded_file = st.file_uploader('画像をアップロードしてください', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # アップロードされた画像を表示する
    st.image(uploaded_file, caption='アップロードされた画像', use_column_width=True)
    # 画像を高画質化するボタンを作成する
    if st.button('画像を高画質化する'):
        # プログレスバーを表示する
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        # 画像を高画質化する
        enhanced_image = enhance_image(uploaded_file)
        # 画像を表示する
        st.image(enhanced_image, caption='高画質化された画像', use_column_width=True)
