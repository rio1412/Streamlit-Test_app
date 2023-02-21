#ライブラリのインポート
import streamlit as st
from PIL import Image
import numpy as np
import cv2

uploaded_file=st.file_uploader("ファイルアップロード", type='png')
img=Image.open(uploaded_file)
img_array = np.array(img)
st.image(img_array,caption = 'サムネイル画像',use_column_width = True)


img = cv2.imread(uploaded_file)

#画像の解像度を上げる
scale_percent = 200 # 200%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)


img_array = np.array(resized)
st.image(img_array,caption = 'サムネイル画像',use_column_width = True)
