#ライブラリのインポート
import cv2
import streamlit as st
from PIL import Image
import numpy as np

uploaded_file=st.file_uploader("ファイルアップロード", type='png')
image=Image.open(uploaded_file)

#画像の読み込み
img = cv2.imread(image)

#画像の解像度を上げる
scale_percent = 200 # 200%
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)

img_array = np.array(image)
st.image(img_array,caption = 'サムネイル画像',use_column_width = True) 
