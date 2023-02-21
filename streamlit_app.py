#ライブラリのインポート
import cv2
import streamlit as st
from PIL import Image
import numpy as np

uploaded_file=st.file_uploader("ファイルアップロード", type='png')
img=Image.open(uploaded_file)
