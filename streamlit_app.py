#ライブラリのインポート
import streamlit as st
from PIL import Image
import numpy as np
import cv2

uploaded_file=st.file_uploader("ファイルアップロード", type='png')
img=Image.open(uploaded_file)
img_array = np.array(img)
st.image(img_array,caption = 'サムネイル画像',use_column_width = True)


image = cv2.imread("icon.png")

#画像の解像度を上げる
scale_percent = 200 # 200%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)


img_array1 = np.array(resized)
st.image(img_array1,caption = 'サムネイル画像',use_column_width = True)

#@title **画像全体オリジナル**
import os
import shutil
from PIL import Image

pic ='images.jpg'#@param {type:"string"} 
input_folder = 'inputs/whole_imgs_original'
reset_folder(input_folder)
im = icon.open('/content/drive/MyDrive/pic/'+pic)
im.save(input_folder+'/'+os.path.splitext(pic)[0]+'.png')

w = 0.6 #@param {type:"slider", min:0.1, max:0.9, step:0.1}


! python inference_codeformer.py --w $w\
                                  --test_path $input_folder\
                                  --bg_upsampler realesrgan\
                                  --face_upsample

clear_output()
result_folder = 'results/whole_imgs_original_'+str(w)+'/final_results'
display_result(input_folder, result_folder )
