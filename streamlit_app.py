import streamlit as st
uploaded_file = st.file_uploader("ファイルをアップロードしてください。")

import pandas as pd
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(
        image, caption='upload images',
        use_column_width=True
    )
