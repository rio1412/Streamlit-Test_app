import streamlit as st
uploaded_file = st.file_uploader("アクセスログをアップロードしてください。")

import pandas as pd
if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file,
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
        engine='python',
        na_values='-',
        header=None)
    
    st.markdown('### アクセスログ（先頭5件）')
    st.write(df.head(5))

st.set_page_config(page_title="メインページ", page_icon='1.png')
st.title("Multiple OSS Access Log Analyzer")
