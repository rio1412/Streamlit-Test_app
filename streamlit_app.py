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
