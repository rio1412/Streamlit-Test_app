import streamlit as st


uploaded_file = st.file_uploader("アクセスログをアップロードしてください。")
if uploaded_file is not None:

    df_tmp = pd.read_csv(
        copy.deepcopy(uploaded_file),
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
        engine='python',
        na_values='-',
        header=None)
    
    st.markdown('### アクセスログ（先頭5件）')
    st.write(df_tmp.head(5))
