import copy
import pytz
import woothee
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import *
from pandas import DataFrame
from datetime import datetime
import re
import json
import pycountry
from urllib.request import urlopen
import ipaddress
from ipaddress import ip_address

st.set_page_config(page_title="main", page_icon='1.png')
st.title("Multiple OSS Access Log Analyzer")


uploaded_file = st.file_uploader("アクセスログをアップロードしてください。")
if uploaded_file is not None:

    df_tmp = pd.read_csv(
        copy.deepcopy(uploaded_file),
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
        engine='python',
        na_values='-',
        header=None)

    default_usecols = []
    default_names = []

    define_usecols(df_tmp, default_usecols, default_names)

    st.markdown('### アクセスログ（先頭5件）')
    st.write(df_tmp.head(5))
