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

st.set_page_config(page_title="MOSSALA", page_icon='1.png')
st.title("Multiple OSS Access Log Analyzer")
st.image('1.png')

'''
**以下のOSSのような一般的な出力形式のアクセスログを解析できます。**
 
* Apache
* Nginx
* Tomcat
* WildFly
'''

ALL_METHODS = ['GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'CONNECT', 'OPTIONS', 'TRACE', 'LINK', 'UNLINK']

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

    help_txt = '''
        以下のフォーマット文字列を解析可能です。
        | 列名 | フォーマット文字列 | 説明 | 
        |:-----|:-----:|:-----|
        | Remote Host | `%h` | リモートホスト |
        | Time | `%t` | リクエストを受付けた時刻 | 
        | Request | `\"%r\"` | リクエストの最初の行 | 
        | Status | `%>s` | ステータス | 
        | Size | `%b` | レスポンスのバイト数 | 
        | User Agent | `\"%{User-agent}i\"` | リクエストのUser-agentヘッダの内容 | 
        | Response Time (ms) | `%D` | リクエストを処理するのにかかった時間（ミリ秒） |         
        | Response Time (s) | `%T` | リクエストを処理するのにかかった時間（秒） |         
        | Method | `%m` | リクエストメソッド | 
        | URL | `%U` | リクエストされたURLパス | 
        | Version | `%H` | リクエストプロトコル | 
        
        詳細については、各OSSの公式ドキュメントを参照して下さい。Apacheの公式ドキュメントを参照する場合は、[ここ](https://httpd.apache.org/docs/2.4/ja/mod/mod_log_config.html)をクリックして下さい。
        '''
