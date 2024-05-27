import streamlit as st
import pandas as pd
import streamlit_pandas as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import io
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(
    page_title="Home",
    page_icon="🏚️"
)

#file = open(r"C:\Users\hacio\OneDrive\Desktop\Project\Teknofest Kopya.png", "rb")
#contents = file.read()
#img_str = base64.b64encode(contents).decode("utf-8")
#buffer = io.BytesIO()
#file.close()
#img_data = base64.b64decode(img_str)
#img = Image.open(io.BytesIO(img_data))
#resized_img = img.resize((200  , 80))  # x, y
#resized_img.save(buffer, format="PNG")
#img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
#
#st.markdown(
#        f"""
#        <style>
#            [data-testid="stSidebarNav"] {{
#                background-image: url('data:image/png;base64,{img_b64}');
#                background-repeat: no-repeat;  /* Adjust as needed */
#                background-position:center top;  /* Adjust position */
#                background-size: contain;  /* Adjust scaling */
#                padding-top: 20px;
#                height:150px;
#            }}
#        </style>
#        """,
#        unsafe_allow_html=True,
#    )

st.image(r"\Project\Logo.png",width=400)

st.title("Türkiye Influenza Cases")

option = st.selectbox(
    "Please Select Dataset range you would like to anlayze",
    ("Before Pandemic : 2015-2020", "After Pandemic : 2021-2024"))

if option=="Before Pandemic : 2015-2020":
    tab_titles = ["Graph","Dataframe","Histogram","Monthly Analyses","Quarterly Analysis"]
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    file = r"\Project\Needed\No Anomaly\2015_2020.csv"
    df = load_data(file)

    tabs=st.tabs(tab_titles)

    with tabs[0]:
        st.header("Türkiye Influenza Cases During 2015-2020")
        fig = go.Figure([go.Scatter(x=df['Week Start Date'], y=df['Positive'])])
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Influenza cases")
        st.plotly_chart(fig)

    with tabs[1]:
        st.header("2015-2020 DataFrame")
        #st.write(df)
        st.dataframe(df, use_container_width=True)

    with tabs[2]:
        st.header("2015-2020 Histogram")
        fig, ax = plt.subplots()
        ax.hist(df["Positive"], bins=20)
        st.pyplot(fig)
        
        skew_value=df["Positive"].skew()
        st.markdown(f"As seen in the histogram above skew value calculated as **{skew_value}**. That shows that our dataset is right skewed.")

    with tabs[3]:
        dataset = ["Week Start Date","Positive"]
        df= df[dataset]
        df=df.set_index("Week Start Date")
        df.index=pd.to_datetime(df.index, format='%d/%m/%Y')

        def create_features(df):
            """
            Create time series features based on time series index.
            """
            df = df.copy()
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['dayofyear'] = df.index.dayofyear
            df['dayofmonth'] = df.index.day
            df['weekofyear'] = df.index.isocalendar().week
            return df

        df = create_features(df)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x='month', y='Positive')
        ax.set_title('Monthly Distribution of Influenza Cases Over the Year')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='2015-2020 Monthly Distribution of Influenza Cases Over the Year')

    with tabs[4]:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x='quarter', y='Positive')
        ax.set_title('Quarterly Distribution of Influenza Cases Over the Year')

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='2015-2020 Quarterly Distribution of Influenza Cases Over the Year')

else :
    tab_titles = ["Graph","Dataframe","Histogram","Monthly Analyses","Quarterly Analysis"]
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    file1 = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\No Anomaly\2021_2024.csv"
    df1 = load_data(file1)

    tabs=st.tabs(tab_titles)

    with tabs[0]:
        st.header("Türkiye Influenza Cases During 2021-2024")
        fig = go.Figure([go.Scatter(x=df1['Week Start Date'], y=df1['Positive'])])
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Influenza cases")
        st.plotly_chart(fig)

    with tabs[1]:
        st.header("2021-2024 DataFrame")
        #st.write(df)
        st.dataframe(df1, use_container_width=True)

    with tabs[2]:
        st.header("2021-2024 Histogram")
        fig, ax = plt.subplots()
        ax.hist(df1["Positive"], bins=20)
        st.pyplot(fig)
        
        skew_value=df1["Positive"].skew()
        st.markdown(f"As seen in the histogram above skew value calculated as **{skew_value}**. That shows that our dataset is right skewed.")

    with tabs[3]:
        dataset = ["Week Start Date","Positive"]
        df1= df1[dataset]
        df1=df1.set_index("Week Start Date")
        df1.index=pd.to_datetime(df1.index, format='%d/%m/%Y')

        def create_features(df):
            """
            Create time series features based on time series index.
            """
            df = df.copy()
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['dayofyear'] = df.index.dayofyear
            df['dayofmonth'] = df.index.day
            df['weekofyear'] = df.index.isocalendar().week
            return df

        df1 = create_features(df1)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df1, x='month', y='Positive')
        ax.set_title('Monthly Distribution of Influenza Cases Over the Year')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='2021-2024 Monthly Distribution of Influenza Cases Over the Year')
    
    with tabs[4]:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df1, x='quarter', y='Positive')
        ax.set_title('Quarterly Distribution of Influenza Cases Over the Year')

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='2021-2024 Quarterly Distribution of Influenza Cases Over the Year')




