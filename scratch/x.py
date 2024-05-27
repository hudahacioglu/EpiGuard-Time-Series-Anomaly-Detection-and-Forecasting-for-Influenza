import streamlit as st
import pandas as pd
import streamlit_pandas as sp
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight',)
import io
from PIL import Image
import base64
from io import BytesIO
from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')
def load_data(file):
        df = pd.read_csv(file)
        return df


st.header("Forecasting Using Prophet")

file1 = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\No Anomaly\2015_2020.csv"
df1 = load_data(file1)

dataset = ["Week Start Date","Positive"]
df1= df1[dataset]
df1["Week Start Date"]=pd.to_datetime(df1["Week Start Date"],format='%d/%m/%Y')
df1=df1.set_index("Week Start Date")
df1.index=pd.to_datetime(df1.index)
split_date = '04-02-2019'
df_train =df1.loc[df1.index <= split_date].copy()
df_test = df1.loc[df1.index > split_date].copy()


df_test \
    .rename(columns={'Positive': 'TEST SET'}) \
    .join(df_train.rename(columns={'Positive': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(10, 5), title='Influenza', style='-', ms=8)

# Format data for prophet model using ds and y
df_train_prophet = df_train.reset_index() \
    .rename(columns={'Week Start Date':'ds',
                     'Positive':'y'})

model = Prophet()
model.fit(df_train_prophet)

# Predict on test set with model
df_test_prophet = df_test.reset_index() \
    .rename(columns={'Week Start Date':'ds',
                     'Positive':'y'})

df_test_fcst = model.predict(df_test_prophet)


fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(df_test_fcst, ax=ax)
ax.set_title('Prophet Forecast')
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.image(buf, caption='Forecast')


fig = model.plot_components(df_test_fcst)
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.image(buf, caption='Components')


# Plot the forecast with the actuals
fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(df_test.index, df_test['Positive'], color='r')
fig = model.plot(df_test_fcst, ax=ax)
ax.set_title('Prophet Results Comparison')
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.image(buf, caption='2021-2024 Quarterly Distribution of Influenza Cases Over the Year')

import plotly.express as px

future = model.make_future_dataframe(periods=365,freq="W")
forecast = model.predict(future)
plot_plotly(model ,forecast)
st.plotly_chart(plot_plotly(model ,forecast))


st.plotly_chart(plot_components_plotly(model, forecast))








#dataset = ["Week Start Date","Positive"]
#df1= df1[dataset]
#df1["Week Start Date"]=pd.to_datetime(df1["Week Start Date"])
#df1=df1.set_index("Week Start Date")
#df1.index=pd.to_datetime(df1.index, format='%d/%m/%Y')
### Plot train and test so you can see where we have split
#split_date = '04-02-2019'
#df_train =df1.loc[df1.index <= split_date].copy()
#df_test = df1.loc[df1.index > split_date].copy()
#df_test \
#.rename(columns={'Positive': 'TEST SET'}) \
#.join(df_train.rename(columns={'Positive': 'TRAINING SET'}),
#      how='outer')
#df_train_prophet = df_train.reset_index() \
#.rename(columns={'Week Start Date':'ds',
#                 'Positive':'y'})
#model = Prophet()
#model.fit(df_train_prophet)
#df_test_prophet = df_test.reset_index() \
#.rename(columns={'Week Start Date':'ds',
#                 'Positive':'y'})
#df_test_fcst = model.predict(df_test_prophet)
#fig, ax = plt.subplots(figsize=(10, 5))
#fig = model.plot(df_test_fcst, ax=ax)
#ax.set_title('Prophet Forecast')
#st.plotly_chart(fig)













































