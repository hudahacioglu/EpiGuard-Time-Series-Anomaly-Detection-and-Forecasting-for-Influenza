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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')


st.set_page_config(
    page_title="Early Warning",
    page_icon="❗"
)


st.image("Logo.png",width=400)


@st.cache_data
def load_data(file):
        df = pd.read_csv(file)
        return df

st.title("Early Warning System")

st.header("Türkiye 2015-2026 Influenza Outlier Forecasting and Detection")

#st.markdown("Pay Attention* there is a possible Influneza outbreak expected on 12 December 2024. Precautions are suggested to be taken.")

st.markdown("""
    <div style="font-size:24px; font-weight:bold; color:red;">
        Pay Attention!
    </div>
    There is a possible Influenza outbreak expected on 12 December 2024. Precautions are suggested to be taken.
    """, unsafe_allow_html=True)


file = "2015_2020.csv"
df = load_data(file)
dataset = ["Week Start Date","Positive"]
df1= df[dataset]

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


future = model.make_future_dataframe(periods=365,freq="W")
forecast = model.predict(future)


# Calculate mean and standard deviation
mean = np.mean(forecast["yhat"])
std = np.std(forecast["yhat"])

# Calculate the z-score (assuming no missing values)
z_score = mean + std * 1.3

forecast["Z_Score Anomaly"] = forecast["yhat"] > z_score
df_true = forecast[forecast["Z_Score Anomaly"] == True]

fig = go.Figure([go.Scatter(x=forecast['ds'], y=forecast['yhat'],name="Influenza cases")])
fig.update_layout(title='Türkiye 2015-2026 Influenza Cases')
fig.update_layout(title=dict(x=0.45, y=0.9))
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Influenza cases")


fig.add_trace(go.Scatter(mode='markers',
                             x=df_true["ds"],
                             y=df_true["yhat"],
                             marker=dict(color='red', size=10),
                             name='Predicted Anomaly'))

fig.update_layout(autosize=True, width=1000, height=600,showlegend=True)
st.plotly_chart(fig)
