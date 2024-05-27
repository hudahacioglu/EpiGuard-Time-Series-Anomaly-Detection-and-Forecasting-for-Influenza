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


st.set_page_config(
    page_title="Forecasting",
    page_icon="🧐"
)


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data
def load_data(file):
        df = pd.read_csv(file)
        return df    

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

file1 = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\No Anomaly\2021_2024.csv"
df1 = load_data(file1)

st.header("Forecasting Using XGBOOST")
dataset = ["Week Start Date","Positive"]
df1= df1[dataset]
df1=df1.set_index("Week Start Date")
df1.index=pd.to_datetime(df1.index, format='%d/%m/%Y')
train = df1.loc[df1.index < "06-26-2023"]
test = df1.loc[df1.index >= "06-26-2023"]

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('06-26-2023', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
st.plotly_chart(fig)

#df1 = create_features(df1)
#train= create_features(train)
#test= create_features(test)
#FEATURES=['Positive','dayofweek', 'quarter', 'month', 'year',
#    'dayofyear', 'dayofmonth', 'weekofyear']
#TARGET="Positive"
#reg=xgb.XGBRegressor(n_estimators=1000,
#                    early_stopping_rounds=50,
#                    learning_rate=0.01
#                    )
#X_train=train[FEATURES]
#y_train=train[TARGET]
#X_test=test[FEATURES]
#y_test=test[TARGET]
#reg.fit( X_train, y_train,
#    eval_set=[(X_train, y_train),(X_test,y_test)],
#    verbose=200)
#test["prediction"]=reg.predict(X_test)
#df = df1.merge(test[["prediction"]],how="left",left_index=True,right_index=True)
#
#fig1, ax1 = plt.subplots(figsize=(15, 5))  # Create a figure and axes
#ax1.plot(df["Positive"], color='b', label='Truth Data', linewidth=2)
#ax1.plot(df["prediction"], color='yellow', label='Predictions', linewidth=2)
## Set labels and title
#ax1.set_xlabel("Date")
#ax1.set_ylabel("Values")
#ax1.set_title("Real Data and Prediction Comparison")
## Add legend
#plt.legend()
#st.plotly_chart(fig1)