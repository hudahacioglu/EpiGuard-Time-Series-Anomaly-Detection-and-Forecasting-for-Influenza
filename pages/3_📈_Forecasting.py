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
    page_title="Forecasting",
    page_icon="📈"
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
#                background-position:0px -20px;  /* Adjust position */
#                background-size: contain;  /* Adjust scaling */
#                padding-top: 80px;
#            }}
#        </style>
#        """,
#        unsafe_allow_html=True,
#    )

st.image("Logo.png",width=400)


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data
def load_data(file):
        df = pd.read_csv(file)
        return df

st.title("Forecasting")
option = st.selectbox(
    "Please Select Dataset range you would detect anomaly",
    ("Before Pandemic : 2015-2020", "After Pandemic : 2021-2024"))

if option=="Before Pandemic : 2015-2020":
    tab_titles = ["Train Test Split","XGBOOST","Prophet","Results Comparison"]
    tabs=st.tabs(tab_titles)

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

    file = r"2015_2020.csv"
    df = load_data(file)

    with tabs[0]:
        st.header("Train Test Split")

        dataset = ["Week Start Date", "Positive"]
        df = df[dataset]
        df = df.set_index("Week Start Date")
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y')

        train = df.loc[df.index < "04-02-2019"]
        test = df.loc[df.index >= "04-02-2019"]

        # Create a Plotly figure
        fig = go.Figure()
        # Add traces for the training set
        fig.add_trace(go.Scatter(x=train.index, y=train['Positive'], mode='lines', name='Training Set',line=dict(color='blue')))
        # Add traces for the test set
        fig.add_trace(go.Scatter(x=test.index, y=test['Positive'], mode='lines', name='Test Set',line=dict(color='red')))
        # Add a vertical line to mark the split
        fig.add_shape(
        type="line",
        x0="2019-04-02", y0=0, x1="2019-04-02", y1=max(df['Positive']),
        line=dict(color="Black", dash="dash")
    )
        # Update layout
        fig.update_layout(
            title="Data Train/Test Split",
            xaxis_title="Week Start Date",
            yaxis_title="Positive",
            legend_title="Legend"
        )

        # Display the plot
        st.plotly_chart(fig)
    
    with tabs[1]:
        st.header("Forecasting Using XGBOOST")
        df = create_features(df)
        train= create_features(train)
        test= create_features(test)
        FEATURES=['Positive','dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']
        TARGET="Positive"
        reg=xgb.XGBRegressor(n_estimators=1000,
                            early_stopping_rounds=50,
                            learning_rate=0.01
                            )
        X_train=train[FEATURES]
        y_train=train[TARGET]
        X_test=test[FEATURES]
        y_test=test[TARGET]
        reg.fit( X_train, y_train,
            eval_set=[(X_train, y_train),(X_test,y_test)],
            verbose=200)

        test["prediction"]=reg.predict(X_test)
        df = df.merge(test[["prediction"]],how="left",left_index=True,right_index=True)
        
        fig1, ax1 = plt.subplots(figsize=(15, 5))  # Create a figure and axes
        ax1.plot(df["Positive"], color='b', label='Truth Data', linewidth=2)
        ax1.plot(df["prediction"], color='red', label='Predictions', linewidth=2)
        # Set labels and title
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Values")
        ax1.set_title("Real Data and Prediction Comparison")

        # Add legend
        plt.legend()
        st.plotly_chart(fig1)
            
        predictions = reg.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        smape = 2 * np.mean(np.abs(predictions - y_test) / (np.abs(predictions) + np.abs(y_test)))
        medae = median_absolute_error(y_test, predictions)

    with tabs[2]:

        st.header("Forecasting Using Prophet")

        file1 = "2015_2020.csv"
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
        ax.set_title('Prophet Forecast Results')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='Prophet Forecast Results')

        fig = model.plot_components(df_test_fcst)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='Relevant components of the trained model')

        # Plot the forecast with the actuals
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(df_test.index, df_test['Positive'], color='r')
        fig = model.plot(df_test_fcst, ax=ax)
        ax.set_title('Prophet Results Comparison')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='Prophet Results Comparison')

        future = model.make_future_dataframe(periods=365,freq="W")
        forecast = model.predict(future)
        plot_plotly(model ,forecast)
        st.plotly_chart(plot_plotly(model ,forecast))
        st.markdown(
        "<p style='text-align: center; color: #a0b5c3; font-size: 14px;'>Prediction of the next 5 years</p>",unsafe_allow_html=True)    
        st.plotly_chart(plot_components_plotly(model, forecast))
        st.markdown(
        "<p style='text-align: center; color: #a0b5c3; font-size: 14px;'>Relevant components of the predicted dataset</p>",unsafe_allow_html=True)  

        y_true = df_test['Positive'].values
        y_pred = df_test_fcst['yhat'].values

        mae_1 = mean_absolute_error(y_true, y_pred)
        mse_1 = mean_squared_error(y_true, y_pred)
        rmse_1 = np.sqrt(mse_1)
        r2_1 = r2_score(y_true, y_pred)
        

        # Additional metrics
        def mean_absolute_percentage_error(y_true, y_pred): 
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def symmetric_mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

        def median_absolute_error(y_true, y_pred):
            return np.median(np.abs(y_true - y_pred))

        mape_1 = mean_absolute_percentage_error(y_true, y_pred)
        smape_1 = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        medae_1 = median_absolute_error(y_true, y_pred)
        
    with tabs[3]:
        
        st.header("Results Comparison")

        # Metrics for XGBoost model (assumed values)
        xgboost_metrics = {
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error(MSE)", "Root Mean Square Error (RMSE)", "Coefficient of determination (R2)", "Mean Absolute Percentage Error (MAPE)", "Symmetric Mean Absolute Percentage Error (SMAPE)", "Median Absolute Error (MedAE)"],
            "XGBoost": [mae, mse, rmse, r2,"infinity", smape, medae]
        }

        # Metrics for Prophet model (assumed values)
        prophet_metrics = {
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error(MSE)", "Root Mean Square Error (RMSE)", "Coefficient of determination (R2)", "Mean Absolute Percentage Error (MAPE)", "Symmetric Mean Absolute Percentage Error (SMAPE)", "Median Absolute Error (MedAE)"],
            "Prophet": [mae_1, mse_1,rmse_1,r2_1, "infinity", smape_1, medae_1]
        }
        # Create DataFrames
        prophet_df = pd.DataFrame(prophet_metrics)
        xgboost_df = pd.DataFrame(xgboost_metrics)

        # Merge DataFrames
        comparison_df = pd.merge(prophet_df, xgboost_df, on="Metric")

        st.dataframe(comparison_df, use_container_width=True)


else:

    tab_titles = ["Train Test Split","XGBOOST","Prophet","Results Comparison"]
    tabs=st.tabs(tab_titles)

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

    file1 = "2021_2024.csv"
    df1 = load_data(file1)

    with tabs[0]:
        
        st.header("Train Test Split")

        dataset1 = ["Week Start Date", "Positive"]
        df1 = df1[dataset1]
        df1 = df1.set_index("Week Start Date")
        df1.index = pd.to_datetime(df1.index, format='%d/%m/%Y')

        train = df1.loc[df1.index < "2023-06-26"]
        test = df1.loc[df1.index >= "2023-06-26"]

        # Create a Plotly figure
        fig = go.Figure()
        # Add traces for the training set
        fig.add_trace(go.Scatter(x=train.index, y=train['Positive'], mode='lines', name='Training Set',line=dict(color='blue')))
        # Add traces for the test set
        fig.add_trace(go.Scatter(x=test.index, y=test['Positive'], mode='lines', name='Test Set',line=dict(color='red')))
        # Add a vertical line to mark the split
        fig.add_vline(x='2023-06-26', line=dict(color='black', dash='dash'))
        # Update layout
        fig.update_layout(
            title="Data Train/Test Split",
            xaxis_title="Week Start Date",
            yaxis_title="Positive",
            legend_title="Legend"
        )

        # Display the plot
        st.plotly_chart(fig)

    with tabs[1]:
        df1 = create_features(df1)
        train= create_features(train)
        test= create_features(test)
        FEATURES=['Positive','dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']
        TARGET="Positive"
        reg=xgb.XGBRegressor(n_estimators=1000,
                            early_stopping_rounds=50,
                            learning_rate=0.01
                            )
        X_train=train[FEATURES]
        y_train=train[TARGET]
        X_test=test[FEATURES]
        y_test=test[TARGET]
        reg.fit( X_train, y_train,
            eval_set=[(X_train, y_train),(X_test,y_test)],
            verbose=200)

        test["prediction"]=reg.predict(X_test)
        df = df1.merge(test[["prediction"]],how="left",left_index=True,right_index=True)
        
        fig1, ax1 = plt.subplots(figsize=(15, 5))  # Create a figure and axes
        ax1.plot(df["Positive"], color='b', label='Truth Data', linewidth=2)
        ax1.plot(df["prediction"], color='red', label='Predictions', linewidth=2)
        # Set labels and title
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Values")
        ax1.set_title("Real Data and Prediction Comparison")

        # Add legend
        plt.legend()
        st.plotly_chart(fig1)

        predictions = reg.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        smape = 2 * np.mean(np.abs(predictions - y_test) / (np.abs(predictions) + np.abs(y_test)))
        medae = median_absolute_error(y_test, predictions)

    with tabs[2]:
        st.header("Forecasting Using Prophet")
    
        file1 = "2021_2024.csv"
        df1 = load_data(file1)
    
        dataset = ["Week Start Date","Positive"]
        df1= df1[dataset]
        df1["Week Start Date"]=pd.to_datetime(df1["Week Start Date"],format='%d/%m/%Y')
        df1=df1.set_index("Week Start Date")
        df1.index=pd.to_datetime(df1.index)
        split_date = '06-26-2023'
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
        ax.set_title('Prophet Forecast Results')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='Prophet Forecast Results')
    
        fig = model.plot_components(df_test_fcst)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='Relevant components of the trained model')
    
        # Plot the forecast with the actuals
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(df_test.index, df_test['Positive'], color='r')
        fig = model.plot(df_test_fcst, ax=ax)
        ax.set_title('Prophet Results Comparison')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption='Prophet Results Comparison')
        
        future = model.make_future_dataframe(periods=365,freq="W")
        forecast = model.predict(future)
        plot_plotly(model ,forecast)
        st.plotly_chart(plot_plotly(model ,forecast))
        st.markdown(
        "<p style='text-align: center; color: #a0b5c3; font-size: 14px;'>Prediction of the next 5 years</p>",unsafe_allow_html=True)    
        st.plotly_chart(plot_components_plotly(model, forecast))
        st.markdown(
        "<p style='text-align: center; color: #a0b5c3; font-size: 14px;'>Relevant components of the predicted dataset</p>",unsafe_allow_html=True)  
        
        y_true = df_test['Positive'].values
        y_pred = df_test_fcst['yhat'].values

        mae_1 = mean_absolute_error(y_true, y_pred)
        mse_1 = mean_squared_error(y_true, y_pred)
        rmse_1 = np.sqrt(mse_1)
        r2_1 = r2_score(y_true, y_pred)

        # Additional metrics
        def mean_absolute_percentage_error(y_true, y_pred): 
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def symmetric_mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

        def median_absolute_error(y_true, y_pred):
            return np.median(np.abs(y_true - y_pred))

        mape_1 = mean_absolute_percentage_error(y_true, y_pred)
        smape_1 = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        medae_1 = median_absolute_error(y_true, y_pred)

    with tabs[3]:
        
        st.header("Results Comparison")
        
        # Metrics for XGBoost model (assumed values)
        xgboost_metrics = {
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error(MSE)", "Root Mean Square Error (RMSE)", "Coefficient of determination (R2)", "Mean Absolute Percentage Error (MAPE)", "Symmetric Mean Absolute Percentage Error (SMAPE)", "Median Absolute Error (MedAE)"],
            "XGBoost": [mae, mse, rmse, r2,"infinity", smape, medae]
        }

        # Metrics for Prophet model (assumed values)
        prophet_metrics = {
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error(MSE)", "Root Mean Square Error (RMSE)", "Coefficient of determination (R2)", "Mean Absolute Percentage Error (MAPE)", "Symmetric Mean Absolute Percentage Error (SMAPE)", "Median Absolute Error (MedAE)"],
            "Prophet": [mae_1, mse_1,rmse_1,r2_1, "infinity", smape_1, medae_1]
        }
        # Create DataFrames
        prophet_df = pd.DataFrame(prophet_metrics)
        xgboost_df = pd.DataFrame(xgboost_metrics)

        # Merge DataFrames
        comparison_df = pd.merge(prophet_df, xgboost_df, on="Metric")

        st.dataframe(comparison_df, use_container_width=True)     