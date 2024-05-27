import streamlit as st
import streamlit_pandas as sp
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from scipy import stats
import io
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔍"
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

st.image(r"C:\Users\hacio\OneDrive\Desktop\Project\Teknofest Kopya.png",width=400)



@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


st.title("Anomaly Detection")

option = st.selectbox(
    "Please Select Dataset range you would detect anomaly",
    ("Before Pandemic : 2015-2020", "After Pandemic : 2021-2024"))

if option=="Before Pandemic : 2015-2020":

    file = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\No Anomaly\2015_2020.csv"
    df = load_data(file)


    tab_titles = ["Threshold","Z-Score","Tukey's Model","Results Comparison"]
    tabs=st.tabs(tab_titles)

    with tabs[0]:
        df["Treshold Anomaly"] = df["Positive"] > 77.5
        df_true = df[df["Treshold Anomaly"] == True]
        
        st.header("Anomalies Detected Using Treshold")

        st.markdown(f"Treshold was set to **77.5**.")

        fig = go.Figure([go.Scatter(x=df['Week Start Date'], y=df['Positive'],name="Influenza cases")])
        fig.update_layout(title='Türkiye 2015-2020')
        fig.update_layout(title=dict(x=0.45, y=0.9))
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Influenza cases")

        fig.add_shape(
            type="line",
            x0=0,  # Set the starting point of the line
            y0=77.5,
            x1=233,  # Set the ending point of the line
            y1=77.5,
            line=dict(
                color="red",  # You can change the color of the line if you want
                width=2,
                dash="dash",
                ),
                name="Treshold"
        )

        # Define the text content and position
        text = "Threshold line"  # Customize the text as needed
        x_pos = 116.5  # Adjust x-position for better placement
        y_pos = 82

        # Add the annotation
        fig.add_annotation(
            text=text,
            x=x_pos,  # X-coordinate relative to data points
            y=y_pos,  # Y-coordinate relative to data points
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red"),  # Customize font size and color
            xref="x",  # Reference to the x-axis
            yref="y"   # Reference to the y-axis
        )

        fig.add_trace(go.Scatter(mode='markers',
                                    x=df_true["Week Start Date"],
                                    y=df_true["Positive"],
                                    marker=dict(color='red', size=10),
                                    name='Treshold Anomaly'))

        fig.update_layout(autosize=True, width=1500, height=600,showlegend=True)

        st.plotly_chart(fig)

    with tabs[1]:
        st.header("Anomalies Detected Using Z-Score")
        # Calculate mean and standard deviation
        mean = np.mean(df["Positive"])
        std = np.std(df["Positive"])
        # Calculate the z-score (assuming no missing values)
        z_score = mean + std * 1.3

        st.markdown(f"Mean was found equal to **{mean}**.")
        st.markdown(f"Standard Deviation was found equal to **{std}**.")
        st.markdown(f"Z Score  was found equal to **{z_score}**.")

        df["Z_Score Anomaly"] = df["Positive"] > z_score
        df_true = df[df["Z_Score Anomaly"] == True]

        fig_2 = go.Figure([go.Scatter(x=df['Week Start Date'], y=df['Positive'],name="Influenza cases")])
        fig_2.update_layout(title='Türkiye 2015-2020')
        fig_2.update_layout(title=dict(x=0.45, y=0.9))
        fig_2.update_xaxes(title_text="Time")
        fig_2.update_yaxes(title_text="Influenza cases")
        fig_2.add_shape(
            type="line",
            x0=0,  # Set the starting point of the line
            y0=53.34,
            x1=233,  # Set the ending point of the line
            y1=53.34,
            line=dict(
                color="red",  # You can change the color of the line if you want
                width=2,
                dash="dash",
                ),
                name="Treshold"
        )
        # Define the text content and position
        text = "Z_score line"  # Customize the text as needed
        x_pos = 100  # Adjust x-position for better placement
        y_pos = 58
        # Add the annotation
        fig_2.add_annotation(
            text=text,
            x=x_pos,  # X-coordinate relative to data points
            y=y_pos,  # Y-coordinate relative to data points
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red"),  # Customize font size and color
            xref="x",  # Reference to the x-axis
            yref="y"   # Reference to the y-axis
        )
        fig_2.add_trace(go.Scatter(mode='markers',
                                    x=df_true["Week Start Date"],
                                    y=df_true["Positive"],
                                    marker=dict(color='red', size=10),
                                    name='Z_score Anomaly'))

        fig_2.update_layout(autosize=True, width=1000, height=600,showlegend=True)
        st.plotly_chart(fig_2)

    with tabs[2]:
        st.header("Anomalies Detected Using Tukey's Model")

        q75 = df["Positive"].quantile(0.75)
        q = 0.4
        iqr = df["Positive"].quantile(0.75) - df["Positive"].quantile(0.25)
        tukey= q75+q*iqr

        st.markdown(f"The interquartile range was found equal to **{iqr}**.")
        st.markdown(f"The third quartile was found equal to **{q75}**.")
        st.markdown(f"Q was selected to be **{q}**.")
        st.markdown(f"Tukey Value was found equal to **{tukey}**.")

        df["Tukey's Model Anomaly"] = df["Positive"] > tukey
        df_true = df[df["Tukey's Model Anomaly"] == True]

        fig_3 = go.Figure([go.Scatter(x=df['Week Start Date'], y=df['Positive'],name="Influenza cases")])

        fig_3.update_layout(title='Türkiye 2015-2020')
        fig_3.update_layout(title=dict(x=0.45, y=0.9))
        fig_3.update_xaxes(title_text="Time")
        fig_3.update_yaxes(title_text="Influenza cases")

        fig_3.add_shape(
            type="line",
            x0=0,  # Set the starting point of the line
            y0=36.4,
            x1=233,  # Set the ending point of the line
            y1=36.4,
            line=dict(
                color="red",  # You can change the color of the line if you want
                width=2,
                dash="dash",
                ),
                name="Treshold"
        )

        # Define the text content and position
        text = "Tukey's Model line"  # Customize the text as needed
        x_pos = 95  # Adjust x-position for better placement
        y_pos = 40

        # Add the annotation
        fig_3.add_annotation(
            text=text,
            x=x_pos,  # X-coordinate relative to data points
            y=y_pos,  # Y-coordinate relative to data points
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red"),  # Customize font size and color
            xref="x",  # Reference to the x-axis
            yref="y"   # Reference to the y-axis
        )

        fig_3.add_trace(go.Scatter(mode='markers',
                                    x=df_true["Week Start Date"],
                                    y=df_true["Positive"],
                                    marker=dict(color='red', size=10),
                                    name="Tukey's Model Anomaly"))

        fig_3.update_layout(autosize=True, width=1000, height=600,showlegend=True)
        st.plotly_chart(fig_3)
    
    with tabs[3]:
        st.header("2015-2020 Anomaly Detection Results Comparison")

        file1 = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\With Anomaly\2015_2020_Anomaly_Detected.csv"
        df1 = load_data(file1)
        
        st.dataframe(df1, use_container_width=True)
else:
    file1 = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\No Anomaly\2021_2024.csv"
    df1 = load_data(file1)

    tab_titles = ["Threshold","Z-Score","Tukey's Model","Results Comparison"]
    tabs=st.tabs(tab_titles)

    with tabs[0]:
        df1["Treshold Anomaly"] = df1["Positive"] > 47.5
        df_true = df1[df1["Treshold Anomaly"] == True]
        
        st.header("Anomalies Detected Using Treshold")

        st.markdown(f"Treshold was set to **47.5**.")


        fig = go.Figure([go.Scatter(x=df1['Week Start Date'], y=df1['Positive'],name="Influenza cases")])
        fig.update_layout(title='Türkiye 2021-2024')
        fig.update_layout(title=dict(x=0.45, y=0.9))
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Influenza cases")

        fig.add_shape(
            type="line",
            x0=0,  # Set the starting point of the line
            y0=47.5,
            x1=160,  # Set the ending point of the line
            y1=47.5,
            line=dict(
                color="red",  # You can change the color of the line if you want
                width=2,
                dash="dash",
                ),
                name="Treshold"
        )

        # Define the text content and position
        text = "Threshold line"  # Customize the text as needed
        x_pos = 116.5  # Adjust x-position for better placement
        y_pos = 50

        # Add the annotation
        fig.add_annotation(
            text=text,
            x=x_pos,  # X-coordinate relative to data points
            y=y_pos,  # Y-coordinate relative to data points
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red"),  # Customize font size and color
            xref="x",  # Reference to the x-axis
            yref="y"   # Reference to the y-axis
        )

        fig.add_trace(go.Scatter(mode='markers',
                                    x=df_true["Week Start Date"],
                                    y=df_true["Positive"],
                                    marker=dict(color='red', size=10),
                                    name='Treshold Anomaly'))

        fig.update_layout(autosize=True, width=1500, height=600,showlegend=True)

        st.plotly_chart(fig)

    with tabs[1]:
        st.header("Anomalies Detected Using Z-Score")
        # Calculate mean and standard deviation
        mean = np.mean(df1["Positive"])
        std = np.std(df1["Positive"])
        # Calculate the z-score (assuming no missing values)
        z_score = mean + std * 1.3

        st.markdown(f"Mean was found equal to **{mean}**.")
        st.markdown(f"Standard Deviation was found equal to **{std}**.")
        st.markdown(f"Z Score  was found equal to **{z_score}**.")

        df1["Z_Score Anomaly"] = df1["Positive"] > z_score
        df_true = df1[df1["Z_Score Anomaly"] == True]

        fig_2 = go.Figure([go.Scatter(x=df1['Week Start Date'], y=df1['Positive'],name="Influenza cases")])
        fig_2.update_layout(title='Türkiye 2021-2024')
        fig_2.update_layout(title=dict(x=0.45, y=0.9))
        fig_2.update_xaxes(title_text="Time")
        fig_2.update_yaxes(title_text="Influenza cases")
        fig_2.add_shape(
            type="line",
            x0=0,  # Set the starting point of the line
            y0=41.039,
            x1=160,  # Set the ending point of the line
            y1=41.039,
            line=dict(
                color="red",  # You can change the color of the line if you want
                width=2,
                dash="dash",
                ),
                name="Treshold"
        )
        # Define the text content and position
        text = "Z_score line"  # Customize the text as needed
        x_pos = 120  # Adjust x-position for better placement
        y_pos = 43
        # Add the annotation
        fig_2.add_annotation(
            text=text,
            x=x_pos,  # X-coordinate relative to data points
            y=y_pos,  # Y-coordinate relative to data points
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red"),  # Customize font size and color
            xref="x",  # Reference to the x-axis
            yref="y"   # Reference to the y-axis
        )
        fig_2.add_trace(go.Scatter(mode='markers',
                                    x=df_true["Week Start Date"],
                                    y=df_true["Positive"],
                                    marker=dict(color='red', size=10),
                                    name='Z_score Anomaly'))

        fig_2.update_layout(autosize=True, width=1000, height=600,showlegend=True)
        st.plotly_chart(fig_2)

    with tabs[2]:
        st.header("Anomalies Detected Using Tukey's Model")

        q75 = df1["Positive"].quantile(0.75)
        q = 0.4
        iqr = df1["Positive"].quantile(0.75) - df1["Positive"].quantile(0.25)
        tukey= q75+q*iqr

        st.markdown(f"The interquartile range was found equal to **{iqr}**.")
        st.markdown(f"The third quartile was found equal to **{q75}**.")
        st.markdown(f"Q was selected to be **{q}**.")
        st.markdown(f"Tukey Value was found equal to **{tukey}**.")

        df1["Tukey's Model Anomaly"] = df1["Positive"] > tukey
        df_true = df1[df1["Tukey's Model Anomaly"] == True]

        fig_3 = go.Figure([go.Scatter(x=df1['Week Start Date'], y=df1['Positive'],name="Influenza cases")])

        fig_3.update_layout(title='Türkiye 2021-2024')
        fig_3.update_layout(title=dict(x=0.45, y=0.9))
        fig_3.update_xaxes(title_text="Time")
        fig_3.update_yaxes(title_text="Influenza cases")

        fig_3.add_shape(
            type="line",
            x0=0,  # Set the starting point of the line
            y0=25.2,
            x1=160,  # Set the ending point of the line
            y1=25.2,
            line=dict(
                color="red",  # You can change the color of the line if you want
                width=2,
                dash="dash",
                ),
                name="Treshold"
        )

        # Define the text content and position
        text = "Tukey's Model line"  # Customize the text as needed
        x_pos = 65  # Adjust x-position for better placement
        y_pos = 28

        # Add the annotation
        fig_3.add_annotation(
            text=text,
            x=x_pos,  # X-coordinate relative to data points
            y=y_pos,  # Y-coordinate relative to data points
            showarrow=False,  # Hide the arrow
            font=dict(size=12, color="red"),  # Customize font size and color
            xref="x",  # Reference to the x-axis
            yref="y"   # Reference to the y-axis
        )

        fig_3.add_trace(go.Scatter(mode='markers',
                                    x=df_true["Week Start Date"],
                                    y=df_true["Positive"],
                                    marker=dict(color='red', size=10),
                                    name="Tukey's Model Anomaly"))

        fig_3.update_layout(autosize=True, width=1000, height=600,showlegend=True)
        st.plotly_chart(fig_3)
    
    with tabs[3]:
        st.header("2021-2024 Anomaly Detection Results Comparison")

        file2 = r"C:\Users\hacio\OneDrive\Desktop\Project\Needed\With Anomaly\2021_2024_Anomaly_Detected.csv"
        df2 = load_data(file2)
        st.dataframe(df2, use_container_width=True)






