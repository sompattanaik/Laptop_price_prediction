import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and data
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Laptop Price Predictor.')
# Selection dropdowns
brand = st.selectbox('Brand',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
Ram = st.selectbox('RAM (in GB)',df['Ram'].unique())
touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS', ['No','Yes'])
weight = st.number_input('Weight of the Laptop')
screensize = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    if screensize > 0:
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screensize
    else:
        st.error("Screen size must be greater than zero.")
        st.stop()

    query = np.array([brand,type,Ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1, -1)

    # Predict and display the result
    predicted_price = np.exp(pipe.predict(query)[0])
    st.title(f"The Predicted Price Of A Laptop With These Specifications is Approximately : {int(predicted_price)} INR")
