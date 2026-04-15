import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title='Bike Predictor', page_icon='Bike', layout='wide')

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'Dataset.csv')
    df = pd.read_csv(data_path)

    df.replace('?', np.nan, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()

    df['season'] = df['season'].replace('springer', 'spring')
    df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)

    for col in ['temp','atemp','hum','windspeed','casual','registered','cnt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna().drop('instant', axis=1)

@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('features.pkl')
    return model, feature_cols

model, feature_cols = load_model()
df_raw = load_data()

st.title('Bike Rental Predictor')

with st.sidebar:
    st.header('Input Parameters')

    hour = st.slider('Hour', 0, 23, 12)
    day = st.selectbox('Day', ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], 1)
    month = st.selectbox('Month', ['January','February','March','April','May','June',
                                  'July','August','September','October','November','December'], 5)
    year = st.radio('Year', [2011, 2012], 1)

    temp = st.slider('Temp (C)', -10.0, 45.0, 20.0)
    humidity = st.slider('Humidity (%)', 0, 100, 50)
    windspeed = st.slider('Wind (km/h)', 0.0, 50.0, 10.0)
    weather = st.selectbox('Weather', ['Clear','Mist','Light Snow','Heavy Rain'])

    season = st.selectbox('Season', ['Spring','Summer','Fall','Winter'], 2)
    holiday = st.checkbox('Holiday')
    working = st.checkbox('Working Day', True)

    predict = st.button('Predict')

if predict:
    data = {
        'hr': hour,
        'weekday': ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'].index(day),
        'mnth': ['January','February','March','April','May','June','July','August','September','October','November','December'].index(month)+1,
        'yr': str(year),
        'temp': temp/41.0,
        'hum': humidity/100.0,
        'windspeed': windspeed/67.0,
        'cnt': 0,
        'season': season.lower(),
        'holiday': 'yes' if holiday else 'no',
        'workingday': 'working day' if working else 'no work',
        'weathersit': weather.lower()
    }

    df_in = pd.DataFrame([data])

    df_in['time_of_day'] = pd.cut(df_in['hr'], bins=[-1,6,12,18,24],
                                 labels=['Night','Morning','Afternoon','Evening'])
    df_in['is_weekend'] = 1 if day in ['Sunday','Saturday'] else 0

    df_enc = df_in.copy()

    for s in ['spring','summer','winter']:
        df_enc[f'season_{s}'] = (df_enc['season']==s).astype(int)

    df_enc['yr_2012'] = (df_enc['yr']=='2012').astype(int)

    for m in range(2,13):
        df_enc[f'mnth_{m}'] = (df_enc['mnth']==m).astype(int)

    df_enc['holiday_yes'] = (df_enc['holiday']=='yes').astype(int)
    df_enc['workingday_working day'] = (df_enc['workingday']=='working day').astype(int)

    for w in ['mist','light snow','heavy rain']:
        df_enc[f'weathersit_{w}'] = (df_enc['weathersit']==w).astype(int)

    for t in ['Morning','Afternoon','Evening']:
        df_enc[f'time_of_day_{t}'] = (df_enc['time_of_day']==t).astype(int)

    df_enc = df_enc.drop(['season','yr','mnth','holiday','workingday','weathersit','time_of_day'], axis=1)

    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0

    df_enc = df_enc[feature_cols]

    pred = model.predict(df_enc)[0]

    st.write('Predicted Rentals:', int(pred))

    if pred > 500:
        st.write('High Demand')
    elif pred > 200:
        st.write('Moderate Demand')
    else:
        st.write('Low Demand')

