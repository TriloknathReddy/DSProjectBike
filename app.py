import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

st.set_page_config(page_title='Bike Predictor', layout='wide')

#  LOAD DATA 
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset.csv')
    df.replace('?', np.nan, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()

    df['season'] = df['season'].replace('springer', 'spring')
    df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)

    for col in ['temp','atemp','hum','windspeed','casual','registered','cnt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna().drop('instant', axis=1)

#    TRAIN MODEL     
@st.cache_resource
def train_model():
    df = pd.read_csv('Dataset.csv')
    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    df['season'] = df['season'].replace('springer', 'spring')
    df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)
    for col in ['temp','atemp','hum','windspeed','casual','registered','cnt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().drop('instant', axis=1)

    df['time_of_day'] = pd.cut(df['hr'], bins=[-1, 6, 12, 18, 24], labels=['Night','Morning','Afternoon','Evening'])
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)

    df_encoded = pd.get_dummies(df, drop_first=True)
    leakage_cols = ['casual','registered','dteday']
    drop_cols = [c for c in leakage_cols if c in df_encoded.columns]
    model_df = df_encoded.drop(columns=drop_cols)

    X = model_df.drop(columns=['cnt'])
    y = model_df['cnt']
    feature_cols = list(X.columns)

    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return model, feature_cols

model, feature_cols = train_model()

st.title("Bike Rental Prediction")

#    INPUT  
with st.sidebar:
    hour = st.slider('Hour', 0, 23, 12)
    day = st.selectbox('Day', ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
    month = st.selectbox('Month', ['January','February','March','April','May','June',
                                  'July','August','September','October','November','December'])
    year = st.radio('Year', [2011, 2012])

    temp = st.slider('Temperature (C)', -10.0, 45.0, 25.0)
    humidity = st.slider('Humidity (%)', 0, 100, 50)
    windspeed = st.slider('Wind Speed (km/h)', 0.0, 50.0, 10.0)

    weather = st.selectbox('Weather', ['Clear','Mist','Light Snow','Heavy Rain'])
    season = st.selectbox('Season', ['Spring','Summer','Fall','Winter'])

    holiday = st.checkbox('Holiday')
    working = st.checkbox('Working Day', True)

    predict = st.button("Predict")

#  DASHBOARD 
st.markdown("---")
st.subheader("Input Dashboard")

# Time & Date
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
with row1_col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{hour:02d}:00</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Hour</p>
    </div>
    """, unsafe_allow_html=True)
with row1_col2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #11998e, #38ef7d); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{day[:3]}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Day</p>
    </div>
    """, unsafe_allow_html=True)
with row1_col3:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fc4a1a, #f7b733); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{month[:3]}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Month</p>
    </div>
    """, unsafe_allow_html=True)
with row1_col4:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eb3349, #f45c43); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{year}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Year</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Weather & Conditions
row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
temp_color = "#eb3349" if temp > 30 else "#2193b0" if temp < 10 else "#11998e"
with row2_col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {temp_color}, #6dd5ed); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{temp:.1f}°C</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Temperature</p>
    </div>
    """, unsafe_allow_html=True)
with row2_col2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2193b0, #6dd5ed); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{humidity}%</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Humidity</p>
    </div>
    """, unsafe_allow_html=True)
with row2_col3:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{windspeed:.1f}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Wind km/h</p>
    </div>
    """, unsafe_allow_html=True)
with row2_col4:
    weather_colors = {"Clear": ("#11998e", "#38ef7d"), "Mist": ("#fc4a1a", "#f7b733"), "Light Snow": ("#2193b0", "#6dd5ed"), "Heavy Rain": ("#eb3349", "#f45c43")}
    wc1, wc2 = weather_colors.get(weather, ("#667eea", "#764ba2"))
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {wc1}, {wc2}); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.4rem; margin: 0; color: white; font-weight: bold;">{weather}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Weather</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Season & Status
row3_col1, row3_col2, row3_col3 = st.columns(3)
with row3_col1:
    season_colors = {"Spring": "#11998e", "Summer": "#fc4a1a", "Fall": "#667eea", "Winter": "#2193b0"}
    sc = season_colors.get(season, "#667eea")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {sc}, #764ba2); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{season}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Season</p>
    </div>
    """, unsafe_allow_html=True)
with row3_col2:
    h_color = "#fc4a1a" if holiday else "#11998e"
    h_text = "Yes" if holiday else "No"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {h_color}, #38ef7d); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{h_text}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Holiday</p>
    </div>
    """, unsafe_allow_html=True)
with row3_col3:
    w_color = "#11998e" if working else "#fc4a1a"
    w_text = "Yes" if working else "No"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {w_color}, #38ef7d); padding: 15px; border-radius: 10px; text-align: center;">
        <p style="font-size: 1.8rem; margin: 0; color: white; font-weight: bold;">{w_text}</p>
        <p style="font-size: 0.9rem; margin: 0; color: rgba(255,255,255,0.8);">Working Day</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

#  PREDICTION 
if predict:

    data = {
        'hr': hour,
        'weekday': ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'].index(day),
        'mnth': ['January','February','March','April','May','June',
                 'July','August','September','October','November','December'].index(month) + 1,
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

    # Feature engineering
    df_in['time_of_day'] = pd.cut(df_in['hr'], bins=[-1,6,12,18,24],
                                 labels=['Night','Morning','Afternoon','Evening'])
    df_in['is_weekend'] = 1 if day in ['Sunday','Saturday'] else 0

    df_enc = df_in.copy()

    # Encoding
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

    # Align columns
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0

    df_enc = df_enc[feature_cols]

    # Model prediction
    pred = model.predict(df_enc)[0]

    weather_factor = {
        "Clear": 1.0,
        "Mist": 0.9,
        "Light Snow": 0.75,
        "Heavy Rain": 0.6
    }

    pred = pred * weather_factor.get(weather, 1.0)

    #  OUTPUT 
    st.write("Predicted Rentals:", int(pred))

    if pred > 500:
        st.success("High Demand")
    elif pred > 200:
        st.info("Moderate Demand")
    else:
        st.warning("Low Demand")