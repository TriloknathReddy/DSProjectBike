import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title='Bike Predictor', page_icon='Bike', layout='wide')

# Colorful Dashboard CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.4);
    }
    .metric-card.orange {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        box-shadow: 0 8px 25px rgba(252, 74, 26, 0.4);
    }
    .metric-card.red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 8px 25px rgba(235, 51, 73, 0.4);
    }
    .metric-card.blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        box-shadow: 0 8px 25px rgba(33, 147, 176, 0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.9);
        margin-top: 5px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(17, 153, 142, 0.4);
    }
    .prediction-box.high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 15px 50px rgba(235, 51, 73, 0.4);
    }
    .prediction-box.medium {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        box-shadow: 0 15px 50px rgba(252, 74, 26, 0.4);
    }
    .prediction-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    .prediction-label {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-top: 10px;
    }
    .indicator-box {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #667eea;
    }
    .indicator-box.green { border-left-color: #11998e; background: #e8f5e9; }
    .indicator-box.orange { border-left-color: #fc4a1a; background: #fff3e0; }
    .indicator-box.red { border-left-color: #eb3349; background: #ffebee; }
</style>
""", unsafe_allow_html=True)

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
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    features_path = os.path.join(os.path.dirname(__file__), 'features.pkl')

    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
    else:
        # Train and save model
        df = load_data()
        df['time_of_day'] = pd.cut(df['hr'], bins=[-1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
        df_enc = pd.get_dummies(df, drop_first=True)
        drop_cols = [c for c in ['casual', 'registered', 'dteday', 'atemp'] if c in df_enc.columns]
        model_df = df_enc.drop(columns=drop_cols)
        X, y = model_df.drop('cnt', axis=1), model_df['cnt']

        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        feature_cols = X.columns.tolist()

        joblib.dump(model, model_path)
        joblib.dump(feature_cols, features_path)

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

    predict = st.button('Predict', use_container_width=True)

st.subheader('Dashboard')

# Time & Date Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card blue">
        <p class="metric-value">{hour:02d}:00</p>
        <p class="metric-label">Hour</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{day[:3]}</p>
        <p class="metric-label">Day</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{month[:3]}</p>
        <p class="metric-label">Month</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{year}</p>
        <p class="metric-label">Year</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Weather Row
col5, col6, col7, col8 = st.columns(4)
with col5:
    color_class = "red" if temp > 30 else "blue" if temp < 10 else "green"
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <p class="metric-value">{temp:.1f}°C</p>
        <p class="metric-label">Temperature</p>
    </div>
    """, unsafe_allow_html=True)
with col6:
    st.markdown(f"""
    <div class="metric-card blue">
        <p class="metric-value">{humidity}%</p>
        <p class="metric-label">Humidity</p>
    </div>
    """, unsafe_allow_html=True)
with col7:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{windspeed:.1f}</p>
        <p class="metric-label">Wind km/h</p>
    </div>
    """, unsafe_allow_html=True)
with col8:
    weather_color = "green" if weather == "Clear" else "orange" if weather == "Mist" else "red"
    st.markdown(f"""
    <div class="metric-card {weather_color}">
        <p class="metric-value">{weather}</p>
        <p class="metric-label">Weather</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Status Row
col9, col10, col11 = st.columns(3)
with col9:
    season_color = "green" if season in ["Summer", "Fall"] else "blue"
    st.markdown(f"""
    <div class="metric-card {season_color}">
        <p class="metric-value">{season}</p>
        <p class="metric-label">Season</p>
    </div>
    """, unsafe_allow_html=True)
with col10:
    holiday_color = "orange" if holiday else "green"
    holiday_text = "Yes" if holiday else "No"
    st.markdown(f"""
    <div class="metric-card {holiday_color}">
        <p class="metric-value">{holiday_text}</p>
        <p class="metric-label">Holiday</p>
    </div>
    """, unsafe_allow_html=True)
with col11:
    work_color = "green" if working else "orange"
    work_text = "Yes" if working else "No"
    st.markdown(f"""
    <div class="metric-card {work_color}">
        <p class="metric-value">{work_text}</p>
        <p class="metric-label">Working Day</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

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

    # Colorful Prediction Display
    pred_class = "high" if pred > 500 else "medium" if pred > 200 else ""
    indicator_class = "red" if pred > 500 else "orange" if pred > 200 else "green"
    status_text = "HIGH DEMAND" if pred > 500 else "MODERATE DEMAND" if pred > 200 else "LOW DEMAND"
    
    st.markdown(f"""
    <div class="prediction-box {pred_class}">
        <p class="prediction-label">Predicted Bike Rentals</p>
        <p class="prediction-number">{int(pred):,}</p>
        <p class="prediction-label">bikes this hour</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Indicator
    st.markdown(f"""
    <div class="indicator-box {indicator_class}">
        <h3 style="margin:0;color:#333;">{status_text}</h3>
        <p style="margin:5px 0 0 0;color:#666;">
            {"Ensure maximum bike availability at all stations" if pred > 500 else 
             "Standard operations - monitor popular stations" if pred > 200 else 
             "Good time for maintenance and bike redistribution"}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
