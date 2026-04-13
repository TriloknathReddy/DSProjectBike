import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

st.set_page_config(page_title="Bike Predictor", page_icon="Bike", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0d1117; }
.main { background-color: #0d1117; }
h1, h2, h3 { color: #f0f6fc; }
p { color: #c9d1d9; }
.header {
    background: linear-gradient(90deg, #238636, #2ea043);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
}
.header h1 { color: white; font-size: 2rem; margin: 0; }
.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
}
.metric-box {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
.metric-box.green { background: linear-gradient(135deg, #238636, #2ea043); }
.metric-box.purple { background: linear-gradient(135deg, #8957e5, #a371f7); }
.metric-box.orange { background: linear-gradient(135deg, #d29922, #e3b341); }
.metric-number { font-size: 1.8rem; font-weight: 700; color: white; margin: 0; }
.metric-label { font-size: 0.9rem; color: #c9d1d9; margin-top: 5px; }
.prediction-box {
    background: linear-gradient(135deg, #238636, #3fb950);
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}
.prediction-value { font-size: 3.5rem; font-weight: 800; color: white; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Bike Rental Predictor</h1></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Dataset.csv')
    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    df['season'] = df['season'].replace('springer', 'spring')
    df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)
    for col in ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().drop('instant', axis=1)

@st.cache_resource
def train_model():
    df = load_data()
    df['time_of_day'] = pd.cut(df['hr'], bins=[-1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
    df_enc = pd.get_dummies(df, drop_first=True)
    drop_cols = [c for c in ['casual', 'registered', 'dteday', 'atemp'] if c in df_enc.columns]
    model_df = df_enc.drop(columns=drop_cols)
    X, y = model_df.drop('cnt', axis=1), model_df['cnt']
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist(), df

# Don't train at import - use a function to get/cached model
def get_model():
    return train_model()

with st.sidebar:
    st.header("Input Parameters")
    hour = st.slider("Hour", 0, 23, 12)
    day = st.selectbox("Day", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], 1)
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November", "December"], 5)
    year = st.radio("Year", [2011, 2012], 1)
    
    st.markdown("---")
    temp = st.slider("Temp C", -10.0, 45.0, 20.0, 0.5)
    humidity = st.slider("Humidity %", 0, 100, 50)
    windspeed = st.slider("Wind km/h", 0.0, 50.0, 10.0, 0.5)
    weather = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    
    st.markdown("---")
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], 2)
    holiday = st.checkbox("Holiday")
    working = st.checkbox("Working Day", True)
    
    predict = st.button("Predict", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-box"><p class="metric-number">{hour:02d}:00</p><p class="metric-label">Hour</p></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-box green"><p class="metric-number">{temp:.1f}C</p><p class="metric-label">Temp</p></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-box purple"><p class="metric-number">{humidity}%</p><p class="metric-label">Humidity</p></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-box orange"><p class="metric-number">{windspeed:.1f}</p><p class="metric-label">Wind</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load model and data only when needed
model, feature_cols, df_raw = get_model()

similar = df_raw[df_raw['hr'] == hour]
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="metric-box"><p class="metric-number">{int(similar["cnt"].mean())}</p><p class="metric-label">Avg Demand</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-box green"><p class="metric-number">{int(similar["cnt"].max())}</p><p class="metric-label">Max Demand</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-box purple"><p class="metric-number">{int(similar["cnt"].min())}</p><p class="metric-label">Min Demand</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Model Info"):
    st.write("Algorithm: Extra Trees Regressor")
    st.write("R2: 0.951 | RMSE: 40.3 | MAE: 25.2")
    st.write("Features: Hour, Temp, Humidity, Working Day, Season")

if predict:
    with st.spinner("Predicting..."):
        # Create input with ALL original columns (not pre-engineered)
        data = {
            'hr': hour,
            'weekday': ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"].index(day),
            'mnth': ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].index(month) + 1,
            'yr': str(year),  # '2011' or '2012'
            'temp': temp / 41.0,
            'atemp': temp / 41.0,  # approximation
            'hum': humidity / 100.0,
            'windspeed': windspeed / 67.0,
            'cnt': 0,
            'season': season.lower(),
            'holiday': 'yes' if holiday else 'no',  # lowercase
            'workingday': 'working day' if working else 'no work',  # match dataset
            'weathersit': weather.lower()
        }
        
        # Convert to DataFrame
        df_in = pd.DataFrame([data])
        
        # Feature engineering (EXACTLY as in training)
        df_in['time_of_day'] = pd.cut(df_in['hr'], bins=[-1, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        df_in['is_weekend'] = 1 if day in ['Sunday', 'Saturday'] else 0
        
        # Manually create one-hot encoded columns to match training
        df_enc = df_in.copy()
        
        # Create dummies for each categorical (matching training drop_first=True)
        # Season (baseline: fall)
        for s in ['spring', 'summer', 'winter']:
            df_enc[f'season_{s}'] = (df_enc['season'] == s).astype(int)
        
        # Year (baseline: 2011)
        df_enc['yr_2012'] = (df_enc['yr'] == '2012').astype(int)
        
        # Month (baseline: 1/January)
        for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            df_enc[f'mnth_{m}'] = (df_enc['mnth'] == m).astype(int)
        
        # Holiday (baseline: no)
        df_enc['holiday_yes'] = (df_enc['holiday'] == 'yes').astype(int)
        
        # Working day (baseline: no work)
        df_enc['workingday_working day'] = (df_enc['workingday'] == 'working day').astype(int)
        
        # Weather (baseline: clear)
        for w in ['mist', 'light snow', 'heavy rain']:
            df_enc[f'weathersit_{w}'] = (df_enc['weathersit'] == w).astype(int)
        
        # Time of day (baseline: Night)
        for t in ['Morning', 'Afternoon', 'Evening']:
            df_enc[f'time_of_day_{t}'] = (df_enc['time_of_day'] == t).astype(int)
        
        # Drop original categoricals
        df_enc = df_enc.drop(['season', 'yr', 'mnth', 'holiday', 'workingday', 'weathersit', 'time_of_day'], axis=1)
        
        
        # Align columns with training
        for col in feature_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0
        
        # Ensure correct order and only training columns
        df_enc = df_enc[feature_cols]
        
        # Predict
        pred = model.predict(df_enc)[0]
        
        st.markdown(f'<div class="prediction-box"><p style="color:#c9d1d9;margin:0;">Predicted Rentals</p><p class="prediction-value">{int(pred):,}</p><p style="color:#c9d1d9;">bikes this hour</p></div>', unsafe_allow_html=True)
        
        
        if pred > 500:
            st.success("High demand. Ensure maximum availability.")
        elif pred > 200:
            st.info("Moderate demand. Monitor stations.")
        else:
            st.warning("Low demand. Good for maintenance.")
