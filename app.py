import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bike Rental Predictor", layout="wide")

# Load trained model
model = joblib.load("model.pkl")

st.title("🚴 Bike Rental Prediction")

# =========================
# INPUT
# =========================
with st.sidebar:
    hour = st.slider("Hour", 0, 23, 12)
    weekday = st.selectbox("Weekday (0=Sun)", list(range(7)))
    month = st.selectbox("Month", list(range(1,13)))
    year = st.selectbox("Year", [0,1])  # 0=2011,1=2012

    temp = st.slider("Temp (normalized)", 0.0, 1.0, 0.5)
    atemp = temp
    humidity = st.slider("Humidity (0-1)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Wind (0-1)", 0.0, 1.0, 0.2)

    season = st.selectbox("Season", ["spring","summer","fall","winter"])
    holiday = st.selectbox("Holiday", ["no","yes"])
    workingday = st.selectbox("Working Day", ["no work","working day"])
    weather = st.selectbox("Weather", ["clear","mist","light snow","heavy rain"])

    predict = st.button("Predict")

# =========================
# PREDICTION
# =========================
if predict:
    data = pd.DataFrame([{
        "hr": hour,
        "weekday": weekday,
        "mnth": month,
        "yr": year,
        "temp": temp,
        "atemp": atemp,
        "hum": humidity,
        "windspeed": windspeed,
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weathersit": weather,
        "time_of_day": (
            "night" if hour <= 6 else
            "morning" if hour <= 12 else
            "afternoon" if hour <= 18 else
            "evening"
        ),
        "is_weekend": 1 if weekday in [0,6] else 0
    }])

    pred = model.predict(data)[0]

    st.success(f"Predicted Rentals: {int(pred)}")

    if pred > 500:
        st.info("High demand")
    elif pred > 200:
        st.warning("Moderate demand")
    else:
        st.error("Low demand")
