import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

df = pd.read_csv('Dataset.csv')
df.replace('?', np.nan, inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.lower()
df['season'] = df['season'].replace('springer', 'spring')
df['dteday'] = pd.to_datetime(df['dteday'], dayfirst=True)
for col in ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().drop('instant', axis=1)

df['time_of_day'] = pd.cut(df['hr'], bins=[-1, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df['is_weekend'] = df['weekday'].isin([0, 6]).astype(int)
df_enc = pd.get_dummies(df, drop_first=True)
drop_cols = [c for c in ['casual', 'registered', 'dteday', 'atemp'] if c in df_enc.columns]
model_df = df_enc.drop(columns=drop_cols)
X, y = model_df.drop('cnt', axis=1), model_df['cnt']

# Smaller model: 20 trees, max_depth 12
model = ExtraTreesRegressor(n_estimators=20, random_state=42, max_depth=12, min_samples_split=5)
model.fit(X, y)

# Compress with joblib (much better than pickle)
joblib.dump({'model': model, 'feature_cols': X.columns.tolist(), 'df_raw': df}, 'model_compressed.pkl', compress=3)

import os
size = os.path.getsize('model_compressed.pkl') / (1024 * 1024)
print(f"Model saved: {size:.1f} MB")
