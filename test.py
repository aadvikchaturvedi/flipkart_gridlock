"""
test.py — Load trained XGBoost artifacts and generate submission.csv
Run after train.py
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import pygeohash as pgh
from sklearn.impute import SimpleImputer

# ── Load artifacts ────────────────────────────────────────────────────────────
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('gh_map.pkl', 'rb') as f:
    gh_artifacts = pickle.load(f)
    gh_map       = gh_artifacts['gh_map']
    gh_fallback  = gh_artifacts['fallback']

with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

print("Artifacts loaded.")

# ── Load & preprocess test.csv ────────────────────────────────────────────────
df_test = pd.read_csv('dataset/test.csv')

# Impute
road_imputer    = SimpleImputer(strategy='most_frequent')
weather_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

df_test['RoadType']    = road_imputer.fit_transform(df_test[['RoadType']]).ravel()
df_test['Weather']     = weather_imputer.fit_transform(df_test[['Weather']]).ravel()
df_test['Temperature'] = df_test['Temperature'].fillna(df_test['Temperature'].mean())
df_test['Landmarks']   = (df_test['Landmarks'] == "Yes").astype(int)

# One-hot encode
df_test = pd.get_dummies(df_test, columns=['RoadType', 'LargeVehicles', 'Weather'], dtype=int)

# ── Feature Engineering (same as train.py) ────────────────────────────────────
def safe_decode(gh):
    try:
        lat, lon = pgh.decode(gh)
        return pd.Series([lat, lon])
    except Exception:
        return pd.Series([np.nan, np.nan])

df_test['timestamp']   = pd.to_datetime(df_test['timestamp'])
df_test['hour']        = df_test['timestamp'].dt.hour
df_test['day_of_week'] = df_test['timestamp'].dt.dayofweek
df_test['month']       = df_test['timestamp'].dt.month
df_test['is_weekend']  = (df_test['day_of_week'] >= 5).astype(int)
df_test['is_peak']     = df_test['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

df_test['hour_sin']  = np.sin(2 * np.pi * df_test['hour'] / 24)
df_test['hour_cos']  = np.cos(2 * np.pi * df_test['hour'] / 24)
df_test['dow_sin']   = np.sin(2 * np.pi * df_test['day_of_week'] / 7)
df_test['dow_cos']   = np.cos(2 * np.pi * df_test['day_of_week'] / 7)
df_test['month_sin'] = np.sin(2 * np.pi * df_test['month'] / 12)
df_test['month_cos'] = np.cos(2 * np.pi * df_test['month'] / 12)

df_test[['lat', 'lon']] = df_test['geohash'].apply(safe_decode)
df_test['lat'].fillna(df_test['lat'].mean(), inplace=True)
df_test['lon'].fillna(df_test['lon'].mean(), inplace=True)

df_test['temp_x_peak']          = df_test['Temperature'] * df_test['is_peak']
df_test['lanes_x_highway']      = df_test['NumberofLanes'] * df_test.get('RoadType_Highway', 0)
df_test['temp_x_weather_rainy'] = df_test['Temperature'] * df_test.get('Weather_Rainy', 0)
df_test['temp_x_weather_snowy'] = df_test['Temperature'] * df_test.get('Weather_Snowy', 0)
df_test['temp_x_weather_foggy'] = df_test['Temperature'] * df_test.get('Weather_Foggy', 0)

# Geohash target encoding using train map
df_test['geohash_mean_demand'] = df_test['geohash'].map(gh_map).fillna(gh_fallback)

# ── Align columns with training ───────────────────────────────────────────────
for col in feature_cols:
    if col not in df_test.columns:
        df_test[col] = 0

X_test_final = df_test[feature_cols]

print("Test shape:", X_test_final.shape)
print("Nulls:", X_test_final.isnull().sum().sum())

# ── Predict ───────────────────────────────────────────────────────────────────
preds_log  = model.predict(X_test_final)
preds_orig = np.expm1(preds_log)

print(f"Pred range: {preds_orig.min():.4f} – {preds_orig.max():.4f}")

# ── Submission ────────────────────────────────────────────────────────────────
# Create submission with test Index and predictions
submission = pd.DataFrame({
    'Index': df_test['Index'].values,
    'demand': preds_orig
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved. Shape:", submission.shape)
print(submission.head())