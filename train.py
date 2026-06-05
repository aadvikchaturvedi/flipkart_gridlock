"""
train.py — Full ML Pipeline + saves XGBoost model artifacts for inference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

import pygeohash as pgh

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ── Load ──────────────────────────────────────────────────────────────────────
encoded_df = pd.read_csv('encoded_df.csv')

# ── Feature Engineering ───────────────────────────────────────────────────────
fe_df = encoded_df.copy()

fe_df['timestamp']   = pd.to_datetime(fe_df['timestamp'])
fe_df['hour']        = fe_df['timestamp'].dt.hour
fe_df['day_of_week'] = fe_df['timestamp'].dt.dayofweek
fe_df['month']       = fe_df['timestamp'].dt.month
fe_df['is_weekend']  = (fe_df['day_of_week'] >= 5).astype(int)
fe_df['is_peak']     = fe_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

fe_df['hour_sin']  = np.sin(2 * np.pi * fe_df['hour'] / 24)
fe_df['hour_cos']  = np.cos(2 * np.pi * fe_df['hour'] / 24)
fe_df['dow_sin']   = np.sin(2 * np.pi * fe_df['day_of_week'] / 7)
fe_df['dow_cos']   = np.cos(2 * np.pi * fe_df['day_of_week'] / 7)
fe_df['month_sin'] = np.sin(2 * np.pi * fe_df['month'] / 12)
fe_df['month_cos'] = np.cos(2 * np.pi * fe_df['month'] / 12)

def safe_decode(gh):
    try:
        lat, lon = pgh.decode(gh)
        return pd.Series([lat, lon])
    except Exception:
        return pd.Series([np.nan, np.nan])

fe_df[['lat', 'lon']] = fe_df['geohash'].apply(safe_decode)
fe_df['lat'].fillna(fe_df['lat'].mean(), inplace=True)
fe_df['lon'].fillna(fe_df['lon'].mean(), inplace=True)

fe_df['temp_x_peak']          = fe_df['Temperature'] * fe_df['is_peak']
fe_df['lanes_x_highway']      = fe_df['NumberofLanes'] * fe_df['RoadType_Highway']
fe_df['temp_x_weather_rainy'] = fe_df['Temperature'] * fe_df['Weather_Rainy']
fe_df['temp_x_weather_snowy'] = fe_df['Temperature'] * fe_df['Weather_Snowy']
fe_df['temp_x_weather_foggy'] = fe_df['Temperature'] * fe_df['Weather_Foggy']

fe_df['geohash_mean_demand'] = np.nan  # filled post-split

print("Feature engineering done. Shape:", fe_df.shape)

# ── Train/Test Split + Target Encoding ───────────────────────────────────────
drop_cols    = ['geohash', 'timestamp', 'demand']
feature_cols = [c for c in fe_df.columns if c not in drop_cols]

X = fe_df[feature_cols]
y = np.log1p(fe_df['demand'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_idx = X_train.index
test_idx  = X_test.index

# Build geohash map from train only — save for test.py
gh_map = fe_df.loc[train_idx].groupby('geohash')['demand'].mean()
gh_fallback = fe_df.loc[train_idx, 'demand'].mean()

X_train = X_train.copy()
X_test  = X_test.copy()
X_train['geohash_mean_demand'] = fe_df.loc[train_idx, 'geohash'].map(gh_map).values
X_test['geohash_mean_demand']  = fe_df.loc[test_idx, 'geohash'].map(gh_map).fillna(gh_fallback).values

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Ridge":         Ridge(alpha=1.0),
    "Lasso":         Lasso(alpha=0.001, max_iter=5000),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42),
    "KNN":           KNeighborsRegressor(n_neighbors=10, weights='distance'),
    "SVM":           SVR(kernel='rbf', C=10, epsilon=0.05),
    "ANN":           MLPRegressor(
                         hidden_layer_sizes=(256, 128, 64),
                         activation='relu', max_iter=500,
                         learning_rate_init=0.001,
                         early_stopping=True, random_state=42
                     ),
    "XGBoost":       XGBRegressor(
                         n_estimators=800, learning_rate=0.05,
                         max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5,
                         reg_alpha=0.1, reg_lambda=1.0,
                         random_state=42, tree_method='hist'
                     ),
}

# ── Train & Evaluate ──────────────────────────────────────────────────────────
SCALED  = {"Ridge", "Lasso", "KNN", "SVM", "ANN"}
results = []

for name, model in models.items():
    Xtr, Xte = (X_train_sc, X_test_sc) if name in SCALED else (X_train, X_test)
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    preds_orig  = np.expm1(preds)
    y_test_orig = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae  = mean_absolute_error(y_test_orig, preds_orig)
    r2   = r2_score(y_test, preds)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
    print(f"{name:<15}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

results_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
print("\n", results_df.to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
palette   = sns.color_palette("viridis", len(results_df))
metrics   = ["R²", "RMSE", "MAE"]
ascending = [False, True, True]

for ax, metric, asc in zip(axes, metrics, ascending):
    sdf  = results_df.sort_values(metric, ascending=asc)
    bars = ax.barh(sdf["Model"], sdf[metric], color=palette)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.invert_yaxis()

plt.suptitle("Model Comparison — Demand Forecasting (with Feature Engineering)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("model_comparison_fe.png", dpi=150, bbox_inches='tight')
plt.show()

# XGBoost feature importance
xgb_model   = models["XGBoost"]
importances = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color=sns.color_palette("viridis", 20))
plt.gca().invert_yaxis()
plt.title("XGBoost — Top 20 Feature Importances", fontweight='bold')
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

# Residuals
Xte_use    = X_test
best_preds = np.expm1(xgb_model.predict(Xte_use))
residuals  = np.expm1(y_test.values) - best_preds

plt.figure(figsize=(8, 4))
plt.scatter(best_preds, residuals, alpha=0.3, s=5)
plt.axhline(0, color='red', lw=1.5)
plt.xlabel("Predicted Demand")
plt.ylabel("Residual")
plt.title("Residuals — XGBoost", fontweight='bold')
plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Retrain XGBoost on FULL data + save artifacts ────────────────────────────
print("\nRetraining XGBoost on full dataset...")

X_full = fe_df[feature_cols].copy()
y_full = np.log1p(fe_df['demand'])

gh_map_full     = fe_df.groupby('geohash')['demand'].mean()
gh_fallback_full = fe_df['demand'].mean()
X_full['geohash_mean_demand'] = fe_df['geohash'].map(gh_map_full).values

final_model = XGBRegressor(
    n_estimators=800, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, tree_method='hist'
)
final_model.fit(X_full, y_full)

# Save artifacts
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('gh_map.pkl', 'wb') as f:
    pickle.dump({'gh_map': gh_map_full, 'fallback': gh_fallback_full}, f)

with open('feature_cols.pkl', 'wb') as f:
    pickle.dump(X_full.columns.tolist(), f)

print("Saved: xgb_model.pkl, gh_map.pkl, feature_cols.pkl")