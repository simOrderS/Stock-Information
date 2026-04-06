#!/usr/bin/env python
# coding: utf-8
"""
Run this manually to inspect which features are driving the model.
Usage: python shap_analysis.py
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import everything from your main script
from next_day_INDEX_prediction import (
    engineer_daily_features,
    make_sample_weights,
    _fit_lgb,
    DAILY_FEATURE_COLS,
    ROLLING_TRAIN_WINDOW,
    WEIGHT_HALF_LIFE_DAYS,
    WF_MIN_TRAIN,
    DATA_FILE_INDEX,
)
import index_information

# ── Load and prepare data (same as main()) ────────────────────────────────────
df_stocks = pd.read_csv(DATA_FILE_INDEX)
df_stocks["date"] = pd.to_datetime(df_stocks["date"]).dt.tz_localize(None)
df_stocks = index_information.calculate_indicators(df_stocks, include_rsi=True)
df_stocks = index_information.add_supertrend(df_stocks)

# ── Pick which ticker to analyse ──────────────────────────────────────────────
TICKER = "DAX"   # ← change this to any ticker in your data

df = df_stocks[df_stocks["ticker"] == TICKER].sort_values("date").copy()
df = engineer_daily_features(df)

# ATR-based threshold: scales with recent volatility instead of fixed 0.4%
atr_pct   = (df["high"].rolling(14).max() - df["low"].rolling(14).min()) / df["close"]
threshold = (atr_pct.shift(1) * 0.4).fillna(0.004)

ret = df["close"].shift(-1) / df["close"] - 1
df["target"] = np.where(
    ret >  threshold, 1,
    np.where(ret < -threshold, 0, np.nan)
)

required = ["momentum_norm", "RSI14", "ADX14", "supertrend_dir", "vol_factor"]
df_model = df.dropna(subset=required + ["target"]).copy()

if ROLLING_TRAIN_WINDOW and len(df_model) > ROLLING_TRAIN_WINDOW:
    df_model = df_model.tail(ROLLING_TRAIN_WINDOW).copy()

if len(df_model) < WF_MIN_TRAIN:
    print(f"Not enough data for {TICKER} ({len(df_model)} rows)")
    exit()

X = df_model[DAILY_FEATURE_COLS].astype("float64").to_numpy()
y = df_model["target"].astype("int64").to_numpy()

# ── Fit the final model on all data except the last row ───────────────────────
w           = make_sample_weights(len(X) - 1, WEIGHT_HALF_LIFE_DAYS)
final_model = _fit_lgb(X[:-1], y[:-1], w)

# ── SHAP ──────────────────────────────────────────────────────────────────────
explainer  = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# If shap_values is a list (binary classification), take the positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

X_df = pd.DataFrame(X, columns=DAILY_FEATURE_COLS)

# ── Plot 1: overall feature importance (mean absolute SHAP) ───────────────────
shap.summary_plot(
    shap_values, X_df,
    plot_type = "bar",
    show      = False,
)
plt.title(f"{TICKER} — Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.savefig(f"shap_importance_{TICKER}.png", dpi=150)
plt.close()
print(f"Saved: shap_importance_{TICKER}.png")

# ── Plot 2: dot plot showing direction of each feature's effect ───────────────
shap.summary_plot(
    shap_values, X_df,
    show = False,
)
plt.title(f"{TICKER} — Feature Effects")
plt.tight_layout()
plt.savefig(f"shap_effects_{TICKER}.png", dpi=150)
plt.close()
print(f"Saved: shap_effects_{TICKER}.png")

# ── Print a simple table so you can read it without opening the images ─────────
import numpy as np
importance = pd.DataFrame({
    "feature":    DAILY_FEATURE_COLS,
    "mean_shap":  np.abs(shap_values).mean(axis=0),
}).sort_values("mean_shap", ascending=False)

print(f"\nFeature importance for {TICKER}:")
print(importance.to_string(index=False))
print(f"\nFeatures with near-zero importance (candidates to remove):")
print(importance[importance["mean_shap"] < 0.005].to_string(index=False))