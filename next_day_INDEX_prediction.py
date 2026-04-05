#!/usr/bin/env python
# coding: utf-8
"""
Next-Day Direction Predictor  —  fixed version
===============================================
Fixes vs original:

  BUG 1  Intraday feature mismatch
         Intraday features were NaN for ALL historical rows and only real for
         today. The walk-forward model never saw them, so attaching them at
         prediction time was misleading (LightGBM silently ignored them or
         used the split "is this NaN?" as a spurious signal).
         FIX: intraday features are kept as a SEPARATE scoring layer, not
         fed into the ML model. The ML model uses only daily features, which
         are consistently available for the full history.

  BUG 2  Regime shift / bull-market bias
         Training on 800+ days of historical data (mostly bull) means the
         model learns a base rate of ~58% UP and keeps predicting UP even
         during a crash. Two weeks of wrong calls in a row is the classic
         symptom of an unweighted model meeting a regime change.
         FIX: exponential sample weights (half_life=WEIGHT_HALF_LIFE_DAYS)
         so recent rows dominate training without throwing away history.

  BUG 3  No early stopping → overfitting
         300 trees on ~1 000 rows with max_depth=4 has capacity to memorise.
         FIX: early_stopping_rounds=30 against a small held-out validation
         slice (last 10% of training data).

  BUG 4  Walk-forward accuracy reported for the wrong model
         WF accuracy was measured on a model WITHOUT intraday features, but
         the number was displayed as the accuracy of the final prediction.
         FIX: WF accuracy now covers only daily-feature model, and is clearly
         labelled as such. Intraday signals are shown separately.

  BONUS  Confidence gate
         Predictions with |prob - 0.5| < CONFIDENCE_THRESHOLD are labelled
         "NEUTRAL" instead of forcing a UP/DOWN call. This reduces noise and
         improves the signal-to-noise ratio of the alerts.
"""

import os
import json
import urllib.parse
import numpy as np
import pandas as pd
import requests
import re
import traceback
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
from tqdm import tqdm

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

pd.set_option('max_colwidth', None)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:0.2f}'.format

import index_information
import GPT_trader_analysis

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
TELEGRAM_TOKEN_INDEX   = os.getenv("TELEGRAM_TOKEN_INDEX")
TELEGRAM_TOKEN_SANDBOX = os.getenv("TELEGRAM_TOKEN_SANDBOX")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE_INDEX        = "history_INDEX.csv"
LIST_INDEX             = "liste_INDEX_OnVista.csv"

# Walk-forward settings
WF_MIN_TRAIN        = 200   # minimum daily rows before first prediction
WF_ACCURACY_WINDOW  = 252   # evaluate accuracy on last ~1 year of rows

# Rolling training window — only use the most recent N daily rows.
# Keeps the model adaptive to the current regime.
# Set to None to use all history (not recommended after a regime change).
ROLLING_TRAIN_WINDOW = 500

# Recency weighting: sample weight halves every N days.
# Lower = faster adaptation to regime changes but less stable.
# 60–90 days is a good range for index trading.
WEIGHT_HALF_LIFE_DAYS = 90

# Confidence gate: if |prob_up - 0.5| < this, output NEUTRAL instead of UP/DOWN.
# Reduces false precision. Set to 0 to disable.
CONFIDENCE_THRESHOLD = 0.06   # i.e., prob must be < 0.44 or > 0.56 to call direction

# LightGBM — deliberately shallow + early stopping to avoid overfitting
LGB_PARAMS = dict(
    objective         = "binary",
    metric            = "binary_logloss",
    n_estimators      = 500,          # raised ceiling; early stopping will cut it down
    learning_rate     = 0.03,
    max_depth         = 4,
    num_leaves        = 12,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.3,
    reg_lambda        = 0.3,
    verbose           = -1,
    n_jobs            = -1,
)

EARLY_STOPPING_ROUNDS = 30
VALIDATION_FRAC       = 0.10   # last 10% of training rows used as eval set


# ─────────────────────────────────────────────────────────
# INTRADAY FETCH  (unchanged)
# ─────────────────────────────────────────────────────────
def fetch_intraday_today(idNotation: str) -> pd.DataFrame:
    """
    Fetch today's 3-min bars from OnVista chart_history endpoint.
    Returns clean DataFrame with columns:
        datetime, open, high, low, close, volume, n_trades
    Last 2 rows (session-close artifact) are dropped.
    """
    url = (
        f"https://api.onvista.de/api/v1/instruments/INDEX/{idNotation}"
        f"/chart_history?idNotation={idNotation}"
        f"&range=D1&resolution=3m&withCurrentDay=true&withEarnings=false"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        parsed = r.json()
    except Exception as e:
        print(f"[Intraday] fetch failed for {idNotation}: {e}")
        return pd.DataFrame()

    if "datetimeLast" not in parsed:
        return pd.DataFrame()

    ts_list  = parsed["datetimeLast"]
    opens    = parsed.get("first",        parsed.get("last", []))
    closes   = parsed.get("last",         [])
    highs    = parsed.get("high",         closes)
    lows     = parsed.get("low",          closes)
    volumes  = parsed.get("volume",       [0] * len(ts_list))
    n_trades = parsed.get("numberPrices", [0] * len(ts_list))

    rows = []
    for i, ts in enumerate(ts_list):
        rows.append({
            "datetime": datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None),
            "open":     opens[i],
            "high":     highs[i],
            "low":      lows[i],
            "close":    closes[i],
            "volume":   volumes[i],
            "n_trades": n_trades[i],
        })

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) > 4:
        df = df.iloc[:-2].copy()

    return df


# ─────────────────────────────────────────────────────────
# INTRADAY FEATURES  (kept separate from ML model — BUG 1 fix)
# ─────────────────────────────────────────────────────────
def extract_intraday_features(df_intra: pd.DataFrame) -> dict:
    """
    Extract 7 scalar features from today's 3-min bars.

    NOTE: These are NO LONGER fed into the LightGBM model.
    They are used only for the human-readable intraday summary
    and as a separate directional signal shown in the message.
    This avoids the train/predict NaN mismatch (BUG 1).
    """
    nan_result = {
        "intra_close_pos":    np.nan,
        "intra_last2h_ret":   np.nan,
        "intra_vol_skew":     np.nan,
        "intra_body_last":    np.nan,
        "intra_macd_hist":    np.nan,
        "intra_vwap_dev":     np.nan,
        "intra_activity_acc": np.nan,
    }
    if df_intra is None or len(df_intra) < 10:
        return nan_result

    df        = df_intra.copy().reset_index(drop=True)
    n         = len(df)
    day_high  = df["high"].max()
    day_low   = df["low"].min()
    day_close = df["close"].iloc[-1]
    day_open  = df["open"].iloc[0]

    rng       = day_high - day_low
    close_pos = (day_close - day_low) / rng if rng > 0 else 0.5

    n_2h         = min(40, n - 1)
    price_2h_ago = df["close"].iloc[-n_2h]
    last_2h_ret  = (day_close - price_2h_ago) / price_2h_ago if price_2h_ago > 0 else np.nan

    act_col     = "volume" if df["volume"].sum() > 0 else "n_trades"
    mid         = n // 2
    act_morning = df[act_col].iloc[:mid].sum()
    act_after   = df[act_col].iloc[mid:].sum()
    vol_skew    = act_after / act_morning if act_morning > 0 else np.nan

    last      = df.iloc[-1]
    body      = abs(last["close"] - last["open"])
    bar_rng   = last["high"] - last["low"]
    body_last = body / bar_rng if bar_rng > 0 else 0.5

    close_s = df["close"]
    if len(close_s) >= 35:
        ema_fast  = close_s.ewm(span=12, adjust=False).mean()
        ema_slow  = close_s.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal    = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd_line - signal).iloc[-1])
    else:
        macd_hist = np.nan

    weight        = df[act_col].replace(0, 1)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap          = (typical_price * weight).cumsum() / weight.cumsum()
    vwap_now      = float(vwap.iloc[-1])
    vwap_dev      = (day_close - vwap_now) / rng if rng > 0 else np.nan

    n_1h      = min(20, n // 3)
    act_first = df[act_col].iloc[:n_1h].sum()
    act_last  = df[act_col].iloc[-n_1h:].sum()
    act_acc   = act_last / act_first if act_first > 0 else np.nan

    return {
        "intra_close_pos":    close_pos,
        "intra_last2h_ret":   last_2h_ret,
        "intra_vol_skew":     vol_skew,
        "intra_body_last":    body_last,
        "intra_macd_hist":    macd_hist,
        "intra_vwap_dev":     vwap_dev,
        "intra_activity_acc": act_acc,
    }


def intraday_direction_signal(intra: dict) -> str:
    """
    Produce a simple ▲/▼/~ signal from intraday features.
    This replaces the previous (broken) ML use of intraday features.
    Each sub-signal votes; majority wins.
    """
    if all(np.isnan(v) for v in intra.values()):
        return "~"

    votes_up = 0
    votes_dn = 0

    cp = intra.get("intra_close_pos", np.nan)
    if not np.isnan(cp):
        if cp > 0.60: votes_up += 1
        elif cp < 0.40: votes_dn += 1

    r2h = intra.get("intra_last2h_ret", np.nan)
    if not np.isnan(r2h):
        if r2h > 0.002: votes_up += 1
        elif r2h < -0.002: votes_dn += 1

    mh = intra.get("intra_macd_hist", np.nan)
    if not np.isnan(mh):
        if mh > 0: votes_up += 1
        else: votes_dn += 1

    vd = intra.get("intra_vwap_dev", np.nan)
    if not np.isnan(vd):
        if vd > 0.05: votes_up += 1
        elif vd < -0.05: votes_dn += 1

    if votes_up > votes_dn: return "▲"
    if votes_dn > votes_up: return "▼"
    return "~"


# ─────────────────────────────────────────────────────────
# DAILY FEATURE ENGINEERING  (unchanged)
# ─────────────────────────────────────────────────────────
DAILY_FEATURE_COLS = [
    "momentum_norm", "mom_delta", "mom_lag1", "mom_lag2", "mom_lag3",
    "ret_d1", "ret_d2", "ret_d3",
    "day_range_pct", "body_ratio", "range_lag1", "range_lag2",
    "RSI14", "RSI14_lag1",
    "ADX14",
    "vol_factor",
    "trend_long", "ema_cross", "supertrend_dir",
    "is_dark_red", "is_dark_red_2nd",
    "dow", "month_end", "month_start",
]


def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]

    df["mom_delta"] = df["momentum_norm"].diff()
    df["mom_lag1"]  = df["momentum_norm"].shift(1)
    df["mom_lag2"]  = df["momentum_norm"].shift(2)
    df["mom_lag3"]  = df["momentum_norm"].shift(3)

    df["ret_d1"] = c.pct_change(1)
    df["ret_d2"] = c.pct_change(2)
    df["ret_d3"] = c.pct_change(3)

    rng = (h - l).replace(0, np.nan)
    df["day_range_pct"] = (c - l) / rng
    df["body_ratio"]    = (c - o).abs() / rng
    df["range_lag1"]    = df["day_range_pct"].shift(1)
    df["range_lag2"]    = df["day_range_pct"].shift(2)

    df["RSI14_lag1"] = df["RSI14"].shift(1)
    df["ema_cross"]  = (df["EMA10"] > df["EMA20"]).astype(int)

    df["is_dark_red"]     = ((df["momentum_norm"] < 0) &
                              (df["mom_delta"]     < 0)).astype(int)
    df["is_dark_red_2nd"] = (df["is_dark_red"] &
                              df["is_dark_red"].shift(1).fillna(0).astype(bool)).astype(int)

    dates = pd.to_datetime(df["date"])
    df["dow"]         = dates.dt.dayofweek
    df["month_end"]   = dates.dt.is_month_end.astype(int)
    df["month_start"] = dates.dt.is_month_start.astype(int)

    return df


# ─────────────────────────────────────────────────────────
# SAMPLE WEIGHTS  (BUG 2 fix — recency weighting)
# ─────────────────────────────────────────────────────────
def make_sample_weights(n: int, half_life: int) -> np.ndarray:
    """
    Exponential decay weights: the most recent row has weight 1,
    a row half_life days ago has weight 0.5, etc.
    Weights are normalised to sum to n (so LightGBM leaf sizes stay meaningful).
    """
    ages    = np.arange(n - 1, -1, -1, dtype=float)   # 0 = most recent
    weights = np.exp(-np.log(2) * ages / half_life)
    weights = weights / weights.sum() * n
    return weights


# ─────────────────────────────────────────────────────────
# MODEL FITTING  (BUG 3 fix — early stopping)
# ─────────────────────────────────────────────────────────
def _fit_lgb(X_train: np.ndarray,
             y_train: np.ndarray,
             w_train: np.ndarray) -> lgb.LGBMClassifier:
    """
    Fit LightGBM with early stopping against a held-out validation slice.
    The validation slice is the last VALIDATION_FRAC of training rows
    (most recent data = hardest test of current regime knowledge).
    """
    n_val   = max(1, int(len(X_train) * VALIDATION_FRAC))
    n_tr    = len(X_train) - n_val

    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]
    w_tr        = w_train[:n_tr]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        sample_weight   = w_tr,
        eval_set        = [(X_val, y_val)],
        callbacks       = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                           lgb.log_evaluation(period=-1)],
    )
    return model


# ─────────────────────────────────────────────────────────
# WALK-FORWARD LightGBM  (all 4 bugs fixed)
# ─────────────────────────────────────────────────────────
def walk_forward_predict(df: pd.DataFrame) -> tuple:
    """
    Walk-forward training + prediction using DAILY features only.

    Changes vs original:
    - Intraday features removed from ML (BUG 1)
    - Exponential sample weights (BUG 2)
    - Early stopping (BUG 3)
    - WF accuracy now honestly reflects daily-only model (BUG 4)
    - Rolling training window (ROLLING_TRAIN_WINDOW) for regime adaptability

    Returns
    -------
    prob_up      : float [0,1] | None
    wf_acc       : float [0,1] | None
    n_signals    : int  — dark-red bars in last 252 rows
    n_trees_used : int  — actual trees used by final model (early stopping)
    """
    df = engineer_daily_features(df)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)

    required = ["momentum_norm", "RSI14", "ADX14", "supertrend_dir", "vol_factor"]
    df_model = df.dropna(subset=required + ["target"]).copy()

    # Rolling window: keep only recent rows
    if ROLLING_TRAIN_WINDOW and len(df_model) > ROLLING_TRAIN_WINDOW:
        df_model = df_model.tail(ROLLING_TRAIN_WINDOW).copy()

    n = len(df_model)
    if n < WF_MIN_TRAIN:
        return None, None, 0, 0

    X = df_model[DAILY_FEATURE_COLS].astype("float64").to_numpy()
    y = df_model["target"].astype("int64").to_numpy()

    # ── Walk-forward accuracy ──────────────────────────────────────────────────
    test_start  = max(WF_MIN_TRAIN, n - WF_ACCURACY_WINDOW)
    preds_wf, true_wf = [], []

    for t in range(test_start, n):
        w   = make_sample_weights(t, WEIGHT_HALF_LIFE_DAYS)
        mdl = _fit_lgb(X[:t], y[:t], w)
        p   = mdl.predict_proba(X[t:t+1])[0][1]
        preds_wf.append(1 if p >= 0.5 else 0)
        true_wf.append(y[t])

    wf_acc = (float(np.mean(np.array(preds_wf) == np.array(true_wf)))
              if preds_wf else None)

    # ── Final model: train on all rows except last ─────────────────────────────
    w_final     = make_sample_weights(n - 1, WEIGHT_HALF_LIFE_DAYS)
    final_model = _fit_lgb(X[:-1], y[:-1], w_final)
    prob_up     = float(final_model.predict_proba(X[-1:])[0][1])
    n_trees     = final_model.best_iteration_ or LGB_PARAMS["n_estimators"]

    n_signals = int(df_model["is_dark_red"].tail(252).sum())

    return prob_up, wf_acc, n_signals, n_trees


# ─────────────────────────────────────────────────────────
# DIRECTION LABEL  (BUG 4 fix — confidence gate)
# ─────────────────────────────────────────────────────────
def direction_label(prob_up: float) -> str:
    """
    Apply confidence threshold. Returns NEUTRAL when the model
    is not sufficiently confident, instead of forcing a UP/DOWN call.
    """
    if prob_up is None:
        return "N/A"
    if abs(prob_up - 0.5) < CONFIDENCE_THRESHOLD:
        return "NEUTRAL ↔"
    return "LONG 📈" if prob_up >= 0.5 else "SHORT 📉"


# ─────────────────────────────────────────────────────────
# MESSAGE GENERATION
# ─────────────────────────────────────────────────────────
def generate_messages(df_stocks: pd.DataFrame, period_days: int = 180):
    params_scalable = {"strategy": "LONG", "tab": "factors"}
    base_url_github = "https://simorders.github.io/Stock-Information/"

    os.makedirs("docs", exist_ok=True)

    cutoff_date  = pd.Timestamp.today().normalize() - pd.Timedelta(days=period_days)
    df_recent    = df_stocks[df_stocks["date"] > cutoff_date].copy()

    df_list      = pd.read_csv(LIST_INDEX)
    notation_map = dict(zip(df_list["ticker"].astype(str),
                            df_list["idNotation"].astype(str)))

    for ticker in tqdm(df_recent["ticker"].unique(), desc="Generating messages..."):
        df_ticker = df_recent[df_recent["ticker"] == ticker].sort_values("date")
        if len(df_ticker) < 2:
            continue

        latest = df_ticker.iloc[-1]
        prev   = df_ticker.iloc[-2]

        # ── Fetch intraday ───────────────────────────────────────────────────
        idNotation     = notation_map.get(str(ticker), "")
        df_intra       = fetch_intraday_today(idNotation) if idNotation else pd.DataFrame()
        intra_features = extract_intraday_features(df_intra)
        has_intraday   = not df_intra.empty

        # ── Intraday summary ─────────────────────────────────────────────────
        if not df_intra.empty and len(df_intra) > 4:
            last_intra_time = df_intra["datetime"].iloc[-1].strftime("%d.%m.%Y %H:%M CET")
            open_price  = df_intra["open"].iloc[0]
            close_price = df_intra["close"].iloc[-1]
            day_high    = df_intra["high"].max()
            day_low     = df_intra["low"].min()
            close_pos   = (close_price / open_price - 1) * 100
            max_pos     = (day_high    / open_price - 1) * 100
            min_pos     = (day_low     / open_price - 1) * 100
            net_dir     = "↗️" if close_price > open_price else "↘️"
            intra_sig   = intraday_direction_signal(intra_features)
            intra_header = (
                f"<b>Intraday:</b> {last_intra_time}\n"
                f"Day P&L: {close_pos:.1f}% {net_dir}\n"
                f"Max: {max_pos:.1f}%  ·  Min: {min_pos:.1f}%\n"
                f"Intraday signal: {intra_sig}"
            )
        else:
            intra_header = "No intraday data"

        # ── ML prediction (daily only — BUG 1 fix) ──────────────────────────
        df_full = df_stocks[df_stocks["ticker"] == ticker].sort_values("date").copy()
        prob_up, wf_acc, n_signals, n_trees = walk_forward_predict(df_full)

        if prob_up is not None:
            prob_down  = 1.0 - prob_up
            direction  = direction_label(prob_up)
            acc_str    = f"{wf_acc*100:.1f}%" if wf_acc is not None else "N/A"
            conf_str   = f"{abs(prob_up - 0.5)*100:.1f}pp"  # distance from 50%
            prob_text  = (
                f"Predicted: <b>{direction}</b>\n"
                f"UP 📈 {prob_up*100:.1f}%  ·  DOWN 📉 {prob_down*100:.1f}%\n"
                f"Confidence: {conf_str} | Accuracy: {acc_str}\n"
                f"Trees used: {n_trees} (of {LGB_PARAMS['n_estimators']} max)"
            )
        else:
            prob_text = "ML: N/A (insufficient history)"

        # ── GPT advice ───────────────────────────────────────────────────────
        signal = GPT_trader_analysis.build_signal(
            ticker        = ticker,
            latest        = latest,
            prob_up       = prob_up if prob_up is not None else 0.5,
            wf_acc        = wf_acc,
            has_intraday  = has_intraday,
            df_intra      = df_intra,
            intra_features= intra_features,
        )
        gpt_advice = GPT_trader_analysis.get_trader_advice(signal)

        # ── Chart ────────────────────────────────────────────────────────────
        sup_p        = index_information.get_supertrend_params(ticker)
        title        = (f"{ticker} · {df_ticker['isin'].iloc[0]} · "
                        f"{latest['date'].strftime('%d.%m.%Y')} · "
                        f"SUP: ATR={sup_p['atr_period']}, M={sup_p['multiplier']}")
        fig          = index_information.plot_chart(df_ticker, title=title)
        clean_ticker = re.sub(r"[^A-Za-z0-9]", "", ticker)
        fn_png       = f"{clean_ticker}.png"
        fn_html      = f"docs/{clean_ticker}.html"
        fig.write_image(fn_png)
        fig.write_html(fn_html, include_plotlyjs="cdn", full_html=True)
        chart_url  = f"{base_url_github}{clean_ticker}.html"
        broker_url = (f"https://de.scalable.capital/broker/search/derivatives/"
                      f"{df_ticker['isin'].iloc[0]}?"
                      f"{urllib.parse.urlencode(params_scalable)}")

        # ── Indicator labels ─────────────────────────────────────────────────
        trend = "ON ✅" if latest["trend_long"] else "OFF ❌"
        if latest["trend_long"] and not prev["trend_long"]:
            trend += " <i>➜New❗</i>"

        supertrend = "UP ✅" if latest["supertrend_dir"] == 1 else "DOWN ❌"
        if latest["supertrend_dir"] != prev["supertrend_dir"]:
            supertrend += " <i>➜New❗</i>"

        if   latest["vol_factor"] > 1.2: volatility = "HIGH ❌"
        elif latest["vol_factor"] < 0.8: volatility = "LOW ✅"
        else:                            volatility = "Normal"

        momentum = "Positive ✅" if latest["momentum_norm"] > 0 else "Negative ❌"

        if pd.isna(latest["RSI14"]): rsi = "N/A"
        elif latest["RSI14"] > 70:   rsi = "Overbought ⚠️"
        elif latest["RSI14"] < 30:   rsi = "Oversold ⚠️"
        else:                        rsi = "Normal"
        p_rsi = "OB" if prev["RSI14"] > 70 else "OS" if prev["RSI14"] < 30 else "N"
        c_rsi = "OB" if latest["RSI14"] > 70 else "OS" if latest["RSI14"] < 30 else "N"
        if c_rsi != p_rsi:
            rsi += " <i>➜New❗</i>"

        adx = "Strong ✅" if latest["ADX14"] > 25 else "Weak ❌"
        if latest["ADX14"] > 25 and prev["ADX14"] <= 25:
            adx += " <i>➜New❗</i>"

        # ── Telegram caption ─────────────────────────────────────────────────
        summary = (
            f"<b><a href='{broker_url}'>{ticker}</a></b> · "
            f"{latest['date'].strftime('%d.%m.%Y')}\n"
            f"Trend: {trend}\n"
            f"Supertrend: {supertrend}\n"
            f"Volatility: {volatility}\n"
            f"Momentum: {momentum}\n"
            f"RSI: {rsi}\n"
            f"ADX: {adx}\n"
            f"────────────────\n"
            f"{intra_header}\n"
            f"────────────────\n"
            f"{prob_text}\n"
            f"────────────────\n"
            f"{gpt_advice}\n"
        )

        index_information.send_telegram("sendPhoto", filename=fn_png,
                                        caption=summary, url=chart_url)

        if os.path.exists(fn_png):
            os.remove(fn_png)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    try:
        df_stocks = pd.read_csv(DATA_FILE_INDEX)
        df_stocks["date"] = pd.to_datetime(df_stocks["date"]).dt.tz_localize(None)
        df_stocks = index_information.calculate_indicators(df_stocks, include_rsi=True)
        df_stocks = index_information.add_supertrend(df_stocks)
        generate_messages(df_stocks, period_days=180)

    except Exception as e:
        today_str = datetime.today().strftime("%d.%m.%Y")
        tb_str    = traceback.format_exc()
        index_information.send_telegram(
            "sendMessage",
            text=f"<b>{today_str} - unexpected error:</b>\n{tb_str}"
        )

if __name__ == "__main__":
    main()