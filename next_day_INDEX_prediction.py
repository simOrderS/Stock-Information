#!/usr/bin/env python
# coding: utf-8
"""
Next-Day Direction Predictor
============================
Drop-in replacement for the logistic regression block in the existing script.
Keeps all original functions (setup_database, plot_chart, send_telegram, etc.)
and replaces compute_logistic_probability() with a full walk-forward LightGBM
pipeline that also uses today's intraday 3-min bars.

Run at ~17:35 CET (after DAX close).

New dependency:
    pip install lightgbm
Everything else is already in the original requirements.
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

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
TELEGRAM_TOKEN_INDEX   = os.getenv("TELEGRAM_TOKEN_INDEX")
TELEGRAM_TOKEN_SANDBOX = os.getenv("TELEGRAM_TOKEN_SANDBOX")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE_INDEX        = "history_INDEX.csv"
LIST_INDEX             = "liste_INDEX_OnVista.csv"
#DATA_FILE_INDEX        = "history_DAX40.csv"
#LIST_INDEX             = "liste_DAX40_OnVista.csv"

# Walk-forward settings
WF_MIN_TRAIN        = 200   # minimum daily rows before first prediction
WF_ACCURACY_WINDOW  = 252   # evaluate accuracy on last ~1 year of rows

# LightGBM — deliberately shallow to avoid overfitting on ~1 000 rows per ticker
LGB_PARAMS = dict(
    objective         = "binary",
    metric            = "binary_logloss",
    n_estimators      = 300,
    learning_rate     = 0.03,
    max_depth         = 4,
    num_leaves        = 12,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.2,
    reg_lambda        = 0.2,
    verbose           = -1,
    n_jobs            = -1,
)


# ─────────────────────────────────────────────────────────
# INTRADAY FETCH
# ─────────────────────────────────────────────────────────
def fetch_intraday_today(idNotation: str) -> pd.DataFrame:
    """
    Fetch today's 3-min bars from OnVista chart_history endpoint.
    URL pattern:
        /instruments/INDEX/{idNotation}/chart_history
        ?idNotation={idNotation}&range=D1&resolution=3m
        &withCurrentDay=true&withEarnings=false

    Returns clean DataFrame with columns:
        datetime, open, high, low, close, volume, n_trades
    Last 2 rows (session-close artifact) are dropped.
    """
    url = (
        f"https://api.onvista.de/api/v1/instruments/STOCK/{idNotation}"
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

    # Drop last 2 rows — OnVista session-close artifact (tiny partial bars)
    if len(df) > 4:
        df = df.iloc[:-2].copy()

    return df


# ─────────────────────────────────────────────────────────
# INTRADAY FEATURES
# ─────────────────────────────────────────────────────────
def extract_intraday_features(df_intra: pd.DataFrame) -> dict:
    """
    Extract 7 scalar features from today's 3-min bars.
    Returns NaN for all features if data is unavailable.

    Features
    --------
    intra_close_pos    : close position in day range [0=bottom, 1=top]
    intra_last2h_ret   : price return over last 2 hours of session
    intra_vol_skew     : afternoon / morning activity (>1 = late acceleration)
    intra_body_last    : body / range of last bar (conviction of final move)
    intra_macd_hist    : 3-min MACD histogram value at session close
    intra_vwap_dev     : (close - VWAP) / day-range — positive = above VWAP
    intra_activity_acc : last-hour activity / first-hour activity
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

    # 1. Close position in day range  [0, 1]
    rng       = day_high - day_low
    close_pos = (day_close - day_low) / rng if rng > 0 else 0.5

    # 2. Last-2h return  (~40 bars × 3 min = 120 min)
    n_2h         = min(40, n - 1)
    price_2h_ago = df["close"].iloc[-n_2h]
    last_2h_ret  = ((day_close - price_2h_ago) / price_2h_ago
                    if price_2h_ago > 0 else np.nan)

    # 3. Activity skew: use real volume for indices that have it, else n_trades
    act_col     = "volume" if df["volume"].sum() > 0 else "n_trades"
    mid         = n // 2
    act_morning = df[act_col].iloc[:mid].sum()
    act_after   = df[act_col].iloc[mid:].sum()
    vol_skew    = act_after / act_morning if act_morning > 0 else np.nan

    # 4. Last bar body / range
    last      = df.iloc[-1]
    body      = abs(last["close"] - last["open"])
    bar_rng   = last["high"] - last["low"]
    body_last = body / bar_rng if bar_rng > 0 else 0.5

    # 5. Intraday MACD histogram on 3-min close series
    close_s = df["close"]
    if len(close_s) >= 35:
        ema_fast  = close_s.ewm(span=12, adjust=False).mean()
        ema_slow  = close_s.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal    = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd_line - signal).iloc[-1])
    else:
        macd_hist = np.nan

    # 6. VWAP deviation normalised by day range
    weight        = df[act_col].replace(0, 1)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap          = (typical_price * weight).cumsum() / weight.cumsum()
    vwap_now      = float(vwap.iloc[-1])
    vwap_dev      = (day_close - vwap_now) / rng if rng > 0 else np.nan

    # 7. Activity acceleration: last-1h vs first-1h  (~20 bars each)
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


# ─────────────────────────────────────────────────────────
# DAILY FEATURE ENGINEERING
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

INTRA_FEATURE_COLS = [
    "intra_close_pos", "intra_last2h_ret", "intra_vol_skew",
    "intra_body_last", "intra_macd_hist", "intra_vwap_dev",
    "intra_activity_acc",
]

ALL_FEATURE_COLS = DAILY_FEATURE_COLS + INTRA_FEATURE_COLS


def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all lagged/structural daily features. Strictly no lookahead."""
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
    df["day_range_pct"] = (c - l) / rng      # 0 = close at low, 1 = close at high
    df["body_ratio"]    = (c - o).abs() / rng
    df["range_lag1"]    = df["day_range_pct"].shift(1)
    df["range_lag2"]    = df["day_range_pct"].shift(2)

    df["RSI14_lag1"] = df["RSI14"].shift(1)
    df["ema_cross"]  = (df["EMA10"] > df["EMA20"]).astype(int)

    # Dark-red bar: momentum negative AND falling
    df["is_dark_red"]     = ((df["momentum_norm"] < 0) &
                              (df["mom_delta"]     < 0)).astype(int)
    # Second consecutive dark-red
    df["is_dark_red_2nd"] = (df["is_dark_red"] &
                              df["is_dark_red"].shift(1).fillna(0).astype(bool)).astype(int)

    dates = pd.to_datetime(df["date"])
    df["dow"]         = dates.dt.dayofweek
    df["month_end"]   = dates.dt.is_month_end.astype(int)
    df["month_start"] = dates.dt.is_month_start.astype(int)

    return df


# ─────────────────────────────────────────────────────────
# WALK-FORWARD LightGBM  (replaces logistic regression)
# ─────────────────────────────────────────────────────────
def _fit_lgb(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def walk_forward_predict(df: pd.DataFrame,
                         intra_features: dict) -> tuple:
    """
    Walk-forward training + prediction.

    Training data  : all daily rows with valid features, minus the last row
    Prediction     : last row (= today) → probability of tomorrow being green

    Walk-forward accuracy is estimated over the last WF_ACCURACY_WINDOW rows:
    at each step, train on everything before, predict the next row.

    Intraday features are attached only to today's row.
    LightGBM handles the NaN values natively for all historical rows.

    Returns
    -------
    prob_up   : float [0,1] | None
    wf_acc    : float [0,1] | None
    n_signals : int   — dark-red bars in last 252 rows
    """
    df = engineer_daily_features(df)

    # Target: next-day candle is green (close > open)
    #df["target"] = (df["close"].shift(-1) > df["open"].shift(-1)).astype(float)
    # → "will tomorrow be a green candle internally?"
    df["target"] = (df["close"].shift(-1) > df["close"])
    # → "will tomorrow's close be higher than today's close?"

    # Attach intraday features — NaN for history, live values for today (last row)
    for col in INTRA_FEATURE_COLS:
        df[col] = np.nan
    for col in INTRA_FEATURE_COLS:
        df.loc[df.index[-1], col] = intra_features.get(col, np.nan)

    # Require core daily features; drop rows that lack them
    required = ["momentum_norm", "RSI14", "ADX14", "supertrend_dir", "vol_factor"]
    df_model = df.dropna(subset=required + ["target"]).copy()

    n = len(df_model)
    if n < WF_MIN_TRAIN:
        return None, None, 0

    X = df_model[ALL_FEATURE_COLS].astype("float64").to_numpy()
    y = df_model["target"].astype("int64").to_numpy()

    # ── Walk-forward accuracy over last WF_ACCURACY_WINDOW rows ──
    test_start = max(WF_MIN_TRAIN, n - WF_ACCURACY_WINDOW)
    preds_wf, true_wf = [], []

    for t in range(test_start, n):
        mdl = _fit_lgb(X[:t], y[:t])
        p   = mdl.predict_proba(X[t:t+1])[0][1]
        preds_wf.append(1 if p >= 0.5 else 0)
        true_wf.append(y[t])

    wf_acc = (float(np.mean(np.array(preds_wf) == np.array(true_wf)))
              if preds_wf else None)

    # ── Final model: train on all rows except last (no target yet) ──
    final_model = _fit_lgb(X[:-1], y[:-1])
    prob_up     = float(final_model.predict_proba(X[-1:])[0][1])

    n_signals = int(df_model["is_dark_red"].tail(252).sum())

    return prob_up, wf_acc, n_signals


# ─────────────────────────────────────────────────────────
# MESSAGE GENERATION  (updated)
# ─────────────────────────────────────────────────────────
def generate_messages(df_stocks: pd.DataFrame, period_days: int = 180):
    params_scalable = {"strategy": "LONG", "tab": "factors"}
    base_url_github = "https://simorders.github.io/Stock-Information/"

    os.makedirs("docs", exist_ok=True)

    cutoff_date  = pd.Timestamp.today().normalize() - pd.Timedelta(days=period_days)
    df_recent    = df_stocks[df_stocks["date"] > cutoff_date].copy()

    # Build ticker → idNotation map from list file
    df_list      = pd.read_csv(LIST_INDEX)
    notation_map = dict(zip(df_list["ticker"].astype(str),
                            df_list["idNotation"].astype(str)))

    for ticker in tqdm(df_recent["ticker"].unique(), desc="Generating messages..."):
        df_ticker = df_recent[df_recent["ticker"] == ticker].sort_values("date")
        if len(df_ticker) < 2:
            continue

        latest = df_ticker.iloc[-1]
        prev   = df_ticker.iloc[-2]

        # ── Fetch today's intraday bars ──────────────────────────
        idNotation     = notation_map.get(str(ticker), "")
        df_intra       = fetch_intraday_today(idNotation) if idNotation else pd.DataFrame()
        intra_features = extract_intraday_features(df_intra)
        has_intraday   = not df_intra.empty

        # ── INTRADAY SUMMARY + GLIMPSE ───────────────────────────
        last_intra_time = "N/A"
        day_glimpse     = "N/A"

        if not df_intra.empty and len(df_intra) > 4:
            # Last record timestamp (last valid 3-min bar)
            last_intra_time = df_intra["datetime"].iloc[-1].strftime("%d.%m.%Y %H:%M CET")
            
            # Day glimpse calculation
            open_price  = df_intra["open"].iloc[0]
            close_price = df_intra["close"].iloc[-1]
            day_high    = df_intra["high"].max()
            day_low     = df_intra["low"].min()
            day_range   = day_high - day_low
            
            net_dir = "↗️" if close_price > open_price else "↘️"
            close_pos = ((close_price / open_price) -1) * 100
            max_pos = ((day_high / open_price) -1) * 100
            min_pos = ((day_low / open_price) -1) * 100
            
            intra_header = f"<b>Intraday:</b> {last_intra_time}"
        else:
            intra_header = f"No intraday data"

        # ── Walk-forward LightGBM ────────────────────────────────
        df_full = df_stocks[df_stocks["ticker"] == ticker].sort_values("date").copy()
        prob_up, wf_acc, n_signals = walk_forward_predict(df_full, intra_features)

        # ── Prediction text ──────────────────────────────────────
        if prob_up is not None:
            prob_down = 1.0 - prob_up
            direction = "LONG 📈" if prob_up >= 0.5 else "SHORT 📉"
            acc_str   = f"{wf_acc*100:.1f}%" if wf_acc is not None else "N/A"
            
            prob_text = (
                f"{intra_header}\n"
                f"Day P&L: {close_pos:.1f}% {net_dir}\n"
                f"Max: {max_pos:.1f}%  ·  Min: {min_pos:.1f}%\n"
                f"Predicted: <b>{direction}</b>\n"
                f"UP 📈 {prob_up*100:.1f}%  ·  DOWN 📉 {prob_down*100:.1f}%\n"
                f"Accuracy: {acc_str}"
            )
        else:
            prob_text = (
                f"{intra_header}\n\n"
                f"ML: N/A (insufficient history)"
            )

        # ── Chart ────────────────────────────────────────────────
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

        # ── Indicator labels (unchanged logic) ───────────────────
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
        p_rsi = ("OB" if prev["RSI14"] > 70 else "OS" if prev["RSI14"] < 30 else "N")
        c_rsi = ("OB" if latest["RSI14"] > 70 else "OS" if latest["RSI14"] < 30 else "N")
        if c_rsi != p_rsi:
            rsi += " <i>➜New❗</i>"

        adx = "Strong ✅" if latest["ADX14"] > 25 else "Weak ❌"
        if latest["ADX14"] > 25 and prev["ADX14"] <= 25:
            adx += " <i>➜New❗</i>"

        # ── Telegram caption ─────────────────────────────────────
        summary = (
            f"<b><a href='{broker_url}'>{ticker}</a></b> · "
            f"{latest['date'].strftime('%d.%m.%Y')}\n"
            f"Trend: {trend}\n"
            f"Supertrend: {supertrend}\n"
            f"Volatility: {volatility}\n"
            f"Momentum: {momentum}\n"
            f"RSI: {rsi}\n"
            f"ADX: {adx}\n"
            f"────────────────" + "\n"
            f"{prob_text}\n"
        )

        index_information.send_telegram("sendPhoto", filename=fn_png, caption=summary, url=chart_url)

        if os.path.exists(fn_png):
            os.remove(fn_png)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    try:
        # LOAD PRE-UPDATED DATA ONLY
        df_stocks = pd.read_csv(DATA_FILE_INDEX)

        df_stocks["date"] = pd.to_datetime(df_stocks["date"]).dt.tz_localize(None)

        # Rebuild indicators (still required!)
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