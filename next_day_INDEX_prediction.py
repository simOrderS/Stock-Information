#!/usr/bin/env python
# coding: utf-8

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
WF_MIN_TEST_ROWS    = 30    # FIX 1: guarantee at least this many test rows

# Rolling training window — only use the most recent N daily rows.
# Set to None to use all history.
ROLLING_TRAIN_WINDOW = 500

# Recency weighting: sample weight halves every N days.
WEIGHT_HALF_LIFE_DAYS = 90

# Confidence gate: if |prob_up - 0.5| < this, output NEUTRAL instead of UP/DOWN.
# Set to 0 to disable.
CONFIDENCE_THRESHOLD = 0.06

# FIX 2: Cap the ATR-based target threshold so high-volatility regimes
# don't neutralise almost all labels and collapse the training set.
ATR_THRESHOLD_MULTIPLIER = 0.4
ATR_THRESHOLD_CAP        = 0.015   # max allowed threshold (1.5 % daily move)
ATR_THRESHOLD_DEFAULT    = 0.004   # fallback when ATR is not yet available

# LightGBM — shallow + early stopping to avoid overfitting
LGB_PARAMS = dict(
    objective         = "binary",
    metric            = "binary_logloss",
    n_estimators      = 500,
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
# INTRADAY FETCH
# ─────────────────────────────────────────────────────────
def fetch_intraday_today(idNotation: str) -> pd.DataFrame:
    """
    Fetch today's 3-min bars from OnVista chart_history endpoint.
    Returns a clean DataFrame or an empty DataFrame when the market is closed
    or no data is available.
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

    ts_list = parsed.get("datetimeLast", [])
    closes  = parsed.get("last", [])

    if not ts_list or not closes:
        print("[Intraday] No intraday data → market closed or holiday")
        return pd.DataFrame()

    opens    = parsed.get("first", [])
    highs    = parsed.get("high", [])
    lows     = parsed.get("low", [])
    volumes  = parsed.get("volume", [])
    n_trades = parsed.get("numberPrices", [])

    n = min(len(ts_list), len(closes))

    rows = []
    for i in range(n):
        rows.append({
            "datetime": datetime.fromtimestamp(ts_list[i], tz=timezone.utc).astimezone().replace(tzinfo=None),
            "open":     opens[i] if i < len(opens) else closes[i],
            "high":     highs[i] if i < len(highs) else closes[i],
            "low":      lows[i] if i < len(lows) else closes[i],
            "close":    closes[i],
            "volume":   volumes[i] if i < len(volumes) else 0,
            "n_trades": n_trades[i] if i < len(n_trades) else 0,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) > 10:
        df = df.iloc[:-2].copy()

    # Reject stale data: the API sometimes returns the previous session's bars
    # (e.g. DAX on a Sunday returning Friday data). Treat that as no data.
    today = datetime.now().date()
    if df["datetime"].iloc[-1].date() != today:
        print(f"[Intraday] Last bar date {df['datetime'].iloc[-1].date()} ≠ today ({today}) → discarding")
        return pd.DataFrame()

    return df


# ─────────────────────────────────────────────────────────
# INTRADAY FEATURES
# ─────────────────────────────────────────────────────────
def extract_intraday_features(df_intra: pd.DataFrame) -> dict:
    """
    Extract scalar features from today's 3-min bars.
    Used only for the human-readable intraday summary and as a separate
    directional signal — not fed into the LightGBM model.
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
    Each sub-signal votes; majority wins.
    """
    if all(np.isnan(v) for v in intra.values()):
        return "~"

    score = 0

    cp = intra.get("intra_close_pos", np.nan)
    if not np.isnan(cp):
        if cp > 0.6: score += 1
        elif cp < 0.4: score -= 1

    r2h = intra.get("intra_last2h_ret", np.nan)
    if not np.isnan(r2h):
        if r2h > 0.002: score += 1
        elif r2h < -0.002: score -= 1

    mh = intra.get("intra_macd_hist", np.nan)
    if not np.isnan(mh):
        score += 2 if mh > 0 else -2

    vd = intra.get("intra_vwap_dev", np.nan)
    if not np.isnan(vd):
        if vd > 0.05: score += 2
        elif vd < -0.05: score -= 2

    if score >= 2: return "▲"
    if score <= -2: return "▼"
    return "~"


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
    "dow",
    # removed: "trend_long", "ema_cross", "supertrend_dir",
    # removed: "is_dark_red", "is_dark_red_2nd",
    # removed: "month_end", "month_start",
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
# SAMPLE WEIGHTS
# ─────────────────────────────────────────────────────────
def make_sample_weights(n: int, half_life: int) -> np.ndarray:
    """
    Exponential decay weights: the most recent row has weight 1,
    a row half_life days ago has weight 0.5.
    Normalised to sum to n so LightGBM leaf sizes remain meaningful.
    """
    ages    = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2) * ages / half_life)
    weights = weights / weights.sum() * n
    return weights


# ─────────────────────────────────────────────────────────
# MODEL FITTING
# ─────────────────────────────────────────────────────────
def _fit_lgb(X_train: np.ndarray,
             y_train: np.ndarray,
             w_train: np.ndarray) -> lgb.LGBMClassifier:
    """
    Fit LightGBM with early stopping against a held-out validation slice
    (last VALIDATION_FRAC of training rows).
    LightGBM's raw probabilities are well-calibrated enough for directional
    trading signals, so no additional calibration step is needed.
    Returns the fitted LGBMClassifier.
    """
    n_val = max(1, int(len(X_train) * VALIDATION_FRAC))
    n_tr  = len(X_train) - n_val

    X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
    y_tr, y_val = y_train[:n_tr], y_train[n_tr:]
    w_tr        = w_train[:n_tr]

    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(
        X_tr, y_tr,
        sample_weight = w_tr,
        eval_set      = [(X_val, y_val)],
        callbacks     = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(period=-1)],
    )

    return model


# ─────────────────────────────────────────────────────────
# WALK-FORWARD LIGHTGBM
# ─────────────────────────────────────────────────────────
def walk_forward_predict(df: pd.DataFrame) -> tuple:
    """
    Walk-forward training and prediction using daily features only.

    Returns
    -------
    prob_up      : float [0,1] | None
    wf_acc       : float [0,1] | None  (-1.0 sentinel means "insufficient history")
    n_signals    : int  — dark-red bars in last 252 rows
    n_trees_used : int  — actual trees used by final model after early stopping
    """
    df = engineer_daily_features(df)

    # ── FIX 2: cap ATR-based threshold so high-volatility regimes don't
    #    neutralise almost all labels and collapse the training set. ──────────
    atr_pct   = (df["high"].rolling(14).max() - df["low"].rolling(14).min()) / df["close"]
    threshold = (
        (atr_pct.shift(1) * ATR_THRESHOLD_MULTIPLIER)
        .fillna(ATR_THRESHOLD_DEFAULT)
        .clip(upper=ATR_THRESHOLD_CAP)          # ← cap added here
    )

    ret = df["close"].shift(-1) / df["close"] - 1
    df["target"] = np.where(
        ret >  threshold, 1,
        np.where(ret < -threshold, 0, np.nan)
    )

    required  = ["momentum_norm", "RSI14", "ADX14", "supertrend_dir", "vol_factor"]
    df_model  = df.dropna(subset=required + ["target"]).copy()

    if ROLLING_TRAIN_WINDOW and len(df_model) > ROLLING_TRAIN_WINDOW:
        df_model = df_model.tail(ROLLING_TRAIN_WINDOW).copy()

    n = len(df_model)
    if n < WF_MIN_TRAIN:
        # FIX 3: return sentinel -1.0 so callers can distinguish "not computed"
        # from a genuine 0 % accuracy score.
        return None, -1.0, 0, 0

    X = df_model[DAILY_FEATURE_COLS].astype("float64").to_numpy()
    y = df_model["target"].astype("int64").to_numpy()

    # ── FIX 1: guarantee a minimum test window ────────────────────────────────
    # Without this, when n == WF_MIN_TRAIN the loop range is empty and
    # wf_acc is never computed, which gets reported as 0 %.
    raw_test_start = max(WF_MIN_TRAIN, n - WF_ACCURACY_WINDOW)
    test_start     = min(raw_test_start, n - WF_MIN_TEST_ROWS)
    test_start     = max(0, test_start)   # safety clamp — never go negative

    preds_wf, true_wf = [], []
    wf_log = []

    RETRAIN_EVERY = 5
    mdl = None
    for t in range(test_start, n):
        if mdl is None or (t - test_start) % RETRAIN_EVERY == 0:
            w   = make_sample_weights(t, WEIGHT_HALF_LIFE_DAYS)
            mdl = lgb.LGBMClassifier(**LGB_PARAMS)
            mdl.fit(X[:t], y[:t], sample_weight=w)

        p = mdl.predict_proba(X[t:t+1])[0][1]
        preds_wf.append(1 if p >= 0.5 else 0)
        true_wf.append(y[t])

        wf_log.append({
            "date":      df_model["date"].iloc[t],
            "prob_up":   round(p, 4),
            "predicted": 1 if p >= 0.5 else 0,
            "actual":    y[t],
            "correct":   int((1 if p >= 0.5 else 0) == y[t]),
            "neutral":   int(abs(p - 0.5) < CONFIDENCE_THRESHOLD),
        })

    wf_acc = (float(np.mean(np.array(preds_wf) == np.array(true_wf)))
              if preds_wf else -1.0)   # FIX 3: use -1.0 sentinel, not None

    # ── Save log to CSV (one file per ticker) ─────────────────────────────────
    if wf_log:
        ticker_name = df_model["ticker"].iloc[0] if "ticker" in df_model.columns else "unknown"
        clean_name  = re.sub(r"[^A-Za-z0-9]", "", str(ticker_name))
        log_path    = f"wf_log_{clean_name}.csv"
        pd.DataFrame(wf_log).to_csv(log_path, index=False)

    # ── Final model: train on all rows, predict the last row ──────────────────
    w_final     = make_sample_weights(n - 1, WEIGHT_HALF_LIFE_DAYS)
    final_model = _fit_lgb(X[:-1], y[:-1], w_final)
    prob_up     = float(final_model.predict_proba(X[-1:])[0][1])

    n_trees   = getattr(final_model, "best_iteration_", LGB_PARAMS["n_estimators"])
    n_signals = int(df_model["is_dark_red"].tail(252).sum())

    return prob_up, wf_acc, n_signals, n_trees


# ─────────────────────────────────────────────────────────
# DIRECTION LABEL
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

    df_list      = pd.read_csv(LIST_INDEX)
    notation_map = dict(zip(df_list["ticker"].astype(str),
                            df_list["idNotation"].astype(str)))

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=period_days)
    df_recent   = df_stocks[df_stocks["date"] > cutoff_date].copy()
    today_str   = datetime.today().strftime("%d.%m.%Y")

    # Collect tickers whose intraday data is missing or stale (closed market, holiday).
    # These are reported in one grouped message; no chart or GPT is triggered for them.
    closed_tickers = []

    for ticker in tqdm(df_recent["ticker"].unique(), desc="Generating messages..."):
        df_ticker = df_recent[df_recent["ticker"] == ticker].sort_values("date")
        if len(df_ticker) < 2:
            continue

        latest = df_ticker.iloc[-1]
        prev   = df_ticker.iloc[-2]

        # ── Fetch intraday — empty means no today data (stale or market closed) ──
        idNotation = notation_map.get(str(ticker), "")
        df_intra   = fetch_intraday_today(idNotation) if idNotation else pd.DataFrame()

        if df_intra.empty:
            closed_tickers.append(ticker)
            continue   # skip chart, GPT, and full message for this ticker

        # ── Intraday features & summary ──────────────────────────────────────
        intra_features  = extract_intraday_features(df_intra)
        tz_label        = datetime.now().astimezone().strftime("%Z")
        last_intra_time = df_intra["datetime"].iloc[-1].strftime(f"%d.%m.%Y %H:%M {tz_label}")
        open_price      = df_intra["open"].iloc[0]
        close_price     = df_intra["close"].iloc[-1]
        day_high        = df_intra["high"].max()
        day_low         = df_intra["low"].min()
        close_pos       = (close_price / open_price - 1) * 100
        max_pos         = (day_high    / open_price - 1) * 100
        min_pos         = (day_low     / open_price - 1) * 100
        net_dir         = "↗️" if close_price > open_price else "↘️"
        intra_sig       = intraday_direction_signal(intra_features)
        intra_header    = (
            f"<b>Intraday:</b> {last_intra_time}\n"
            f"Day P&L: {close_pos:.1f}% {net_dir}\n"
            f"Max: {max_pos:.1f}%  ·  Min: {min_pos:.1f}%\n"
            f"Intraday signal: {intra_sig}"
        )

        # ── ML prediction (daily features only) ─────────────────────────────
        df_full = df_stocks[df_stocks["ticker"] == ticker].sort_values("date").copy()
        prob_up, wf_acc, n_signals, n_trees = walk_forward_predict(df_full)

        # FIX 3: treat the -1.0 sentinel as "not available" downstream
        wf_acc_safe = wf_acc if (wf_acc is not None and wf_acc >= 0) else None

        if prob_up is not None:
            final_score = prob_up - 0.5
            if intra_sig == "▲":
                final_score += 0.05
            elif intra_sig == "▼":
                final_score -= 0.05

            prob_combined = min(max(0.5 + final_score, 0.01), 0.99)
            prob_up       = prob_combined
            prob_down     = 1.0 - prob_combined

            direction = direction_label(prob_up)
            acc_str   = f"{wf_acc_safe*100:.1f}%" if wf_acc_safe is not None else "N/A"
            conf_str  = f"{abs(prob_up - 0.5)*100:.1f}pp"
            prob_text = (
                f"Predicted: <b>{direction}</b>\n"
                f"UP 📈 {prob_up*100:.1f}%  ·  DOWN 📉 {prob_down*100:.1f}%\n"
                #f"Confidence: {conf_str} | Accuracy: {acc_str}\n"
                f"Accuracy: {acc_str}\n"
                #f"Trees used: {n_trees} (of {LGB_PARAMS['n_estimators']} max)"
            )
        else:
            prob_text = "ML: N/A (insufficient history)"

        # ── GPT advice ───────────────────────────────────────────────────────
        signal = GPT_trader_analysis.build_signal(
            ticker         = ticker,
            latest         = latest,
            prob_up        = prob_up if prob_up is not None else 0.5,
            wf_acc         = wf_acc_safe,   # FIX 3: pass None, not -1.0
            has_intraday   = True,
            df_intra       = df_intra,
            intra_features = intra_features,
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
            #f"{prob_text}\n"
            #f"────────────────\n"
            f"{gpt_advice}\n"
        )

        index_information.send_telegram("sendPhoto", filename=fn_png,
                                        caption=summary, url=chart_url)

        if os.path.exists(fn_png):
            os.remove(fn_png)

    # ── Single grouped message for all tickers with no today data ────────────
    if closed_tickers:
        ticker_list = "\n".join(f"  · {t}" for t in closed_tickers)
        index_information.send_telegram(
            "sendMessage",
            text=(
                f"<b>{today_str} — No intraday data 🏖️</b>\n"
                f"The following indices returned no data for today.\n"
                f"Market may be closed (holiday or weekend):\n\n"
                f"{ticker_list}"
            ),
        )


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