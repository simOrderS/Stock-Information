#!/usr/bin/env python
# coding: utf-8

"""
WEEKLY PARAMETER OPTIMIZER (NO SIGNAL GENERATION)
================================================

Standalone script for weekly Supertrend parameter optimization.

Run this script every Sunday to find optimal ATR period and
multiplier for each index based on 5 years of historical data.

Output: optimal_parameters.json
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from tqdm import tqdm
from itertools import product

# ============================================================
# Optional external data functions
# ============================================================

try:
    from index_information import get_index_data, update_index_data, calculate_indicators
    FUNCTIONS_AVAILABLE = True
except ImportError:
    print("WARNING: index_information.py not found.")
    FUNCTIONS_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

INDEX_LIST_FILE = 'liste_INDEX_OnVista.csv'
HISTORY_FILE = 'history_INDEX.csv'
OUTPUT_FILE = 'optimal_parameters.json'
LOG_FILE = 'params_optimization_log.txt'
LEVERAGE = 5

PARAM_GRID = {
    'atr_periods': [10, 12, 14, 16, 18, 21],
    'multipliers': [2.5, 2.75, 3.0, 3.25, 3.5, 4.0]
}

OPTIMIZATION_YEARS = 5
MIN_DATA_POINTS = 252


# ============================================================
# Logging
# ============================================================

def log_message(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_separator():
    log_message("=" * 70)


# ============================================================
# Data Management
# ============================================================

def load_index_list():
    if not os.path.exists(INDEX_LIST_FILE):
        log_message("ERROR: Index list file not found")
        return None
    df = pd.read_csv(INDEX_LIST_FILE)
    log_message(f"Loaded {len(df)} indices")
    return df


def setup_database():
    log_message("Setting up database")

    if not os.path.exists(HISTORY_FILE):
        if not FUNCTIONS_AVAILABLE:
            return None

        start_date = (datetime.now() - timedelta(days=OPTIMIZATION_YEARS * 365)).strftime("%Y-%m-%d")
        df = get_index_data(start_date, f"Y{OPTIMIZATION_YEARS}")

        if df is None or df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.sort_values(["ticker", "date"], inplace=True)
        df.to_csv(HISTORY_FILE, index=False)

    df = pd.read_csv(HISTORY_FILE)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    last_available = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)

    if df["date"].max().normalize() < last_available and FUNCTIONS_AVAILABLE:
        df = update_index_data(INDEX_LIST_FILE, df)
        df.to_csv(HISTORY_FILE, index=False)

    return df


def get_ticker_data(df_all, ticker, years):
    cutoff = datetime.now() - timedelta(days=years * 365)
    df = df_all[df_all["ticker"] == ticker]
    df = df[df["date"] >= cutoff]
    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# Indicators
# ============================================================

def compute_supertrend(df, atr_period, multiplier):
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )

    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
    hl2 = (high + low) / 2

    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr

    up1, dn1 = up.copy(), dn.copy()

    for i in range(1, len(df)):
        up1.iloc[i] = max(up.iloc[i], up1.iloc[i - 1]) if close.iloc[i - 1] > up1.iloc[i - 1] else up.iloc[i]
        dn1.iloc[i] = min(dn.iloc[i], dn1.iloc[i - 1]) if close.iloc[i - 1] < dn1.iloc[i - 1] else dn.iloc[i]

    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if trend[i - 1] == -1 and close.iloc[i] > dn1.iloc[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close.iloc[i] < up1.iloc[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    df["supertrend_dir"] = trend
    df["supertrend"] = np.where(trend == 1, up1, dn1)
    df["atr"] = atr

    return df


def calculate_ema(df, period=200):
    df[f"EMA{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


# ============================================================
# Market Regime
# ============================================================

def detect_market_regime(df, lookback=60):
    recent = df.tail(min(len(df), lookback))
    returns = recent["close"].pct_change()
    volatility = returns.std() * np.sqrt(252)

    x = np.arange(len(recent))
    y = recent["close"].values

    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        y_pred = coeffs[0] * x + coeffs[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2 = 0

    if volatility > 0.30:
        regime = "volatile"
    elif r2 > 0.7:
        regime = "trending"
    else:
        regime = "ranging"

    return regime, volatility, r2


# ============================================================
# Backtesting
# ============================================================

def backtest_parameters(df, atr_period, multiplier, leverage=5):
    """
    Backtest Supertrend strategy for leveraged factor certificates (e.g., 5x)
    with optional 2-day trend confirmation.
    """
    df = compute_supertrend(df, atr_period, multiplier)
    df = calculate_ema(df, 200)

    equity = 100.0  # starting equity
    equity_curve = [equity]
    trades = []
    position = False
    entry = 0.0

    # start index to avoid NaN from indicators
    start_idx = max(atr_period * 3, 200)

    for i in range(start_idx, len(df)):
        row = df.iloc[i]

        # LONG entry: 2-day confirmation of Supertrend LONG
        if i >= 1:
            prev_row = df.iloc[i - 1]
            if not position and row["supertrend_dir"] == 1 and prev_row["supertrend_dir"] == 1 and row["close"] > row["EMA200"]:
                position = True
                entry = row["close"]

        # Exit condition
        if position and (row["supertrend_dir"] == -1 or row["close"] < row["EMA200"]):
            # Calculate return with leverage
            ret = ((row["close"] - entry) / entry) * leverage
            trades.append(ret)
            equity *= (1 + ret)
            position = False

        equity_curve.append(equity)

    if not trades:
        return None

    equity_curve = pd.Series(equity_curve)
    daily_returns = equity_curve.pct_change().dropna()

    # Metrics
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
    win_rate = len([t for t in trades if t > 0]) / len(trades)

    # Custom score (same as before)
    score = (
        np.tanh((equity - 100) / 50) * 30 +
        np.tanh(sharpe / 2) * 30 +
        win_rate * 20 -
        abs(drawdown.min()) * 50
    )

    return {
        "atr_period": atr_period,
        "multiplier": round(multiplier, 2),
        "total_return": round(equity - 100, 2),
        "num_trades": len(trades),
        "win_rate": round(win_rate, 3),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(drawdown.min() * 100, 2),
        "score": round(score, 2)
    }


# ============================================================
# Optimization Engine
# ============================================================

def optimize_index(ticker, df_all):
    df = get_ticker_data(df_all, ticker, OPTIMIZATION_YEARS)

    if df is None or len(df) < MIN_DATA_POINTS:
        return None

    regime, vol, r2 = detect_market_regime(df)

    results = []
    for atr, mult in product(PARAM_GRID["atr_periods"], PARAM_GRID["multipliers"]):
        res = backtest_parameters(df.copy(), atr, mult, leverage=LEVERAGE)
        if res:
            results.append(res)

    if not results:
        return None

    best = max(results, key=lambda x: x["score"])

    return {
        "ticker": ticker,
        "optimal_params": {
            "atr_period": best["atr_period"],
            "multiplier": best["multiplier"],
            "ema_period": 200
        },
        "performance": best,
        "market_regime": {
            "regime": regime,
            "volatility": round(vol * 100, 1),
            "trend_strength": round(r2, 3)
        },
        "top_3_alternatives": sorted(results, key=lambda x: x["score"], reverse=True)[:3],
        "optimization_date": datetime.now().isoformat(),
        "data_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
            "days": len(df)
        }
    }


# ============================================================
# Main
# ============================================================

def run_weekly_optimization():
    log_separator()
    log_message("WEEKLY PARAMETER OPTIMIZATION STARTED")
    log_separator()

    index_list = load_index_list()
    df_all = setup_database()

    results = {}

    #for _, row in index_list.iterrows():
    for _, row in tqdm(index_list.iterrows(), total=len(index_list), desc="Calculating parameters..."):

        res = optimize_index(row["ticker"], df_all)
        if res:
            results[row["ticker"]] = res

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print_optimization_summary(results)

    log_message("OPTIMIZATION COMPLETED")
    return results


def print_optimization_summary(results):
    log_separator()
    log_message("OPTIMIZATION SUMMARY PER INDEX")
    log_separator()

    for ticker, res in results.items():
        p = res["optimal_params"]
        perf = res["performance"]
        regime = res["market_regime"]

        print(
            f"\n{ticker}"
            f"\n  Market regime : {regime['regime']} | Vol {regime['volatility']}% | R² {regime['trend_strength']}"
            f"\n  Best params   : ATR {p['atr_period']} | Mult {p['multiplier']} | EMA {p['ema_period']}"
            f"\n  Performance   : Return {perf['total_return']}% | Sharpe {perf['sharpe']} | "
            f"DD {perf['max_drawdown']}% | Trades {perf['num_trades']} | Win {perf['win_rate']}"
            f"\n  Score         : {perf['score']}"
        )


if __name__ == "__main__":
    run_weekly_optimization()
