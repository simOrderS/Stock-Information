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
from datetime import datetime, timezone
from copy import deepcopy
from tqdm import tqdm

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('max_colwidth', None)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:0.2f}'.format

TELEGRAM_TOKEN_NASDAQ = os.getenv("TELEGRAM_TOKEN_NASDAQ")
TELEGRAM_TOKEN_SANDBOX = os.getenv("TELEGRAM_TOKEN_SANDBOX")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DATA_FILE_NASDAQ = "history_NASDAQ100_gettex.csv"
LIST_NASDAQ = 'liste_NASDAQ100_OnVista.csv'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.onvista.de/"
}

def send_telegram(method_name, text=None, url=None, document=None, filename=None, caption=None):
    TOKEN = TELEGRAM_TOKEN_NASDAQ
    API_URL = f'https://api.telegram.org/bot{TOKEN}/{method_name}'

    keyboard = {
        "inline_keyboard": [
            [{"text": "interactive chart", "url": url}]
        ]
    }
    params_text = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'HTML'
    }
    params_file = {
        'chat_id': TELEGRAM_CHAT_ID,
        'caption': caption,
        'parse_mode': 'HTML',
        'disable_notification': True,
        'reply_markup': json.dumps(keyboard)
    }

    try:
        if method_name == 'sendMessage':
            r = requests.post(url=API_URL, params=params_text)
        elif method_name == 'sendDocument':
            r = requests.post(url=API_URL, params=params_file, files={'document': open(filename, 'rb')})
        elif method_name == 'sendPhoto':
            r = requests.post(url=API_URL, params=params_file, files={'photo': open(filename, 'rb')})
        else:
            print("Unknown method_name:", method_name)
            return False

        resp_json = r.json()
        if resp_json.get('ok'):
            msg_link = f"https://t.me/{resp_json['result']['from']['username']}/{resp_json['result']['message_id']}"
            return True
        else:
            print("Telegram API error:", resp_json)
            return False

    except Exception as e:
        print("Telegram sending failed:", e)
        return False


def get_stock_data(list_file, start_date, range_):
    df_list = pd.read_csv(list_file)

    df_stocks = pd.DataFrame(columns=[
        'ticker','isin','branche','idNotation','isoCurrency',
        'date','open','close','high','low','volume'
    ])

    for ind in tqdm(df_list.index, desc='Retrieving data...'):
        idNotation = df_list.loc[ind, 'idNotation']
        url = (
            f"https://api.onvista.de/api/v1/instruments/STOCK/{idNotation}"
            f"/eod_history?idNotation={idNotation}&range={range_}&startDate={start_date}"
        )

        r = requests.get(url, headers=HEADERS, timeout=10)
        parsed = r.json()

        if "datetimeLast" not in parsed:
            continue

        rows = []
        for i, ts in enumerate(parsed["datetimeLast"]):
            rows.append({
                "ticker": df_list.loc[ind, "ticker"],
                "isin": df_list.loc[ind, "isin"],
                "branche": df_list.loc[ind, "branche"],
                "idNotation": idNotation,
                "isoCurrency": parsed.get("isoCurrency"),
                "date": datetime.fromtimestamp(ts, tz=timezone.utc),
                "open": parsed["first"][i],
                "close": parsed["last"][i],
                "high": parsed["high"][i],
                "low": parsed["low"][i],
                "volume": parsed["volume"][i] if "volume" in parsed else None
            })

        df_stocks = pd.concat([df_stocks, pd.DataFrame(rows)])

    df_stocks.sort_values(["ticker","date"], inplace=True)
    #df_indexes.to_csv(data_file, index=False)
    
    return df_stocks


def update_stock_data(list_file, df):
    start_date = df.date.max()

    df_market = pd.read_csv(list_file)

    df_update_list = []

    for ind in tqdm(df_market.index, desc='Retrieving data...'):
        ticker = df_market.loc[ind, 'ticker']
        isin = df_market.loc[ind, 'isin']
        branche = df_market.loc[ind, 'branche']
        idNotation = df_market.loc[ind, 'idNotation']

        url = (
            f"https://api.onvista.de/api/v1/instruments/STOCK/{idNotation}"
            f"/eod_history?idNotation={idNotation}&range=M1&startDate={start_date}"
        )

        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            continue

        parsed = r.json()
        if "datetimeLast" not in parsed:
            continue

        results = []
        for idx, ts in enumerate(parsed["datetimeLast"]):
            results.append({
                "ticker": ticker,
                "isin": isin,
                "branche": branche,
                "idNotation": idNotation,
                "isoCurrency": parsed.get("isoCurrency"),
                "date": datetime.fromtimestamp(ts, tz=timezone.utc),
                "open": parsed["first"][idx],
                "close": parsed["last"][idx],
                "high": parsed["high"][idx],
                "low": parsed["low"][idx],
                # indexes often have no real volume
                "volume": parsed["volume"][idx] if "volume" in parsed else None
            })

        if results:
            df_update_list.append(pd.DataFrame(results))

    # --- Merge with existing history ---
    if df_update_list:
        df_update = pd.concat(df_update_list, ignore_index=True)
        df_update["date"] = pd.to_datetime(df_update["date"]).dt.tz_localize(None)

        df_stocks = pd.concat([df, df_update], ignore_index=True)
    else:
        df_stocks = df.copy()

    # --- Cleanup ---
    df_stocks.sort_values(by=["ticker", "date"], inplace=True)
    df_stocks.drop_duplicates(subset=["ticker", "date"], inplace=True)
    df_stocks.reset_index(drop=True, inplace=True)

    #df_indexes.to_csv(data_file, index=False)

    return df_stocks


def setup_database(list_file, data_file):    
    # Step 1: create file if not exists
    if not os.path.exists(data_file):
        df_initial = get_stock_data(list_file, '2024-01-01', 'Y1')
        df_initial['date'] = pd.to_datetime(df_initial['date']).dt.tz_localize(None)
        df_initial.to_csv(data_file, index=False)
        print(f"Initial file {data_file} generated")
    
    # Step 2: read existing data
    df_stocks = pd.read_csv(data_file, dtype={
        'ticker': 'string', 'isin': 'string', 'market': 'string', 'idNotation': 'string', 'isoCurrency': 'string',
        'open': 'Float64', 'close': 'Float64', 'high': 'Float64', 'low': 'Float64', 'volume': 'Float64',
        'numberPrices': 'Float64'
    })
    df_stocks['date'] = pd.to_datetime(df_stocks['date']).dt.tz_localize(None)
    
    # Step 3: update if needed
    last_available_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    if df_stocks.date.max().normalize() < last_available_date:
        df_stocks = update_stock_data(list_file, df_stocks)
        df_stocks.to_csv(data_file, index=False)
        print(f"Stock data updated to {df_stocks.date.max()}")
    
    # Step 4: calculate indicators
    df_stocks = calculate_indicators(df_stocks, include_rsi=True)
    df_stocks = add_supertrend(df_stocks, atr_period=10, multiplier=3.0)

    # Step 5: Support & Resistance
    '''
    dfs = []
    for ticker in df_stocks["ticker"].unique():
        dft = df_stocks[df_stocks["ticker"] == ticker].sort_values("date")
        dft = add_pivots(dft, left=3, right=3)
        dft = add_sr_curves(dft, span=20)
        dfs.append(dft)

    df_stocks = pd.concat(dfs, ignore_index=True)
    '''

    print(f'Indicators calculated up to {df_stocks.date.max()}')
    return df_stocks


def calculate_indicators(df, include_rsi=True):
    dfs = []

    for ticker in tqdm(df["ticker"].unique(), desc="Calculating indicators"):
        df_ = df[df["ticker"] == ticker].copy()
        df_ = df_.sort_values("date").reset_index(drop=True)

        # Skip if not enough rows for largest indicator
        min_rows_needed = 200  # EMA200 is largest window
        if len(df_) < min_rows_needed:
            print(f"Skipping {ticker}, not enough data ({len(df_)} rows)")
            continue

        # --- Ensure numeric ---
        for col in ["open", "high", "low", "close"]:
            df_[col] = pd.to_numeric(df_[col], errors="coerce")

        # --- Trend ---
        df_["EMA10"] = EMAIndicator(df_["close"], window=10).ema_indicator()
        df_["EMA20"] = EMAIndicator(df_["close"], window=20).ema_indicator()
        df_["EMA200"] = EMAIndicator(df_["close"], window=200).ema_indicator()

        df_["trend_long"] = df_["close"] > df_["EMA200"]

        # --- Volatility ---
        atr = AverageTrueRange(
            high=df_["high"],
            low=df_["low"],
            close=df_["close"],
            window=14
        )
        df_["ATR14"] = atr.average_true_range()

        # --- Volatility baseline ---
        df_["ATR14_mean"] = df_["ATR14"].rolling(50, min_periods=20).mean()
        df_["vol_factor"] = df_["ATR14"] / df_["ATR14_mean"]

        # --- Momentum ---
        macd = MACD(df_["close"], window_fast=12, window_slow=26, window_sign=9)
        df_["MACDhist"] = macd.macd_diff()

        df_["momentum_norm"] = df_["MACDhist"] / df_["ATR14"]

        # --- Regime ---
        df_[["high", "low", "close"]] = df_[["high", "low", "close"]].astype(float)

        adx = ADXIndicator(
            high=df_["high"].ffill(),
            low=df_["low"].ffill(),
            close=df_["close"].ffill(),
            window=14
        )
        df_["ADX14"] = adx.adx()

        # --- Optional RSI ---
        if include_rsi:
            df_["RSI14"] = RSIIndicator(df_["close"], window=14).rsi()

        dfs.append(df_)

    return pd.concat(dfs, ignore_index=True)


def add_pivots(df, left=3, right=3):
    df = df.copy()
    df["pivot_high"] = np.nan
    df["pivot_low"] = np.nan

    highs = df["high"].values
    lows = df["low"].values

    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i-left:i+right+1]):
            df.iloc[i, df.columns.get_loc("pivot_high")] = highs[i]

        if lows[i] == min(lows[i-left:i+right+1]):
            df.iloc[i, df.columns.get_loc("pivot_low")] = lows[i]

    return df


def add_sr_curves(df, span=20):
    df = df.copy()

    df["resistance_curve"] = (
        df["pivot_high"]
        .ffill()
        .ewm(span=span, adjust=False)
        .mean()
    )

    df["support_curve"] = (
        df["pivot_low"]
        .ffill()
        .ewm(span=span, adjust=False)
        .mean()
    )

    return df


def compute_supertrend(df, atr_period=10, multiplier=3.0):
    df = df.copy()

    # --- Ensure no missing values at start ---
    df["close"] = df["close"].ffill()
    df["high"] = df["high"].ffill()
    df["low"] = df["low"].ffill()

    high, low, close = df["high"], df["low"], df["close"]

    # --- True Range ---
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    tr.iloc[0] = high.iloc[0] - low.iloc[0]  # first row safe

    # --- ATR (Wilder) ---
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    # --- Base bands ---
    hl2 = (high + low) / 2
    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr

    # --- Adjust bands like Pine Script ---
    up1 = up.copy()
    dn1 = dn.copy()
    up1.iloc[0] = up.iloc[0]
    dn1.iloc[0] = dn.iloc[0]

    for i in range(1, len(df)):
        up1.iloc[i] = max(up.iloc[i], up1.iloc[i-1]) if close.iloc[i-1] > up1.iloc[i-1] else up.iloc[i]
        dn1.iloc[i] = min(dn.iloc[i], dn1.iloc[i-1]) if close.iloc[i-1] < dn1.iloc[i-1] else dn.iloc[i]

    # --- Supertrend direction ---
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if trend[i-1] == -1 and close.iloc[i] > dn1.iloc[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close.iloc[i] < up1.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    # --- Supertrend line ---
    supertrend = np.where(trend == 1, up1, dn1)

    df["supertrend_dir"] = trend
    df["supertrend"] = supertrend
    df["supertrend_up"] = up1
    df["supertrend_dn"] = dn1

    return df


def add_supertrend(df, atr_period=10, multiplier=3.0):
    dfs = []
    for ticker in df["ticker"].unique():
        df_t = df[df["ticker"] == ticker].sort_values("date")
        df_t = compute_supertrend(df_t, atr_period, multiplier)
        dfs.append(df_t)

    df = pd.concat(dfs, ignore_index=True)
    df["supertrend_dir_prev"] = df.groupby("ticker")["supertrend_dir"].shift(1)
    return df


# ============================================================
# Supertrend strategy
# ============================================================
def strategy_supertrend(row, position=None):
    # Always HOLD for first row (no previous trend)
    if pd.isna(row["supertrend_dir_prev"]) or pd.isna(row["supertrend_dir"]):
        return "HOLD"

    # Buy signal: trend flips from -1 to 1
    if row["supertrend_dir_prev"] == -1 and row["supertrend_dir"] == 1:
        return "BUY"

    # Sell signal: trend flips from 1 to -1
    if row["supertrend_dir_prev"] == 1 and row["supertrend_dir"] == -1:
        return "SELL"

    return "HOLD"


def momentum_colors(df):
    mom = df['momentum_norm'].fillna(0)
    delta = mom.diff().fillna(0)

    return np.select(
        [
            (mom > 0) & (delta > 0),
            (mom > 0) & (delta <= 0),
            (mom < 0) & (delta < 0),
            (mom < 0) & (delta >= 0),
        ],
        ['lime', 'green', 'maroon', 'red'],
        default='gray'
    ).tolist()


def plot_chart(df, title):

    # --- Support / Resistance lookback (recent only) ---
    SR_LOOKBACK = 50
    df_sr = df.tail(SR_LOOKBACK)

    strategy = df.get('strategy', pd.Series(index=df.index))

    strategy_symbols = np.select([strategy == 'BUY', strategy == 'SELL'], ['triangle-up', 'triangle-down'], default='circle')
    strategy_colors = np.select([strategy == 'BUY', strategy == 'SELL'], ['green', 'red'], default='rgba(0,0,0,0)'
    )

    volume_colors = np.where(df['close'] >= df['open'], 'green', 'red')
    momentum_bar_colors = momentum_colors(df)

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.45, 0.1, 0.2, 0.15, 0.1])

    # --- PRICE + TREND ---
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 name="Price", showlegend=False), row=1, col=1)

    # EMA traces with legend
    fig.add_trace(go.Scatter(x=df['date'], y=df['EMA10'], line=dict(color='grey', width=1),
                             name="EMA10", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['EMA20'], line=dict(color='black', width=1),
                             name="EMA20", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['EMA200'], line=dict(color='orange', width=2),
                             name="EMA200", showlegend=True), row=1, col=1)
    
    # --- EMA corridor dynamic fill ---
    ema_up = df['EMA10'] > df['EMA20']
    start_idx = 0

    for i in range(1, len(df)):
        # segment ends when the order changes OR we reach the last index
        if ema_up.iloc[i] != ema_up.iloc[start_idx] or i == len(df) - 1:
            # include last index if end of dataframe
            end_idx = i + 1 if i == len(df) - 1 else i

            # slice the segment
            x_seg = df['date'].iloc[start_idx:end_idx]
            y1_seg = df['EMA10'].iloc[start_idx:end_idx]
            y2_seg = df['EMA20'].iloc[start_idx:end_idx]

            # EMA10 trace (invisible)
            fig.add_trace(go.Scatter(x=x_seg, y=y1_seg, line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)

            # EMA20 trace with fill to EMA10
            fig.add_trace(go.Scatter(x=x_seg, y=y2_seg, fill='tonexty', fillcolor='rgba(0,255,0,0.2)' if ema_up.iloc[start_idx] else 'rgba(255,0,0,0.2)',
                                     line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
            start_idx = i

    # --- SUPERTREND ---
    if "supertrend" in df.columns:
        fig.add_trace(go.Scatter(x=df['date'],y=df['supertrend'].where(df['supertrend_dir'] == 1),
                                 mode='lines', line=dict(color='green', width=2), name='Supertrend UP',
                                 showlegend=True), row=1, col=1)

        fig.add_trace(go.Scatter(x=df['date'], y=df['supertrend'].where(df['supertrend_dir'] == -1),
                                 mode='lines', line=dict(color='red', width=2), name='Supertrend DOWN', 
                                 showlegend=True), row=1, col=1)
        
    # --- SUPPORT & RESISTANCE CORRIDOR ---
    '''
    if "support_curve" in df_sr.columns and df_sr["support_curve"].notna().any() \
    and "resistance_curve" in df_sr.columns and df_sr["resistance_curve"].notna().any():

        # Resistance line first
        fig.add_trace(go.Scatter(x=df_sr['date'], y=df_sr['resistance_curve'], mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                 showlegend=False, name='Resistance'), row=1, col=1)

        # Support line, fill to previous y
        fig.add_trace(go.Scatter(x=df_sr['date'], y=df_sr['support_curve'], mode='lines', fill='tonexty',
                                 fillcolor='rgba(0,100,255,0.2)', line=dict(color='rgba(0,0,0,0)'), name='SR Zone',
                                 showlegend=True), row=1, col=1)
    '''

    # --- VOLUME ---
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color=volume_colors,
                         name="Volume", showlegend=False), row=2, col=1)

    # --- MOMENTUM ---
    fig.add_trace(go.Bar(x=df['date'], y=df['momentum_norm'], marker_color=momentum_bar_colors,
                         name="Momentum (norm)", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=[0]*len(df), line=dict(color='black', width=1),
                             name="Zero", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['momentum_norm'], mode='markers',
                             marker=dict(symbol=strategy_symbols, size=12, color=strategy_colors),
                             name="Signals", showlegend=False), row=3, col=1)

    # --- RSI ---
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI14'], line=dict(color='grey', width=2),
                             name="RSI14", showlegend=False), row=4, col=1)

    fig.add_trace(go.Scatter(x=df['date'], y=[RSI_OVERBOUGHT]*len(df),line=dict(color="red", width=1, dash="dash"),
                             name="Overbought", showlegend=False), row=4, col=1)

    fig.add_trace(go.Scatter(x=df['date'], y=[RSI_OVERSOLD]*len(df), line=dict(color="green", width=1, dash="dash"),
                             name="Oversold", showlegend=False), row=4, col=1)

    # --- ADX ---
    ADX_STRONG_THRESHOLD = 25

    fig.add_trace(go.Scatter(x=df['date'], y=df['ADX14'],line=dict(color='purple', width=2),
                             name="ADX", showlegend=False), row=5, col=1)

    fig.add_trace(go.Scatter(x=df['date'], y=[ADX_STRONG_THRESHOLD]*len(df),line=dict(color='red', width=1, dash="dash"),
                             name="Strong Trend", showlegend=False), row=5, col=1)

    # --- Layout with legend at bottom of price row ---
    fig.update_layout(title=title, height=900, width=1200, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(x=0.5, y=1, xanchor='center', yanchor='top', orientation='h', borderwidth=0, bgcolor='rgba(0,0,0,0)'))

    fig.update_xaxes(showgrid=True, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Momentum", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.update_yaxes(title_text="ADX", row=5, col=1)

    price_max = df['high'].max()
    price_min = df['low'].min()
    padding = (price_max - price_min) * 0.2
    fig.update_yaxes(range=[price_min, price_max + padding], row=1, col=1)

    return fig


def generate_messages(df_stocks: pd.DataFrame, period_days: int = 180, plot: bool = False):
    base_url_scalable = "https://de.scalable.capital/broker/security?"
    base_url_github = "https://simorders.github.io/Stock-Information/"

    os.makedirs("docs", exist_ok=True)

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=period_days)
    df_recent = df_stocks[df_stocks["date"] > cutoff_date].copy()

    for ticker in tqdm(df_recent["ticker"].unique(), desc="Generating messages..."):
        df_ticker = df_recent[df_recent["ticker"] == ticker].sort_values("date")

        if len(df_ticker) < 2:
            continue

        latest = df_ticker.iloc[-1]
        prev = df_ticker.iloc[-2]

        # Chart
        title = f"{ticker} · {df_ticker['isin'].iloc[0]} · {latest['date'].strftime('%d.%m.%Y')}"
        fig = plot_chart(df_ticker, title=title)

        clean_ticker = re.sub(r"[^A-Za-z0-9]", "", ticker)

        filename_png = f"{clean_ticker}.png"
        filename_html = f"docs/{clean_ticker}.html"

        fig.write_image(filename_png)
        fig.write_html(filename_html, include_plotlyjs="cdn", full_html=True)

        chart_url = f"{base_url_github}{clean_ticker}.html"

        params = {"isin": df_ticker["isin"].iloc[0]}
        #params = {'isin': df_ticker['isin'].unique()[0], 'model': 'trade', 'security': df_ticker['isin'].unique()[0], 'type': "BUY"}
        broker_url = f"{base_url_scalable}{urllib.parse.urlencode(params)}"

        # Trend
        trend = "ON ✅" if latest["trend_long"] else "OFF ❌"
        if latest["trend_long"] and not prev["trend_long"]:
            trend += " <i>➜New❗</i>"

        # Supertrend
        supertrend = "UP ✅" if latest["supertrend_dir"] == 1 else "DOWN ❌"
        if latest["supertrend_dir"] != prev["supertrend_dir"]:
            supertrend += " <i>➜New❗</i>"

        # Volatility
        if latest["vol_factor"] > 1.2:
            volatility = "HIGH ❌"
        elif latest["vol_factor"] < 0.8:
            volatility = "LOW ✅"
        else:
            volatility = "Normal"

        # Momentum (context
        momentum = "Positive ✅" if latest["momentum_norm"] > 0 else "Negative ❌"
        
        # RSI
        if pd.isna(latest["RSI14"]):
            rsi = "N/A"
        elif latest["RSI14"] > 70:
            rsi = "Overbought ⚠️"
        elif latest["RSI14"] < 30:
            rsi = "Oversold ⚠️"
        else:
            rsi = "Normal"

        prev_rsi_state = (
            "OB" if prev["RSI14"] > 70 else
            "OS" if prev["RSI14"] < 30 else
            "N"
        )
        curr_rsi_state = (
            "OB" if latest["RSI14"] > 70 else
            "OS" if latest["RSI14"] < 30 else
            "N"
        )
        if curr_rsi_state != prev_rsi_state:
            rsi += " <i>➜New❗</i>"

        # ADX
        adx = "Strong ✅" if latest["ADX14"] > 25 else "Weak ❌"
        if latest["ADX14"] > 25 and prev["ADX14"] <= 25:
            adx += " <i>➜New❗</i>"

        # --- Track whether anything actually changed today ---
        has_new_signal = any([
            latest["trend_long"] and not prev["trend_long"],
            latest["supertrend_dir"] != prev["supertrend_dir"],
            curr_rsi_state != prev_rsi_state,
            latest["ADX14"] > 25 and prev["ADX14"] <= 25,
        ])

        if not has_new_signal:
            continue  # skip this ticker, nothing new to report

        summary = (
            f"<b><a href='{broker_url}'>{ticker}</a></b> · {latest['date'].strftime('%d.%m.%Y')}\n"
            f"Trend: {trend}\n"
            f"Supertrend: {supertrend}\n"
            f"Volatility: {volatility}\n"
            f"Momentum: {momentum}\n"
            f"RSI: {rsi}\n"
            f"ADX: {adx}\n"
        )

        send_telegram("sendPhoto", filename=filename_png, caption=summary, url=chart_url,)

        if os.path.exists(filename_png):
            os.remove(filename_png)


def main():

    try:
        # --- Update stock data & indicators --- #
        df_stocks = setup_database(LIST_NASDAQ, DATA_FILE_NASDAQ)
        
        # --- LIVE MODE ---
        generate_messages(df_stocks, period_days=180, plot=False)

    except Exception as e:
        today_str = datetime.today().strftime("%d.%m.%Y")
        tb_str = traceback.format_exc()
        send_telegram('sendMessage', text=f"<b>{today_str} - unexpected error:</b>\n{tb_str}")
        return


if __name__ == "__main__":

    main()

