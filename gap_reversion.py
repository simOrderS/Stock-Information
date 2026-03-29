#!/usr/bin/env python
# coding: utf-8

import os
import re
import json
import pytz
import requests
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

pd.set_option('max_colwidth', None)
pd.options.display.max_rows = 20
pd.options.display.float_format = '{:0.2f}'.format

# --- Config ---
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN_GAP_REV")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
LIST_STOCKS      = ['liste_DAX40_OnVista.csv', 'liste_MDAX_OnVista.csv', 'liste_NASDAQ100_OnVista.csv']
GERMANY_TZ       = pytz.timezone("Europe/Berlin")
GAP_THRESHOLD    = float(os.getenv("GAP_THRESHOLD", "3.0"))


# ─────────────────────────────────────────────
# 1. TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(method_name, text=None, url=None, filename=None, caption=None):
    API_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method_name}'
    keyboard = {"inline_keyboard": [[{"text": "Open chart", "url": url}]]} if url else None

    params_text = {'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}
    params_file = {
        'chat_id': TELEGRAM_CHAT_ID, 'caption': caption,
        'parse_mode': 'HTML', 'disable_notification': True,
    }
    if keyboard:
        params_file['reply_markup'] = json.dumps(keyboard)

    try:
        if method_name == 'sendMessage':
            r = requests.post(API_URL, params=params_text)
        elif method_name == 'sendPhoto':
            r = requests.post(API_URL, params=params_file, files={'photo': open(filename, 'rb')})
        elif method_name == 'sendDocument':
            r = requests.post(API_URL, params=params_file, files={'document': open(filename, 'rb')})
        else:
            print(f"Unknown method_name: {method_name}")
            return False

        resp = r.json()
        if resp.get('ok'):
            return True
        print("Telegram API error:", resp)
        return False

    except Exception as e:
        print("Telegram sending failed:", e)
        return False


# ─────────────────────────────────────────────
# 2. REALTIME DATA
# ─────────────────────────────────────────────

def _parse_datetime(value):
    """Convert ISO string or Unix timestamp to aware datetime."""
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            pass
    return None


def fetch_quote(row, session):
    try:
        url = f"https://api.onvista.de/api/v1/notations/{row['idNotation']}/quote"
        r = session.get(url, timeout=5)
        r.raise_for_status()
        q = r.json()
        return {
            "ticker":        row['ticker'],
            "isin":          row['isin'],
            "branche":       row['branche'],
            "idNotation":    row['idNotation'],
            "isoCurrency":   q.get("isoCurrency"),
            "marketIsOpen":  q.get("marketIsOpen"),
            "date":          _parse_datetime(q.get("datetimeLast")),
            "open":          q.get("first"),
            "close":         q.get("last"),
            "high":          q.get("high"),
            "low":           q.get("low"),
            "bid":           q.get("bid"),
            "ask":           q.get("ask"),
            "previousClose": q.get("previousLast"),
            # ✅ Use API values directly — no need to recompute
            "performance":    q.get("performance"),
            "gap_pct":        (q.get("performancePct") or 0) * 100,
            # ✅ Both volume metrics
            "volumeDay":     q.get("volumeDay"),   # shares
            "moneyDay":      q.get("moneyDay"),    # EUR turnover
            "numberTrades":  int(q.get("numberTrades") or 0),
        }
    except Exception as e:
        print(f"[{row['ticker']}] Request failed: {e}")
        return None


def get_stock_realtime_data(list_file):
    df_list = pd.read_csv(list_file)
    headers = {
        "User-Agent": "Mozilla/5.0 ...",
        "Accept": "application/json",
        "Referer": "https://www.onvista.de/",
    }
    session = requests.Session()
    session.headers.update(headers)
    rows = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_quote, df_list.loc[i], session) for i in df_list.index]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {list_file}"):
            result = f.result()
            if result:
                rows.append(result)

    df_stocks = pd.DataFrame(rows)
    if not df_stocks.empty:
        df_stocks.sort_values("ticker", inplace=True)
        df_stocks.reset_index(drop=True, inplace=True)
    return df_stocks


# ─────────────────────────────────────────────
# 3. FILTER PRICE SHOCKS
# ─────────────────────────────────────────────

def filter_price_shocks(df_stocks, threshold=5.0, min_turnover=50_000.0):
    """
    Return stocks whose gap_pct (from API) meets the threshold,
    optionally filtered by minimum EUR turnover to suppress illiquid noise.
    """
    df = df_stocks.copy()

    # ✅ gap_pct already populated in fetch_quote — no recomputation needed
    df_shocks = df[
        (df["gap_pct"].abs() >= threshold)
    ].copy()

    df_shocks.sort_values("gap_pct", ascending=True, inplace=True)
    df_shocks.reset_index(drop=True, inplace=True)
    return df_shocks


# ─────────────────────────────────────────────
# 4. GENERATE & SEND MESSAGE
# ─────────────────────────────────────────────

def generate_messages(df_shocks: pd.DataFrame, threshold: float):
    base_url_scalable = "https://de.scalable.capital/broker/security?"
    base_url_github = "https://simorders.github.io/Stock-Information/"

    now = datetime.now(GERMANY_TZ)

    if df_shocks.empty:
        send_telegram("sendMessage", text=f"<b>{now.strftime('%d.%m.%Y')}:</b> No stocks moved.")
        return

    df = df_shocks.copy()
    df.sort_values("gap_pct", ascending=True, inplace=True)

    df["Time"] = df["date"].apply(
        lambda x: x.astimezone(GERMANY_TZ).strftime("%H:%M") if pd.notna(x) else "–"
    )
    df["ISIN"] = df["isin"].apply(
        lambda x: f'<a href="{base_url_scalable}isin={x}" target="_blank">{x}</a>'
    )
    df["ticker"] = df["ticker"].apply(
    lambda x: f'<a href="{base_url_github}{re.sub(r"[^A-Za-z0-9]", "", x)}.html" target="_blank">{x}</a>'
    )

    df = df[["ticker", "ISIN", "previousClose", "close", "gap_pct", "moneyDay", "Time"]]
    df.rename(columns={
        "ticker":        "Ticker",
        "close":         "Price",
        "previousClose": "Prev Close",
        "gap_pct":       "Gap %",
        "moneyDay":      "Turnover",
    }, inplace=True)

    def highlight_row(row):
        color = "background-color:salmon;color:white" if row["Gap %"] < 0 else "background-color:lightgreen;color:black"
        return [color] * len(row)

    filename = "price_shocks.html"
    try:
        (
            df.style
            .format({
                "Price":      "{:,.2f}",
                "Prev Close": "{:,.2f}",
                "Gap %":      "{:+.1f}%",
                "Turnover": "{:,.0f}",
            })
            .apply(highlight_row, axis=1)
            .set_caption(f"Price shocks >= {threshold:.0f}%")
            .set_table_styles([
                {'selector': 'caption', 'props': [('font-size', '42px'), ('font-weight', 'bold'), ('padding', '0.5em')]},
                {'selector': 'th',      'props': [('padding', '0.5em'), ('font-size', '24px'), ('text-align', 'center'), ('background-color', '#222'), ('color', 'white')]},
                {'selector': 'td',      'props': [('padding', '0.5em'), ('font-size', '24px'), ('text-align', 'center')]},
            ])
            .to_html(filename, escape=False)
        )
        caption = f"{now.strftime('%d.%m.%Y')}: {len(df)} ticker(s) ≥ {threshold:.0f}%"
        send_telegram("sendDocument", filename=filename, caption=caption)
    finally:
        if os.path.exists(filename):          # ✅ always clean up
            os.remove(filename)


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    now_str = datetime.now(GERMANY_TZ).strftime("%d.%m.%Y %H:%M %Z")
    print(f"[{now_str}] Scanning for price shocks ≥ {GAP_THRESHOLD}%")

    df_stocks_all = pd.concat(
        [get_stock_realtime_data(f) for f in LIST_STOCKS],
        ignore_index=True
    )

    if df_stocks_all.empty:
        print("No data retrieved, exiting.")
        return

    df_shocks = filter_price_shocks(df_stocks_all, threshold=GAP_THRESHOLD,)

    print(f"Stocks with |gap| ≥ {GAP_THRESHOLD}%: {len(df_shocks)}")
    if not df_shocks.empty:
        print(df_shocks[["ticker", "close", "previousClose", "gap_pct", "moneyDay"]].to_string())

    generate_messages(df_shocks, threshold=GAP_THRESHOLD)


if __name__ == "__main__":
    main()