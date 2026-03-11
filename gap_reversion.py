#!/usr/bin/env python
# coding: utf-8

import os
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
LIST_STOCKS      = ['liste_DAX40_OnVista.csv', 'liste_MDAX_OnVista.csv', 'liste_SDAX_OnVista.csv']
GERMANY_TZ       = pytz.timezone("Europe/Berlin")
GAP_THRESHOLD    = float(os.getenv("GAP_THRESHOLD", "5.0"))  # % default 5.0


# ─────────────────────────────────────────────
# 1. TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(method_name, text=None, url=None, filename=None, caption=None):
    """
    Send a Telegram message, photo or document.

    Parameters
    ----------
    method_name : str  — 'sendMessage', 'sendPhoto' or 'sendDocument'
    text        : str  — HTML message body (sendMessage)
    url         : str  — inline keyboard button URL (sendPhoto/sendDocument)
    filename    : str  — local file path (sendPhoto/sendDocument)
    caption     : str  — HTML caption (sendPhoto/sendDocument)
    """
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
            r = requests.post(API_URL, params=params_file,
                              files={'photo': open(filename, 'rb')})
        elif method_name == 'sendDocument':
            r = requests.post(API_URL, params=params_file,
                              files={'document': open(filename, 'rb')})
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
    """Convert ISO string or Unix timestamp (int/float/str) to aware datetime."""
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


from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_quote(row, session):
    try:
        url = f"https://api.onvista.de/api/v1/notations/{row['idNotation']}/quote"
        r = session.get(url, timeout=5)
        r.raise_for_status()
        q = r.json()
        return {
            "ticker": row['ticker'],
            "isin": row['isin'],
            "branche": row['branche'],
            "idNotation": row['idNotation'],
            "isoCurrency": q.get("isoCurrency"),
            "marketIsOpen": q.get("marketIsOpen"),
            "date": _parse_datetime(q.get("datetimeLast")),
            "open": q.get("first"),
            "close": q.get("last"),
            "high": q.get("high"),
            "low": q.get("low"),
            "volume": q.get("volumeDay"),
            "bid": q.get("bid"),
            "ask": q.get("ask"),
            "previousClose": q.get("previousLast"),
            "performance": q.get("performance"),
            "performancePct": q.get("performancePct"),
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
        for f in tqdm(as_completed(futures), total=len(futures), desc="Fetching realtime quotes..."):
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

def filter_price_shocks(df_stocks, threshold=5.0):
    """
    Return stocks whose current price deviates from previousClose by at
    least `threshold` percent.

    No marketIsOpen filter — works at any time of day.

    Parameters
    ----------
    df_stocks : pd.DataFrame — output of get_stock_realtime_data()
    threshold : float        — minimum absolute gap in % (default 5.0)

    Returns
    -------
    pd.DataFrame with added column gap_pct, sorted by gap_pct ascending
    (largest drops first, largest gains last).
    """
    df = df_stocks.copy()

    df["gap_pct"] = (
        (df["close"] - df["previousClose"]) / df["previousClose"] * 100
    )

    df_shocks = df[df["gap_pct"].abs() >= threshold].copy()
    df_shocks.sort_values("gap_pct", ascending=True, inplace=True)
    df_shocks.reset_index(drop=True, inplace=True)

    return df_shocks


# ─────────────────────────────────────────────
# 4. GENERATE & SEND MESSAGE
# ─────────────────────────────────────────────

def generate_messages(df_shocks: pd.DataFrame, threshold: float):
    """
    Generate a styled HTML table of price shocks and send it to Telegram.
    
    df_shocks : pd.DataFrame — must contain 'ticker', 'isin', 'close', 'previousClose', 'gap_pct', 'date'
    threshold : float        — minimum gap % to show in caption
    """

    base_url_scalable = "https://de.scalable.capital/broker/security?"

    now = datetime.now(GERMANY_TZ)
    now_str = now.strftime("%d.%m.%Y %H:%M")

    if df_shocks.empty:
        send_telegram("sendMessage", 
                      text=f"<b>{now.strftime('%d.%m.%Y')}: Price shocks ≥ {threshold:.0f}%</b>\n\nNo stocks moved.")
        return

    # Sort by gap ascending (largest drop first)
    df = df_shocks.copy()
    df.sort_values("gap_pct", ascending=True, inplace=True)

    # Format trade time
    df["Time"] = df["date"].apply(
        lambda x: x.astimezone(GERMANY_TZ).strftime("%H:%M") if pd.notna(x) else "–"
    )

    # Make ISIN clickable
    df["ISIN"] = df["isin"].apply(
        lambda x: f'<a href="{base_url_scalable}isin={x}" target="_blank">{x}</a>'
    )

    # Select & rename columns
    df = df[["ticker", "ISIN", "previousClose", "close", "gap_pct", "Time"]]
    df.rename(
        columns={
            "ticker": "Ticker",
            "close": "Price",
            "previousClose": "Prev Close",
            "gap_pct": "Gap %",
        },
        inplace=True,
    )

    # Row styling
    def highlight_row(row):
        if row["Gap %"] < 0:
            return ["background-color:salmon;color:white"] * len(row)
        else:
            return ["background-color:lightgreen;color:black"] * len(row)

    caption = f"{now.strftime('%d.%m.%Y %H:%M')}h: {len(df)} ticker(s)"
    filename = f"price_shocks.html"

    # Apply pandas styling
    (
        df.style
        .format({
            "Price": "{:,.2f}",
            "Prev Close": "{:,.2f}",
            "Gap %": "{:+.2f}%",
        })
        .apply(highlight_row, axis=1)
        .set_caption(f"Price shocks >= {threshold:.0f}%")
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '42px'), ('font-weight', 'bold'), ('padding', '0.5em')]},
            {'selector': 'th', 'props': [('padding', '0.5em 0.5em'), ('font-size', '24px'), ('text-align', 'center'), ('background-color', '#222'), ('color', 'white')]},
            {'selector': 'td', 'props': [('padding', '0.5em 0.5em'), ('font-size', '24px'), ('text-align', 'center')]}
        ])
        .to_html(filename, escape=False)
    )

    # Send HTML as Telegram document
    send_telegram("sendDocument", filename=filename, caption=caption)

    # Remove file
    os.remove(os.path.join(os.getcwd(), filename))


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    now_str = datetime.now(GERMANY_TZ).strftime("%d.%m.%Y %H:%M %Z")
    print(f"[{now_str}] Scanning for price shocks ≥ {GAP_THRESHOLD}% vs previous close")

    df_stocks = get_stock_realtime_data(LIST_STOCKS)

    if df_stocks.empty:
        print("No data retrieved, exiting.")
        return

    df_shocks = filter_price_shocks(df_stocks, threshold=GAP_THRESHOLD)

    print(f"Stocks with |gap| ≥ {GAP_THRESHOLD}%: {len(df_shocks)}")
    if not df_shocks.empty:
        print(df_shocks[["ticker", "close", "previousClose", "gap_pct"]].to_string())

    generate_messages(df_shocks, threshold=GAP_THRESHOLD)


if __name__ == "__main__":
    main()