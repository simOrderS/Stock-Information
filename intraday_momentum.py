#!/usr/bin/env python3
"""
Intraday momentum monitor — single run, designed for cron every 15 minutes.

The onvista W1 tick endpoint returns one price sample every ~15 minutes,
so each cron run sees exactly one new data point. The full W1 history
(~240 points) is fetched fresh on every run — no state file needed.

Z-score is computed over the rolling window of W1 returns per stock.
A signal fires when |Z| >= ZSCORE_THRESHOLD.

Cron example (every 15 min, Mon–Fri 08:00–21:45 CET):
    */15 8-21 * * 1-5 /usr/bin/python3 /home/pi/monitor_momentum.py >> /home/pi/momentum.log 2>&1

Environment variables:
    TELEGRAM_TOKEN_GAP_REV  – bot token
    TELEGRAM_CHAT_ID        – destination chat / channel ID
    ZSCORE_THRESHOLD        – alert threshold (default 2.5)
    WINDOW                  – rolling window in candles (default 20 = ~5 h)

Input files (produced by fetch_dax40.py):
    liste_DAX40_OnVista.csv  (adjust LIST_STOCKS as needed)
"""

import os
import re
import json
import pytz
import requests
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

pd.set_option("max_colwidth", None)
pd.options.display.float_format = "{:0.2f}".format

# ── Config ────────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN_GAP_REV")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LIST_STOCKS = [
    "liste_DAX40_OnVista.csv",
]

GERMANY_TZ       = pytz.timezone("Europe/Berlin")
ZSCORE_THRESHOLD = float(os.getenv("ZSCORE_THRESHOLD", "2.5"))
WINDOW           = int(os.getenv("WINDOW", "20"))   # ~5 hours of 15-min candles

TICK_URL = (
    "https://api.onvista.de/api/v1/instruments/STOCK/{entity_value}"
    "/simple_chart_history"
    "?chartType=PRICE&idNotation={id_notation}&range=W1&"
)


# ── 1. TELEGRAM ───────────────────────────────────────────────────────────────

def send_telegram(method_name, text=None, url=None, filename=None, caption=None):
    API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method_name}"
    keyboard = (
        {"inline_keyboard": [[{"text": "Open chart", "url": url}]]} if url else None
    )

    params_text = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    params_file = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "HTML",
        "disable_notification": True,
    }
    if keyboard:
        params_file["reply_markup"] = json.dumps(keyboard)

    try:
        if method_name == "sendMessage":
            r = requests.post(API_URL, params=params_text)
        elif method_name == "sendPhoto":
            r = requests.post(
                API_URL, params=params_file, files={"photo": open(filename, "rb")}
            )
        elif method_name == "sendDocument":
            r = requests.post(
                API_URL,
                params=params_file,
                files={"document": open(filename, "rb")},
            )
        else:
            print(f"Unknown method_name: {method_name}")
            return False

        resp = r.json()
        if resp.get("ok"):
            return True
        print("Telegram API error:", resp)
        return False

    except Exception as e:
        print("Telegram sending failed:", e)
        return False


# ── 2. LOAD STOCK LISTS ───────────────────────────────────────────────────────

def load_all_stocks() -> pd.DataFrame:
    frames = []
    for f in LIST_STOCKS:
        if not os.path.exists(f):
            print(f"  WARNING: {f} not found – skipped")
            continue
        df = pd.read_csv(f)
        df["source_list"] = (
            os.path.basename(f)
            .replace("_notations.csv", "")
            .replace("_OnVista.csv", "")
            .replace("liste_", "")
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No notation CSV files found.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["idNotation"].notna() & (df_all["idNotation"] != "")]
    df_all["idNotation"] = df_all["idNotation"].astype(str)
    df_all.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df_all)} stocks from {len(frames)} list(s).")
    return df_all


# ── 3. FETCH W1 TICKS & COMPUTE Z-SCORE ──────────────────────────────────────

def fetch_and_score(row: pd.Series, session: requests.Session) -> dict | None:
    """
    Fetch the full W1 tick series for one stock, compute the Z-score of the
    latest 15-min return vs. the rolling window, and return a result dict
    (or None if data is insufficient or fetch fails).
    """
    isin        = str(row["isin"])
    id_notation = str(row["idNotation"])
    ticker      = row["ticker"]

    # Resolve entityValue from snapshot (needed for tick URL path)
    try:
        snap_url = f"https://api.onvista.de/api/v1/stocks/ISIN:{isin}/snapshot"
        r = session.get(snap_url, timeout=8)
        r.raise_for_status()
        entity_value = r.json().get("instrument", {}).get("entityValue")
        if not entity_value:
            return None
    except Exception as e:
        print(f"  [{ticker}] snapshot failed: {e}")
        return None

    # Fetch W1 tick series
    try:
        tick_url = TICK_URL.format(
            entity_value=entity_value,
            id_notation=id_notation,
        )
        r = session.get(tick_url, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [{ticker}] tick fetch failed: {e}")
        return None

    timestamps = data.get("datetimeTick", [])
    prices     = [float(p) for p in data.get("tick", [])]

    # Need at least WINDOW + 2 points: WINDOW for history + 1 prev + 1 latest
    if len(prices) < WINDOW + 2:
        return None

    # Only use the most recent WINDOW+1 prices to keep the window fresh
    recent = prices[-(WINDOW + 1):]

    # Percentage returns over the window
    returns = [
        (recent[i] - recent[i - 1]) / recent[i - 1] * 100
        for i in range(1, len(recent))
    ]

    latest_return  = returns[-1]
    window_returns = returns[-WINDOW:]
    mean           = sum(window_returns) / len(window_returns)
    variance       = sum((r - mean) ** 2 for r in window_returns) / len(window_returns)
    std            = variance ** 0.5

    if std == 0:
        return None

    z_score = (latest_return - mean) / std

    if abs(z_score) < ZSCORE_THRESHOLD:
        return None

    return {
        "ticker":      ticker,
        "isin":        isin,
        "branche":     row.get("branche", ""),
        "source_list": row.get("source_list", ""),
        "last_price":  recent[-1],
        "prev_price":  recent[-2],
        "ret_pct":     latest_return,
        "z_score":     z_score,
        "direction":   "LONG" if z_score > 0 else "SHORT",
        "last_ts":     timestamps[-1] if timestamps else None,
    }


# ── 4. SCAN ALL STOCKS ────────────────────────────────────────────────────────

def scan_for_signals(df_stocks: pd.DataFrame, session: requests.Session) -> list[dict]:
    signals = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_and_score, df_stocks.loc[i], session): i
            for i in df_stocks.index
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                signals.append(result)
    return signals


# ── 5. GENERATE & SEND ALERT ──────────────────────────────────────────────────

def generate_alert(signals: list[dict], threshold: float) -> None:
    if not signals:
        return

    now = datetime.now(GERMANY_TZ)
    df  = pd.DataFrame(signals)
    df.sort_values("z_score", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    base_url_scalable = "https://de.scalable.capital/broker/security?"

    df["Time"] = df["last_ts"].apply(
        lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc)
                          .astimezone(GERMANY_TZ)
                          .strftime("%H:%M")
        if x else "–"
    )
    df["ISIN_link"] = df["isin"].apply(
        lambda x: f'<a href="{base_url_scalable}isin={x}">{x}</a>'
    )
    df["Ticker_link"] = df["ticker"].apply(
        lambda x: (
            f'<a href="https://www.onvista.de/suche?searchValue='
            f'{re.sub(r"[^A-Za-z0-9]", "", x)}">{x}</a>'
        )
    )

    display = df[[
        "Ticker_link", "ISIN_link", "source_list",
        "last_price", "ret_pct", "z_score", "direction", "Time",
    ]].rename(columns={
        "Ticker_link": "Ticker",
        "ISIN_link":   "ISIN",
        "source_list": "Index",
        "last_price":  "Price",
        "ret_pct":     "Ret %",
        "z_score":     "Z-Score",
        "direction":   "Signal",
        "Time":        "Time",
    })

    def highlight_row(row):
        color = (
            "background-color:#c0392b;color:white"
            if row["Signal"].startswith("📉")
            else "background-color:#27ae60;color:white"
        )
        return [color] * len(row)

    filename = "momentum_signals.html"
    try:
        (
            display.style
            .format({
                "Price":   "{:,.2f}",
                "Ret %":   "{:+.3f}%",
                "Z-Score": "{:+.2f}",
            })
            .apply(highlight_row, axis=1)
            .set_caption(
                f"Momentum signals  |  Z ≥ {threshold:.1f}  |  "
                f"{now.strftime('%d.%m.%Y %H:%M')} CET"
            )
            .set_table_styles([
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "36px"),
                        ("font-weight", "bold"),
                        ("padding", "0.5em"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("padding", "0.5em"),
                        ("font-size", "22px"),
                        ("text-align", "center"),
                        ("background-color", "#111"),
                        ("color", "white"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("padding", "0.5em"),
                        ("font-size", "22px"),
                        ("text-align", "center"),
                    ],
                },
            ])
            .to_html(filename, escape=False)
        )
        caption = (
            f"{now.strftime('%d.%m.%Y')}: {len(df)} momentum signal(s) \nZ ≥ {threshold:.1f}"
        )
        send_telegram("sendDocument", filename=filename, caption=caption)
        print(f"  Alert sent: {len(df)} signal(s)")

    finally:
        if os.path.exists(filename):
            os.remove(filename)


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

def main():
    now_str = datetime.now(GERMANY_TZ).strftime("%d.%m.%Y %H:%M %Z")
    print(
        f"[{now_str}] Momentum scan  |  "
        f"Z≥{ZSCORE_THRESHOLD}  window={WINDOW} candles (~{WINDOW//4}h)"
    )

    df_stocks = load_all_stocks()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept":     "application/json",
        "Referer":    "https://www.onvista.de/",
    })

    signals = scan_for_signals(df_stocks, session)
    print(f"  {len(signals)} signal(s) found")

    generate_alert(signals, ZSCORE_THRESHOLD)


if __name__ == "__main__":
    main()