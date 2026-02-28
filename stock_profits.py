#!/usr/bin/env python3
"""
Scalable Capital Broker - Profit Calculator
============================================
Calculates realized profits for stock/ETF sells within a configurable lookback window.

Usage:
    python scalable_profit_calculator.py [csv_file] [--days N] [--telegram]

    csv_file   : Path to the CSV file (optional - will auto-detect by filename pattern)
    --days N   : Number of days to look back for sells (default: 7)
    --telegram : Send the profit summary to Telegram

CSV filename format: yyyy-mm-dd_hh-mm-ss_ScalableCapital-Broker-Transactions.csv

Environment variables required for Telegram:
    TELEGRAM_TOKEN_INDEX  — your bot token
    TELEGRAM_CHAT_ID     — your chat/channel ID
"""

import csv
import argparse
import glob
import json
import os
import sys
import requests
from datetime import datetime, timedelta
from collections import defaultdict


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_LOOKBACK_DAYS = 7          # How many days back to look for sells
CSV_DELIMITER         = ";"        # Column delimiter in the CSV
DATE_FORMAT           = "%Y-%m-%d" # Date format in the CSV
FILE_PATTERN          = "Scalable/*.csv"

# ── Telegram credentials ──────────────────────────────────────────────────────
#    Set these as environment variables, or replace the fallback strings below.
TELEGRAM_TOKEN_INDEX = os.environ.get("TELEGRAM_TOKEN_INDEX", "")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID",     "")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_amount(value: str) -> float:
    """Parse German-formatted numbers like '1.230,20' -> 1230.20"""
    if not value or value.strip() == "":
        return 0.0
    cleaned = value.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_datetime(date_str: str, time_str: str) -> datetime:
    """Combine date and time strings into a datetime object."""
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def find_latest_csv() -> str | None:
    """Auto-detect the most recent matching CSV file in the Scalable subfolder."""
    files = glob.glob(FILE_PATTERN)
    if not files:
        return None
    files.sort(reverse=True)
    return files[0]


# ── Core Logic ────────────────────────────────────────────────────────────────

def load_transactions(filepath: str) -> list[dict]:
    """Read all transactions from the CSV file into a list of dicts."""
    transactions = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        for row in reader:
            transactions.append(row)
    return transactions


def calculate_profits(transactions: list[dict], lookback_days: int) -> list[dict]:
    """
    For each recent sell, compute realized profit using FIFO lot matching.
    Buy lots are consumed in chronological order (oldest first), supporting
    partial consumption across multiple buy orders.
    Returns a list of result dicts, one per sell within the lookback window.
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)

    # ── 1. Sort all transactions chronologically ───────────────────────────
    all_sorted = sorted(
        transactions,
        key=lambda x: parse_datetime(x["date"], x["time"])
    )

    # ── 2. Build FIFO lot queues ───────────────────────────────────────────
    buy_lots = defaultdict(list)
    for t in all_sorted:
        if t.get("status") != "Executed" or t.get("assetType") != "Security":
            continue
        key = t.get("isin", "").strip() or t.get("description", "").strip()
        if t.get("type") == "Buy":
            shares = parse_amount(t.get("shares", "0"))
            price  = parse_amount(t.get("price",  "0"))
            if shares > 0:
                buy_lots[key].append({
                    "shares"   : shares,
                    "price"    : price,
                    "date"     : parse_datetime(t["date"], t["time"]),
                    "original" : shares,
                })

    # ── 3. Match each sell against the FIFO queue ──────────────────────────
    results = []

    for t in all_sorted:
        if t.get("status") != "Executed" or t.get("assetType") != "Security":
            continue
        if t.get("type") != "Sell":
            continue

        sell_dt     = parse_datetime(t["date"], t["time"])
        sell_shares = parse_amount(t.get("shares", "0"))
        sell_price  = parse_amount(t.get("price",  "0"))
        sell_amount = parse_amount(t.get("amount", "0"))
        currency    = t.get("currency", "EUR")
        key         = t.get("isin", "").strip() or t.get("description", "").strip()

        # ── Consume lots FIFO ──────────────────────────────────────────────
        remaining  = sell_shares
        cost_basis = 0.0
        lots_used  = []

        for lot in buy_lots[key]:          # already in chronological order
            if remaining <= 0:
                break
            consumed = min(lot["shares"], remaining)
            cost_basis += consumed * lot["price"]
            lots_used.append({
                "date"     : lot["date"],
                "price"    : lot["price"],
                "consumed" : consumed,
            })
            lot["shares"] -= consumed
            remaining     -= consumed

        # Remove fully exhausted lots
        buy_lots[key] = [l for l in buy_lots[key] if l["shares"] > 0]

        matched = (remaining == 0)   # False if history didn't cover all shares

        # ── Only include sells within the lookback window in the report ────
        if sell_dt < cutoff:
            continue

        if matched:
            profit = sell_amount - cost_basis

            # Build a compact summary of which lots were used
            buy_date_summary = ", ".join(
                f"{l['consumed']:.4g} @ {l['price']:.4f} ({l['date'].strftime('%Y-%m-%d')})"
                for l in lots_used
            )

            results.append({
                "description" : t.get("description"),
                "isin"        : t.get("isin"),
                "sell_date"   : sell_dt.strftime("%Y-%m-%d %H:%M"),
                "sell_shares" : sell_shares,
                "sell_price"  : sell_price,
                "sell_amount" : sell_amount,
                "buy_date"    : buy_date_summary,
                "buy_shares"  : sum(l["consumed"] for l in lots_used),
                "buy_price"   : cost_basis / sell_shares if sell_shares else 0,  # weighted avg
                "cost_basis"  : round(cost_basis, 4),
                "profit"      : round(profit, 4),
                "currency"    : currency,
                "matched"     : True,
            })
        else:
            results.append({
                "description" : t.get("description"),
                "isin"        : t.get("isin"),
                "sell_date"   : sell_dt.strftime("%Y-%m-%d %H:%M"),
                "sell_shares" : sell_shares,
                "sell_price"  : sell_price,
                "sell_amount" : sell_amount,
                "buy_date"    : "N/A (incomplete buy history)",
                "buy_shares"  : None,
                "buy_price"   : None,
                "cost_basis"  : None,
                "profit"      : None,
                "currency"    : currency,
                "matched"     : False,
            })

    return results


# ── Output / Reporting ────────────────────────────────────────────────────────

def print_report(results: list[dict], lookback_days: int, csv_file: str):
    """Pretty-print the profit report to stdout."""
    print()
    print("=" * 72)
    print("  Scalable Capital -- Realized Profit Calculator")
    print(f"  File        : {os.path.basename(csv_file)}")
    print(f"  Lookback    : last {lookback_days} day(s)")
    print(f"  Run at      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    if not results:
        print("\n  No executed sells found in the selected time window.\n")
        return

    matched      = [r for r in results if r["matched"]]
    unmatched    = [r for r in results if not r["matched"]]
    total_profit = sum(r["profit"] for r in matched)

    print(f"\n  {'ASSET':<35} {'SELL DATE':<17} {'SH':>4}  {'BUY EUR/sh':>10}  {'SELL EUR/sh':>11}  {'PROFIT':>10}")
    print(f"  {'-'*35} {'-'*17} {'-'*4}  {'-'*10}  {'-'*11}  {'-'*10}")

    for r in matched:
        profit_str = f"{r['profit']:+.2f} {r['currency']}"
        buy_p  = f"{r['buy_price']:.4f}"  if r['buy_price']  else "N/A"
        sell_p = f"{r['sell_price']:.4f}" if r['sell_price'] else "N/A"
        shares = int(r['sell_shares']) if r['sell_shares'] == int(r['sell_shares']) else r['sell_shares']
        print(
            f"  {r['description']:<35} {r['sell_date']:<17} {shares:>4}  "
            f"{buy_p:>10}  {sell_p:>11}  {profit_str:>13}"
        )
        print(
            f"    Buy: {r['buy_date']}  |  cost basis: {r['cost_basis']:.2f}  |  sell proceeds: {r['sell_amount']:.2f}"
        )

    print()
    print(f"  {'─'*68}")
    print(f"  {'TOTAL REALIZED PROFIT / LOSS':.<50} {total_profit:>+10.2f} EUR")
    print(f"  {'─'*68}")

    if unmatched:
        print(f"\n  WARNING: Could not find a matching Buy for {len(unmatched)} sell(s):")
        for r in unmatched:
            print(f"     * {r['description']} -- sold {r['sell_date']}  ({r['sell_amount']:.2f} {r['currency']})")
        print()

    print("=" * 72)
    print()


# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(method_name, text=None, url=None, document=None, filename=None, caption=None):
    TOKEN   = TELEGRAM_TOKEN_INDEX
    API_URL = f'https://api.telegram.org/bot{TOKEN}/{method_name}'

    keyboard = {
        "inline_keyboard": [
            [{"text": "interactive chart", "url": url}]
        ]
    }
    params_text = {
        'chat_id'    : TELEGRAM_CHAT_ID,
        'text'       : text,
        'parse_mode' : 'HTML'
    }
    params_file = {
        'chat_id'              : TELEGRAM_CHAT_ID,
        'caption'              : caption,
        'parse_mode'           : 'HTML',
        'disable_notification' : True,
        'reply_markup'         : json.dumps(keyboard)
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
            return True
        else:
            print(f"  Telegram API error {resp_json.get('error_code')}: {resp_json.get('description')}")
            return False

    except Exception as e:
        print(f"  Telegram sending failed: {e}")
        return False


def build_telegram_summary(results: list[dict], lookback_days: int) -> str:
    """Build the HTML-formatted Telegram summary message."""
    matched   = [r for r in results if r["matched"]]
    unmatched = [r for r in results if not r["matched"]]

    profits = [r for r in matched if r["profit"] >= 0]
    losses  = [r for r in matched if r["profit"] <  0]

    total_profit = sum(r["profit"] for r in profits)
    total_loss   = sum(r["profit"] for r in losses)
    total_net    = total_profit + total_loss

    date_to    = datetime.now()
    date_from  = date_to - timedelta(days=lookback_days)
    date_range = f"{date_from.strftime('%d.%m.%y')} - {date_to.strftime('%d.%m.%y')}"

    net_sign  = "+" if total_net >= 0 else ""
    net_emoji = "🟢" if total_net >= 0 else "🔴"

    lines = [
        "📊 <b>Stocks Profitability</b>",
        f"Last {lookback_days} days · {date_range}",
        "",
        f"✅ Profit:  <b>+{total_profit:.2f} EUR</b>  ({len(profits)} order{'s' if len(profits) != 1 else ''})",
        f"❌ Loss:    <b>{total_loss:.2f} EUR</b>  ({len(losses)} order{'s' if len(losses) != 1 else ''})",
        "─" * 18,
        f"{net_emoji} Total:   <b>{net_sign}{total_net:.2f} EUR</b>",
    ]

    if unmatched:
        lines += [
            "",
            f"⚠️ <i>{len(unmatched)} sell(s) without matching buy (older history needed)</i>",
        ]

    return "\n".join(lines)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calculate realized profits from Scalable Capital broker CSV exports."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="Path to the CSV file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Lookback window in days for sells (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send the profit summary to Telegram",
    )
    args = parser.parse_args()

    # ── Resolve the CSV file ────────────────────────────────────────────────
    if args.csv_file:
        csv_file = args.csv_file
        if not os.path.isfile(csv_file):
            print(f"ERROR: File not found: {csv_file}", file=sys.stderr)
            sys.exit(1)
    else:
        csv_file = find_latest_csv()
        if not csv_file:
            print(
                f"ERROR: No CSV file found matching '{FILE_PATTERN}'.\n"
                "Please pass the file path as an argument or run the script\n"
                "from the directory containing your transaction CSV.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected file: {csv_file}")

    # ── Load & calculate ────────────────────────────────────────────────────
    transactions = load_transactions(csv_file)
    results      = calculate_profits(transactions, args.days)
    print_report(results, args.days, csv_file)

    # ── Send Telegram summary ───────────────────────────────────────────────
    if args.telegram:
        # Validate credentials before attempting to send
        missing = []
        if not TELEGRAM_TOKEN_INDEX: missing.append("TELEGRAM_TOKEN_INDEX")
        if not TELEGRAM_CHAT_ID:     missing.append("TELEGRAM_CHAT_ID")
        if missing:
            print(f"ERROR: Telegram not configured -- missing env var(s): {', '.join(missing)}", file=sys.stderr)
            print("Set them with: export TELEGRAM_TOKEN_INDEX=... && export TELEGRAM_CHAT_ID=...", file=sys.stderr)
            sys.exit(1)

        summary = build_telegram_summary(results, args.days)
        print("Sending Telegram summary...")
        print(f"  Token : ...{TELEGRAM_TOKEN_INDEX[-6:]}")  # show last 6 chars only for safety
        print(f"  Chat  : {TELEGRAM_CHAT_ID}")
        ok = send_telegram("sendMessage", text=summary)
        if ok:
            print("✓ Telegram message sent successfully.")
        else:
            print("✗ Telegram message failed -- check token and chat ID.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()