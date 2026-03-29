#!/usr/bin/env python
# coding: utf-8
"""
gpt_trader_analysis.py
======================
Drop-in module for both next_day_INDEX_prediction.py and next_day_DAX_prediction.py.

Usage:
    import gpt_trader_analysis
    advice = gpt_trader_analysis.get_trader_advice(signal)
    # returns a short HTML-formatted string ready for Telegram

Requires:
    pip install openai
    export OPENAI_API_KEY="sk-..."
"""

import os
import json
import traceback
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL      = "gpt-4o-mini"   # fast + cheap; swap to "gpt-4o" for higher quality
MAX_TOKENS     = 220              # keeps Telegram caption short


# ─────────────────────────────────────────────────────────
# SIGNAL DATACLASS  — populated by generate_messages()
# ─────────────────────────────────────────────────────────
@dataclass
class TradingSignal:
    # Identity
    ticker:       str
    date:         str           # "27.03.2026"

    # Daily indicators
    trend_long:   bool          # close > EMA200
    supertrend:   str           # "UP" | "DOWN"
    volatility:   str           # "HIGH" | "NORMAL" | "LOW"
    momentum:     str           # "Positive" | "Negative"
    rsi:          float         # 0–100
    adx:          float         # 0–100+

    # Prediction
    prob_up:      float         # 0.0–1.0
    wf_accuracy:  float         # 0.0–1.0
    has_intraday: bool

    # Intraday (optional — NaN when weekend/holiday)
    intra_day_pnl:  Optional[float] = None   # e.g. -1.8  (percent)
    intra_max_pnl:  Optional[float] = None   # e.g. +0.1
    intra_min_pnl:  Optional[float] = None   # e.g. -2.1
    intra_close_pos: Optional[float] = None  # 0=bottom of range, 1=top


# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────
def _build_prompt(s: TradingSignal) -> str:
    """
    Builds a compact, structured prompt for GPT.
    Keeps token count low while providing all relevant context.
    """
    prob_down  = 1.0 - s.prob_up
    direction  = "LONG" if s.prob_up >= 0.5 else "SHORT"
    conf       = max(s.prob_up, prob_down) * 100
    trend_str  = "above EMA200 (bullish)" if s.trend_long else "below EMA200 (bearish)"
    rsi_zone   = "overbought" if s.rsi > 70 else "oversold" if s.rsi < 30 else "neutral"
    adx_str    = "strong trend" if s.adx > 25 else "weak/no trend"
    vol_str    = s.volatility.lower()

    intra_str = "No intraday data available (weekend or holiday)."
    if s.has_intraday and s.intra_day_pnl is not None:
        close_zone = ""
        if s.intra_close_pos is not None:
            pct = s.intra_close_pos * 100
            if pct >= 75:   close_zone = " — closed near HIGH of day"
            elif pct <= 25: close_zone = " — closed near LOW of day"
            else:           close_zone = " — closed mid-range"
        intra_str = (
            f"Day P&L: {s.intra_day_pnl:+.1f}%{close_zone}. "
            f"Intraday range: high {s.intra_max_pnl:+.1f}%, low {s.intra_min_pnl:+.1f}%."
        )

    prompt = f"""You are a professional short-term trader. Analyze this end-of-day signal and give brief, direct trading advice for the NEXT trading day using a factor certificate on gettex (tradeable 09:00–22:00 CET).

INSTRUMENT: {s.ticker} — {s.date}

DAILY INDICATORS:
- Price vs EMA200: {trend_str}
- Supertrend: {s.supertrend}
- Momentum: {s.momentum}
- Volatility: {vol_str} (vs 50-day average)
- RSI: {s.rsi:.0f} ({rsi_zone})
- ADX: {s.adx:.0f} ({adx_str})

INTRADAY (today's underlying session):
{intra_str}

ML MODEL:
- Predicted direction: {direction} ({conf:.0f}% confidence)
- UP probability: {s.prob_up*100:.1f}% | DOWN probability: {prob_down*100:.1f}%
- Walk-forward accuracy (last ~1 year): {s.wf_accuracy*100:.1f}%

CONTEXT:
- Factor certificate: leveraged, daily reset, buy at today's close / tomorrow's open
- Maximum hold: 2 trading days
- Platform: Scalable Capital / gettex

Respond in 4–6 sentences maximum. Structure: (1) Trade or no trade and why. (2) If trade: direction, entry timing, stop level. (3) Position sizing given volatility. (4) One honest risk caveat. Be direct, no disclaimers."""

    return prompt


# ─────────────────────────────────────────────────────────
# MAIN FUNCTION — call this from generate_messages()
# ─────────────────────────────────────────────────────────
def get_trader_advice(signal: TradingSignal) -> str:
    """
    Sends the signal to GPT and returns a short HTML string
    ready to append to the Telegram caption.

    Returns a fallback string on any error — never raises.
    """
    if not OPENAI_API_KEY:
        return "🤖 <i>GPT advice unavailable (OPENAI_API_KEY not set)</i>"

    # Skip GPT when model has no demonstrated edge
    if signal.wf_accuracy is not None and signal.wf_accuracy < 0.52:
        return (
            f"🤖 <i>GPT skipped — model accuracy {signal.wf_accuracy*100:.1f}% "
            f"is at chance level. No tradeable edge detected.</i>"
        )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = _build_prompt(signal)

        response = client.chat.completions.create(
            model      = GPT_MODEL,
            max_tokens = MAX_TOKENS,
            temperature= 0.3,    # low temperature = consistent, focused answers
            messages   = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise, experienced short-term trader. "
                        "Give practical, actionable advice. Never hedge every sentence. "
                        "Respond in plain text — no markdown, no bullet points, no headers."
                    )
                },
                {"role": "user", "content": prompt},
            ],
        )

        advice = response.choices[0].message.content.strip()

        # Sanitise for Telegram HTML — angle brackets would break parse_mode
        advice = advice.replace("<", "‹").replace(">", "›")

        return f"🤖 <b>Trader view (GPT):</b>\n<i>{advice}</i>"

    except Exception:
        tb = traceback.format_exc()
        print(f"[GPT] Error for {signal.ticker}:\n{tb}")
        return "🤖 <i>GPT advice unavailable (API error)</i>"


# ─────────────────────────────────────────────────────────
# HELPER — build TradingSignal from the variables already
# computed inside generate_messages()
# ─────────────────────────────────────────────────────────
def build_signal(
    ticker:       str,
    latest:       object,   # df_ticker.iloc[-1]
    prob_up:      float,
    wf_acc:       float,
    has_intraday: bool,
    df_intra      = None,   # the raw intraday DataFrame
    intra_features: dict = field(default_factory=dict),
) -> TradingSignal:
    """
    Convenience constructor — pulls values from the same variables
    already available in generate_messages() so you don't have to
    re-extract anything manually.
    """
    # Intraday metrics
    intra_day_pnl  = None
    intra_max_pnl  = None
    intra_min_pnl  = None

    if has_intraday and df_intra is not None and len(df_intra) > 4:
        open_p = df_intra["open"].iloc[0]
        if open_p and open_p > 0:
            intra_day_pnl = (df_intra["close"].iloc[-1] / open_p - 1) * 100
            intra_max_pnl = (df_intra["high"].max()      / open_p - 1) * 100
            intra_min_pnl = (df_intra["low"].min()       / open_p - 1) * 100

    # Map volatility float → label
    vol_factor = float(latest["vol_factor"]) if latest["vol_factor"] is not None else 1.0
    if vol_factor > 1.2:   vol_label = "HIGH"
    elif vol_factor < 0.8: vol_label = "LOW"
    else:                  vol_label = "NORMAL"

    return TradingSignal(
        ticker        = str(ticker),
        date          = latest["date"].strftime("%d.%m.%Y"),
        trend_long    = bool(latest["trend_long"]),
        supertrend    = "UP" if latest["supertrend_dir"] == 1 else "DOWN",
        volatility    = vol_label,
        momentum      = "Positive" if float(latest["momentum_norm"]) > 0 else "Negative",
        rsi           = float(latest["RSI14"]) if latest["RSI14"] is not None else 50.0,
        adx           = float(latest["ADX14"]) if latest["ADX14"] is not None else 0.0,
        prob_up       = prob_up,
        wf_accuracy   = wf_acc if wf_acc is not None else 0.0,
        has_intraday  = has_intraday,
        intra_day_pnl = intra_day_pnl,
        intra_max_pnl = intra_max_pnl,
        intra_min_pnl = intra_min_pnl,
        intra_close_pos = intra_features.get("intra_close_pos") if intra_features else None,
    )