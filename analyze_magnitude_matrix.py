#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm

from index_information import (
    setup_database,
    momentum_colors,
    LIST_INDEX,
    DATA_FILE_INDEX
)


# =========================================
# Helpers
# =========================================

def detect_sequences(colors, target_color, length):
    seq = (pd.Series(colors) == target_color).values
    result = np.zeros(len(seq), dtype=bool)

    for i in range(len(seq) - length + 1):
        if seq[i:i+length].all():
            result[i] = True

    return result


def forward_consecutive_candles(df, length, bullish=True):
    result = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        ok = True

        for j in range(1, length + 1):
            if i + j >= len(df):
                ok = False
                break

            if bullish:
                if not (df["close"].iloc[i+j] > df["open"].iloc[i+j]):
                    ok = False
                    break
            else:
                if not (df["close"].iloc[i+j] < df["open"].iloc[i+j]):
                    ok = False
                    break

        result[i] = ok

    return result


def forward_return(df, days):
    return (df["close"].shift(-days) / df["close"] - 1)


# =========================================
# Core Analysis
# =========================================

def analyze_index(df, ticker):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["mom_color"] = momentum_colors(df)

    results = []

    for trend_state in [True, False]:
        df_trend = df[df["trend_long"] == trend_state].copy()
        if len(df_trend) < 20:
            continue

        for signal_len in [1, 2, 3]:
            for fwd_len in [1, 2, 3]:

                returns = forward_return(df_trend, fwd_len)

                # =====================================
                # GREEN
                # =====================================
                seq_mask = detect_sequences(df_trend["mom_color"], "lime", signal_len)
                candle_mask = forward_consecutive_candles(df_trend, fwd_len, bullish=True)

                valid = seq_mask & candle_mask
                selected_returns = returns[valid]

                total = seq_mask.sum()
                matches = valid.sum()

                results.append({
                    "ticker": ticker,
                    "type": "GREEN",
                    "trend": "ON" if trend_state else "OFF",
                    "signal_len": signal_len,
                    "fwd_len": fwd_len,
                    "occurrences": int(total),
                    "matches": int(matches),
                    "hit_rate": matches / total if total > 0 else None,

                    # raw stats
                    "avg_return": selected_returns.mean(),
                    "median_return": selected_returns.median(),
                    "std_return": selected_returns.std(),

                    # derived metrics (NEW)
                    "edge_score": selected_returns.mean(),
                    "risk_proxy": abs(selected_returns.std() if len(selected_returns) > 0 else np.nan),
                })

                # =====================================
                # RED
                # =====================================
                seq_mask = detect_sequences(df_trend["mom_color"], "maroon", signal_len)
                candle_mask = forward_consecutive_candles(df_trend, fwd_len, bullish=False)

                valid = seq_mask & candle_mask
                selected_returns = returns[valid]

                total = seq_mask.sum()
                matches = valid.sum()

                results.append({
                    "ticker": ticker,
                    "type": "RED",
                    "trend": "ON" if trend_state else "OFF",
                    "signal_len": signal_len,
                    "fwd_len": fwd_len,
                    "occurrences": int(total),
                    "matches": int(matches),
                    "hit_rate": matches / total if total > 0 else None,

                    # raw stats
                    "avg_return": selected_returns.mean(),
                    "median_return": selected_returns.median(),
                    "std_return": selected_returns.std(),

                    # derived metrics
                    "edge_score": selected_returns.mean(),
                    "risk_proxy": abs(selected_returns.std() if len(selected_returns) > 0 else np.nan),
                })

    return pd.DataFrame(results)


# =========================================
# Main
# =========================================

def main():

    print("🚀 Running full analysis...")

    df = setup_database(LIST_INDEX, DATA_FILE_INDEX)

    all_results = []

    for ticker in tqdm(df["ticker"].unique(), desc="Analyzing indices"):
        df_t = df[df["ticker"] == ticker]
        res = analyze_index(df_t, ticker)
        all_results.append(res)

    final_df = pd.concat(all_results, ignore_index=True)

    # =========================================
    # FINAL ENRICHMENT (GLOBAL METRICS)
    # =========================================

    final_df["avg_return"] = final_df["avg_return"].round(4)
    final_df["median_return"] = final_df["median_return"].round(4)
    final_df["std_return"] = final_df["std_return"].round(4)
    final_df["hit_rate"] = final_df["hit_rate"].round(3)

    # Risk-adjusted edge (important for factor certs)
    final_df["risk_adj_edge"] = final_df["edge_score"] / (final_df["risk_proxy"] + 1e-6)

    # Simple quality score
    final_df["quality_score"] = final_df["matches"] / (final_df["occurrences"] + 1e-6)

    # =========================================
    # SINGLE OUTPUT FILE
    # =========================================
    final_df.to_csv("pattern_full_matrix.csv", index=False)

    print("\n✅ ONE FILE GENERATED:")
    print(" - pattern_full_matrix.csv")

    print("\n📊 Preview:")
    print(final_df.head())


if __name__ == "__main__":
    main()