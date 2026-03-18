#!/usr/bin/env python3
"""
Fetches constituents for each index defined in DICT_INDEXES from the onvista
API and enriches each stock with its gettex (market code _TRO) idNotation
from the snapshot endpoint.

Output: one CSV per index, e.g. DAX_notations.csv, MDAX_notations.csv, …
"""

import csv
import time
import requests

CONSTITUENTS_URL = "https://api.onvista.de/api/v2/indices/{index_id}/constituents?order=ASC&sort=NAME"
SNAPSHOT_URL = "https://api.onvista.de/api/v1/stocks/ISIN:{isin}/snapshot"
GETTEX_CODE = "_TRO"
REQUEST_DELAY = 0.2  # seconds between snapshot requests

DICT_INDEXES = {
    "DAX40":    20735,
    "MDAX":   323547,
    "SDAX":   324724,
    "NASDAQ100": 325104,
}


def fetch_constituents(index_id: int) -> list[dict]:
    """Return the list of constituent entries for the given index ID."""
    url = CONSTITUENTS_URL.format(index_id=index_id)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["list"]


def fetch_snapshot_full(isin: str) -> dict:
    """Return the full snapshot JSON for the given ISIN."""
    url = SNAPSHOT_URL.format(isin=isin)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def process_index(index_name: str, index_id: int) -> None:
    """Fetch all constituents for one index and write the enriched CSV."""
    output_file = f"liste_{index_name}_OnVista.csv"

    print(f"\n{'='*60}")
    print(f"Index: {index_name} (id={index_id})")
    print(f"{'='*60}")
    print("Fetching constituents ...")

    constituents = fetch_constituents(index_id)
    total = len(constituents)
    print(f"  -> {total} stocks found\n")

    rows = []
    for i, entry in enumerate(constituents, start=1):
        instrument = entry["instrument"]
        name = instrument["name"]
        isin = instrument["isin"]

        print(f"[{i:0{len(str(total))}d}/{total}] {name} ({isin})", end=" ... ", flush=True)

        try:
            snapshot = fetch_snapshot_full(isin)

            # gettex idNotation
            id_notation = None
            for quote in snapshot.get("quoteList", {}).get("list", []):
                if quote.get("market", {}).get("codeMarket") == GETTEX_CODE:
                    id_notation = quote["market"]["idNotation"]
                    break

            # Branch name (e.g. "Sportartikel")
            branch = (
                snapshot.get("company", {})
                        .get("branch", {})
                        .get("name", "")
            )

            print(f"idNotation={id_notation}, branch={branch}")

        except Exception as exc:
            print(f"ERROR - {exc}")
            id_notation = None
            branch = ""

        rows.append({
            "ticker":     name,
            "isin":       isin,
            "branche":    branch,
            "idNotation": id_notation if id_notation is not None else "",
        })

        time.sleep(REQUEST_DELAY)

    # Write CSV
    with open(output_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["ticker", "isin", "branche", "idNotation"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Results written to {output_file}")

    missing = [r for r in rows if not r["idNotation"]]
    if missing:
        print(f"  WARNING: {len(missing)} stock(s) had no gettex listing:")
        for r in missing:
            print(f"    - {r['ticker']} ({r['isin']})")


def main() -> None:
    for index_name, index_id in DICT_INDEXES.items():
        process_index(index_name, index_id)

    print("\nAll indexes processed.")


if __name__ == "__main__":
    main()