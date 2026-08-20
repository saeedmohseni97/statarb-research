"""Fetch the cross-sector 'pairs demo' universe to data/pairs.csv.

Run this ONCE on a machine with internet access (it uses yfinance):

    python scripts/fetch_pairs.py

It downloads adjusted daily closes for the tickers in config.PAIRS_UNIVERSE over the
default window and caches them to data/pairs.csv (the same format as data/semis.csv).
After it finishes, the dashboard and the results notebook can use this universe offline.

This universe deliberately mixes:
  * near-arbitrage pairs (share classes GOOG/GOOGL; same-index ETFs SPY/IVV/VOO, GLD/IAU)
    where cointegration is essentially guaranteed — a working positive control, and
  * classic economic pairs (KO/PEP, MA/V, XOM/CVX, HD/LOW, GS/MS, WMT/TGT) whose
    cointegration is plausible but must be tested honestly.
"""
from __future__ import annotations

import os
import sys

# make the repo importable whether run from repo root or scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from statarb import config, data


def main() -> None:
    tickers = config.PAIRS_UNIVERSE
    cache = os.path.join(ROOT, config.PAIRS_CACHE)
    print(f"Downloading {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"Window: {config.DEFAULT_START} .. {config.DEFAULT_END}")
    px = data.get_prices(tickers, config.DEFAULT_START, config.DEFAULT_END, cache_path=cache)
    px = data.align_prices(px)
    # basic sanity: warn on any ticker that came back empty
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        print(f"WARNING: no data returned for: {', '.join(missing)} "
              "(check the ticker symbols / your network).")
    print(f"Saved {config.PAIRS_CACHE}: {px.shape[0]} rows x {px.shape[1]} names, "
          f"{px.index.min().date()} .. {px.index.max().date()}")


if __name__ == "__main__":
    main()
