"""Project configuration — the *default* research universe and its metadata.

Design rule: the universe is always a **parameter**, never hard-wired into logic.
Every function downstream (screening, signals, backtest, dashboard) operates on
whatever price frame / ticker list it is handed. What lives here is only the
*default example* the shipped pipeline runs on — a user can supply their own list
and nothing else has to change.

Default study universe: the US semiconductor sector.
Chosen because it is a tightly economically-tethered group (shared chip-cycle
demand, capex, supply chains), which is what makes cointegration *meaningful*
rather than spurious. Note that high correlation across semis does NOT guarantee
cointegration — many pairs move together yet have drifted apart structurally
(so their spread is non-stationary). That distinction is exactly what Phase 1
sets out to measure honestly.
"""
from __future__ import annotations

# --- The default universe -------------------------------------------------
# Grouped by sub-cluster (where cointegration is a priori more vs. less likely).
UNIVERSE_GROUPS: dict[str, list[str]] = {
    # Wafer-fab equipment makers — the tightest sub-group.
    "equipment": ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
    # Analog / embedded / mixed-signal.
    "analog": ["TXN", "ADI", "MCHP", "NXPI", "ON"],
    # Fabless designers.
    "fabless": ["NVDA", "AMD", "QCOM", "AVGO", "MRVL"],
    # Integrated device makers / memory / foundry.
    "idm_foundry": ["INTC", "MU", "TSM", "STM"],
    # Near-identical semiconductor ETFs — a built-in POSITIVE CONTROL: these
    # track almost the same basket and *should* screen as strongly cointegrated.
    "etf_control": ["SMH", "SOXX"],
}

# Flat default universe (order preserved: equipment -> ... -> etf_control).
DEFAULT_UNIVERSE: list[str] = [t for group in UNIVERSE_GROUPS.values() for t in group]

# Default backtest window for the shipped example pipeline.
DEFAULT_START = "2013-01-01"
DEFAULT_END = "2024-12-31"

# Where the cached price CSV for the default universe lives.
DEFAULT_CACHE = "data/semis.csv"


# --- A second, cross-sector universe: classic ECONOMIC pairs ------------------
# The semis universe is the honest null (high correlation, no cointegration). This
# second universe holds economically-motivated pairs — same business, same demand
# driver — chosen a priori (NOT by mining p-values). The point: with an economic
# prior instead of a blind search, the pipeline finds a real, defensible, profitable
# strategy (MA~V, the card-network duopoly, is the featured result). Same design
# rule: just another ticker list; nothing downstream changes.
PAIRS_GROUPS: dict[str, list[str]] = {
    "payments":    ["MA", "V"],            # card-network duopoly — the featured pair
    "staples":     ["KO", "PEP"],          # consumer-staples rivals
    "energy":      ["XOM", "CVX"],         # integrated oil majors
    "home_improv": ["HD", "LOW"],          # home-improvement retail
    "banks":       ["GS", "MS"],           # bulge-bracket banks
    "retail":      ["WMT", "TGT"],         # big-box retail
    "gold_etf":    ["GLD", "IAU"],         # two gold trackers (secondary positive)
}
PAIRS_UNIVERSE: list[str] = [t for group in PAIRS_GROUPS.values() for t in group]
PAIRS_CACHE = "data/pairs.csv"
# The pre-declared headline pair (economic prior, not a search result).
FEATURED_PAIR = ("MA", "V")


def group_of(ticker: str) -> str | None:
    """Return the sub-cluster a ticker belongs to (or None if not in the default set)."""
    for name, members in UNIVERSE_GROUPS.items():
        if ticker in members:
            return name
    return None
