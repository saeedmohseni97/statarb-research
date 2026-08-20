"""Phase 1 — stationarity & cointegration.

You implement four things here:
  1. ADFResult / EGResult   — small result containers (dataclasses).
  2. adf_test(series, ...)   — wrap statsmodels adfuller; report is_stationary.
  3. engle_granger(y, x, ...)— two-step cointegration test + OLS hedge ratio.
  4. screen_pairs(prices,...)— test EVERY pair, then correct for multiple testing.

The notebook (notebooks/01_cointegration.ipynb) has the exact specs, formulas, and a
worked example. Implement until tests/test_stats.py is green, then self-check against
.solutions/stats_reference.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class ADFResult:
    statistic: float
    pvalue: float
    used_lag: int
    nobs: int
    crit: dict           # {"1%": .., "5%": .., "10%": ..}
    is_stationary: bool  # pvalue < alpha


@dataclass
class EGResult:
    tstat: float
    pvalue: float
    crit: dict           # Engle-Granger critical values {"1%","5%","10%"}
    hedge_ratio: float   # OLS slope beta of y on x
    const: float         # OLS intercept
    is_cointegrated: bool


def adf_test(series, regression: str = "c", autolag: str = "AIC", alpha: float = 0.05) -> "ADFResult":
    """Augmented Dickey-Fuller test for a unit root.

    H0: series has a unit root (non-stationary). Small p-value -> reject H0 ->
    stationary. Drop NaNs first. Return an ADFResult with is_stationary=(pvalue<alpha).
    Hint: from statsmodels.tsa.stattools import adfuller
          stat, pvalue, used_lag, nobs, crit, icbest = adfuller(x, regression=..., autolag=...)
    """
    # TODO
    from statsmodels.tsa.stattools import adfuller 
    stat, pvalue, used_lag, nobs, crit, icbest = adfuller(series, regression=regression, autolag=autolag)
    return ADFResult(
        statistic = stat,
        pvalue = pvalue,
        used_lag = used_lag,
        nobs = nobs,
        crit = crit,        # {"1%": .., "5%": .., "10%": ..}
        is_stationary = bool(pvalue<alpha)  # pvalue < alpha
    )
    raise NotImplementedError


def engle_granger(y, x, alpha: float = 0.05) -> "EGResult":
    """Engle-Granger two-step cointegration test of y on x.

    Align y & x (dropna together). Use statsmodels.tsa.stattools.coint(y, x) for the
    (tstat, pvalue, crit) — it applies the correct EG critical values. Separately fit
    an OLS of y on x WITH an intercept to report (const, hedge_ratio=beta); later
    phases build the spread from that hedge ratio. is_cointegrated = (pvalue < alpha).
    NOTE: the test is direction-sensitive — fix the convention y ~ x.
    Hint for the OLS: X = np.column_stack([np.ones_like(x), x]);
                      coef, *_ = np.linalg.lstsq(X, y, rcond=None)  # -> [const, beta]
    """
    # TODO
    from statsmodels.tsa.stattools import coint
    tstat, pvalue, crit = coint(y,x)
    const, beta = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y)[0] 
    return EGResult(
            tstat = tstat,
            pvalue = pvalue, 
            crit = crit,           # Engle-Granger critical values {"1%","5%","10%"}
            hedge_ratio = beta,    # OLS slope beta of y on x
            const = const,          # OLS intercept
            is_cointegrated = bool(pvalue<alpha)
    )



def screen_pairs(prices: pd.DataFrame, alpha: float = 0.05,
                 method: str = "fdr_bh", use_log: bool = True) -> pd.DataFrame:
    """Screen EVERY pair for cointegration, honestly.

    For each unordered pair (use itertools.combinations of the columns; convention
    y=first, x=second) run engle_granger on the (log, if use_log) prices and collect
    the raw p-value, eg_tstat, and hedge_ratio. Then correct for multiple testing with
    statsmodels.stats.multitest.multipletests(pvals, alpha=alpha, method=method), which
    returns (reject, pvals_corrected, ...). With n tickers that's C(n,2) tests, so ~5%
    of *unrelated* pairs would look significant by chance at alpha=0.05 — the correction
    is what keeps the screen honest.

    Return a DataFrame, one row per pair, sorted by raw pvalue ascending, with columns:
        y, x, eg_tstat, hedge_ratio, pvalue, pvalue_adj, significant_raw, significant_adj
    and set df.attrs['n_tests'] = number of pairs tested.
    """
    # TODO
    from itertools import combinations 
    from statsmodels.stats.multitest import multipletests
    px = np.log(prices) if use_log else prices
    cols = list(px.columns)
    rows = []
    for a, b in combinations(cols, 2):
        res = engle_granger(px[a], px[b], alpha=alpha)
        rows.append({"y": a, "x": b, "eg_tstat": res.tstat,
                     "hedge_ratio": res.hedge_ratio, "pvalue": res.pvalue})
    df = pd.DataFrame(rows)

    reject, p_adj, _, _ = multipletests(df["pvalue"].to_numpy(), alpha=alpha, method=method)
    df["pvalue_adj"] = p_adj
    df["significant_raw"] = df["pvalue"] < alpha
    df["significant_adj"] = reject

    df = df.sort_values("pvalue").reset_index(drop=True)
    df.attrs["n_tests"] = len(df)
    return df
    raise NotImplementedError
