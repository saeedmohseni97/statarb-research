"""Phase 1 tests. Fully offline & deterministic: seeded synthetic series are the oracles.

We KNOW the ground truth by construction:
  * a stationary series (white noise) vs a unit-root series (random walk);
  * a cointegrated pair (y = const + beta*x + stationary noise, x a random walk);
  * an independent pair (two unrelated random walks) that must NOT cointegrate.
"""
import numpy as np
import pandas as pd

from statarb import stats


# ---------- synthetic oracles ------------------------------------------------
def _random_walk(seed, n=500, scale=1.0, start=50.0):
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.normal(0.0, scale, n))


def _white_noise(seed, n=500):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n)


def _coint_pair(seed, n=500, beta=1.5, const=3.0, noise=1.0):
    """x is a random walk; y is const + beta*x + STATIONARY noise -> y,x cointegrated."""
    rng = np.random.default_rng(seed)
    x = 50.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    y = const + beta * x + rng.normal(0.0, noise, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


def _indep_pair(seed_y, seed_x, n=500):
    idx = pd.bdate_range("2015-01-01", periods=n)
    return (pd.Series(_random_walk(seed_y, n), index=idx),
            pd.Series(_random_walk(seed_x, n), index=idx))


# ---------- ADF --------------------------------------------------------------
def test_adf_flags_stationary_and_unit_root():
    stationary = stats.adf_test(_white_noise(0))
    unit_root = stats.adf_test(_random_walk(0))
    assert stationary.is_stationary is True
    assert stationary.pvalue < 0.05
    assert unit_root.is_stationary is False
    assert unit_root.pvalue > 0.10
    # crit values are reported for the three standard levels
    assert set(stationary.crit) == {"1%", "5%", "10%"}


# ---------- Engle-Granger ----------------------------------------------------
def test_engle_granger_detects_cointegration_and_hedge():
    y, x = _coint_pair(1)
    res = stats.engle_granger(y, x)
    assert res.is_cointegrated is True
    assert res.pvalue < 0.05
    # OLS should recover the true hedge ratio (beta=1.5) closely
    assert abs(res.hedge_ratio - 1.5) < 0.1


def test_engle_granger_rejects_independent_pair():
    y, x = _indep_pair(10, 20)
    res = stats.engle_granger(y, x)
    assert res.is_cointegrated is False
    assert res.pvalue > 0.10


# ---------- screen_pairs -----------------------------------------------------
def _mixed_universe():
    """4 tickers: A&B are cointegrated; C and D are independent random walks."""
    y, x = _coint_pair(2)          # -> columns A (y), B (x)
    c = pd.Series(_random_walk(30, len(x)), index=x.index)
    d = pd.Series(_random_walk(40, len(x)), index=x.index)
    return pd.DataFrame({"A": y, "B": x, "C": c, "D": d})


def test_screen_pairs_shape_columns_and_counts():
    df = stats.screen_pairs(_mixed_universe())
    # C(4,2) = 6 pairs, one row each
    assert len(df) == 6
    assert df.attrs["n_tests"] == 6
    expected_cols = {"y", "x", "eg_tstat", "hedge_ratio",
                     "pvalue", "pvalue_adj", "significant_raw", "significant_adj"}
    assert expected_cols.issubset(df.columns)


def test_screen_pairs_finds_the_true_pair_first():
    df = stats.screen_pairs(_mixed_universe())
    # sorted by pvalue ascending -> the A/B cointegrated pair is the top row
    top = df.iloc[0]
    assert {top["y"], top["x"]} == {"A", "B"}
    assert bool(top["significant_raw"]) is True


def test_screen_pairs_adjusted_pvalues_are_not_smaller():
    df = stats.screen_pairs(_mixed_universe())
    # multiple-testing correction can only make p-values larger (or equal)
    assert (df["pvalue_adj"] >= df["pvalue"] - 1e-9).all()
    # at least as many raw 'hits' as adjusted 'hits'
    assert int(df["significant_raw"].sum()) >= int(df["significant_adj"].sum())
