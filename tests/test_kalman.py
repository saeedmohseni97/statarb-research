"""Phase 5 tests — kalman.py. Offline, deterministic, seeded synthetic oracles."""
import numpy as np
import pandas as pd
import pytest

from statarb import kalman as K
from statarb.signals import hedge_ratio


def _const_beta(seed=0, n=800, a=2.0, b=1.5, noise=0.5):
    """y = a + b*x + noise, x a random walk. True beta is the constant b."""
    rng = np.random.default_rng(seed)
    x = 50 + np.cumsum(rng.normal(0, 1, n))
    y = a + b * x + rng.normal(0, noise, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, index=idx), pd.Series(x, index=idx), b


def _drift_beta(seed=0, n=1500, a=1.0, b0=1.0, b1=2.5, noise=0.3):
    """y with a beta that drifts linearly from b0 to b1 — the case a static hedge misses."""
    rng = np.random.default_rng(seed)
    x = 50 + np.cumsum(rng.normal(0, 1, n))
    beta_path = np.linspace(b0, b1, n)
    y = a + beta_path * x + rng.normal(0, noise, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, index=idx), pd.Series(x, index=idx), pd.Series(beta_path, index=idx)


# --- kalman_filter_hedge --------------------------------------------------
def test_filter_shape_and_columns():
    y, x, _ = _const_beta(1)
    f = K.kalman_filter_hedge(y, x)
    assert list(f.columns) == ["beta", "alpha", "spread", "var"]
    assert len(f) == len(y)
    assert (f["var"] > 0).all()


def test_recovers_constant_beta():
    y, x, b = _const_beta(2, n=1500, b=1.5, noise=0.3)
    f = K.kalman_filter_hedge(y, x, delta=1e-4, obs_var=1e-3)
    assert abs(f["beta"].iloc[-1] - b) < 0.15


def test_filter_is_causal():
    y, x, _ = _const_beta(3)
    f = K.kalman_filter_hedge(y, x)
    y2 = y.copy(); y2.iloc[-1] += 100.0
    f2 = K.kalman_filter_hedge(y2, x)
    assert np.allclose(f["beta"].iloc[:-1], f2["beta"].iloc[:-1])   # past not rewritten


def test_deterministic():
    y, x, _ = _const_beta(4)
    a = K.kalman_filter_hedge(y, x)
    b = K.kalman_filter_hedge(y, x)
    assert np.allclose(a["beta"], b["beta"]) and np.allclose(a["var"], b["var"])


def test_larger_delta_more_variable_beta():
    y, x, _ = _const_beta(5, n=1200)
    # bar-to-bar jitter in beta (not overall spread) is what delta controls
    lo = K.kalman_filter_hedge(y, x, delta=1e-6)["beta"].diff().std()
    hi = K.kalman_filter_hedge(y, x, delta=1e-2)["beta"].diff().std()
    assert hi > lo                                                   # more process noise -> livelier beta


def test_tracks_drift_better_than_static_ols():
    y, x, bpath = _drift_beta(6, n=1500, b0=1.0, b1=2.5, noise=0.3)
    f = K.kalman_filter_hedge(y, x, delta=1e-3, obs_var=1e-3)
    err_kalman = abs(f["beta"].iloc[-1] - bpath.iloc[-1])
    err_static = abs(hedge_ratio(y, x) - bpath.iloc[-1])
    assert err_kalman < err_static                                  # the whole point of Phase 5


# --- kalman_zscore --------------------------------------------------------
def test_zscore_matches_and_finite():
    y, x, _ = _const_beta(7)
    z = K.kalman_zscore(y, x)
    f = K.kalman_filter_hedge(y, x)
    assert np.allclose(z, f["spread"] / np.sqrt(f["var"]))
    assert np.isfinite(z.iloc[50:]).all()


# --- kalman_backtest ------------------------------------------------------
def test_backtest_structure():
    y, x, _ = _const_beta(8, n=1000)
    res = K.kalman_backtest(y, x, entry=1.0, exit=0.0, cost=0.0, warmup=50)
    for k in ["beta", "alpha", "spread", "z", "positions", "pnl", "equity", "stats"]:
        assert k in res
    assert set(pd.unique(res["positions"])).issubset({-1, 0, 1})
    assert len(res["pnl"]) == len(y)
    assert np.allclose(res["equity"], res["pnl"].cumsum())


def test_backtest_warmup_stays_flat():
    y, x, _ = _const_beta(9)
    res = K.kalman_backtest(y, x, warmup=100)
    assert (res["positions"].iloc[:100] == 0).all()


def test_backtest_no_lookahead():
    y, x, _ = _const_beta(10, n=900)
    res = K.kalman_backtest(y, x, warmup=50)
    y2 = y.copy(); y2.iloc[-1] += 50.0
    res2 = K.kalman_backtest(y2, x, warmup=50)
    assert np.allclose(res["positions"].iloc[:-1], res2["positions"].iloc[:-1])


def test_costs_reduce_pnl():
    y, x, _ = _const_beta(11, n=1000)
    free = K.kalman_backtest(y, x, cost=0.0, warmup=50)["stats"]["total_pnl"]
    charged = K.kalman_backtest(y, x, cost=0.01, warmup=50)["stats"]["total_pnl"]
    assert charged <= free
