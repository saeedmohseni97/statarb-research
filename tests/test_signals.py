"""Phase 2 tests. Offline & deterministic: seeded synthetic series with KNOWN properties.

  * y = const + beta*x + noise -> hedge_ratio should recover beta; spread ~ centered noise.
  * an explicit AR(1) s_t = phi*s_{t-1} + e_t has a known mean-reversion speed -> half-life.
  * a hand-built z-score path exercises the entry/exit hysteresis exactly.
"""
import numpy as np
import pandas as pd

from statarb import signals


# ---------- oracles ----------------------------------------------------------
def _pair(seed, n=600, beta=1.5, const=3.0, noise=1.0):
    rng = np.random.default_rng(seed)
    x = 50.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    y = const + beta * x + rng.normal(0.0, noise, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


def _ar1(seed, n=3000, phi=0.9, sigma=1.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + e[t]
    return pd.Series(s)


# ---------- hedge ratio & spread --------------------------------------------
def test_hedge_ratio_recovers_beta():
    y, x = _pair(1)
    assert abs(signals.hedge_ratio(y, x) - 1.5) < 0.05


def test_spread_is_centered():
    y, x = _pair(2)
    s = signals.spread(y, x)
    assert len(s) == len(x)
    # OLS residuals are mean-zero by construction
    assert abs(float(s.mean())) < 1e-6


# ---------- half-life --------------------------------------------------------
def test_half_life_positive_and_reasonable():
    hl = signals.half_life(_ar1(0, phi=0.9))
    # discrete truth ln(0.5)/ln(0.9) ~ 6.6 periods
    assert 4.0 < hl < 10.0


def test_half_life_faster_reversion_is_shorter():
    fast = signals.half_life(_ar1(0, phi=0.5))   # ~1 period
    slow = signals.half_life(_ar1(0, phi=0.95))  # ~13.5 periods
    assert fast < slow


# ---------- z-score ----------------------------------------------------------
def test_zscore_full_sample_is_standardized():
    y, x = _pair(3)
    z = signals.zscore(signals.spread(y, x))
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std()) - 1.0) < 1e-9


def test_zscore_rolling_matches_manual_last_value():
    s = pd.Series(np.arange(1, 21, dtype=float))   # 1..20
    z = signals.zscore(s, window=5)
    window = s.iloc[-5:]
    expected = (s.iloc[-1] - window.mean()) / window.std()
    assert abs(float(z.iloc[-1]) - float(expected)) < 1e-9
    assert bool(z.iloc[:4].isna().all())           # first (window-1) are NaN


# ---------- signals / hysteresis --------------------------------------------
def test_generate_signals_hysteresis_path():
    z = pd.Series([0, 1, 2.5, 1.5, 0.5, -0.5, -2.5, -1.0, 0.0])
    pos = signals.generate_signals(z, entry=2.0, exit=0.5)
    assert list(pos) == [0, 0, -1, -1, 0, 0, 1, 1, 0]


def test_generate_signals_values_are_valid():
    rng = np.random.default_rng(7)
    z = pd.Series(rng.normal(0, 1.5, 200))
    pos = signals.generate_signals(z)
    assert set(pd.unique(pos)).issubset({-1, 0, 1})
    assert len(pos) == len(z)
