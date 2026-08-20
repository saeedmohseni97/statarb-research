"""Phase 3 tests. Offline & deterministic.

Small hand-built series with PnL/turnover we can compute by pencil, plus a seeded
cointegrated pair for the walk-forward. The walk-forward gets a genuine
no-look-ahead check: perturbing the LAST fold's data must not change any earlier
out-of-sample position.
"""
import numpy as np
import pandas as pd

from statarb import backtest


# ---------- oracles ----------------------------------------------------------
def _pair(seed, n=600, beta=1.5, const=3.0, noise=1.0):
    """x is a random walk; y = const + beta*x + STATIONARY noise -> spread reverts."""
    rng = np.random.default_rng(seed)
    x = 50.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    y = const + beta * x + rng.normal(0.0, noise, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


# hand-built position / spread paths (pencil-checkable)
POS = pd.Series([0, 1, 1, -1, 0])
SPREAD = pd.Series([10.0, 11.0, 13.0, 12.0, 12.0])


# ---------- turnover ---------------------------------------------------------
def test_turnover_counts_position_changes():
    t = backtest.turnover(POS)                      # start flat: |0-0|,|1-0|,|1-1|,|-1-1|,|0+1|
    assert list(t) == [0, 1, 0, 2, 1]
    assert float(t.sum()) == 4.0


# ---------- strategy_returns -------------------------------------------------
def test_strategy_returns_no_lookahead_gross():
    # cost=0 -> net == gross == pos_{t-1} * (s_t - s_{t-1})
    pnl = backtest.strategy_returns(POS, SPREAD, cost=0.0)
    assert np.allclose(pnl.to_numpy(), [0.0, 0.0, 2.0, -1.0, 0.0])


def test_strategy_returns_charges_turnover_cost():
    pnl = backtest.strategy_returns(POS, SPREAD, cost=0.5)   # cost * [0,1,0,2,1]
    assert np.allclose(pnl.to_numpy(), [0.0, -0.5, 2.0, -2.0, -0.5])


# ---------- equity_curve -----------------------------------------------------
def test_equity_curve_is_cumulative():
    pnl = pd.Series([0.0, -0.5, 2.0, -2.0, -0.5])
    eq = backtest.equity_curve(pnl)
    assert np.allclose(eq.to_numpy(), [0.0, -0.5, 1.5, -0.5, -1.0])


# ---------- performance_stats ------------------------------------------------
def test_performance_stats_drawdown_and_hit_rate():
    pnl = pd.Series([1.0, -3.0, 2.0])               # equity [1,-2,0]; peak 1 -> dd min -3
    s = backtest.performance_stats(pnl)
    assert abs(s["max_drawdown"] - (-3.0)) < 1e-12
    assert abs(s["total_pnl"] - 0.0) < 1e-12
    # three non-zero periods, two positive -> hit rate 2/3
    assert abs(s["hit_rate"] - 2.0 / 3.0) < 1e-12


def test_performance_stats_positive_stream():
    pnl = pd.Series([0.02, 0.01, 0.03, 0.02])       # monotone up -> no drawdown
    s = backtest.performance_stats(pnl)
    assert s["max_drawdown"] == 0.0
    assert s["hit_rate"] == 1.0
    assert s["sharpe"] > 0
    assert set(s) >= {"total_pnl", "ann_return", "ann_vol", "sharpe",
                      "max_drawdown", "hit_rate", "n_periods"}


# ---------- walk_forward -----------------------------------------------------
def test_walk_forward_out_of_sample_shape_and_values():
    y, x = _pair(1, n=600)
    res = backtest.walk_forward(y, x, train=250, step=50, entry=2.0, exit=0.5)
    # OOS region is everything after the first train window
    assert len(res["positions"]) == 600 - 250
    assert len(res["pnl"]) == 600 - 250
    assert set(pd.unique(res["positions"])).issubset({-1, 0, 1})
    assert res["n_folds"] == (600 - 250) // 50
    assert "sharpe" in res["stats"]


def test_walk_forward_has_no_lookahead():
    y, x = _pair(2, n=600)
    base = backtest.walk_forward(y, x, train=250, step=50, entry=2.0, exit=0.5)

    # Perturb ONLY the last fold's data (indices 550:600). Earlier folds must not move.
    y2, x2 = y.copy(), x.copy()
    y2.iloc[550:] += 25.0
    x2.iloc[550:] -= 25.0
    perturbed = backtest.walk_forward(y2, x2, train=250, step=50, entry=2.0, exit=0.5)

    # positions are indexed 250..599; the first 300 (250..549) precede the change
    assert base["positions"].iloc[:300].equals(perturbed["positions"].iloc[:300])
    # and the last block is actually different (sanity: the perturbation did something)
    assert not base["positions"].iloc[300:].equals(perturbed["positions"].iloc[300:])


def test_walk_forward_rejects_too_long_train():
    y, x = _pair(3, n=120)
    try:
        backtest.walk_forward(y, x, train=200, step=20)
        assert False, "expected ValueError"
    except ValueError:
        pass
