"""Phase 4 tests — significance.py. Offline, deterministic, seeded synthetic oracles."""
import numpy as np
import pandas as pd
import pytest

from statarb import significance as S


def _normal_pnl(seed=0, n=1000, mu=0.05, sd=1.0):
    """A seeded normal PnL stream with a known per-period Sharpe ~= mu/sd."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n))


# --- sharpe_ratio ---------------------------------------------------------
def test_sharpe_ratio_matches_formula():
    r = _normal_pnl(1, 2000, 0.1, 1.0)
    expected = r.mean() / r.std() * np.sqrt(252)      # pandas std -> ddof=1
    assert abs(S.sharpe_ratio(r) - expected) < 1e-9


def test_sharpe_ratio_zero_vol():
    assert S.sharpe_ratio(pd.Series([2.0, 2.0, 2.0])) == 0.0


# --- bootstrap_sharpe_ci --------------------------------------------------
def test_bootstrap_ci_brackets_point_and_is_deterministic():
    r = _normal_pnl(2, 1500, 0.08, 1.0)
    a = S.bootstrap_sharpe_ci(r, n_boot=1000, seed=42)
    b = S.bootstrap_sharpe_ci(r, n_boot=1000, seed=42)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]   # seeded => reproducible
    assert a["ci_low"] < a["sharpe"] < a["ci_high"]                       # CI brackets the point
    assert a["ci_low"] < a["ci_high"]


def test_bootstrap_ci_excludes_zero_for_strong_signal():
    r = _normal_pnl(3, 3000, 0.2, 1.0)                # per-period SR ~0.2 -> annual ~3.2
    res = S.bootstrap_sharpe_ci(r, n_boot=1500, seed=7)
    assert res["ci_low"] > 0
    assert res["p_value"] < 0.05


def test_bootstrap_block_runs_and_is_deterministic():
    r = _normal_pnl(4, 1000, 0.05, 1.0)
    a = S.bootstrap_sharpe_ci(r, n_boot=500, block=20, seed=1)
    b = S.bootstrap_sharpe_ci(r, n_boot=500, block=20, seed=1)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
    assert a["ci_low"] < a["ci_high"]


# --- probabilistic_sharpe_ratio -------------------------------------------
def test_psr_is_half_at_own_sharpe():
    r = _normal_pnl(5, 2000, 0.1, 1.0)
    sr = float(np.asarray(r).mean() / np.asarray(r).std(ddof=1))
    assert abs(S.probabilistic_sharpe_ratio(r, sr_benchmark=sr) - 0.5) < 1e-9


def test_psr_decreases_with_higher_benchmark():
    r = _normal_pnl(6, 2000, 0.1, 1.0)
    assert S.probabilistic_sharpe_ratio(r, 0.0) > S.probabilistic_sharpe_ratio(r, 0.15)


def test_psr_high_for_strong_signal():
    r = _normal_pnl(7, 3000, 0.15, 1.0)
    assert S.probabilistic_sharpe_ratio(r, 0.0) > 0.9


# --- expected_max_sharpe --------------------------------------------------
def test_expected_max_increases_with_trials():
    assert S.expected_max_sharpe(200, 0.01) > S.expected_max_sharpe(10, 0.01)


def test_expected_max_increases_with_variance():
    assert S.expected_max_sharpe(100, 0.04) > S.expected_max_sharpe(100, 0.01)


def test_expected_max_requires_two_trials():
    with pytest.raises(ValueError):
        S.expected_max_sharpe(1, 0.01)


# --- deflated_sharpe_ratio ------------------------------------------------
def test_dsr_is_bounded_and_below_psr_vs_zero():
    r = _normal_pnl(8, 2000, 0.1, 1.0)
    out = S.deflated_sharpe_ratio(r, n_trials=100, sr_variance=0.02)
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["dsr"] <= out["psr_vs_zero"]           # deflation can only lower confidence
    assert out["sr_star"] > 0


def test_dsr_from_trial_array_reads_n_and_variance():
    r = _normal_pnl(9, 2000, 0.12, 1.0)
    trials = np.random.default_rng(0).normal(0, 0.1, 150)
    out = S.deflated_sharpe_ratio(r, sr_trials=trials)
    assert out["n_trials"] == 150
    assert out["dsr"] <= out["psr_vs_zero"]


def test_dsr_falls_as_trials_grow():
    r = _normal_pnl(10, 2000, 0.1, 1.0)
    few = S.deflated_sharpe_ratio(r, n_trials=10, sr_variance=0.02)["dsr"]
    many = S.deflated_sharpe_ratio(r, n_trials=500, sr_variance=0.02)["dsr"]
    assert many <= few                                # more searching -> higher bar -> lower DSR


def test_dsr_requires_enough_info():
    r = _normal_pnl(11, 500, 0.1, 1.0)
    with pytest.raises(ValueError):
        S.deflated_sharpe_ratio(r)                    # neither trials nor (n, var) given
