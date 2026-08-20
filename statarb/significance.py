"""Phase 4 — significance: is a Sharpe ratio distinguishable from luck?

Six functions, built on a per-period PnL stream (e.g. what walk_forward returns):
  1. _per_period_sharpe        — mean / std(ddof=1); the block everything else uses.
  2. sharpe_ratio              — annualized Sharpe.
  3. bootstrap_sharpe_ci       — resample-based CI (iid or moving-block).
  4. probabilistic_sharpe_ratio— P(true Sharpe > benchmark), skew/kurtosis-aware.
  5. expected_max_sharpe       — the Sharpe a lucky best-of-N posts with no skill.
  6. deflated_sharpe_ratio     — PSR against that best-of-N benchmark.

The one idea underneath all of this: a Sharpe from finite, noisy, non-normal data is
itself a random variable, and if you SEARCHED over many strategies the best one is
biased upward even with zero skill. These tools quantify both effects. Everything is
per-period unless a function annualizes explicitly (× sqrt(periods_per_year)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

# Euler-Mascheroni constant — used in expected_max_sharpe.
EULER_MASCHERONI = 0.5772156649015329


def _clean(pnl) -> np.ndarray:
    """Drop NaNs and return a float ndarray."""
    return np.asarray(pd.Series(pnl).dropna(), dtype=float)


def _per_period_sharpe(pnl) -> float:
    """Per-period Sharpe = mean / std(ddof=1). Returns 0.0 if <2 obs or zero vol."""
    r = _clean(pnl)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def sharpe_ratio(pnl, periods_per_year: int = 252) -> float:
    """Annualized Sharpe = per-period Sharpe * sqrt(periods_per_year)."""
    return _per_period_sharpe(pnl) * np.sqrt(periods_per_year)


def bootstrap_sharpe_ci(pnl, n_boot: int = 2000, block: int = 1, alpha: float = 0.05,
                        periods_per_year: int = 252, seed: int = 0) -> dict:
    """Bootstrap confidence interval for the (annualized) Sharpe ratio.

    Resample the PnL `n_boot` times, recompute the per-period Sharpe on each resample,
    annualize (* sqrt(ppy)), then take the (alpha/2, 1-alpha/2) percentiles.
      * block == 1 -> iid bootstrap (resample individual bars with replacement).
      * block  > 1 -> circular moving-block bootstrap (length-`block` runs, wrap-around)
        so some serial correlation in the PnL survives the resampling.
    Returns dict: sharpe (point), ci_low, ci_high, p_value (share of bootstrap Sharpes
    <= 0), alpha, n_boot, boot (np.ndarray of the bootstrap Sharpes).
    """
    r = _clean(pnl)
    n = r.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    if block < 1:
        raise ValueError("block must be >= 1")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(periods_per_year)
    boot = np.empty(n_boot, dtype=float)
    if block == 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        for b in range(n_boot):
            boot[b] = _per_period_sharpe(r[idx[b]])
    else:
        n_blocks = int(np.ceil(n / block))
        offsets = np.arange(block)
        for b in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            take = (starts[:, None] + offsets[None, :]) % n     # circular wrap
            boot[b] = _per_period_sharpe(r[take.ravel()[:n]])
    boot *= scale
    return {
        "sharpe": _per_period_sharpe(r) * scale,
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "p_value": float(np.mean(boot <= 0.0)),
        "alpha": alpha,
        "n_boot": n_boot,
        "boot": boot,
    }


def probabilistic_sharpe_ratio(pnl, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): P(true SR > sr_benchmark).

        PSR = Phi( (SR_hat - SR*) * sqrt(n - 1)
                   / sqrt(1 - g3*SR_hat + (g4 - 1)/4 * SR_hat^2) )

    with SR_hat the per-period Sharpe, SR* = sr_benchmark (per-period), n = #obs,
    g3 = skewness, g4 = NON-excess kurtosis (3 for a normal). SR_hat == SR* -> 0.5.
    """
    r = _clean(pnl)
    n = r.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    sr = _per_period_sharpe(r)
    g3 = float(skew(r, bias=True))
    g4 = float(kurtosis(r, fisher=False, bias=True))
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2)
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected MAXIMUM per-period Sharpe under the null of zero skill across N trials
    whose Sharpe estimates have cross-sectional variance `sr_variance`:

        E[max SR] ~ sqrt(sr_variance) * [ (1 - gamma)*Phi^-1(1 - 1/N)
                                          + gamma*Phi^-1(1 - 1/(N*e)) ]

    gamma = EULER_MASCHERONI. The bar a lucky best-of-N must clear. Requires N >= 2.
    """
    N = int(n_trials)
    if N < 2:
        raise ValueError("need n_trials >= 2")
    if sr_variance < 0:
        raise ValueError("sr_variance must be >= 0")
    sigma = np.sqrt(sr_variance)
    q1 = norm.ppf(1.0 - 1.0 / N)
    q2 = norm.ppf(1.0 - 1.0 / (N * np.e))
    return float(sigma * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2))


def deflated_sharpe_ratio(pnl, sr_trials=None, n_trials: int | None = None,
                          sr_variance: float | None = None) -> dict:
    """Deflated Sharpe Ratio = PSR evaluated at the expected_max_sharpe benchmark.

    Supply EITHER `sr_trials` (array of the N tried strategies' per-period Sharpes; N
    and their variance come from it) OR both `n_trials` and `sr_variance`. Otherwise
    raise ValueError. Returns dict: dsr, sr_star, n_trials, sr_variance, psr_vs_zero
    (PSR against 0). DSR <= psr_vs_zero always.
    """
    if sr_trials is not None:
        arr = np.asarray(sr_trials, dtype=float)
        N = arr.size
        var = float(np.var(arr, ddof=1)) if N > 1 else 0.0
    else:
        if n_trials is None or sr_variance is None:
            raise ValueError("provide sr_trials, or both n_trials and sr_variance")
        N = int(n_trials)
        var = float(sr_variance)
    sr_star = expected_max_sharpe(N, var)
    return {
        "dsr": probabilistic_sharpe_ratio(pnl, sr_benchmark=sr_star),
        "sr_star": sr_star,
        "n_trials": N,
        "sr_variance": var,
        "psr_vs_zero": probabilistic_sharpe_ratio(pnl, sr_benchmark=0.0),
    }
