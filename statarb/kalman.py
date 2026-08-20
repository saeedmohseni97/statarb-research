
"""Phase 5 — Kalman dynamic hedge: a time-varying hedge ratio.

The static OLS hedge ratio of Phase 2 is ONE number for the whole sample. Real pairs
drift (recall AMAT/LRCX in Phase 3), so we let beta move: model the pair as a
linear-Gaussian state-space system and run a Kalman filter to estimate (alpha_t, beta_t)
online.

    observation:  y_t = alpha_t + beta_t * x_t + v_t,        v_t ~ N(0, R)
    state (rw):   [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w_t,  w_t ~ N(0, Q)

with Q = delta/(1-delta) * I (small delta => slow-varying beta) and R = obs_var. The
one-step forecast error e_t = y_t - (alpha_{t-1} + beta_{t-1} x_t) is the model's
*spread*; it uses only information up to t-1, so it carries no look-ahead.

You implement three functions:
  1. kalman_filter_hedge(y, x, delta, obs_var) -> DataFrame[beta, alpha, spread, var]
  2. kalman_zscore(y, x, ...)                   -> Series  e_t / sqrt(S_t)
  3. kalman_backtest(y, x, ...)                 -> dict (dynamic-hedge, no-look-ahead PnL)

The notebook (notebooks/05_kalman.ipynb) has the exact specs, the recursions, and a
worked example. Implement until tests/test_kalman.py is green, then self-check against
.solutions/kalman_reference.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.signals import generate_signals
from statarb.backtest import turnover, equity_curve, performance_stats


def kalman_filter_hedge(y, x, delta: float = 1e-4, obs_var: float = 1e-3) -> pd.DataFrame:
    """Run the Kalman filter for a time-varying hedge ratio.

    State = [beta, alpha]; observation vector H_t = [x_t, 1] so y_hat = beta*x + alpha.
    Process-noise cov Q = delta/(1-delta) * I(2); observation variance R = obs_var.
    Initialize theta = [0, 0], P = 0. For each t, run predict then update:

        R_pred = P + Q                       # state = random walk, so mean is unchanged
        e_t    = y_t - H_t @ theta           # innovation (forecast error), uses t-1 state
        S_t    = H_t @ R_pred @ H_t + R      # innovation variance (scalar)
        K_t    = R_pred @ H_t / S_t          # Kalman gain (length-2)
        theta  = theta + K_t * e_t           # state update
        P      = R_pred - outer(K_t, H_t) @ R_pred

    Return a DataFrame indexed like y with columns: beta, alpha, spread (= e_t), var (= S_t).
    Hint: work in numpy (y.to_numpy(), x.to_numpy()); H = np.array([x_t, 1.0]).
    """
    Q = (delta/(-delta+1)) * np.identity(2)
    P = np.zeros((2,2))
    idx = y.index
    num_sample = len(idx)

    e = np.zeros((num_sample,))
    S = np.zeros((num_sample,)) 
    beta = np.zeros((num_sample,))
    alpha = np.zeros((num_sample,)) 
    y = y.to_numpy()
    x = x.to_numpy()

    theta = np.zeros(2)          # running state [beta, alpha]; carries across bars

    for t in range(len(idx)):
        R_pred = P + Q
        H_t = np.array([x[t], 1.0])
        e[t] = y[t] - H_t @ theta                 # innovation vs the PREVIOUS state
        S[t] = H_t @ R_pred @ H_t + obs_var
        K = R_pred @ H_t / S[t]
        theta = theta + K * e[t]                   # update the running state
        beta[t], alpha[t] = theta                  # record it, then carry to t+1
        P = R_pred - np.outer(K, H_t) @ R_pred
    df = pd.DataFrame(
    {
        "beta": beta,
        "alpha": alpha,
        "spread": e,
        "var": S
    },
    index=idx
) 

    return df


def kalman_zscore(y, x, delta: float = 1e-4, obs_var: float = 1e-3) -> pd.Series:
    """Standardized innovation z_t = e_t / sqrt(S_t) — the tradable signal.

    Hint: f = kalman_filter_hedge(...); return f["spread"] / np.sqrt(f["var"]).
    """
    f = kalman_filter_hedge(y, x, delta, obs_var)
    return f["spread"] / np.sqrt(f["var"])


def kalman_backtest(y, x, delta: float = 1e-4, obs_var: float = 1e-3, entry: float = 1.0,
                    exit: float = 0.0, cost: float = 0.0, warmup: int = 50,
                    periods_per_year: int = 252) -> dict:
    """Trade the Kalman innovation z-score with an honest dynamic-hedge PnL.

    Steps:
      1. z = kalman_zscore(...); positions = generate_signals(z.reset_index(drop=True),
         entry, exit), then reattach the index and cast to int.
      2. Burn-in: set the first `warmup` positions to 0 (the filter hasn't settled yet).
      3. Dynamic-hedge PnL holding YESTERDAY'S beta (no look-ahead):
             gross_t = pos_{t-1} * (dy_t - beta_{t-1} * dx_t)      # dy=y.diff(), dx=x.diff()
             pnl_t   = gross_t - cost * turnover(positions)        # reuse Phase-3 turnover
         (fill the first bar's NaN gross with 0.)
      4. equity = equity_curve(pnl); stats = performance_stats(pnl, periods_per_year).

    Return dict: beta, alpha, spread, z, positions, pnl, equity, stats.
    """
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)

    # 1. filter once -> state + innovation; standardize to the tradable z-score.
    f = kalman_filter_hedge(y, x, delta, obs_var)
    z = f["spread"] / np.sqrt(f["var"])

    # 2. z -> positions {-1, 0, +1}. Signal on a clean RangeIndex (generate_signals is
    #    label-indexed), then reattach the real dates and cast to int.
    positions = generate_signals(z.reset_index(drop=True), entry, exit)
    positions.index = z.index
    positions = positions.astype(int)

    # 3. Burn-in: hold flat while the filter settles (beta, P still moving).
    if warmup and warmup > 0:
        positions.iloc[:warmup] = 0

    # 4. Dynamic-hedge PnL. Hold YESTERDAY'S position AND yesterday's beta — both were
    #    knowable at t-1's close, so there is no look-ahead.
    beta = f["beta"]
    gross = (positions.shift(1) * (y.diff() - beta.shift(1) * x.diff())).fillna(0.0)
    pnl = gross - cost * turnover(positions)

    # 5. Reuse the Phase-3 machinery so stats are directly comparable to the static backtest.
    return {
        "beta": beta,
        "alpha": f["alpha"],
        "spread": f["spread"],
        "z": z,
        "positions": positions,
        "pnl": pnl,
        "equity": equity_curve(pnl),
        "stats": performance_stats(pnl, periods_per_year),
    }
