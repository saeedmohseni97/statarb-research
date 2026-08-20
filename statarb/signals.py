"""Phase 2 — signals: turn a cointegrated pair into a tradable, mean-reverting series.

You implement five functions:
  1. hedge_ratio(y, x)          — OLS slope beta of y on x.
  2. spread(y, x)               — residual spread s = y - (const + beta*x).
  3. half_life(spread)          — OU/AR(1) half-life of mean reversion.
  4. zscore(spread, window)     — standardize the spread (full-sample or rolling).
  5. generate_signals(z, ...)   — z-score -> positions {-1,0,+1} with entry/exit bands.

The notebook (notebooks/02_signals.ipynb) has the exact specs, formulas, and a worked
example. Implement until tests/test_signals.py is green, then self-check against
.solutions/signals_reference.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_ratio(y, x) -> float:
    """OLS slope beta of y on x (with intercept).

    Align y & x (dropna together). Fit y = const + beta*x by least squares and return beta.
    Hint: X = np.column_stack([np.ones_like(x), x]); coef, *_ = np.linalg.lstsq(X, y, rcond=None)
          -> coef is [const, beta]; return the beta.
    """
    # TODO
    x = np.expand_dims(x,1)
    x = np.concatenate((np.ones_like(x), x),axis=1)
    const, beta = np.linalg.lstsq(x, y)[0] 
    return beta    
    raise NotImplementedError


def spread(y, x) -> pd.Series:
    """Residual spread s_t = y_t - (const + beta * x_t) from OLS of y on x.

    Same OLS as hedge_ratio, but here return the full residual series (indexed like the
    aligned inputs). This mean-reverting series is what every later step trades on.
    """
    # TODO
    X = np.expand_dims(x,1)
    X = np.concatenate((np.ones_like(X), X),axis=1)
    const, beta = np.linalg.lstsq(X, y)[0]
    return y - (const + beta * x)
    raise NotImplementedError


def half_life(spread_series) -> float:
    """Half-life of mean reversion (in periods) via an OU / AR(1) fit.

    Regress the change on the level:  Δs_t = a + b * s_{t-1} + e_t   (drop NaNs).
    Mean-reverting => b < 0, and half-life = -ln(2) / b. Return +inf if b >= 0.
    Hint: s_lag = s.shift(1); ds = s - s_lag; OLS ds on s_lag; b is the slope.
    """
    # TODO
    delta_s = spread_series.diff(1).dropna()
    
    x = np.column_stack([np.ones_like(delta_s), spread_series.shift(1).dropna()]) 
    y = delta_s

    const, beta = np.linalg.lstsq(x, y)[0] 
    return -np.log(2)/beta if beta<0 else np.inf
    raise NotImplementedError


def zscore(spread_series, window: int | None = None) -> pd.Series:
    """Standardize the spread to a z-score.

    window=None -> full-sample: (s - s.mean()) / s.std().
    window=k    -> rolling: subtract a k-period rolling mean, divide by rolling std
                   (causal — uses only past data, no look-ahead).
    Hint: s.rolling(window).mean(), s.rolling(window).std().
    """
    # TODO

    if window:
        return (spread_series - spread_series.rolling(window).mean())/spread_series.rolling(window).std()
    else: 
        return (spread_series - spread_series.mean())/spread_series.std()
    
    
    raise NotImplementedError


def generate_signals(z, entry: float = 2.0, exit: float = 0.0) -> pd.Series:
    """Map a z-score series to positions {-1, 0, +1} with entry/exit HYSTERESIS.

    Rules (hold the position between the bands — don't re-decide every bar):
      * flat (0) and z > entry   -> go short the spread (-1)
      * flat (0) and z < -entry  -> go long the spread  (+1)
      * short (-1) and z <= exit  -> flatten (0)
      * long  (+1) and z >= -exit -> flatten (0)
    Return a Series (same index as z) of ints in {-1, 0, +1}.
    Hint: this is stateful — loop with a `pos` variable, or track state some other way.
    """
    # TODO
    out = z.copy()
    length = z.shape[0]
    
    out[0] = 0
    
    for i in range(1, length):
        if out[i-1] == 0 and z[i]> entry:
            out[i] = -1
        elif out[i-1] == 0 and z[i]< -entry:
            out[i] = +1
        elif out[i-1] == -1 and z[i]<= exit:
            out[i] = 0
        elif out[i-1] == 1 and z[i]>= -exit:
            out[i] = 0
        else:
            out[i] = out[i-1]
    return out
    #print(z)
    #print(out)
    raise NotImplementedError
