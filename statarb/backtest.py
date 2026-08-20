"""Phase 3 — backtest: turn positions into an honest, cost-aware track record.

You implement five functions:
  1. turnover(positions)                 — |Delta position| per bar (start flat).
  2. strategy_returns(positions, spread)  — no-look-ahead spread PnL minus costs.
  3. equity_curve(pnl)                    — cumulative PnL.
  4. performance_stats(pnl)               — Sharpe, drawdown, hit-rate, etc.
  5. walk_forward(y, x, train, step, ...) — rolling out-of-sample backtest.

The one convention everything rests on: a position chosen at the close of bar t
earns the spread's move over the NEXT bar, so honest PnL uses the LAGGED position
(pos_{t-1}). Never let today's position earn today's move — that is look-ahead.

The notebook (notebooks/03_backtest.ipynb) has the exact specs, formulas, and a
worked example. Implement until tests/test_backtest.py is green, then self-check
against .solutions/backtest_reference.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.signals import hedge_ratio, generate_signals


def turnover(positions) -> pd.Series:
    """Per-period turnover = |pos_t - pos_{t-1}|, starting from FLAT (pos_{-1} = 0).

    A flat -> +/-1 entry costs 1 unit; a +1 -> -1 flip costs 2; holding costs 0.
    This is the quantity transaction costs are charged on.
    Hint: prev = pd.Series(positions).shift(1).fillna(0); return (pos - prev).abs().
    """
    # TODO
    pos = pd.Series(positions)
    prev = pos.shift(1).fillna(0) 
    return (pos-prev).abs()
    raise NotImplementedError


def strategy_returns(positions, spread, cost: float = 0.0) -> pd.Series:
    """Net per-period spread PnL with NO look-ahead and linear transaction costs.

        gross_t = pos_{t-1} * (s_t - s_{t-1})      # yesterday's position, today's move
        cost_t  = cost * |pos_t - pos_{t-1}|       # charged on turnover
        net_t   = gross_t - cost_t

    The first bar has no prior position/move, so its gross is 0 (fill the NaN).
    Return the net PnL series, indexed like `positions`.
    Hint: ds = pd.Series(spread).diff(); gross = (pos.shift(1) * ds).fillna(0.0);
          net = gross - cost * turnover(pos).
    """
    # TODO
    pos = pd.Series(positions)
    s = pd.Series(spread) 
    gross_t = pos.shift(1).fillna(0) * (s - s.shift(1).fillna(0))
    cost_t = cost * turnover(positions)
    net_t = gross_t - cost_t
    return net_t
    raise NotImplementedError


def equity_curve(pnl, initial: float = 0.0) -> pd.Series:
    """Cumulative PnL curve (ADDITIVE — these are spread-PnL units, not % returns).

    Hint: initial + pd.Series(pnl).cumsum().
    """
    # TODO
    pnl = pd.Series(pnl) 
    return pnl.cumsum()
    raise NotImplementedError


def performance_stats(pnl, periods_per_year: int = 252) -> dict:
    """Summary metrics computed from a per-period PnL stream. Return a dict with:

        total_pnl     sum of the PnL
        ann_return    mean(pnl) * periods_per_year
        ann_vol       std(pnl)  * sqrt(periods_per_year)
        sharpe        ann_return / ann_vol   (return 0.0 if vol is 0)
        max_drawdown  min over time of (equity - running max of equity)  (<= 0)
        hit_rate      share of NON-ZERO periods that are positive
        n_periods     number of PnL observations

    Hint: eq = pd.Series(pnl).cumsum(); dd = eq - eq.cummax(); max_drawdown = dd.min().
          sharpe = mean/std * sqrt(ppy). nz = pnl[pnl != 0]; hit = (nz > 0).mean().
    """
    # TODO
    pnl = pd.Series(pnl)

    total_pnl = pnl.sum() 
    ann_return = pnl.mean() * periods_per_year 
    ann_vol = pnl.std() * np.sqrt(periods_per_year)
    sharpe = ann_return/ann_vol 

    eq = pnl.cumsum()
    max_drawdown = (eq - eq.cummax()).min()

    nz = pnl[pnl != 0]
    hit_rate = (nz>0).mean() 
    n_periods = pnl.shape[0] 

    out = {
        "total_pnl" : pnl.sum(),     
        "ann_return" : pnl.mean() * periods_per_year,   
        "ann_vol" : pnl.std() * np.sqrt(periods_per_year),      
        "sharpe" : ann_return/ann_vol,        
        "max_drawdown" : (eq - eq.cummax()).min(), 
        "hit_rate" : (nz>0).mean() ,     
        "n_periods" : pnl.shape[0] ,    
    }
    return out 
    raise NotImplementedError


def walk_forward(y, x, train: int, step: int, entry: float = 2.0, exit: float = 0.0,
                 cost: float = 0.0, periods_per_year: int = 252) -> dict:
    """Rolling OUT-OF-SAMPLE backtest of a pair.

    Roll a fixed-length TRAIN window forward in blocks of `step`. On each fold:
      1. estimate beta = hedge_ratio(y_train, x_train)              (Phase-2 function)
      2. form s = y - beta*x; standardize the TEST block with the TRAIN spread's
         mean/std:  z_test = (s_test - mu_train) / sigma_train
      3. positions = generate_signals(z_test, entry, exit)  (starts FLAT each fold)
      4. accrue net PnL on the test block via strategy_returns(positions, s_test, cost)
    Then move `start` forward by `step` and repeat. Because each fold only uses data
    at or before that fold, the concatenated result is a genuine OOS track record.

    You do NOT need the OLS intercept: it only shifts the spread by a constant, which
    cancels when you subtract the train mean.

    Return a dict: positions (int Series over the OOS region, i.e. y.iloc[train:]),
    pnl, equity, stats (from performance_stats), n_folds.

    Look-ahead trap: standardize the test block with TRAIN statistics only — never
    recompute mu/sigma using the test data you are about to trade.

    Robust-indexing tip: call generate_signals on z_test.reset_index(drop=True), then
    reattach the dates (p.index = z_test.index) so it doesn't depend on the index type.
    Raise ValueError if train >= len(y) or step < 1.
    """
    # TODO
    y = pd.Series(y)
    x = pd.Series(x)
    n = len(y)
    if train >= n or step < 1:
        raise ValueError("need train < len(y) and step >= 1")

    pos_list, pnl_list = [], []
    start = 0
    while start + train < n:                      # still at least one unseen test bar
        # 1. slice train / test windows (positional)
        y_tr = y.iloc[start : start + train]
        x_tr = x.iloc[start : start + train]
        y_te = y.iloc[start + train : start + train + step]
        x_te = x.iloc[start + train : start + train + step]

        # 2. fit beta on TRAIN only
        beta = hedge_ratio(y_tr, x_tr)

        # 3. TRAIN spread stats
        s_tr = y_tr - beta * x_tr
        mu, sigma = s_tr.mean(), s_tr.std()

        # 4. TEST spread, standardized with TRAIN stats (no look-ahead)
        s_te = y_te - beta * x_te
        z_te = (s_te - mu) / sigma

        # 5. signals (index-safe), then reattach real dates
        p = generate_signals(z_te.reset_index(drop=True), entry, exit)
        p.index = z_te.index

        # 6. fold PnL (starts flat each fold)
        pnl_fold = strategy_returns(p, s_te, cost)

        pos_list.append(p)
        pnl_list.append(pnl_fold)
        start += step

    positions = pd.concat(pos_list)
    pnl = pd.concat(pnl_list)
    equity = equity_curve(pnl)
    stats = performance_stats(pnl, periods_per_year)

    return {
        "positions": positions,
        "pnl": pnl,
        "equity": equity,
        "stats": stats,
        "n_folds": len(pos_list),
    }
    raise NotImplementedError
