"""Phase 0 - data layer: download, cache, returns, and calendar alignment."""
import os
import numpy as np
import pandas as pd


def download_prices(tickers, start, end):
    """Adjusted daily close via yfinance -> DataFrame (dates x tickers)."""
    # TODO (networked; not unit-tested)
    import yfinance as yf
    return yf.download(tickers, start, end)["Close"].sort_index()
    raise NotImplementedError


def get_prices(tickers, start, end, cache_path, downloader=download_prices):
    """Load from cache_path if it exists, else download -> cache -> return."""
    # TODO
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = downloader(tickers, start, end)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok = True)
            df.to_csv(cache_path)
        return df
        
    raise NotImplementedError


def to_returns(prices, kind="log"):
    """log: ln(p_t/p_{t-1}); simple: p_t/p_{t-1}-1. Drop first row. ValueError otherwise."""
    # TODO
    if kind == "log":
        return np.log((prices/prices.shift(1)).iloc[1:])
    elif kind== "simple":
        return (prices/prices.shift(1)).iloc[1:] - 1
    else:
        raise ValueError
        


def align_prices(prices):
    """Keep rows where ALL tickers have data (dropna any), sorted by date."""
    # TODO
    return prices.dropna(how="any").sort_index()
    raise NotImplementedError
