"""Phase 0 tests. Fully offline: a fake downloader stands in for yfinance."""
import numpy as np
import pandas as pd

from statarb import data


def _synthetic(tickers=("A", "B"), n=5):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({t: (np.arange(n) + 1.0) * (i + 1)
                         for i, t in enumerate(tickers)}, index=idx)


def test_get_prices_downloads_then_caches(tmp_path):
    calls = {"n": 0}

    def fake(tickers, start, end):
        calls["n"] += 1
        return _synthetic(tickers)

    cache = tmp_path / "px.csv"
    df1 = data.get_prices(["A", "B"], "2020-01-01", "2020-02-01", str(cache), downloader=fake)
    assert calls["n"] == 1 and cache.exists()

    def boom(*a, **k):
        raise AssertionError("cache should have been used, not a re-download")

    df2 = data.get_prices(["A", "B"], "2020-01-01", "2020-02-01", str(cache), downloader=boom)
    assert list(df2.columns) == ["A", "B"]
    assert np.allclose(df1.to_numpy(), df2.to_numpy())


def test_to_returns_simple_and_log():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]},
                          index=pd.bdate_range("2020-01-01", periods=3))
    simple = data.to_returns(prices, kind="simple")
    logret = data.to_returns(prices, kind="log")
    assert np.allclose(simple["A"].to_numpy(), [0.1, 0.1])
    assert np.allclose(logret["A"].to_numpy(), [np.log(1.1), np.log(1.1)])
    assert len(simple) == 2                       # first (NaN) row dropped


def test_to_returns_rejects_bad_kind():
    prices = pd.DataFrame({"A": [1.0, 2.0]})
    try:
        data.to_returns(prices, kind="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_align_drops_partial_rows_and_sorts():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0],
                           "B": [5.0, 6.0, 7.0, 8.0]}, index=idx)
    out = data.align_prices(prices)
    assert len(out) == 3
    assert not out.isna().any().any()
    assert out.index.is_monotonic_increasing
