# Dashboard (`app/streamlit_app.py`)

Interactive front end for the study: universe screen → pair explorer → walk-forward
backtest → significance (bootstrap CI + deflated Sharpe). All computation reuses the
tested `statarb` package; nothing is reimplemented here.

## Run locally

```bash
pip install -e .            # or: pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If `data/semis.csv` is absent and there's no network, the app falls back to a
synthetic semiconductor-like panel so every panel still renders.

## Deploy (Streamlit Community Cloud, free)

1. Push the repo to GitHub (public).
2. On https://share.streamlit.io → **New app**, pick this repo/branch.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Deploy. Cloud installs from the repo-root `requirements.txt` automatically.
5. Copy the public `*.streamlit.app` URL into the project README badge.

To ship real prices instead of the synthetic fallback, commit a cached
`data/semis.csv` (the pipeline writes one via `data.get_prices`) so the deployed
app has data without needing network access at runtime.
