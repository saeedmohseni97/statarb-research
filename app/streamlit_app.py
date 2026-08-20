"""statarb-research — interactive dashboard.

A cost-aware pairs-trading study across two universes:
  * Semiconductors — high correlation, but no cointegration survives multiple-testing.
  * Economic pairs — same-business pairs; MA/V (the card networks) is a robust, significant strategy.

Pipeline wired end to end:
  data -> cointegration screen (FDR) -> signals -> backtest (static | kalman) -> significance

Run locally:   streamlit run app/streamlit_app.py
Deploy:        push to GitHub, point Streamlit Community Cloud at app/streamlit_app.py.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from statarb import config, data, stats, signals, backtest, kalman
from statarb import significance as sig

# --- palette (Okabe-Ito) + crisp, consistent figure style -----------------
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERMILION, PURPLE, GRAY, INK = "#D55E00", "#CC79A7", "#8A8F98", "#1B1F24"
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10.5, "font.family": "sans-serif",
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titleweight": "medium",
    "axes.labelsize": 10, "axes.labelcolor": "#3A4048",
    "axes.edgecolor": "#C4C8CE", "axes.linewidth": 0.9,
    "xtick.color": "#5A6069", "ytick.color": "#5A6069",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "lines.solid_capstyle": "round", "lines.antialiased": True,
})

STATIC = "Static (OLS · walk-forward)"
KALMAN = "Kalman (dynamic β)"

UNIVERSES = {
    "Economic pairs": (config.PAIRS_UNIVERSE, config.PAIRS_CACHE, config.FEATURED_PAIR),
    "Semiconductors": (config.DEFAULT_UNIVERSE, config.DEFAULT_CACHE, ("AMAT", "LRCX")),
}


def _style(ax, title=None):
    ax.grid(True, color="#EBEDF0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.01)
    if title:
        ax.set_title(title, pad=8, loc="left")
    return ax


def _show(fig):
    fig.tight_layout()
    st.pyplot(fig)   # native high-DPI render; avoids the deprecated use_container_width
    plt.close(fig)   # release the figure so figures do not accumulate in memory across reruns


def _implemented(module: str) -> bool:
    try:
        mod = __import__(f"statarb.{module}", fromlist=["*"])
    except Exception:
        return False
    probe = {"kalman": lambda: mod.kalman_zscore(
                 pd.Series(np.arange(60.0)), pd.Series(np.arange(60.0) + 1))}
    fn = probe.get(module)
    if fn is None:
        return True
    try:
        fn(); return True
    except NotImplementedError:
        return False
    except Exception:
        return True


def _default_idx(pair_names, featured):
    """Index of the featured pair in the (screen-sorted) name list, else 0."""
    a, b = featured
    for i, nm in enumerate(pair_names):
        if nm in (f"{a}/{b}", f"{b}/{a}"):
            return i
    return 0


# =========================================================================
# Pure compute helpers (no Streamlit dependency — testable in isolation)
# =========================================================================
def _load_panel(tickers, cache, seed_pair):
    try:
        px = data.get_prices(tickers, config.DEFAULT_START, config.DEFAULT_END,
                             cache_path=os.path.join(ROOT, cache))
        px = data.align_prices(px)
        keep = [t for t in tickers if t in px.columns]      # honor the (possibly trimmed) universe
        px = px[keep]
        if px.shape[1] < 2 or len(px) < 400:
            raise ValueError("panel too small")
        return px, "real"
    except Exception:
        n = 2000
        idx = pd.bdate_range("2015-01-01", periods=n)
        rng = np.random.default_rng(7)
        factor = 100 + np.cumsum(rng.normal(0, 0.6, n))
        cols = {t: 20 + rng.uniform(0.8, 1.4) * factor + np.cumsum(rng.normal(0, 0.4, n))
                for t in tickers}
        a, b = seed_pair
        if a in cols and b in cols:
            cols[a] = 15 + 1.20 * factor + rng.normal(0, 1.0, n)
            cols[b] = 22 + 1.15 * factor + rng.normal(0, 1.0, n)
        return pd.DataFrame(cols, index=idx), "synthetic"


def _run_screen(px):
    return stats.screen_pairs(px, alpha=0.05)


def _pair_series(px, y, x, entry, exit_):
    ly, lx = np.log(px[y]), np.log(px[x])
    s = signals.spread(ly, lx)
    z = signals.zscore(s)
    hl = signals.half_life(s)
    pos = signals.generate_signals(z.reset_index(drop=True), entry, exit_)
    pos.index = z.index
    return ly, lx, s, z, float(hl), pos


def _backtest(px, y, x, engine, entry, exit_, cost, train, step, delta, obs_var, warmup):
    ly, lx = np.log(px[y]), np.log(px[x])
    if engine == KALMAN:
        return kalman.kalman_backtest(ly, lx, delta=delta, obs_var=obs_var,
                                      entry=entry, exit=exit_, cost=cost, warmup=warmup)
    return backtest.walk_forward(ly, lx, train=train, step=step,
                                 entry=entry, exit=exit_, cost=cost)


def _money(pnl, capital):
    p = pd.Series(pnl).fillna(0.0)
    equity = capital * np.exp(p.cumsum())
    total_ret = float(np.exp(p.sum()) - 1.0)
    ann_ret = float(np.exp(p.mean() * 252) - 1.0)
    return equity, total_ret, ann_ret


def _pair_sharpe(px, a, b, engine, entry, exit_, cost, train, step, delta, obs_var, warmup, ppy=1):
    r = _backtest(px, a, b, engine, entry, exit_, cost, train, step, delta, obs_var, warmup)
    return sig.sharpe_ratio(r["pnl"], periods_per_year=ppy)


def _all_pair_trials(px, engine, entry, exit_, cost, train, step, delta, obs_var, warmup):
    out = {}
    for a, b in combinations(px.columns, 2):
        try:
            out[(a, b)] = _pair_sharpe(px, a, b, engine, entry, exit_, cost,
                                       train, step, delta, obs_var, warmup, ppy=1)
        except Exception:
            pass
    return out


def _sensitivity(px, y, x, engine, entries, costs, exit_, train, step, delta, obs_var, warmup):
    M = np.full((len(entries), len(costs)), np.nan)
    P = np.full((len(entries), len(costs)), np.nan)
    for i, e in enumerate(entries):
        for j, c in enumerate(costs):
            try:
                r = _backtest(px, y, x, engine, e, exit_, c, train, step, delta, obs_var, warmup)
                M[i, j] = sig.sharpe_ratio(r["pnl"], periods_per_year=252)
                P[i, j] = sig.probabilistic_sharpe_ratio(r["pnl"], 0.0)
            except Exception:
                pass
    return M, P


# =========================================================================
# Cached wrappers
# =========================================================================
@st.cache_data(show_spinner="Loading prices…")
def load_panel(universe_label):
    tickers, cache, seed = UNIVERSES[universe_label]
    return _load_panel(tickers, cache, seed)


@st.cache_data(show_spinner="Screening every pair (Engle–Granger + FDR)…")
def run_screen(px):
    return _run_screen(px)


@st.cache_data(show_spinner="Backtesting every pair for the deflation…")
def all_pair_trials(px, engine, entry, exit_, cost, train, step, delta, obs_var, warmup):
    return _all_pair_trials(px, engine, entry, exit_, cost, train, step, delta, obs_var, warmup)


@st.cache_data(show_spinner="Sweeping the entry × cost grid…")
def sensitivity(px, y, x, engine, entries, costs, exit_, train, step, delta, obs_var, warmup):
    return _sensitivity(px, y, x, engine, list(entries), list(costs), exit_,
                        train, step, delta, obs_var, warmup)


# =========================================================================
# UI
# =========================================================================
def _stat_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def main():
    st.set_page_config(page_title="statarb-research", page_icon="📈", layout="wide")
    st.title("Statistical-Arbitrage Research Lab")
    st.caption("Cointegration pairs trading with FDR-screened pairs, walk-forward out-of-sample "
               "backtests, transaction costs, and significance testing. New here? Start with the "
               "**Start Here** tab.")

    with st.sidebar:
        st.header("① Universe")
        universe_label = st.radio("Universe", list(UNIVERSES.keys()), label_visibility="collapsed")
        st.header("② Hedge engine")
        engine = st.radio("Engine", [STATIC, KALMAN], label_visibility="collapsed")
        st.header("③ Signal / cost")
        entry = st.slider("Entry band (z / s)", 0.5, 3.0, 1.5 if engine == STATIC else 1.0, 0.25,
                          help="How far the spread must stretch (in std-devs) before a trade opens.")
        exit_ = st.slider("Exit band", 0.0, 1.5, 0.5, 0.25,
                          help="How close to the mean before a trade closes (hysteresis).")
        cost = st.select_slider("Cost per unit turnover",
                                options=[0.0, 0.0005, 0.001, 0.002, 0.005], value=0.0005,
                                help="Transaction cost per trade. Drag it up on the Backtest tab and watch the Sharpe fall.")
        capital = st.number_input("Notional capital ($)", 10_000, 10_000_000, 100_000, 10_000,
                                  help="Hypothetical book size for the dollar figures on the Backtest tab.")
        if engine == STATIC:
            st.header("Walk-forward")
            train = st.slider("Train window (days)", 126, 504, 252, 21)
            step = st.slider("Rebalance step (days)", 5, 63, 63, 1)
            delta, obs_var, warmup = 1e-4, 1e-3, 60
        else:
            st.header("Kalman")
            delta = st.select_slider("δ (adaptivity)", options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
            obs_var = st.select_slider("obs_var (R)", options=[1e-4, 1e-3, 1e-2], value=1e-3)
            warmup = st.slider("Warm-up (days)", 20, 120, 60, 10)
            train, step = 252, 63

    px, source = load_panel(universe_label)
    screen = run_screen(px)
    hits = screen[screen["significant_adj"]]
    pair_names = [f"{r.y}/{r.x}" for r in screen.itertuples()]
    featured = UNIVERSES[universe_label][2]
    fidx = _default_idx(pair_names, featured)
    n_tests = screen.attrs.get("n_tests", len(screen))
    n_adj = int(screen["significant_adj"].sum())

    with st.sidebar:
        st.divider()
        st.caption(f"Data source: **{source}**  \n{px.shape[1]} names · {len(px):,} days · "
                   f"{px.index.min().year}–{px.index.max().year}")
        if source == "synthetic":
            st.info("No cached CSV for this universe and no network — showing a synthetic panel. "
                    "Run the fetch step for this universe and reload for real data.")

    tabs = st.tabs(["🚀 Start Here", "🔎 Universe & Screen", "🔗 Pair Explorer",
                    "📉 Backtest", "🎲 Significance", "🌡️ Robustness", "ℹ️ About"])

    # ---- Tab 0: Start Here ----------------------------------------------
    with tabs[0]:
        st.subheader("What this is")
        st.markdown(
            "A research study of whether cointegration pairs trades survive realistic costs and "
            "statistical testing. It runs the full pipeline — screen → signal → backtest → "
            "significance — and reports the result, positive or null.")
        st.subheader("How to use it")
        st.markdown(
            "1. **① Universe** — *Semiconductors* (no edge survives testing) vs *Economic pairs* "
            "(where MA/V does).\n"
            "2. **② Hedge engine** — *Static* OLS hedge vs the *Kalman* dynamic hedge. Every result tab "
            "recomputes for the engine you pick.\n"
            "3. **③ Signal / cost** — entry/exit bands and trading cost. Drag them and watch the "
            "**Backtest** tab react.\n"
            "4. Walk the tabs left → right: **Screen** → **Pair Explorer** → **Backtest** → "
            "**Significance** → **Robustness**.")
        st.subheader("The three numbers to read")
        st.markdown("**Sharpe** — return per unit of risk (annualized):")
        st.latex(r"\mathrm{Sharpe} \;=\; \frac{\bar r}{\sigma_r}\,\sqrt{252}")
        st.markdown("**PSR** — probability the *true* Sharpe beats zero, skew/kurtosis-aware "
                    r"($\Phi$ = normal CDF, $\hat{SR}$ the estimate, $n$ the sample size):")
        st.latex(r"\mathrm{PSR}=\Phi\!\left(\frac{(\hat{SR}-SR^{*})\sqrt{n-1}}"
                 r"{\sqrt{1-\gamma_3\hat{SR}+\tfrac{\gamma_4-1}{4}\hat{SR}^{2}}}\right)")
        st.markdown(r"**Deflated Sharpe** — the same, but the benchmark $SR^{*}$ is the best a "
                    r"*skill-less search of $N$ strategies* would post. Near $1.0$ = a defensible edge; "
                    r"near $0.5$ = indistinguishable from luck.")
        st.info("The **Significance** and **Robustness** tabs run heavier compute, so they wait behind a "
                "**Run** button. Everything else updates the moment you move a slider.")

    # ---- Tab 1: Universe & Screen ---------------------------------------
    with tabs[1]:
        st.caption("**What this shows:** every pair tested for cointegration (Engle–Granger) with a "
                   "Benjamini–Hochberg FDR correction over the $C(n,2)$ tests. Green = survives "
                   "correction. *Not affected by the entry/cost sliders.*")
        _stat_row([
            ("Pairs tested", f"{n_tests}"),
            ("FDR-significant", f"{n_adj}"),
            ("Raw-significant", f"{int(screen['significant_raw'].sum())}"),
            ("Best adj p", f"{screen['pvalue_adj'].min():.2f}"),
        ])
        st.markdown(rf"About $0.05\times{n_tests}\approx{n_tests*0.05:.0f}$ pairs would look "
                    "significant at raw 5% by chance alone.")
        st.dataframe(screen.head(12)[["y", "x", "eg_tstat", "hedge_ratio",
                                      "pvalue", "pvalue_adj", "significant_adj"]], hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        d = screen.head(15)
        labels = d["y"] + "/" + d["x"]
        vals = -np.log10(d["pvalue_adj"].clip(lower=1e-12))
        colors = [GREEN if s else GRAY for s in d["significant_adj"]]
        ax.bar(range(len(d)), vals, color=colors, width=0.72, zorder=3)
        ax.axhline(-np.log10(0.05), ls="--", lw=1, color=VERMILION, zorder=2)
        ax.text(len(d) - 0.5, -np.log10(0.05), "  α=0.05", va="center", color=VERMILION, fontsize=8)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("$-\\log_{10}(p_{\\mathrm{adj}})$")
        _style(ax, "Cointegration strength (higher = more significant; green = survives FDR)")
        _show(fig)
        if not len(hits):
            st.warning("No pair survives FDR correction: over this window the universe shows no "
                       "statistically robust cointegration. High correlation does not imply a "
                       "stationary, tradeable spread.")
        else:
            st.success(f"{n_adj} pair(s) survive FDR correction. Note that a small screen can still "
                       "surface economically spurious pairs — worth checking each against a real thesis.")

    # ---- Tab 2: Pair Explorer -------------------------------------------
    with tabs[2]:
        st.caption("**What this shows:** the two log-price series and the spread z-score for one pair. "
                   "The dashed lines are your entry bands — move the **Entry band** slider and watch them shift.")
        st.latex(r"s_t = y_t-(\alpha+\beta x_t),\qquad z_t=\frac{s_t-\mu_s}{\sigma_s}")
        choice = st.selectbox("Pair", pair_names, index=fidx)
        y, x = choice.split("/")
        ly, lx, s, z, hl, pos = _pair_series(px, y, x, entry, exit_)
        _stat_row([
            ("Hedge ratio β", f"{signals.hedge_ratio(ly, lx):.3f}"),
            ("Half-life (days)", "∞" if not np.isfinite(hl) else f"{hl:.0f}"),
            ("Spread σ", f"{s.std():.3f}"),
            ("|z| > entry today", "yes" if abs(z.iloc[-1]) > entry else "no"),
        ])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        a1.plot(ly.index, ly, color=BLUE, lw=1.3, label=f"log {y}")
        a1.plot(lx.index, lx, color=ORANGE, lw=1.3, label=f"log {x}")
        a1.legend(frameon=False, loc="upper left")
        _style(a1, f"{y} vs {x} — log prices")
        a2.plot(z.index, z, color=PURPLE, lw=1.0)
        for lvl in (entry, -entry):
            a2.axhline(lvl, ls="--", lw=0.9, color=VERMILION)
        for lvl in (exit_, -exit_):
            a2.axhline(lvl, ls=":", lw=0.8, color=GRAY)
        a2.axhline(0, lw=0.8, color=GRAY)
        _style(a2, "Spread z-score with entry (dashed) / exit (dotted) bands")
        _show(fig)

    # ---- Tab 3: Backtest -------------------------------------------------
    with tabs[3]:
        st.caption(f"**What this shows:** an out-of-sample, cost-aware backtest of one pair with the "
                   f"**{engine}** engine. Dollar figures assume a \\${capital:,.0f} notional spread book.")
        st.latex(r"\text{gross}_t=\text{pos}_{t-1}\,(\Delta y_t-\beta_{t-1}\,\Delta x_t)"
                 r"\;-\;\text{cost}\cdot|\Delta\text{pos}_t|")
        pick = st.selectbox("Pair to backtest", pair_names, index=fidx, key="bt_pair")
        y, x = pick.split("/")
        res = _backtest(px, y, x, engine, entry, exit_, cost, train, step, delta, obs_var, warmup)
        s = res["stats"]
        eq_usd, total_ret, ann_ret = _money(res["pnl"], capital)
        _stat_row([
            ("Sharpe (ann.)", f"{s['sharpe']:.2f}"),
            ("Ann. return", f"{ann_ret*100:.1f}%"),
            ("Total return", f"{total_ret*100:.1f}%"),
            ("Max drawdown", f"{s['max_drawdown']:.3f}"),
        ])
        _stat_row([
            (f"${capital:,.0f} →", f"${eq_usd.iloc[-1]:,.0f}"),
            ("Hit rate", f"{s['hit_rate']*100:.0f}%"),
            ("Trades", f"{int((res['positions'].diff().abs() > 0).sum())}"),
            ("OOS days", f"{s['n_periods']:,}"),
        ])
        if engine == KALMAN:
            fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                                             gridspec_kw={"height_ratios": [2, 3, 1]})
            b = res["beta"]
            a0.plot(b.index, b, color=BLUE, lw=1.3, label="Kalman β_t")
            a0.axhline(signals.hedge_ratio(np.log(px[y]), np.log(px[x])), ls="--",
                       lw=1, color=ORANGE, label="static OLS β")
            a0.legend(frameon=False)
            _style(a0, f"{y} / {x}: time-varying hedge ratio")
        else:
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
        a1.plot(eq_usd.index, eq_usd, color=GREEN, lw=1.6)
        a1.fill_between(eq_usd.index, eq_usd.values, capital, where=(eq_usd.values >= capital),
                        color=GREEN, alpha=0.08)
        a1.axhline(capital, ls=":", lw=0.9, color=GRAY)
        _style(a1, f"{y} / {x}: equity of a ${capital:,.0f} book (after costs)")
        p = res["positions"]
        a2.plot(p.index, p, drawstyle="steps-post", color=BLUE, lw=1.0)
        a2.set_ylim(-1.6, 1.6); a2.set_yticks([-1, 0, 1])
        _style(a2, "Position  (+1 long / −1 short / 0 flat)")
        _show(fig)
        n_trades = int((res['positions'].diff().abs() > 0).sum())
        if engine == KALMAN and n_trades < 10:
            st.warning(f"The Kalman hedge fired only {n_trades} trade(s) here — too few to judge. The "
                       "dynamic hedge helps when β genuinely drifts; on a stable pair it adapts away the "
                       "signal and rarely trades. Try the Static engine, or lower the entry band / raise δ.")
        st.caption("Dollar figures are illustrative: PnL is the spread's log-return on the notional book, "
                   "compounded. What decides whether an edge is real is the **Sharpe and its significance**, "
                   "not the dollar total — a large number on a thin, insignificant spread is still noise.")

    # ---- Tab 4: Significance --------------------------------------------
    with tabs[4]:
        st.caption("**What this shows:** is the backtest's Sharpe real or luck? The bootstrap CI and PSR "
                   "are instant; the **deflated Sharpe** (penalizing for searching every pair) is behind a button.")
        pick = st.selectbox("Pair", pair_names, index=fidx, key="sig_pair")
        y, x = pick.split("/")
        res = _backtest(px, y, x, engine, entry, exit_, cost, train, step, delta, obs_var, warmup)
        ci = sig.bootstrap_sharpe_ci(res["pnl"], n_boot=2000, block=5, seed=0)
        psr = sig.probabilistic_sharpe_ratio(res["pnl"], 0.0)
        _stat_row([
            ("Sharpe (ann.)", f"{ci['sharpe']:.2f}"),
            ("95% CI", f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"),
            ("PSR vs 0", f"{psr:.2f}"),
            ("P(SR ≤ 0)", f"{ci['p_value']:.2f}"),
        ])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(ci["boot"], bins=45, color=BLUE, alpha=0.85, zorder=3)
        ax.axvline(ci["sharpe"], color=INK, lw=1.6, label="point Sharpe", zorder=4)
        ax.axvline(ci["ci_low"], color=INK, ls="--", lw=1, label="95% CI", zorder=4)
        ax.axvline(ci["ci_high"], color=INK, ls="--", lw=1, zorder=4)
        ax.axvline(0, color=VERMILION, lw=1.6, label="zero", zorder=4)
        ax.legend(frameon=False)
        ax.set_xlabel("annualized Sharpe")
        _style(ax, f"{y} / {x}: bootstrap distribution of the Sharpe ({engine})")
        _show(fig)
        st.divider()
        st.markdown(rf"**Deflated Sharpe** — pays for having *searched* all {len(pair_names)} pairs. "
                    r"It is $\mathrm{PSR}$ evaluated at $SR^{*}=\mathbb{E}[\max_N \widehat{SR}\mid\text{no skill}]$.")
        if st.button("Compute deflated Sharpe", key="run_dsr"):
            trials = np.array(list(all_pair_trials(px, engine, entry, exit_, cost,
                                                   train, step, delta, obs_var, warmup).values()))
            dsr = sig.deflated_sharpe_ratio(res["pnl"], sr_trials=trials)
            _stat_row([
                (f"Deflated SR ({dsr['n_trials']} trials)", f"{dsr['dsr']:.2f}"),
                ("Luck bar SR* (/period)", f"{dsr['sr_star']:.3f}"),
                ("PSR vs 0", f"{dsr['psr_vs_zero']:.2f}"),
            ])
            st.caption("Near 0.5 = indistinguishable from luck; near 1.0 = a defensible, real edge. "
                       "A blind search over many pairs deflates hard; a pre-declared economic pair need not.")

    # ---- Tab 5: Robustness (cost × entry sweep) -------------------------
    with tabs[5]:
        st.caption("**What this shows:** the Sharpe across the whole entry × cost grid, so a result "
                   "can be judged on the full surface rather than a single favourable setting.")
        pick = st.selectbox("Pair", pair_names, index=fidx, key="rob_pair")
        y, x = pick.split("/")
        entries = (0.75, 1.0, 1.5, 2.0, 2.5) if engine == KALMAN else (1.0, 1.5, 2.0, 2.5, 3.0)
        costs = (0.0, 0.0005, 0.001, 0.002, 0.005)
        if not st.button("Run entry × cost sweep", key="run_sweep"):
            st.info("Click to sweep the grid (a few seconds).")
        else:
            M, P = sensitivity(px, y, x, engine, entries, costs, exit_,
                               train, step, delta, obs_var, warmup)
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
            im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(costs))); ax.set_xticklabels([f"{c:g}" for c in costs])
            ax.set_yticks(range(len(entries))); ax.set_yticklabels([f"{e:g}" for e in entries])
            ax.set_xlabel("cost per unit turnover"); ax.set_ylabel("entry band")
            for i in range(len(entries)):
                for j in range(len(costs)):
                    if np.isfinite(M[i, j]):
                        mark = "*" if (np.isfinite(P[i, j]) and P[i, j] > 0.95) else ""
                        ax.text(j, i, f"{M[i, j]:.2f}{mark}", ha="center", va="center",
                                fontsize=8, color=INK)
            _style(ax, "Annualized Sharpe by entry × cost  (* = PSR>0.95)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _show(fig)
            best = np.nanmax(M) if np.isfinite(M).any() else float("nan")
            any_sig = bool(np.isfinite(P).any() and (P[np.isfinite(P)] > 0.95).any())
            st.caption(f"Best cell: Sharpe **{best:.2f}**. "
                       + ("Multiple cells clear PSR>0.95 — the edge is robust to the cost/band choice."
                          if any_sig else
                          "**No cell clears PSR>0.95** — nothing here beats zero, at any cost or band."))

    # ---- Tab 6: About ---------------------------------------------------
    with tabs[6]:
        st.markdown("#### Pipeline status")
        rows = [
            ("Phase 0 — data layer", "✅ shipped"),
            ("Phase 1 — cointegration screen (FDR)", "✅ shipped"),
            ("Phase 2 — signals (OU half-life, z-score, bands)", "✅ shipped"),
            ("Phase 3 — walk-forward backtest (no look-ahead, costs)", "✅ shipped"),
            ("Phase 4 — significance (bootstrap CI, PSR, deflated Sharpe)", "✅ shipped"),
            ("Phase 5 — Kalman dynamic hedge",
             "✅ shipped" if _implemented("kalman") else "⏳ in progress"),
            ("PCA stat-arb (Avellaneda–Lee)", "🔭 future work — not built"),
        ]
        st.table(pd.DataFrame(rows, columns=["Component", "Status"]))
        st.markdown(
            "**Summary of findings.** On semiconductors (2013–2024), corrected for multiple testing and "
            "judged out-of-sample, no pair shows a tradeable edge. On a set of pre-declared economic "
            "pairs, the same pipeline finds one: **MA/V** (Mastercard / Visa) is a robust, significant "
            "spread strategy. The value of the project is a reproducible, tested pipeline that "
            "distinguishes a real edge from a lucky one.")
        st.caption("All computation runs on the tested `statarb` package; the dashboard adds no new logic.")


if __name__ == "__main__":
    main()
