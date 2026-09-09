"""
Streamlit dashboard for the Quantitative Trading System.

Everything on this page is pulled live from the same Alpaca paper account
and the same trained model the bot itself uses -- nothing here is
hardcoded, simulated, or faked. It exists so people looking at the GitHub
repo (who don't have access to Sean's private Alpaca login) can still see
real-time portfolio performance, per-stock signals, and the full trade
history.

Run locally:
    streamlit run dashboard.py

Deploy for free (so anyone with the link can view it):
    Streamlit Community Cloud (share.streamlit.io) -- connect this GitHub
    repo, set ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL as
    "Secrets" in the app settings (never committed to the repo), done.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from watchlist import WATCHLIST
from execution.broker import AlpacaBroker
from data_pipeline.market_data import get_ohlcv
from indicators.technical import add_all_indicators
from model.features import FEATURE_COLUMNS
from model.predict import load_model

st.set_page_config(page_title="Quantitative Trading System", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F916\U0001F4C8 Quantitative Trading System — Live Dashboard")
st.caption(
    "This bot only trades a paper (practice) Alpaca account. Every number "
    "below is read live from that account and from the bot's own trained "
    "model -- nothing on this page is hardcoded or simulated."
)


@st.cache_resource
def get_broker():
    return AlpacaBroker()


try:
    broker = get_broker()
except Exception as exc:
    st.error(f"Could not connect to Alpaca: {exc}")
    st.info(
        "If you're running this yourself, copy .env.example to .env (or set "
        "the ALPACA_API_KEY / ALPACA_SECRET_KEY secrets if deployed) with "
        "your own **paper trading** keys from "
        "https://app.alpaca.markets/paper/dashboard/overview"
    )
    st.stop()

# --------------------------------------------------------------------------
# Portfolio performance
# --------------------------------------------------------------------------
st.header("Portfolio Performance")

account = broker.get_account()
equity = float(account.equity)
last_equity = float(account.last_equity)
day_pnl = equity - last_equity
day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${equity:,.2f}")
c2.metric("Cash", f"${float(account.cash):,.2f}")
c3.metric("Buying Power", f"${float(account.buying_power):,.2f}")
c4.metric("Today's P&L", f"${day_pnl:,.2f}", delta=f"{day_pnl_pct:.2f}%")

try:
    history = broker.get_portfolio_history()
    hist_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(history.timestamp, unit="s"),
            "equity": history.equity,
        }
    ).dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_df["timestamp"],
            y=hist_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#2ca02c"),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        yaxis_title="Account Equity ($)",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info(
        "The equity curve will fill in here as the account accumulates "
        "more trading days -- Alpaca's portfolio history needs a bit of "
        "history to plot."
    )

positions = broker.get_all_positions()
st.subheader(f"Open Positions ({len(positions)})")
if positions:
    pos_df = pd.DataFrame(
        [
            {
                "Ticker": p.symbol,
                "Qty": p.qty,
                "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
                "Current Price": f"${float(p.current_price):,.2f}",
                "Unrealized P&L": f"${float(p.unrealized_pl):,.2f}",
                "Unrealized P&L %": f"{float(p.unrealized_plpc) * 100:.2f}%",
            }
            for p in positions
        ]
    )
    st.dataframe(pos_df, use_container_width=True, hide_index=True)
else:
    st.write("No open positions right now.")

# --------------------------------------------------------------------------
# Per-stock signal charts
# --------------------------------------------------------------------------
st.header("Individual Stock Signals")
st.caption(
    "The model's real BUY/SELL signal at every point in history, computed "
    "the same way the live bot computes it each morning -- probability of "
    f"tomorrow closing higher, thresholded at {config.BUY_PROB_THRESHOLD:.0%} "
    f"(BUY) / {config.SELL_PROB_THRESHOLD:.0%} (SELL)."
)

ticker = st.selectbox("Choose a stock from the watchlist", WATCHLIST)


@st.cache_data(ttl=900)
def load_signal_history(ticker: str) -> pd.DataFrame:
    raw = get_ohlcv(ticker, period="1y", interval=config.DATA_INTERVAL)
    enriched = add_all_indicators(raw)
    model = load_model()

    feat = enriched[FEATURE_COLUMNS].dropna()
    probabilities = model.predict_proba(feat)[:, 1]

    sig_df = enriched.loc[feat.index].copy()
    sig_df["probability"] = probabilities
    sig_df["signal"] = "HOLD"
    sig_df.loc[sig_df["probability"] >= config.BUY_PROB_THRESHOLD, "signal"] = "BUY"
    sig_df.loc[sig_df["probability"] <= config.SELL_PROB_THRESHOLD, "signal"] = "SELL"
    return sig_df


sig_df = load_signal_history(ticker)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.06,
    subplot_titles=(f"{ticker} — price with the model's actual signals", "RSI"),
)

fig.add_trace(
    go.Scatter(x=sig_df.index, y=sig_df["close"], mode="lines", name="Close",
               line=dict(color="#4c78a8")),
    row=1, col=1,
)

buys = sig_df[sig_df["signal"] == "BUY"]
sells = sig_df[sig_df["signal"] == "SELL"]
fig.add_trace(
    go.Scatter(x=buys.index, y=buys["close"], mode="markers", name="BUY signal",
               marker=dict(symbol="triangle-up", color="green", size=10)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=sells.index, y=sells["close"], mode="markers", name="SELL signal",
               marker=dict(symbol="triangle-down", color="red", size=10)),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(x=sig_df.index, y=sig_df["rsi"], mode="lines", name="RSI",
               line=dict(color="#e45756")),
    row=2, col=1,
)
fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

fig.update_layout(height=620, legend=dict(orientation="h", y=1.08), margin=dict(t=60))
st.plotly_chart(fig, use_container_width=True)

latest = sig_df.iloc[-1]
st.write(
    f"**Latest signal for {ticker}:** {latest['signal']} — model's estimated "
    f"probability tomorrow closes higher: **{latest['probability']:.1%}**"
)

# --------------------------------------------------------------------------
# Trade history
# --------------------------------------------------------------------------
st.header("Complete Trade History")
st.caption(
    "Every decision the bot has logged since it went live -- not just "
    "trades that executed, but skipped signals too (and why they were "
    "skipped: volatility filter, earnings blackout, sentiment gate, etc.)."
)

if os.path.isfile(config.TRADE_LOG_PATH):
    log_df = pd.read_csv(config.TRADE_LOG_PATH)
    if "timestamp" in log_df.columns:
        log_df = log_df.sort_values("timestamp", ascending=False)

    show_all = st.checkbox("Show every logged decision (including HOLD / no-action)", value=False)
    if not show_all and "action" in log_df.columns:
        log_df = log_df[log_df["action"] != "NONE"]

    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.info(
        "No trades logged yet -- the bot writes to this log every weekday "
        "at 9:35 AM US/Eastern once it runs against the live market."
    )

st.divider()
st.caption(
    "Data sources: Alpaca's paper trading API (account equity, open "
    "positions, portfolio history) and this project's own logs/trade_log.csv "
    "(every BUY/SELL/HOLD decision the bot has made). Nothing on this page "
    "is a mock, a placeholder, or a projection."
)
