"""Online dashboard for the PSX signal bot (Streamlit).  Run locally: streamlit run app.py"""
import streamlit as st
import pandas as pd
import psx_bot as bot

st.set_page_config(page_title="PSX Signal Bot", page_icon="📈", layout="wide")
st.title("📈 PSX Daily Signal Bot")
st.caption("Signals only - not financial advice. Verify prices on your broker terminal before trading.")

with st.sidebar:
    st.header("Settings")
    capital = st.number_input("Capital (PKR)", 50_000, 100_000_000, 500_000, step=50_000)
    risk = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100
    top = st.slider("Rows to show", 3, 27, 10)
    stock_news = st.checkbox("Per-stock news (slower, better)", True)
    demo = st.checkbox("Demo mode (synthetic data)", False)
    run = st.button("Run analysis", type="primary", width="stretch")


@st.cache_data(ttl=900, show_spinner=False)  # cache 15 min so reloads are instant
def analyse(capital, risk, stock_news, demo):
    syms = list(bot.UNIVERSE)
    frames = bot.synthetic_prices(syms) if demo else bot.load_prices(syms)
    if not frames:
        return None
    frames = {k: bot.add_indicators(v) for k, v in frames.items()}
    if demo:
        market, sn, heads = 0.0, {s: 0.0 for s in frames}, {}
    else:
        market, sn, heads = bot.gather_news(list(frames), per_stock=stock_news)
    model, auc = bot.train_model(frames)
    sig = bot.build_signals(frames, market, sn, heads, model, capital, risk)
    return sig, frames, market, auc


if run or "result" not in st.session_state:
    with st.spinner("Fetching prices and news..."):
        st.session_state["result"] = analyse(capital, risk, stock_news, demo)

res = st.session_state["result"]
if res is None:
    st.error("Could not load price data. The PSX endpoint may be down or blocking this server.")
    st.stop()

sig, frames, market, auc = res
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market news sentiment", f"{market:+.0f}/100")
c2.metric("BUY signals", int((sig.signal == "BUY").sum()))
c3.metric("ML holdout AUC", f"{auc:.2f}" if auc else "n/a")
c4.metric("Latest data", sig.data_date.max())
if auc is not None and auc < 0.53:
    st.warning("ML model has little predictive power right now - lean on technicals and risk control.")


def colour(v):
    return {"BUY": "background-color:#1b5e20;color:white",
            "AVOID/SELL": "background-color:#b71c1c;color:white"}.get(v, "")


cols = ["symbol", "signal", "score", "tech", "news", "p_up", "close", "stop", "target", "shares", "notes"]
st.subheader("Ranked signals")
st.dataframe(sig.head(top)[cols].style.map(colour, subset=["signal"]), width="stretch", hide_index=True)
st.download_button("Download CSV", sig.drop(columns="headlines").to_csv(index=False), "psx_signals.csv")

st.subheader("Stock detail")
pick = st.selectbox("Symbol", sig.symbol.tolist())
row = sig[sig.symbol == pick].iloc[0]
d = frames[pick].tail(120).set_index("date")
st.line_chart(d[["close", "ema9", "ema21", "ema50"]])
st.write(f"**{pick}**: {row.signal} | entry {row.close} | stop {row.stop} | target {row.target} | "
         f"qty {row.shares} | RSI {row.rsi} | volume x{row.vol_ratio}")
for h in row.headlines:
    st.markdown(f"- {h}")
