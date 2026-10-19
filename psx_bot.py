#!/usr/bin/env python3
"""
PSX Daily Signal Bot
====================
Combines  (1) technical analysis on PSX price/volume data,
          (2) news sentiment from Pakistani + global sources,
          (3) a small ML model (probability of an up-day)
into one ranked list of BUY / WATCH / AVOID signals with entry, stop,
target and risk-based position size.

SIGNALS ONLY - this bot does not place orders. Not financial advice.

Usage:
    python psx_bot.py --demo                 # offline test with synthetic data
    python psx_bot.py                        # live run
    python psx_bot.py --capital 1000000 --risk 0.01 --top 5
    python psx_bot.py --backtest             # test the technical rules on history
    python psx_bot.py --loop 30              # re-run every 30 minutes
Telegram alerts (optional): set env TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.
"""
import argparse, datetime as dt, json, logging, os, re, time
import numpy as np
import pandas as pd
import requests

try:
    import feedparser
except ImportError:
    feedparser = None
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
except ImportError:
    GradientBoostingClassifier = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("psx-bot")

# --------------------------------------------------------------------------
# 1. CONFIG  (edit freely)
# --------------------------------------------------------------------------
# symbol -> keywords used to match news headlines
UNIVERSE = {
    "OGDC": ["OGDC", "Oil and Gas Development"], "PPL": ["Pakistan Petroleum", "PPL"],
    "MARI": ["Mari Petroleum"], "POL": ["Pakistan Oilfields"],
    "PSO": ["Pakistan State Oil", "PSO"], "ATRL": ["Attock Refinery"],
    "HUBC": ["Hub Power", "Hubco"], "FFC": ["Fauji Fertilizer", "FFC"],
    "EFERT": ["Engro Fertilizers", "EFERT"], "ENGRO": ["Engro Corporation"],
    "LUCK": ["Lucky Cement"], "FCCL": ["Fauji Cement"], "DGKC": ["DG Khan Cement"],
    "MLCL": ["Maple Leaf Cement"], "CHCC": ["Cherat Cement"],
    "MCB": ["MCB Bank"], "UBL": ["United Bank", "UBL"], "HBL": ["Habib Bank", "HBL"],
    "MEBL": ["Meezan Bank"], "BAHL": ["Bank AL Habib"], "NBP": ["National Bank of Pakistan"],
    "SYS": ["Systems Limited"], "TRG": ["TRG Pakistan"], "NML": ["Nishat Mills"],
    "SNGP": ["Sui Northern", "SNGPL"], "PAEL": ["Pak Elektron"], "EPCL": ["Engro Polymer"],
}
# PSX unofficial public data portal (used by psx.com.pk website itself; may change)
PSX_EOD = "https://dps.psx.com.pk/timeseries/eod/{sym}"

PK_FEEDS = [
    "https://www.dawn.com/feeds/business",
    "https://tribune.com.pk/feed/business",
    "https://www.brecorder.com/feeds/latest-news",
    "https://www.thenews.com.pk/rss/1/8",
]
GLOBAL_QUERIES = ["Pakistan stock exchange", "KSE-100", "SBP policy rate", "Pakistan IMF",
                  "Pakistan economy", "oil prices", "emerging markets stocks", "Federal Reserve rates"]
GNEWS = "https://news.google.com/rss/search?q={q}+when:3d&hl=en-PK&gl=PK&ceid=PK:en"

# Rule-of-thumb costs (brokerage + CVT + taxes + slippage), round trip
ROUND_TRIP_COST = 0.004
MIN_AVG_VOLUME = 200_000       # liquidity filter (shares/day, 20d avg)
MAX_TODAY_MOVE = 0.06          # skip chasing stocks already up >6% (PSX upper circuit is ~7.5%)
HEADERS = {"User-Agent": "Mozilla/5.0 (psx-signal-bot)"}

# --------------------------------------------------------------------------
# 2. DATA
# --------------------------------------------------------------------------
def fetch_psx(sym, session):
    """Daily history [ts, close, volume, open] from the PSX data portal."""
    r = session.get(PSX_EOD.format(sym=sym), headers=HEADERS, timeout=20)
    r.raise_for_status()
    rows = r.json().get("data", [])
    if not rows:
        raise ValueError("empty")
    df = pd.DataFrame([row[:4] for row in rows], columns=["ts", "close", "volume", "open"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    return df.drop(columns="ts").sort_values("date").reset_index(drop=True).astype(
        {"close": float, "volume": float, "open": float, "date": "datetime64[ns]"})


def fetch_yahoo(sym):
    """Fallback: Yahoo Finance uses the .KA suffix for Karachi."""
    import yfinance as yf
    h = yf.Ticker(f"{sym}.KA").history(period="1y", auto_adjust=False)
    if h.empty:
        raise ValueError("empty")
    h = h.reset_index().rename(columns={"Date": "date", "Close": "close",
                                        "Volume": "volume", "Open": "open"})
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)
    return h[["date", "open", "close", "volume"]]


def load_prices(symbols):
    out, s = {}, requests.Session()
    for sym in symbols:
        df = None
        for fn in (lambda: fetch_psx(sym, s), lambda: fetch_yahoo(sym)):
            try:
                df = fn()
                break
            except Exception as e:
                log.debug("%s source failed: %s", sym, e)
        if df is None or len(df) < 80:
            log.warning("Skipping %s (no/short data)", sym)
            continue
        out[sym] = df
        time.sleep(0.3)
    return out


def synthetic_prices(symbols, n=300, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for sym in symbols:
        drift = rng.normal(0.0004, 0.0006)
        r = rng.normal(drift, 0.018, n)
        close = 100 * rng.uniform(0.5, 3) * np.exp(np.cumsum(r))
        vol = rng.lognormal(13, 0.5, n) * (1 + 2 * (np.abs(r) > 0.03))
        out[sym] = pd.DataFrame({
            "date": pd.bdate_range(end=dt.date.today(), periods=n),
            "open": np.r_[close[0], close[:-1]], "close": close, "volume": vol})
    return out

# --------------------------------------------------------------------------
# 3. NEWS SENTIMENT
# --------------------------------------------------------------------------
LEX = {  # keyword -> weight
    "surge": 1, "rally": 1, "gain": 1, "record high": 2, "profit": 1, "growth": 1,
    "dividend": 1, "bonus": 1, "upgrade": 1, "rate cut": 2, "slashes rate": 2,
    "imf approves": 2, "tranche": 1, "bullish": 1, "jump": 1, "rise": 1, "recovery": 1,
    "strong": 1, "beat": 1, "inflow": 1, "investment": 1, "agreement": 1, "stable": 1,
    "fall": -1, "plunge": -2, "loss": -1, "decline": -1, "slump": -1, "downgrade": -1,
    "default": -2, "rate hike": -2, "crisis": -2, "ban": -1, "new tax": -1, "bearish": -1,
    "sell-off": -2, "selloff": -2, "tension": -1, "war": -2, "sanction": -1, "drop": -1,
    "weak": -1, "shutdown": -2, "outflow": -1, "circular debt": -1, "political unrest": -2,
}
_LEX_RE = {k: re.compile(r"\b" + re.escape(k)) for k in LEX}


def sentiment(text):
    t, s = text.lower(), 0
    for k, rx in _LEX_RE.items():
        if rx.search(t):
            s += LEX[k]
    return max(-3, min(3, s)) / 3.0


def parse_feed(url, max_age_h=72):
    if feedparser is None:
        return []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        feed = feedparser.parse(r.content)
    except Exception as e:
        log.warning("Feed failed %s: %s", url[:60], e)
        return []
    items, cutoff = [], time.time() - max_age_h * 3600
    for e in feed.entries[:60]:
        ts = time.mktime(e.published_parsed) if getattr(e, "published_parsed", None) else time.time()
        if ts >= cutoff:
            items.append({"title": e.get("title", ""), "summary": re.sub("<[^>]+>", "", e.get("summary", ""))[:300],
                          "src": url.split("/")[2]})
    return items


def gather_news(symbols, per_stock=True):
    pool = []
    for u in PK_FEEDS:
        pool += parse_feed(u)
    glob = []
    for q in GLOBAL_QUERIES:
        glob += parse_feed(GNEWS.format(q=requests.utils.quote(q)))
        time.sleep(0.3)
    market = [sentiment(i["title"] + " " + i["summary"]) for i in pool + glob]
    market_score = 100 * float(np.mean(market)) if market else 0.0

    stock_scores, headlines = {}, {}
    extra = {}
    if per_stock:
        for sym in symbols:
            extra[sym] = parse_feed(GNEWS.format(q=requests.utils.quote(f'"{UNIVERSE[sym][0]}" Pakistan')))
            time.sleep(0.3)
    for sym in symbols:
        kws = [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in UNIVERSE[sym]]
        hits = [i for i in pool if any(k.search(i["title"]) for k in kws)] + extra.get(sym, [])
        seen, uniq = set(), []
        for h in hits:
            if h["title"] not in seen:
                seen.add(h["title"]); uniq.append(h)
        sc = [sentiment(h["title"] + " " + h["summary"]) for h in uniq]
        n = len(sc)
        stock_scores[sym] = (100 * float(np.mean(sc)) * min(1.0, n / 3)) if n else 0.0
        headlines[sym] = [h["title"] for h in uniq[:3]]
    return market_score, stock_scores, headlines

# --------------------------------------------------------------------------
# 4. INDICATORS + TECHNICAL SCORE
# --------------------------------------------------------------------------
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def add_indicators(df):
    df = df.sort_values("date").reset_index(drop=True).copy()
    c = df["close"]
    for n in (9, 21, 50):
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_h"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["macd_n"] = df["macd_h"] / c
    df["rsi"] = rsi(c)
    for n in (1, 3, 5, 10):
        df[f"ret{n}"] = c.pct_change(n)
    df["volr"] = (df["volume"] / df["volume"].rolling(20).mean()).replace([np.inf, -np.inf], np.nan)
    df["avgvol"] = df["volume"].rolling(20).mean()
    # Portal gives only close/open/volume (no high/low) -> ATR proxy from close-to-close moves
    df["atr"] = c.diff().abs().rolling(14).mean() * 1.25
    df["hh20"] = c.shift(1).rolling(20).max()
    df["vol20"] = df["ret1"].rolling(20).std()
    df["gap"] = df["ema9"] / df["ema21"] - 1
    return df


def tech_score(r):
    s = 0
    s += 20 if r.ema9 > r.ema21 else -20
    s += 15 if r.close > r.ema50 else -15
    s += 15 if r.macd_h > 0 else -15
    if r.rsi > 75:
        s -= 15
    elif r.rsi < 30:
        s += 10
    elif 45 <= r.rsi <= 68:
        s += 10
    if r.volr > 1.5:
        s += 15 if r.ret1 > 0 else -15
    if r.close >= r.hh20:
        s += 15
    return float(np.clip(s, -100, 100))

# --------------------------------------------------------------------------
# 5. ML MODEL  (pooled across stocks, time-ordered split)
# --------------------------------------------------------------------------
FEATS = ["ret1", "ret3", "ret5", "ret10", "rsi", "macd_n", "volr", "gap", "vol20"]


def train_model(frames):
    if GradientBoostingClassifier is None:
        return None, None
    parts = []
    for sym, df in frames.items():
        d = df.copy()
        d["y"] = (d["close"].shift(-1) / d["close"] - 1 > ROUND_TRIP_COST / 2).astype(int)
        parts.append(d.iloc[:-1].dropna(subset=FEATS + ["y"]))
    big = pd.concat(parts).sort_values("date")
    if len(big) < 500:
        return None, None
    cut = int(len(big) * 0.8)
    tr, te = big.iloc[:cut], big.iloc[cut:]
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=1)
    m.fit(tr[FEATS], tr["y"])
    auc = roc_auc_score(te["y"], m.predict_proba(te[FEATS])[:, 1]) if te["y"].nunique() > 1 else None
    m.fit(big[FEATS], big["y"])  # refit on everything for live use
    return m, auc

# --------------------------------------------------------------------------
# 6. SIGNALS
# --------------------------------------------------------------------------
def build_signals(frames, market_score, stock_news, headlines, model, capital, risk_pct):
    rows = []
    for sym, df in frames.items():
        last = df.iloc[-1]
        if last[FEATS + ["atr", "ema50"]].isna().any():
            continue
        t = tech_score(last)
        p_up = float(model.predict_proba(last[FEATS].to_frame().T.astype(float))[0, 1]) if model else None
        news = 0.7 * stock_news.get(sym, 0.0) + 0.3 * market_score
        if p_up is not None:
            score = 0.5 * t + 0.2 * (p_up - 0.5) * 200 + 0.3 * news
        else:
            score = 0.7 * t + 0.3 * news
        score = float(np.clip(score, -100, 100))

        notes = []
        if last.avgvol < MIN_AVG_VOLUME:
            notes.append("illiquid"); score = min(score, 0)
        if last.ret1 > MAX_TODAY_MOVE:
            notes.append("already ran >6% today - don't chase"); score = min(score, 25)
        if last.rsi > 80:
            notes.append("overbought")
        signal = "BUY" if score >= 40 else "AVOID/SELL" if score <= -40 else "WATCH"

        entry = float(last.close)
        stop = entry - 1.5 * float(last.atr)
        target = entry + 2.5 * float(last.atr)
        risk_per_share = max(entry - stop, 1e-6)
        shares = int((capital * risk_pct) / risk_per_share)
        shares = max(0, min(shares, int(capital * 0.25 / entry)))  # cap 25% capital per position
        rows.append(dict(symbol=sym, signal=signal, score=round(score, 1), tech=round(t, 1),
                         news=round(news, 1), p_up=None if p_up is None else round(p_up, 2),
                         close=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
                         shares=shares, rsi=round(float(last.rsi), 1), vol_ratio=round(float(last.volr), 2),
                         data_date=str(last.date.date()), notes="; ".join(notes),
                         headlines=headlines.get(sym, [])))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------------------
# 7. BACKTEST (technical rules only - news history isn't available)
# --------------------------------------------------------------------------
def backtest(frames, threshold=40):
    rets = []
    for sym, df in frames.items():
        d = df.dropna(subset=FEATS + ["atr", "ema50", "hh20"]).copy()
        d["score"] = d.apply(tech_score, axis=1)
        d["next"] = d["close"].shift(-1) / d["close"] - 1
        sel = d[(d.score >= threshold) & (d.avgvol >= MIN_AVG_VOLUME) & d.next.notna()]
        rets += list(sel["next"] - ROUND_TRIP_COST)
    if not rets:
        print("No trades in backtest."); return
    r = np.array(rets)
    print(f"\nBACKTEST (buy at close on score>={threshold}, sell next close, cost {ROUND_TRIP_COST:.1%})")
    print(f"  trades: {len(r)} | win rate: {(r > 0).mean():.1%} | avg/trade: {r.mean():.2%} "
          f"| worst: {r.min():.2%} | best: {r.max():.2%}")
    print("  Past results do not predict future results; costs/slippage vary by broker.\n")

# --------------------------------------------------------------------------
# 8. OUTPUT
# --------------------------------------------------------------------------
def telegram(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": text}, timeout=15)
        except Exception as e:
            log.warning("Telegram failed: %s", e)


def report(sig, market_score, auc, top):
    print("\n" + "=" * 78)
    print(f"PSX SIGNALS  {dt.datetime.now():%Y-%m-%d %H:%M}   market-news sentiment: {market_score:+.0f}/100"
          + (f"   ML holdout AUC: {auc:.2f}" if auc else ""))
    print("=" * 78)
    cols = ["symbol", "signal", "score", "tech", "news", "p_up", "close", "stop", "target", "shares", "notes"]
    print(sig.head(top)[cols].to_string(index=False))
    buys = sig[sig.signal == "BUY"].head(top)
    msg = [f"PSX signals {dt.date.today()} (not financial advice)"]
    for _, r in buys.iterrows():
        msg.append(f"BUY {r.symbol} @ {r.close} | SL {r.stop} | TP {r.target} | qty {r.shares} | score {r.score}")
    if buys.empty:
        msg.append("No BUY signals today - sitting out is a position.")
    print("\n" + "\n".join(msg))
    for _, r in buys.iterrows():
        for h in r.headlines:
            print(f"   [{r.symbol}] {h}")
    print("\nStale-data check: latest data date =", sig.data_date.max())
    return "\n".join(msg)


def run(a):
    syms = list(UNIVERSE)
    frames = synthetic_prices(syms) if a.demo else load_prices(syms)
    if not frames:
        log.error("No price data. Check internet / PSX endpoint."); return
    frames = {k: add_indicators(v) for k, v in frames.items()}
    if a.backtest:
        backtest(frames); return
    if a.demo:
        market, stock_news, heads = 0.0, {s: 0.0 for s in frames}, {}
    else:
        market, stock_news, heads = gather_news(list(frames), per_stock=not a.no_stock_news)
    model, auc = train_model(frames)
    sig = build_signals(frames, market, stock_news, heads, model, a.capital, a.risk)
    msg = report(sig, market, auc, a.top)
    sig.drop(columns="headlines").to_csv("psx_signals.csv", index=False)
    with open("psx_signals.json", "w") as f:
        json.dump(sig.to_dict("records"), f, indent=2, default=str)
    telegram(msg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=500_000, help="trading capital in PKR")
    p.add_argument("--risk", type=float, default=0.01, help="fraction of capital risked per trade")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--demo", action="store_true", help="synthetic data, no internet needed")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--no-stock-news", action="store_true", help="faster: skip per-stock news queries")
    p.add_argument("--loop", type=int, default=0, help="re-run every N minutes")
    a = p.parse_args()
    while True:
        run(a)
        if not a.loop:
            break
        time.sleep(a.loop * 60)
