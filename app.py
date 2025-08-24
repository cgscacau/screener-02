# app.py
# -*- coding: utf-8 -*-
# Super Screener + Predição + Carteira — Visual Moderno — Yahoo Finance (versão robusta)

import os
import io
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ======== Predição (opcional) ========
SKLEARN_OK = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
except Exception:
    SKLEARN_OK = False

# ======== Config Global ========
st.set_page_config(
    page_title="Cacau — Super Screener + Predição + Carteira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======== Estilo (CSS) — Visual Moderno ========
st.markdown("""
<style>
html, body, [class*="css"]  { font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
div.block-container{padding-top:1.1rem; padding-bottom:2rem;}
[data-testid="stMetricValue"] { font-size: 1.4rem; }
[data-testid="stMetric"] {
  background: #0f172a0a; border: 1px solid #e2e8f0; padding: 14px 12px; border-radius: 16px;
  box-shadow: 0 2px 14px rgba(2,6,23,0.04);
}
.stButton>button { border-radius: 12px; padding: 0.55rem 0.9rem; border: 1px solid #e2e8f0; }
.badge { display:inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; letter-spacing:.02em; }
.badge-buy { background:#dcfce7; color:#166534; border: 1px solid #86efac;}
.badge-strongbuy { background:#bbf7d0; color:#14532d; border: 1px solid #4ade80;}
.badge-sell { background:#fee2e2; color:#7f1d1d; border: 1px solid #fca5a5;}
.badge-strongsell { background:#fecaca; color:#7f1d1d; border: 1px solid #f87171;}
.badge-neutral { background:#e2e8f0; color:#0f172a; border: 1px solid #cbd5e1;}
[data-testid="stDataFrame"] div[role="table"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ======== Datas padrão ========
TODAY = datetime.now().date()
DEFAULT_START = TODAY - timedelta(days=365 * 3)
DEFAULT_END = TODAY

# ======== Yahoo Finance (download robusto) ========
import yfinance as yf

@st.cache_data(ttl=60 * 60)
def yf_download(tickers, start=None, end=None, period=None, interval="1d", tries=3, pause=0.8) -> pd.DataFrame:
    """
    Baixa dados de 1..N tickers com retries e normaliza o MultiIndex (ticker, OHLC).
    Pula o ticker se falhar repetidamente (log simples no console).
    """
    if isinstance(tickers, str):
        tick_list = [tickers]
    else:
        tick_list = list(dict.fromkeys([t for t in tickers if t]))

    # Guard especial para intraday (histórico limitado pelo Yahoo)
    if interval in ("1h", "90m", "30m", "15m", "5m", "1m") and start:
        start_dt = pd.to_datetime(start).date()
        max_back = TODAY - timedelta(days=720)  # ~2 anos
        if start_dt < max_back:
            start = str(max_back)

    frames, last_exc = [], None
    for tk in tick_list:
        for _ in range(tries):
            try:
                df = yf.download(
                    tk, start=start, end=end, period=period,
                    interval=interval, auto_adjust=True,
                    group_by="ticker", threads=False, progress=False
                )
                if df is not None and not df.empty:
                    # se veio SingleIndex (1 ticker), envolve como MultiIndex (tk, OHLC)
                    if not isinstance(df.columns, pd.MultiIndex):
                        df = pd.concat({tk: df}, axis=1)
                    frames.append(df)
                    break
            except Exception as e:
                last_exc = e
            time.sleep(pause)
        else:
            print(f"[WARN] falha ao baixar {tk}: {last_exc}")

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, axis=1)
    raw = raw.sort_index(axis=1)

    # Normaliza níveis: garantir (ticker, OHLC)
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        if "Close" in lv0 and "Close" not in lv1:
            raw = raw.swaplevel(axis=1).sort_index(axis=1)
    return raw.dropna(how="all")

def _ensure_multiindex(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        return df
    return pd.concat({ticker: df}, axis=1)

def get_field_wide(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Extrai uma tabela larga (colunas=tickers) para um campo (Close, Volume, etc.)
    de um DataFrame MultiIndex (ticker, OHLC).
    """
    if raw is None or raw.empty:
        raise ValueError("DataFrame vazio (falha no download).")
    if not isinstance(raw.columns, pd.MultiIndex):
        # caso baixado para 1 ticker sem MI
        if field not in raw.columns:
            raise KeyError(f"'{field}' não encontrado.")
        tk = getattr(raw, "name", None) or "TICKER"
        raw = pd.concat({tk: raw}, axis=1)

    if field in raw.columns.get_level_values(-1):
        out = raw.xs(field, axis=1, level=-1)
    elif field in raw.columns.get_level_values(0):
        out = raw.xs(field, axis=1, level=0)
    else:
        raise KeyError(f"'{field}' não encontrado em nenhum nível.")
    if out.empty:
        raise ValueError(f"Série '{field}' vazia após extração.")
    return out

@st.cache_data(ttl=60 * 60)
def download_prices_single(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Versão single-ticker (mais leve), com retry e normalização de colunas.
    """
    raw = yf_download(ticker, start=start, end=end, interval=interval)
    if raw.empty:
        return pd.DataFrame()
    # retorna um DF simples com OHLCV
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            out = raw[ticker].copy()
        except KeyError:
            # se o nível estiver invertido, tente swap
            raw2 = raw.swaplevel(axis=1).sort_index(axis=1)
            out = raw2[ticker].copy()
    else:
        out = raw.copy()
    return out.dropna()

@st.cache_data(ttl=24 * 60 * 60)
def get_fast_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = dict(getattr(t, "fast_info", {}) or {})
        return info
    except Exception:
        return {}

# ======== Indicadores Técnicos ========
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.bfill().clip(0, 100)

def true_range(h, l, c_prev):
    return pd.concat([(h - l), (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(df["High"], df["Low"], df["Close"].shift(1))
    return tr.rolling(n).mean()

def macd(series: pd.Series, fast=12, slow=26, sig=9):
    fast_ = ema(series, fast)
    slow_ = ema(series, slow)
    macd_line = fast_ - slow_
    signal = ema(macd_line, sig)
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger(series: pd.Series, n=20, k=2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    up = ma + k * sd
    lo = ma - k * sd
    return ma, up, lo

def pct_return(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"] = rsi(out["Close"], 14)
    out["ATR14"] = atr(out, 14)
    out["ATR%"] = (out["ATR14"] / out["Close"]) * 100
    m, s, h = macd(out["Close"])
    out["MACD"] = m
    out["MACDsig"] = s
    out["MACDhist"] = h
    ma20, up, lo = bollinger(out["Close"], 20, 2)
    out["BB_MA20"] = ma20
    out["BB_UP"] = up
    out["BB_LO"] = lo
    out["Ret5D"] = pct_return(out["Close"], 5)
    out["Ret21D"] = pct_return(out["Close"], 21)
    out["Ret63D"] = pct_return(out["Close"], 63)
    return out.dropna()

# ======== Score Técnico ========
def technical_score(row, w_trend=40, w_mom=35, w_band=15, w_short=10) -> float:
    score = 0.0
    # Tendência
    trend = 0.0
    trend += 0.6 if row["Close"] > row["EMA200"] else 0.0
    trend += 0.4 if (row["EMA20"] > row["EMA50"] > row["EMA200"]) else 0.0
    score += w_trend * min(trend, 1.0)
    # Momentum
    mom = 0.0
    r = row["RSI14"]
    if 45 <= r <= 65: mom += 0.4
    if r > 65: mom += 0.6
    if row["MACD"] > row["MACDsig"]: mom += 0.4
    score += w_mom * min(mom, 1.0)
    # Bandas / Vol
    bb = 0.0
    if row["BB_LO"] < row["Close"] < row["BB_UP"]: bb += 0.6
    if abs(row["Close"] - row["BB_MA20"]) / row["Close"] <= 0.03: bb += 0.4
    score += w_band * min(bb, 1.0)
    # Curto prazo
    short = 0.0
    if row["Ret5D"] > 0: short += 0.5
    if row["Ret21D"] > 0: short += 0.5
    score += w_short * min(short, 1.0)

    total_w = w_trend + w_mom + w_band + w_short
    if total_w != 100 and total_w > 0:
        score = score * (100 / total_w)
    return float(round(score, 2))

def classify_signal(score: float) -> str:
    if score >= 80: return "Forte Compra"
    if score >= 60: return "Compra"
    if score >= 40: return "Neutro"
    if score >= 20: return "Venda"
    return "Forte Venda"

def render_badge(label: str) -> str:
    m = {
        "Forte Compra": "badge badge-strongbuy",
        "Compra": "badge badge-buy",
        "Neutro": "badge badge-neutral",
        "Venda": "badge badge-sell",
        "Forte Venda": "badge badge-strongsell",
    }
    cls = m.get(label, "badge badge-neutral")
    return f'<span class="{cls}">{label}</span>'

# ======== Predição & Backtest ========
def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame(index=df.index)
    x["ret1"] = df["Close"].pct_change()
    x["ret5"] = df["Close"].pct_change(5)
    x["ret21"] = df["Close"].pct_change(21)
    x["rsi"] = df["RSI14"]
    x["ema20_dist"] = (df["Close"] - df["EMA20"]) / df["Close"]
    x["ema50_dist"] = (df["Close"] - df["EMA50"]) / df["Close"]
    x["atrp"] = df["ATR%"]
    x["macd"] = df["MACD"]
    x["macd_sig"] = df["MACDsig"]
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    y = (df["Close"].shift(-1) > df["Close"]).loc[x.index].astype(int)
    return x, y

def walk_forward_logit(X: pd.DataFrame, y: pd.Series, splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=min(splits, max(2, len(X)//60)))
    aucs = []
    preds = pd.Series(index=X.index, dtype=float)
    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        if len(np.unique(ytr)) < 2:
            continue
        model = LogisticRegression(max_iter=250, n_jobs=None)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        preds.iloc[te_idx] = p
        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass
    auc = float(np.nanmean(aucs)) if aucs else np.nan
    return preds, auc

def simulate_threshold(df_close: pd.Series, proba: pd.Series, th: float = 0.55):
    proba = proba.reindex(df_close.index).fillna(0.5)
    pos = np.where(proba >= th, 1, np.where(proba <= 1 - th, -1, 0))
    ret = df_close.pct_change().fillna(0)
    strat = pd.Series(pos, index=df_close.index).shift(1).fillna(0) * ret
    eq = (1 + strat).cumprod()
    buyhold = (1 + ret).cumprod()
    return pd.DataFrame({"eq": eq, "buyhold": buyhold, "pos": pos, "ret": strat})

# ======== Risco — position sizing ========
def position_size(capital: float, risk_perc: float, atr_value: float, atr_mult: float, price: float):
    risk_amt = capital * (risk_perc / 100.0)
    stop = atr_mult * atr_value
    if stop <= 0 or price <= 0:
        return 0, 0, 0
    qty = math.floor(risk_amt / stop)
    qty = max(qty, 0)
    exposure = qty * price
    return qty, stop, exposure

# ======== Universos dos repositórios ========
def _read_lines_txt(p: Path):
    try:
        return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception:
        return []

def _read_json_list(p: Path, key_guess=("tickers","ativos","symbols","lista","assets")):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(t).strip() for t in data if str(t).strip()]
        if isinstance(data, dict):
            for k in key_guess:
                if k in data and isinstance(data[k], list):
                    return [str(t).strip() for t in data[k] if str(t).strip()]
            out = []
            for v in data.values():
                if isinstance(v, list):
                    out += [str(t).strip() for t in v if str(t).strip()]
            return list(dict.fromkeys(out))
    except Exception:
        pass
    return []

def load_universes_from_repo_folder():
    candidates = [
        Path("ibov_tickers.txt"),
        Path("sp500_tickers.txt"),
        Path("assets_database.json"),
        Path("ativos.json"),
    ]
    tickers = []
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".txt":
                tickers += _read_lines_txt(p)
            elif p.suffix.lower() == ".json":
                tickers += _read_json_list(p)
    tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
    return list(dict.fromkeys(tickers))

REPO_TICKERS = load_universes_from_repo_folder()

# ======== SIDEBAR ========
st.sidebar.title("⚙️ Parâmetros")

with st.sidebar.expander("⛏️ Universos de Ativos", expanded=True):
    st.caption("Se houver arquivos locais (ibov_tickers.txt, sp500_tickers.txt, ativos.json, assets_database.json), carrego automaticamente.")
    default_br = ["PETR4.SA","VALE3.SA","ITUB4.SA","BBDC4.SA","BBAS3.SA","ABEV3.SA","WEGE3.SA","SUZB3.SA","B3SA3.SA","GGBR4.SA"]
    default_us = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"]
    default_cr = ["BTC-USD","ETH-USD","SOL-USD"]

    if REPO_TICKERS:
        st.success(f"Foram encontrados {len(REPO_TICKERS)} tickers em arquivos locais.")
        blist = st.text_area("Lista geral detectada", value=",".join(REPO_TICKERS), height=100)
        user_list = [t.strip() for t in blist.split(",") if t.strip()]
        colb, colu, colc = st.columns(3)
        with colb: st.caption("Sugestões BR"); st.code(", ".join(default_br), language="text")
        with colu: st.caption("Sugestões US"); st.code(", ".join(default_us), language="text")
        with colc: st.caption("Sugestões Cripto"); st.code(", ".join(default_cr), language="text")
    else:
        br = st.text_area("Brasil (.SA)", value=",".join(default_br), key="txt_br")
        us = st.text_area("USA", value=",".join(default_us), key="txt_us")
        cr = st.text_area("Cripto (USD)", value=",".join(default_cr), key="txt_cr")
        user_list = []
        for block in [br, us, cr]:
            user_list += [t.strip() for t in block.split(",") if t.strip()]

    limit_n = st.slider("Limite de ativos (screener)", 5, 120, 30, 5,
                        help="Para evitar travamentos no Streamlit Cloud.", key="limite_screener")

with st.sidebar.expander("⏱️ Janela & Intervalo", expanded=False):
    start = st.date_input("Início", value=DEFAULT_START, key="date_start")
    end = st.date_input("Fim", value=DEFAULT_END, key="date_end")
    interval = st.selectbox("Intervalo", ["1d","1h","1wk"], index=0, key="sel_interval")

with st.sidebar.expander("🧮 Filtros do Screener", expanded=True):
    f_min_score = st.slider("Score mínimo", 0, 100, 0, 5, key="flt_score")
    f_rsi_min, f_rsi_max = st.slider("Faixa de RSI", 0, 100, (0, 100), 1, key="flt_rsi")
    f_price_min, f_price_max = st.slider("Faixa de preço (Close)", 0.0, 1000.0, (0.0, 1000.0), 1.0, key="flt_price")
    f_trend_align = st.checkbox("Exigir EMA20 > EMA50 > EMA200", value=False, key="flt_trend")

with st.sidebar.expander("🎛️ Pesos do Score (avançado)", expanded=False):
    w_trend = st.slider("Peso Tendência", 0, 100, 40, 5, key="w_trend")
    w_mom = st.slider("Peso Momentum", 0, 100, 35, 5, key="w_mom")
    w_band = st.slider("Peso Bandas/Vol", 0, 100, 15, 5, key="w_band")
    w_short = st.slider("Peso Curto Prazo", 0, 100, 10, 5, key="w_short")

with st.sidebar.expander("🧠 Predição (experimental)", expanded=False):
    enable_pred = st.checkbox("Ativar predição (Análise/Backtest)", value=True, key="pred_enable")
    thr = st.slider("Threshold compra (prob. de alta)", 0.50, 0.70, 0.55, 0.01, key="pred_thr")

with st.sidebar.expander("💼 Risco", expanded=False):
    capital = st.number_input("Capital (R$ / USD)", min_value=0.0, value=100000.0, step=1000.0, key="risk_cap")
    risk_perc = st.slider("Risco por trade (%)", 0.1, 5.0, 1.0, 0.1, key="risk_pct")
    atr_mult = st.slider("Stop = ATR x", 0.5, 5.0, 2.0, 0.5, key="risk_atr_mult")

st.sidebar.caption("Dica: reduza o limite de ativos se a execução ficar lenta.")

# ======== Título & Tabs ========
st.title("🧭 Super Screener + Predição + Carteira (Yahoo Finance)")
st.caption("Unindo ideias dos seus projetos — Screener • Análise • Probabilidades • Carteira & Risco — com visual moderno.")

tabs = st.tabs([
    "📊 Screener",
    "📈 Análise do Ativo",
    "🤖 Predição & Backtest",
    "💼 Carteira & Risco",
    "🗂️ Listas / Importação"
])

# ======== 1) Screener ========
with tabs[0]:
    st.subheader("📊 Screener")
    tickers = user_list[:limit_n]
    if len(tickers) == 0:
        st.info("Adicione tickers na barra lateral.")
        st.stop()

    progress = st.progress(0, text="Baixando dados…")
    rows = []
    for i, t in enumerate(tickers, start=1):
        progress.progress(i / len(tickers), text=f"{t} ({i}/{len(tickers)})")
        try:
            df = download_prices_single(t, str(start), str(end), interval)
            if df.empty or len(df) < 60:
                continue
            ind = build_indicators(df)
            if ind.empty:
                continue
            last = ind.iloc[-1]
            sc = technical_score(last, w_trend, w_mom, w_band, w_short)
            sig = classify_signal(sc)
            rows.append({
                "Ticker": t,
                "Close": round(float(last["Close"]), 4),
                "EMA20": round(float(last["EMA20"]), 4),
                "EMA50": round(float(last["EMA50"]), 4),
                "EMA200": round(float(last["EMA200"]), 4),
                "RSI14": round(float(last["RSI14"]), 2),
                "ATR%": round(float(last["ATR%"]), 2),
                "Ret5D%": round(float(last["Ret5D"] * 100), 2),
                "Ret21D%": round(float(last["Ret21D"] * 100), 2),
                "Ret63D%": round(float(last["Ret63D"] * 100), 2),
                "Score": sc,
                "Sinal": sig,
                "EMA Hierarquia?": bool(last["EMA20"] > last["EMA50"] > last["EMA200"]),
            })
        except Exception as e:
            print(f"[WARN] {t}: {e}")
        time.sleep(0.01)

    progress.empty()

    if not rows:
        st.warning("Sem dados válidos. Verifique tickers/intervalo.")
        st.stop()

    df_scr = pd.DataFrame(rows).sort_values(["Score", "Ret21D%"], ascending=[False, False]).reset_index(drop=True)

    # Filtros
    mask = (df_scr["Score"] >= f_min_score) & \
           (df_scr["RSI14"].between(f_rsi_min, f_rsi_max)) & \
           (df_scr["Close"].between(f_price_min, f_price_max))
    if f_trend_align:
        mask &= df_scr["EMA Hierarquia?"]

    df_flt = df_scr.loc[mask].copy()

    # Badge HTML
    df_show = df_flt.copy()
    df_show["Sinal"] = df_show["Sinal"].apply(render_badge)

    st.dataframe(
        df_show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "Sinal": st.column_config.Column("Sinal", help="Classificação baseada no Score"),
            "EMA Hierarquia?": st.column_config.CheckboxColumn("EMA20>EMA50>EMA200"),
        }
    )

    # Export CSV
    csv = df_flt.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV filtrado", csv, "screener.csv", "text/csv")

# ======== 2) Análise do Ativo ========
with tabs[1]:
    st.subheader("📈 Análise do Ativo")
    chosen = st.selectbox("Escolha o ativo", options=user_list, index=0, key="chosen_asset")
    df = download_prices_single(chosen, str(start), str(end), interval)
    if df.empty:
        st.warning("Sem dados para o ativo/período.")
        st.stop()
    ind = build_indicators(df)
    if ind.empty:
        st.warning("Não foi possível calcular indicadores para o período.")
        st.stop()
    last = ind.iloc[-1]

    # Métricas rápidas
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Preço", f"{last['Close']:.2f}")
    with c2: st.metric("RSI(14)", f"{last['RSI14']:.1f}")
    with c3: st.metric("ATR%", f"{last['ATR%']:.2f}%")
    with c4:
        sc = technical_score(last, w_trend, w_mom, w_band, w_short)
        st.metric("Score Técnico", f"{sc:.0f} / 100")

    # Gráfico principal
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ind.index, open=ind["Open"], high=ind["High"], low=ind["Low"], close=ind["Close"],
        name="Candles", opacity=0.85
    ))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA200"], name="EMA200"))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_UP"], name="BB_UP", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_MA20"], name="BB_MA20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_LO"], name="BB_LO", line=dict(dash="dot")))
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        xaxis_rangebreaks=[dict(bounds=["sat", "mon"])],
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Diagnóstico textual + Predição (opcional)
    colA, colB = st.columns([1.3, 1.0], gap="large")
    with colA:
        sinal = classify_signal(sc)
        st.markdown("#### Diagnóstico")
        bullets = [
            f"**Sinal**: {render_badge(sinal)}",
            f"**EMAs**: {'EMA20>EMA50>EMA200 ✅' if (last['EMA20']>last['EMA50']>last['EMA200']) else 'Desalinhadas ⚠️'}",
            f"**MACD**: {'Acima da linha de sinal ✅' if last['MACD']>last['MACDsig'] else 'Abaixo ⚠️'}",
            f"**Retornos**: 5D {last['Ret5D']*100:.1f}%, 21D {last['Ret21D']*100:.1f}%, 63D {last['Ret63D']*100:.1f}%",
        ]
        st.write("\n".join([f"- {b}" for b in bullets]), unsafe_allow_html=True)

    with colB:
        if enable_pred and SKLEARN_OK:
            with st.spinner("Treinando predição (logística, walk-forward)…"):
                X, y = make_features(ind)
                if len(X) > 200 and len(np.unique(y.dropna())) == 2:
                    proba, auc = walk_forward_logit(X, y, splits=5)
                    last_p = float(proba.dropna().iloc[-1]) if proba.dropna().size else float("nan")
                    st.metric("Prob. de alta (D+1)", f"{last_p*100:0.1f}%")
                    st.caption(f"AUC média (WF): {auc:0.3f}  •  (≈0.5 é aleatório)")
                else:
                    st.info("Amostra curta para predição robusta (≥200 barras).")
        elif enable_pred and not SKLEARN_OK:
            st.info("Instale scikit-learn (veja requirements.txt) para habilitar a predição.")

# ======== 3) Predição & Backtest ========
with tabs[2]:
    st.subheader("🤖 Predição & Backtest (didático)")
    chosen2 = st.selectbox("Ativo para predição", options=user_list, index=0, key="pred_asset")
    df2 = download_prices_single(chosen2, str(start), str(end), interval)
    if df2.empty:
        st.warning("Sem dados para o ativo/período.")
        st.stop()
    ind2 = build_indicators(df2)
    if ind2.empty:
        st.warning("Não foi possível calcular indicadores para o período.")
    elif not SKLEARN_OK:
        st.info("Instale scikit-learn para usar esta aba.")
    else:
        X2, y2 = make_features(ind2)
        if len(X2) < 200 or len(np.unique(y2.dropna())) < 2:
            st.warning("Amostra insuficiente para walk-forward (≥200 barras e alvo variado).")
        else:
            with st.spinner("Treinando e simulando…"):
                proba2, auc2 = walk_forward_logit(X2, y2, splits=5)
                sim = simulate_threshold(ind2["Close"], proba2, th=thr)
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("AUC (WF)", f"{auc2:0.3f}")
                with c2: st.metric("Retorno Strat", f"{(sim['eq'].iloc[-1]-1)*100:0.2f}%")
                with c3:
                    dd = (sim["eq"] / sim["eq"].cummax() - 1).min()
                    st.metric("Max Drawdown", f"{dd*100:0.2f}%")

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=sim.index, y=sim["eq"], name="Estratégia (Eq)"))
                fig2.add_trace(go.Scatter(x=sim.index, y=sim["buyhold"], name="Buy&Hold"))
                fig2.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
            st.caption("⚠️ Predição didática, sujeita a overfitting. Use como estudo, não como recomendação.")

# ======== 4) Carteira & Risco ========
with tabs[3]:
    st.subheader("💼 Carteira & Risco (ATR sizing)")

    # Base para carteira: se vier do Screener usa df_flt; se não, calcula rápida
    if "df_flt" in locals() and isinstance(df_flt, pd.DataFrame) and not df_flt.empty:
        base = df_flt[["Ticker","Close","RSI14","ATR%","Score","Sinal"]].copy()
    else:
        base_rows = []
        prog2 = st.progress(0, text="Montando base…")
        for i, t in enumerate(user_list[:limit_n], start=1):
            prog2.progress(i/len(user_list[:limit_n]), text=f"{t} ({i}/{len(user_list[:limit_n])})")
            try:
                dfb = download_prices_single(t, str(start), str(end), interval)
                if dfb.empty or len(dfb) < 60:
                    continue
                indb = build_indicators(dfb)
                if indb.empty:
                    continue
                lastb = indb.iloc[-1]
                scb = technical_score(lastb, w_trend, w_mom, w_band, w_short)
                base_rows.append({
                    "Ticker": t,
                    "Close": float(lastb["Close"]),
                    "RSI14": float(lastb["RSI14"]),
                    "ATR%": float(lastb["ATR%"]),
                    "Score": scb,
                    "Sinal": classify_signal(scb),
                })
            except Exception as e:
                print(f"[WARN] {t}: {e}")
        prog2.empty()
        base = pd.DataFrame(base_rows)

    if base.empty:
        st.info("Sem base para sugerir carteira. Rode o Screener na aba 1 ou verifique seus tickers.")
    else:
        st.markdown("#### Base de seleção")
        st.dataframe(base.reset_index(drop=True), hide_index=True, use_container_width=True)

        method = st.radio("Método de pesos", ["Equal-weight", "Inverso ao ATR"], horizontal=True, key="peso_method")
        n_top = st.slider("Qtd. de ativos (top por Score)", 3, min(25, len(base)), min(10, len(base)), key="peso_n_top")
        picks = base.sort_values("Score", ascending=False).head(n_top).copy()

        if method == "Inverso ao ATR":
            w = 1 / picks["ATR%"].replace(0, np.nan)
            w = w.replace([np.inf, -np.inf], np.nan).fillna(w.mean() if w.notna().any() else 1.0)
            picks["Peso"] = w / w.sum() if w.sum() > 0 else 1.0 / len(picks)
        else:
            picks["Peso"] = 1.0 / len(picks)

        picks["Alocado"] = picks["Peso"] * capital

        c1, c2 = st.columns([1.5, 1.0], gap="large")
        with c1:
            st.markdown("#### Sugestão de Alocação")
            view = picks[["Ticker","Close","ATR%","Score","Sinal","Peso","Alocado"]].copy()
            view["Peso (%)"] = (view["Peso"]*100).round(2)
            st.dataframe(view.drop(columns=["Peso"]), use_container_width=True, hide_index=True)
        with c2:
            figp = go.Figure(data=[go.Pie(labels=picks["Ticker"], values=picks["Peso"], hole=.45)])
            figp.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(figp, use_container_width=True)

        st.markdown("#### Position sizing por ATR (unidade de risco)")
        choose_risk = st.selectbox("Ativo", options=picks["Ticker"].tolist(), key="risk_asset")
        df_pos = download_prices_single(choose_risk, str(start), str(end), interval)
        ind_pos = build_indicators(df_pos)
        last_pos = ind_pos.iloc[-1]
        qty, stop_abs, exposure = position_size(
            capital=capital,
            risk_perc=risk_perc,
            atr_value=float(last_pos["ATR14"]),
            atr_mult=atr_mult,
            price=float(last_pos["Close"]),
        )
        st.write(
            f"- Preço: **{last_pos['Close']:.2f}**  •  ATR(14): **{last_pos['ATR14']:.2f}**  →  Stop = **{atr_mult}×ATR = {stop_abs:.2f}**  \n"
            f"- Risco/trade: **{risk_perc:.1f}%**  →  Quantidade: **{qty}**  •  Exposição: **{exposure:,.2f}**"
        )

# ======== 5) Listas / Importação ========
with tabs[4]:
    st.subheader("🗂️ Listas de Ativos")
    st.write("Baixe a lista atual, edite e reenvie. Ou cole manualmente na sidebar.")
    current = pd.DataFrame({"ticker": user_list})
    st.download_button("⬇️ Baixar lista atual (CSV)", data=current.to_csv(index=False).encode("utf-8"),
                       file_name="lista_tickers.csv", mime="text/csv")

    up = st.file_uploader("Subir CSV com coluna 'ticker'", type=["csv"], key="uploader_csv")
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            if "ticker" in df_up.columns:
                new_list = df_up["ticker"].astype(str).str.strip().tolist()
                st.success(f"Carregado {len(new_list)} tickers. Copie-os para a barra lateral para usar.")
                st.dataframe(df_up.head(), use_container_width=True)
            else:
                st.error("CSV precisa conter a coluna 'ticker'.")
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")

st.caption("© Para estudo. Sem recomendação de investimento.")
