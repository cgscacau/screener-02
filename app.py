# app.py — Finance Hub (Screener • Sinais • Backtest • Carteira)
import io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as pex
import plotly.graph_objs as go
from sklearn.covariance import LedoitWolf

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Finance Hub — Yahoo Finance", layout="wide")

# ============================================================
# Helpers de dados (seguros/robustos)
# ============================================================
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_history(tickers, period="3y", interval="1d"):
    """Baixa histórico do Yahoo e normaliza para MultiIndex [ticker, campo]."""
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(
        tickers, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=True, group_by="ticker"
    )
    if data.empty:
        return data
    if not isinstance(data.columns, pd.MultiIndex):
        t = tickers[0] if tickers else "TICKER"
        data.columns = pd.MultiIndex.from_product([[t], data.columns])
    return data

def split_close_volume(raw, fallback_ticker="TICKER"):
    """Extrai Close (ou Adj Close) e Volume como DataFrames (colunas=tickers)."""
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([[fallback_ticker], raw.columns])
    fields = raw.columns.get_level_values(1).unique().tolist()
    close_key = "Close" if "Close" in fields else ("Adj Close" if "Adj Close" in fields else fields[0])
    close = raw.xs(close_key, axis=1, level=1)
    volume = raw.xs("Volume", axis=1, level=1) if "Volume" in fields else pd.DataFrame(index=raw.index, columns=close.columns)
    return close, volume

def pick_field_series(df_single_ticker: pd.DataFrame, field_name: str):
    """Retorna Series para 'Open/High/Low/Close/Adj Close' mesmo se vier MultiIndex."""
    if df_single_ticker is None or df_single_ticker.empty:
        return None
    if isinstance(df_single_ticker.columns, pd.MultiIndex):
        lv1 = df_single_ticker.columns.get_level_values(1)
        if field_name in lv1:
            s = df_single_ticker.xs(field_name, axis=1, level=1)
            return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        return None
    return df_single_ticker[field_name] if field_name in df_single_ticker.columns else None

def pick_close_series(df_single_ticker: pd.DataFrame):
    """Escolhe Close -> Adj Close -> primeira coluna numérica disponível."""
    s = pick_field_series(df_single_ticker, "Close")
    if s is None:
        s = pick_field_series(df_single_ticker, "Adj Close")
    if s is None:
        # cai para a 1ª coluna numérica
        for c in df_single_ticker.columns:
            series = pd.to_numeric(df_single_ticker[c], errors="coerce")
            if series.notna().any():
                return series
        return None
    return s

def _safe_pct_change(s, periods=1):
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

def compute_atr14(o: pd.Series, hi: pd.Series, lo: pd.Series, c: pd.Series):
    tr1 = (hi - lo).abs()
    tr2 = (hi - c.shift()).abs()
    tr3 = (lo - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(14).mean()

# ============================================================
# Indicadores & Score (usados no Screener e na Análise)
# ============================================================
def indicators(df_close: pd.DataFrame):
    out = {}
    out["sma20"]  = df_close.rolling(20).mean()
    out["sma50"]  = df_close.rolling(50).mean()
    out["sma200"] = df_close.rolling(200).mean()

    delta = df_close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean().replace(0, np.nan))
    out["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df_close.ewm(span=12, adjust=False).mean()
    ema26 = df_close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    ret = _safe_pct_change(df_close).fillna(0)
    out["vol20"] = ret.rolling(20).std() * np.sqrt(252)
    out["ret_1w"] = _safe_pct_change(df_close, 5)
    out["ret_1m"] = _safe_pct_change(df_close, 21)

    roll_max = df_close.rolling(252).max()
    roll_min = df_close.rolling(252).min()
    out["dist_52w_high"] = (df_close / roll_max) - 1.0
    out["dist_52w_low"]  = (df_close / roll_min) - 1.0
    return out

def build_signal_score(close: pd.DataFrame, ind: dict):
    last = close.tail(1); s20 = ind["sma20"].tail(1); s50 = ind["sma50"].tail(1); s200 = ind["sma200"].tail(1)
    rsi = ind["rsi14"].tail(1); macd = ind["macd"].tail(1); macd_sig = ind["macd_signal"].tail(1)
    vol20 = ind["vol20"].tail(1); ret1w = ind["ret_1w"].tail(1)
    dist_high = ind["dist_52w_high"].tail(1); dist_low = ind["dist_52w_low"].tail(1)

    rows = []
    for t in last.columns:
        sc = 50
        if (last[t].iloc[0] > s20[t].iloc[0] > s50[t].iloc[0] > s200[t].iloc[0]): sc += 15
        elif (last[t].iloc[0] < s20[t].iloc[0] < s50[t].iloc[0] < s200[t].iloc[0]): sc -= 15

        val = rsi[t].iloc[0]
        if pd.notna(val):
            if 50 <= val <= 60: sc += 5
            if 40 <= val < 50:  sc -= 5
            if val < 30:        sc += 10
            if val > 70:        sc -= 10

        m = macd[t].iloc[0]; ms = macd_sig[t].iloc[0]
        if pd.notna(m) and pd.notna(ms): sc += 8 if m > ms else -8

        if pd.notna(dist_high[t].iloc[0]) and dist_high[t].iloc[0] > -0.02: sc += 4
        if pd.notna(dist_low[t].iloc[0]) and dist_low[t].iloc[0] < 0.02:    sc += 6

        if pd.notna(ret1w[t].iloc[0]):
            if ret1w[t].iloc[0] > 0.03:  sc += 4
            if ret1w[t].iloc[0] < -0.03: sc -= 4

        if pd.notna(vol20[t].iloc[0]) and vol20[t].iloc[0] > 0.6: sc -= 5

        rows.append((t, float(np.clip(sc, 0, 100))))

    out = pd.DataFrame(rows, columns=["Ticker", "Score"]).set_index("Ticker")
    out["Sinal"] = pd.cut(out["Score"], [-0.1, 30, 45, 55, 70, 100.1],
                          labels=["VENDA FORTE","VENDA","NEUTRO","COMPRA","COMPRA FORTE"])
    return out

# ============================================================
# Sidebar — universo e parâmetros
# ============================================================
st.sidebar.title("Universo de Ativos (Yahoo)")
st.sidebar.caption("Ex.: PETR4.SA, VALE3.SA, AAPL, MSFT, BTC-USD")

default_b3   = "PETR4.SA, VALE3.SA, BBAS3.SA, ITUB4.SA, B3SA3.SA, WEGE3.SA, HGLG11.SA"
default_us   = "AAPL, MSFT, NVDA, TSLA, AMZN, SPY, QQQ"
default_crypto = "BTC-USD, ETH-USD, SOL-USD"

universe_choice = st.sidebar.selectbox("Base inicial", ["Custom","B3 (exemplo)","EUA (exemplo)","Cripto (exemplo)"], index=1)
base_text = default_b3 if universe_choice=="B3 (exemplo)" else default_us if universe_choice=="EUA (exemplo)" else default_crypto if universe_choice=="Cripto (exemplo)" else ""
tickers_text = st.sidebar.text_area("Tickers (separados por vírgula)", value=base_text, height=90)
tickers = sorted(list({t.strip() for t in tickers_text.split(",") if t.strip()}))

period   = st.sidebar.selectbox("Período histórico", ["1y","2y","3y","5y","10y","max"], index=2)
interval = st.sidebar.selectbox("Intervalo", ["1d","1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros globais")
risk_free   = st.sidebar.number_input("Selic anual (proxy RF)", value=0.105, step=0.005, format="%.3f")
slippage_bp = st.sidebar.number_input("Slippage (bps por trade)", value=5, step=1)

if not tickers:
    st.info("Adicione ao menos 1 ticker para começar.")
    st.stop()

# ============================================================
# Abas
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Screener", "📈 Ativo & Sinais", "🧪 Backtest & Risco", "📊 Carteira"])

# ------------------------------------------------------------
# 1) Screener
# ------------------------------------------------------------
with tab1:
    st.header("🔎 Screener Multi-Ativos (Yahoo Finance)")
    with st.spinner("Baixando e processando cotações..."):
        raw = fetch_history(tickers, period=period, interval=interval)
        if raw.empty:
            st.error("Sem dados retornados pelo Yahoo."); st.stop()

        close, volume = split_close_volume(raw, fallback_ticker=(tickers[0] if tickers else "TICKER"))
        ind = indicators(close)
        ranking = build_signal_score(close, ind)

        st.subheader("Filtros Rápidos")
        c1, c2, c3, c4 = st.columns(4)
        rsi_min = c1.slider("RSI mín", 0, 100, 0)
        rsi_max = c2.slider("RSI máx", 0, 100, 100)
        ret_min = c3.slider("Retorno 1M mín (%)", -50, 50, -50)
        vol_max = c4.slider("Volatilidade 20d máx (%, a.a.)", 5, 200, 200)

        last_row      = close.tail(1).T.rename(columns=lambda _: "Último")
        rsi_row       = ind["rsi14"].tail(1).T.rename(columns=lambda _: "RSI14")
        ret1m_row     = (ind["ret_1m"].tail(1).T * 100).rename(columns=lambda _: "Ret_1M_%")
        vol_row       = (ind["vol20"].tail(1).T * 100).rename(columns=lambda _: "Vol20_%")
        d52h_row      = (ind["dist_52w_high"].tail(1).T * 100).rename(columns=lambda _: "Dist_52w_High_%")
        d52l_row      = (ind["dist_52w_low"].tail(1).T  * 100).rename(columns=lambda _: "Dist_52w_Low_%")
        vol_mean_row  = volume.tail(20).mean().rename("Vol_Médio_20")

        table = (ranking.join(last_row, how="left")
                        .join(rsi_row,  how="left")
                        .join(ret1m_row,how="left")
                        .join(vol_row,  how="left")
                        .join(d52h_row, how="left")
                        .join(d52l_row, how="left")
                        .join(vol_mean_row, how="left")
                        .sort_values("Score", ascending=False))

        mask = (
            table["RSI14"].between(rsi_min, rsi_max, inclusive="both") &
            (table["Ret_1M_%"].fillna(-1e9) >= ret_min) &
            (table["Vol20_%"].fillna(1e9) <= vol_max)
        )
        table_f = table[mask].copy()

        st.caption("Dica: clique nas colunas para ordenar.")
        st.dataframe(
            table_f.style.format({"Último":"{:.2f}","RSI14":"{:.1f}","Ret_1M_%":"{:.1f}",
                                  "Vol20_%":"{:.1f}","Dist_52w_High_%":"{:.1f}",
                                  "Dist_52w_Low_%":"{:.1f}", "Vol_Médio_20":"{:,.0f}"}),
            use_container_width=True, height=480
        )

        st.subheader("Ranking por Score")
        st.plotly_chart(pex.bar(table_f.reset_index(), x="Ticker", y="Score", color="Sinal", height=420),
                        use_container_width=True)
        st.markdown("**Legenda:** 0–30 (Venda forte) · 30–45 (Venda) · 45–55 (Neutro) · 55–70 (Compra) · 70–100 (Compra forte)")

# ------------------------------------------------------------
# 2) Ativo & Sinais
# ------------------------------------------------------------
with tab2:
    st.header("📈 Análise do Ativo — Gráfico + Indicadores + Regras")
    sel = st.selectbox("Escolha um ativo", tickers, index=0)

    with st.spinner("Baixando OHLCV do ativo…"):
        df1 = yf.download(sel, period=period, interval=interval, auto_adjust=False, progress=False, threads=True)

    if df1.empty:
        st.warning("Sem dados para o ticker."); st.stop()

    pclose = pick_close_series(df1)
    if pclose is None or pclose.dropna().empty:
        st.error("Não foi possível determinar a série de fechamento."); st.stop()

    o     = pick_field_series(df1, "Open")
    hi    = pick_field_series(df1, "High")
    lo    = pick_field_series(df1, "Low")
    c_plot= pick_field_series(df1, "Close") or pick_field_series(df1, "Adj Close")

    if any(s is None or s.dropna().empty for s in [o, hi, lo, c_plot]):
        st.warning("OHLC incompleto para Candlestick. Mostrando linha do fechamento.")
        fig_line = pex.line(pclose.rename("Close"))
        fig_line.update_layout(height=520)
        st.plotly_chart(fig_line, use_container_width=True)
        atr_last = np.nan
    else:
        tmp = pd.DataFrame({"Open":o, "High":hi, "Low":lo, "Close":c_plot}).dropna()
        atr_last = float(compute_atr14(tmp["Open"], tmp["High"], tmp["Low"], tmp["Close"]).iloc[-1])

        fig_c = go.Figure()
        fig_c.add_trace(go.Candlestick(x=tmp.index, open=tmp["Open"], high=tmp["High"],
                                       low=tmp["Low"],  close=tmp["Close"], name="Preço"))
        sma20  = pclose.rolling(20).mean().reindex(tmp.index)
        sma50  = pclose.rolling(50).mean().reindex(tmp.index)
        sma200 = pclose.rolling(200).mean().reindex(tmp.index)
        fig_c.add_trace(go.Scatter(x=tmp.index, y=sma20,  name="SMA20",  mode="lines"))
        fig_c.add_trace(go.Scatter(x=tmp.index, y=sma50,  name="SMA50",  mode="lines"))
        fig_c.add_trace(go.Scatter(x=tmp.index, y=sma200, name="SMA200", mode="lines"))
        fig_c.update_layout(height=520, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig_c, use_container_width=True)

    # Métricas
    ema12  = pclose.ewm(span=12, adjust=False).mean()
    ema26  = pclose.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    macd_s = macd.ewm(span=9,  adjust=False).mean()
    delta  = pclose.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rsi14 = 100 - 100/(1 + (up.rolling(14).mean() / down.rolling(14).mean()))

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Preço", f"{float(pclose.iloc[-1]):.2f}")
    cB.metric("ATR14", f"{atr_last:.2f}" if np.isfinite(atr_last) else "—")
    cC.metric("RSI14", f"{float(rsi14.iloc[-1]):.1f}" if not np.isnan(rsi14.iloc[-1]) else "—")
    cD.metric("MACD-σ", f"{float(macd.iloc[-1]-macd_s.iloc[-1]):.3f}")

    st.subheader("Score & Sinal (régua do Screener)")
    ind_one = indicators(pclose.to_frame(sel))
    score_one = build_signal_score(pclose.to_frame(sel), ind_one)
    st.dataframe(score_one, use_container_width=True)

# ------------------------------------------------------------
# 3) Backtest & Risco
# ------------------------------------------------------------
with tab3:
    st.header("🧪 Backtest Rápido + Gestão de Risco")
    sel2      = st.selectbox("Ativo para backtest", tickers, index=min(1, len(tickers)-1))
    entry_l   = st.selectbox("Entrada", ["Cruzamento SMA20>50", "MACD > Sinal", "RSI<30"], index=0)
    exit_l    = st.selectbox("Saída",   ["Stop ATR & Take Profit R", "Cruzamento SMA20<50", "MACD < Sinal", "RSI>70"], index=0)
    atr_mult  = st.slider("Stop ATR (x)", 0.5, 6.0, 2.0, 0.5)
    take_R    = st.slider("Take Profit (R múltiplos)", 0.5, 6.0, 2.0, 0.5)
    fixed_risk= st.number_input("Risco fixo por trade (% do capital)", 0.5, 10.0, 1.0, 0.5)
    slippage  = (slippage_bp / 10000.0)

    with st.spinner("Preparando dados…"):
        df_bt = yf.download(sel2, period=period, interval=interval, auto_adjust=False, progress=False, threads=True)
        if df_bt.empty:
            st.warning("Sem dados para backtest."); st.stop()

        o = pick_field_series(df_bt, "Open"); hi = pick_field_series(df_bt, "High")
        lo = pick_field_series(df_bt, "Low");  c = pick_close_series(df_bt)
        if any(s is None or s.dropna().empty for s in [o, hi, lo, c]):
            st.warning("Dados insuficientes p/ backtest."); st.stop()

        df_bt = pd.DataFrame({"Open":o, "High":hi, "Low":lo, "Close":c}).dropna()
        df_bt["SMA20"] = df_bt["Close"].rolling(20).mean()
        df_bt["SMA50"] = df_bt["Close"].rolling(50).mean()
        ema12 = df_bt["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df_bt["Close"].ewm(span=26, adjust=False).mean()
        df_bt["MACD"] = ema12 - ema26
        df_bt["MACD_SIG"] = df_bt["MACD"].ewm(span=9, adjust=False).mean()
        delta = df_bt["Close"].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
        df_bt["RSI"] = 100 - 100/(1 + (up.rolling(14).mean() / down.rolling(14).mean()))
        df_bt["ATR"] = compute_atr14(df_bt["Open"], df_bt["High"], df_bt["Low"], df_bt["Close"])

        if entry_l == "Cruzamento SMA20>50":
            df_bt["ENTRY"] = (df_bt["SMA20"] > df_bt["SMA50"]) & (df_bt["SMA20"].shift(1) <= df_bt["SMA50"].shift(1))
        elif entry_l == "MACD > Sinal":
            df_bt["ENTRY"] = (df_bt["MACD"] > df_bt["MACD_SIG"]) & (df_bt["MACD"].shift(1) <= df_bt["MACD_SIG"].shift(1))
        else:
            df_bt["ENTRY"] = (df_bt["RSI"] < 30) & (df_bt["RSI"].shift(1) >= 30)

        if exit_l == "Cruzamento SMA20<50":
            df_bt["EXIT"]  = (df_bt["SMA20"] < df_bt["SMA50"]) & (df_bt["SMA20"].shift(1) >= df_bt["SMA50"].shift(1))
        elif exit_l == "MACD < Sinal":
            df_bt["EXIT"]  = (df_bt["MACD"] < df_bt["MACD_SIG"]) & (df_bt["MACD"].shift(1) >= df_bt["MACD_SIG"].shift(1))
        elif exit_l == "RSI>70":
            df_bt["EXIT"]  = (df_bt["RSI"] > 70) & (df_bt["RSI"].shift(1) <= 70)
        else:
            df_bt["EXIT"]  = False

        trades = []; in_pos = False; capital = 100000.0
        for i in range(1, len(df_bt)):
            row = df_bt.iloc[i]
            if (not in_pos) and row["ENTRY"]:
                stop = row["Close"] - atr_mult * row["ATR"]
                risk_per_share = max(row["Close"] - stop, 1e-6)
                qty = max(int((capital*(fixed_risk/100.0))/risk_per_share), 1)
                entry_px = row["Close"] * (1 + slippage)
                stop_px  = stop
                tp_px    = entry_px + take_R * (entry_px - stop_px)
                in_pos = True
                trades.append({"open_time": df_bt.index[i], "open_px": entry_px, "qty": qty,
                               "stop_px": stop_px, "tp_px": tp_px,
                               "close_time": None, "close_px": None, "ret": None})
                continue

            if in_pos:
                low  = row["Low"]  * (1 - slippage)
                high = row["High"] * (1 - slippage)
                exit_px = None
                if low <= trades[-1]["stop_px"]:
                    exit_px = trades[-1]["stop_px"]
                elif high >= trades[-1]["tp_px"]:
                    exit_px = trades[-1]["tp_px"]
                elif row["EXIT"]:
                    exit_px = row["Close"] * (1 - slippage)

                if exit_px is not None:
                    trades[-1]["close_time"] = df_bt.index[i]
                    trades[-1]["close_px"]   = exit_px
                    pnl = (exit_px - trades[-1]["open_px"]) * trades[-1]["qty"]
                    trades[-1]["ret"] = pnl / capital
                    capital *= (1 + trades[-1]["ret"])
                    in_pos = False

        if trades and trades[-1]["close_time"] is None:
            last_px = df_bt["Close"].iloc[-1] * (1 - slippage)
            trades[-1]["close_time"] = df_bt.index[-1]
            trades[-1]["close_px"]   = last_px
            pnl = (last_px - trades[-1]["open_px"]) * trades[-1]["qty"]
            trades[-1]["ret"] = pnl / capital

        trdf = pd.DataFrame(trades)
        if trdf.empty:
            st.warning("Nenhuma operação gerada.")
        else:
            st.subheader("Resumo dos Trades")
            st.dataframe(
                trdf[["open_time","open_px","close_time","close_px","ret"]]
                .style.format({"open_px":"{:.2f}","close_px":"{:.2f}","ret":"{:.2%}"}),
                use_container_width=True
            )
            st.caption(f"Capital final simulado (teórico): **R$ {capital:,.2f}**".replace(",", "X").replace(".", ",").replace("X","."))

            curve = pd.Series(100000.0, index=df_bt.index, dtype=float)
            cap = 100000.0
            for _, tr in trdf.iterrows():
                if pd.notna(tr["close_time"]):
                    cap *= (1 + tr["ret"])
                    curve.loc[tr["close_time"]:] = cap

            st.plotly_chart(pex.line(curve.rename("Equity Curve"), title="Curva de Capital (aprox.)"),
                            use_container_width=True)

# ------------------------------------------------------------
# 4) Carteira — Markowitz (Ledoit-Wolf)
# ------------------------------------------------------------
with tab4:
    st.header("📊 Carteira — Markowitz (Ledoit-Wolf) com simulação")
    with st.spinner("Coletando dados…"):
        raw2 = fetch_history(tickers, period=period, interval="1d")
        if raw2.empty:
            st.error("Sem dados para os tickers."); st.stop()
        close2, _ = split_close_volume(raw2, fallback_ticker=(tickers[0] if tickers else "TICKER"))
        close2 = close2.dropna(how="all")
        rets   = close2.pct_change().dropna(how="any")

    mu = rets.mean() * 252
    cov_lw = LedoitWolf().fit(rets.values).covariance_
    cov = pd.DataFrame(cov_lw, index=rets.columns, columns=rets.columns) * 252

    sims = st.slider("Número de simulações de carteira", 200, 5000, 1500, 100)
    Ws = np.random.dirichlet(np.ones(len(rets.columns)), size=sims)
    port_mu  = Ws @ mu.values
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", Ws, cov.values, Ws))
    sharpe   = (port_mu - risk_free) / np.where(port_vol==0, np.nan, port_vol)

    df_ports = pd.DataFrame({"Vol":port_vol, "Retorno":port_mu, "Sharpe":sharpe})
    st.plotly_chart(pex.scatter(df_ports, x="Vol", y="Retorno", color="Sharpe",
                                title="Fronteira (simulada)", height=520),
                    use_container_width=True)

    best_idx = int(np.nanargmax(sharpe))
    best_w = pd.Series(Ws[best_idx], index=rets.columns).sort_values(ascending=False)

    st.subheader("Pesos sugeridos (máx. Sharpe simulado)")
    st.dataframe((best_w*100).to_frame("Peso_%").style.format("{:.2f}"), use_container_width=True)

    buff = io.StringIO()
    (best_w*100).to_csv(buff, header=["Peso_%"])
    st.download_button("Baixar pesos em CSV", data=buff.getvalue(),
                       file_name="pesos_portfolio.csv", mime="text/csv")

st.success("Pronto! App limpo com Screener, Análise, Backtest e Carteira (com covariância Ledoit-Wolf).")
