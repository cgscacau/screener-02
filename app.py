import time
import io
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.ar_model import AutoReg
import streamlit as st

st.set_page_config(page_title="Finance Hub — Screener, Sinais, Previsões e Carteira", layout="wide")

# =============================
# Helpers de dados (robustos)
# =============================
@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_history(tickers, period="3y", interval="1d"):
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
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([[fallback_ticker], raw.columns])
    fields = raw.columns.get_level_values(1).unique().tolist()
    close_key = "Close" if "Close" in fields else ("Adj Close" if "Adj Close" in fields else fields[0])
    close = raw.xs(close_key, axis=1, level=1)
    volume = raw.xs("Volume", axis=1, level=1) if "Volume" in fields else pd.DataFrame(index=raw.index, columns=close.columns)
    return close, volume

def pick_field_series(df_single_ticker, field_name):
    """Retorna uma Series (indexada por data) para o field pedido, mesmo se MultiIndex."""
    if df_single_ticker.empty:
        return None
    if isinstance(df_single_ticker.columns, pd.MultiIndex):
        lv1 = df_single_ticker.columns.get_level_values(1)
        if field_name in lv1:
            s = df_single_ticker.xs(field_name, axis=1, level=1)
            # pega a primeira coluna (é o próprio ticker)
            return s.iloc[:, 0]
        return None
    else:
        return df_single_ticker[field_name] if field_name in df_single_ticker.columns else None

def pick_close_series(df_single_ticker):
    """Escolhe Close -> Adj Close -> primeira coluna numérica."""
    s = pick_field_series(df_single_ticker, "Close")
    if s is None:
        s = pick_field_series(df_single_ticker, "Adj Close")
    if s is None:
        # fallback extremo: primeira coluna numérica
        for c in df_single_ticker.columns:
            try:
                return pd.to_numeric(df_single_ticker[c], errors="coerce")
            except Exception:
                continue
        return None
    return s

def _safe_pct_change(s, periods=1):
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

def compute_atr14(o, h, l, c):
    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(14).mean()

# =============================
# Indicadores & Score
# =============================
def indicators(df_close: pd.DataFrame):
    out = {}
    out["sma20"] = df_close.rolling(20).mean()
    out["sma50"] = df_close.rolling(50).mean()
    out["sma200"] = df_close.rolling(200).mean()
    delta = df_close.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean().replace(0, np.nan))
    out["rsi14"] = 100 - (100 / (1 + rs))
    ema12 = df_close.ewm(span=12, adjust=False).mean()
    ema26 = df_close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd - out["macd_signal"]
    ret = _safe_pct_change(df_close).fillna(0)
    out["vol20"] = ret.rolling(20).std() * np.sqrt(252)
    out["ret_1d"] = _safe_pct_change(df_close, 1)
    out["ret_1w"] = _safe_pct_change(df_close, 5)
    out["ret_1m"] = _safe_pct_change(df_close, 21)
    roll_max = df_close.rolling(252).max()
    roll_min = df_close.rolling(252).min()
    out["dist_52w_high"] = (df_close / roll_max) - 1.0
    out["dist_52w_low"] = (df_close / roll_min) - 1.0
    return out

def build_signal_score(close, ind):
    last = close.tail(1); s20 = ind["sma20"].tail(1); s50 = ind["sma50"].tail(1); s200 = ind["sma200"].tail(1)
    rsi = ind["rsi14"].tail(1); macd = ind["macd"].tail(1); macd_sig = ind["macd_signal"].tail(1)
    vol20 = ind["vol20"].tail(1); ret1w = ind["ret_1w"].tail(1)
    dist_high = ind["dist_52w_high"].tail(1); dist_low = ind["dist_52w_low"].tail(1)
    scores = []
    for col in last.columns:
        sc = 50
        if (last[col].iloc[0] > s20[col].iloc[0] > s50[col].iloc[0] > s200[col].iloc[0]): sc += 15
        elif (last[col].iloc[0] < s20[col].iloc[0] < s50[col].iloc[0] < s200[col].iloc[0]): sc -= 15
        val = rsi[col].iloc[0]
        if pd.notna(val):
            if 50 <= val <= 60: sc += 5
            if 40 <= val < 50:  sc -= 5
            if val < 30:        sc += 10
            if val > 70:        sc -= 10
        m = macd[col].iloc[0]; ms = macd_sig[col].iloc[0]
        if pd.notna(m) and pd.notna(ms): sc += 8 if m > ms else -8
        if pd.notna(dist_high[col].iloc[0]) and dist_high[col].iloc[0] > -0.02: sc += 4
        if pd.notna(dist_low[col].iloc[0]) and dist_low[col].iloc[0] < 0.02:    sc += 6
        if pd.notna(ret1w[col].iloc[0]):
            if ret1w[col].iloc[0] > 0.03:  sc += 4
            if ret1w[col].iloc[0] < -0.03: sc -= 4
        if pd.notna(vol20[col].iloc[0]) and vol20[col].iloc[0] > 0.6: sc -= 5
        scores.append((col, float(np.clip(sc, 0, 100))))
    out = pd.DataFrame(scores, columns=["Ticker", "Score"]).set_index("Ticker")
    out["Sinal"] = pd.cut(out["Score"], [-0.1,30,45,55,70,100.1], labels=["VENDA FORTE","VENDA","NEUTRO","COMPRA","COMPRA FORTE"])
    return out

# =============================
# Sidebar
# =============================
st.sidebar.title("Universo de Ativos (Yahoo)")
st.sidebar.caption("Exemplos: PETR4.SA, VALE3.SA, AAPL, MSFT, BTC-USD")

default_b3 = "PETR4.SA, VALE3.SA, BBAS3.SA, ITUB4.SA, B3SA3.SA, WEGE3.SA, HGLG11.SA"
default_us = "AAPL, MSFT, NVDA, TSLA, AMZN, SPY, QQQ"
default_crypto = "BTC-USD, ETH-USD, SOL-USD"

universe_choice = st.sidebar.selectbox("Base inicial", ["Custom","B3 (exemplo)","EUA (exemplo)","Cripto (exemplo)"], index=1)
base = default_b3 if universe_choice=="B3 (exemplo)" else default_us if universe_choice=="EUA (exemplo)" else default_crypto if universe_choice=="Cripto (exemplo)" else ""
tickers_text = st.sidebar.text_area("Tickers (separados por vírgula)", value=base, height=90)
tickers = sorted(list({t.strip() for t in tickers_text.split(",") if t.strip()}))
period = st.sidebar.selectbox("Período histórico", ["1y","2y","3y","5y","10y","max"], index=2)
interval = st.sidebar.selectbox("Intervalo", ["1d","1wk"], index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros globais")
risk_free = st.sidebar.number_input("Selic anual (proxy RF)", value=0.105, step=0.005, format="%.3f")
slippage_bp = st.sidebar.number_input("Slippage (bps por trade)", value=5, step=1)

if not tickers:
    st.info("Adicione ao menos 1 ticker para começar."); st.stop()

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Screener", "📈 Ativo & Sinais", "🧪 Backtest & Risco", "📊 Carteira"])

# -----------------------------
# TAB 1 — Screener
# -----------------------------
with tab1:
    st.header("🔎 Screener Multi-Ativos (Yahoo Finance)")
    with st.spinner("Baixando e processando cotações..."):
        raw = fetch_history(tickers, period=period, interval=interval)
        if raw.empty:
            st.error("Sem dados retornados pelo Yahoo."); st.stop()
        close, volume = split_close_volume(raw, fallback_ticker=(tickers[0] if tickers else "TICKER"))
        ind = indicators(close)
        score_df = build_signal_score(close, ind)

        st.subheader("Filtros Rápidos")
        colf1, colf2, colf3, colf4 = st.columns(4)
        rsi_min = colf1.slider("RSI mín", 0, 100, 0)
        rsi_max = colf2.slider("RSI máx", 0, 100, 100)
        mom_min = colf3.slider("Retorno 1M mín (%)", -50, 50, -50)
        vol_max = colf4.slider("Volatilidade 20d máx (%, anual.)", 5, 200, 200)

        last_row = close.tail(1).T.rename(columns=lambda x: "Último")
        rsi_row = ind["rsi14"].tail(1).T.rename(columns=lambda x: "RSI14")
        ret1m_row = (ind["ret_1m"].tail(1).T * 100).rename(columns=lambda x: "Ret_1M_%")
        vol_row = (ind["vol20"].tail(1).T * 100).rename(columns=lambda x: "Vol20_%")
        dist_high_row = (ind["dist_52w_high"].tail(1).T * 100).rename(columns=lambda x: "Dist_52w_High_%")
        dist_low_row = (ind["dist_52w_low"].tail(1).T * 100).rename(columns=lambda x: "Dist_52w_Low_%")
        vol_mean = volume.tail(20).mean().rename("Vol_Médio_20")

        table = (score_df.join(last_row, how="left")
                        .join(rsi_row, how="left")
                        .join(ret1m_row, how="left")
                        .join(vol_row, how="left")
                        .join(dist_high_row, how="left")
                        .join(dist_low_row, how="left")
                        .join(vol_mean, how="left")
                        .sort_values("Score", ascending=False))

        mask = (
            table["RSI14"].between(rsi_min, rsi_max, inclusive="both") &
            (table["Ret_1M_%"].fillna(-1e9) >= mom_min) &
            (table["Vol20_%"].fillna(1e9) <= vol_max)
        )
        table_f = table[mask].copy()

        st.caption("Dica: clique nas colunas para ordenar.")
        st.dataframe(
            table_f.style.format({"Último":"{:.2f}","RSI14":"{:.1f}","Ret_1M_%":"{:.1f}","Vol20_%":"{:.1f}",
                                  "Dist_52w_High_%":"{:.1f}","Dist_52w_Low_%":"{:.1f}","Vol_Médio_20":"{:,.0f}"}),
            use_container_width=True, height=480
        )

        st.subheader("Ranking (Score → 0–100)")
        st.plotly_chart(px.bar(table_f.reset_index(), x="Ticker", y="Score", color="Sinal", height=420), use_container_width=True)
        st.markdown("**Legenda:** 0–30 (Venda forte) · 30–45 (Venda) · 45–55 (Neutro) · 55–70 (Compra) · 70–100 (Compra forte)")

# -----------------------------
# TAB 2 — Ativo & Sinais (corrigido)
# -----------------------------
with tab2:
    st.header("📈 Análise do Ativo — Gráfico + Indicadores + Regras")
    sel = st.selectbox("Escolha um ativo", tickers, index=0)

    with st.spinner("Baixando OHLCV completo do ativo…"):
        df = yf.download(sel, period=period, interval=interval, auto_adjust=False, progress=False, threads=True)

    if df.empty:
        st.warning("Sem dados para o ticker selecionado."); st.stop()

    # Extrai séries para OHLC (com fallback) e Close para cálculos
    o = pick_field_series(df, "Open")
    h = pick_field_series(df, "High")
    l = pick_field_series(df, "Low")
    c_plot = pick_field_series(df, "Close") or pick_field_series(df, "Adj Close")
    px = pick_close_series(df)  # usado nos indicadores

    # Se algum OHLC faltar, não quebra: plota linha do close
    if any(s is None or s.dropna().empty for s in [o, h, l, c_plot]):
        st.warning("OHLC incompleto para Candlestick. Mostrando linha do fechamento.")
        fig = px.line(px.rename("Close"))
    else:
        tmp = pd.DataFrame({"Open":o, "High":h, "Low":l, "Close":c_plot}).dropna()
        # ATR usa OHLC coerente
        df_ind = tmp.copy()
        df_ind["ATR14"] = compute_atr14(df_ind["Open"], df_ind["High"], df_ind["Low"], df_ind["Close"])
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=tmp.index, open=tmp["Open"], high=tmp["High"], low=tmp["Low"], close=tmp["Close"], name="Preço"))
        fig.update_layout(height=520, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        # métricas usam px e ATR da série coerente se existir
        atr_last = float(df_ind["ATR14"].iloc[-1]) if "ATR14" in df_ind.columns else float("nan")
        price_last = float(px.iloc[-1])

        # Indicadores sobre px
        sma20 = px.rolling(20).mean(); sma50 = px.rolling(50).mean(); sma200 = px.rolling(200).mean()
        ema12 = px.ewm(span=12, adjust=False).mean(); ema26 = px.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26; macd_sig = macd.ewm(span=9, adjust=False).mean()
        delta = px.diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
        rsi = 100 - 100 / (1 + (up.rolling(14).mean() / down.rolling(14).mean()))

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Preço", f"{price_last:.2f}")
        colB.metric("ATR14", f"{atr_last:.2f}" if np.isfinite(atr_last) else "-")
        colC.metric("RSI14", f"{float(rsi.iloc[-1]):.1f}" if not np.isnan(rsi.iloc[-1]) else "-")
        colD.metric("MACD-σ", f"{float(macd.iloc[-1]-macd_sig.iloc[-1]):.3f}")

        st.subheader("Score & Sinal (mesma régua do Screener)")
        tmp_close = px.to_frame(sel)
        ind_one = indicators(tmp_close)
        score_one = build_signal_score(tmp_close, ind_one)
        st.dataframe(score_one, use_container_width=True)

# -----------------------------
# TAB 3 — Backtest & Risco
# -----------------------------
with tab3:
    st.header("🧪 Backtest Rápido + Gestão de Risco (ATR/Stop/TP)")
    sel2 = st.selectbox("Ativo para backtest", tickers, index=min(1, len(tickers)-1))
    entry_logic = st.selectbox("Lógica de entrada", ["Cruzamento SMA20>50", "MACD > Sinal", "RSI<30 (reversão)"], index=0)
    exit_logic = st.selectbox("Saída", ["Stop ATR & Take Profit R", "Cruzamento SMA20<50", "MACD < Sinal", "RSI>70"], index=0)
    atr_mult = st.slider("Stop ATR (x)", 0.5, 6.0, 2.0, 0.5)
    take_R = st.slider("Take Profit (R múltiplos)", 0.5, 6.0, 2.0, 0.5)
    fixed_risk = st.number_input("Risco fixo por trade (% do capital)", 0.5, 10.0, 1.0, 0.5)
    slippage = (slippage_bp / 10000.0)

    with st.spinner("Preparando dados para backtest…"):
        df2 = yf.download(sel2, period=period, interval=interval, auto_adjust=False, progress=False, threads=True)
        if df2.empty:
            st.warning("Sem dados para backtest.")
        else:
            o = pick_field_series(df2, "Open"); h = pick_field_series(df2, "High")
            l = pick_field_series(df2, "Low"); c = pick_close_series(df2)
            if any(s is None or s.dropna().empty for s in [o,h,l,c]):
                st.warning("Dados insuficientes p/ backtest."); st.stop()
            df2 = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c}).dropna()

            df2["SMA20"] = df2["Close"].rolling(20).mean()
            df2["SMA50"] = df2["Close"].rolling(50).mean()
            ema12 = df2["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df2["Close"].ewm(span=26, adjust=False).mean()
            df2["MACD"] = ema12 - ema26
            df2["MACD_SIG"] = df2["MACD"].ewm(span=9, adjust=False).mean()
            delta = df2["Close"].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
            df2["RSI"] = 100 - 100 / (1 + (up.rolling(14).mean() / down.rolling(14).mean()))
            df2["ATR"] = compute_atr14(df2["Open"], df2["High"], df2["Low"], df2["Close"])

            if entry_logic == "Cruzamento SMA20>50":
                df2["ENTRY"] = (df2["SMA20"] > df2["SMA50"]) & (df2["SMA20"].shift(1) <= df2["SMA50"].shift(1))
            elif entry_logic == "MACD > Sinal":
                df2["ENTRY"] = (df2["MACD"] > df2["MACD_SIG"]) & (df2["MACD"].shift(1) <= df2["MACD_SIG"].shift(1))
            else:
                df2["ENTRY"] = (df2["RSI"] < 30) & (df2["RSI"].shift(1) >= 30)

            if exit_logic == "Cruzamento SMA20<50":
                df2["EXIT"] = (df2["SMA20"] < df2["SMA50"]) & (df2["SMA20"].shift(1) >= df2["SMA50"].shift(1))
            elif exit_logic == "MACD < Sinal":
                df2["EXIT"] = (df2["MACD"] < df2["MACD_SIG"]) & (df2["MACD"].shift(1) >= df2["MACD_SIG"].shift(1))
            elif exit_logic == "RSI>70":
                df2["EXIT"] = (df2["RSI"] > 70) & (df2["RSI"].shift(1) <= 70)
            else:
                df2["EXIT"] = False

            trades = []; in_pos = False; capital = 100000.0
            for i in range(1, len(df2)):
                row = df2.iloc[i]
                if (not in_pos) and row["ENTRY"]:
                    stop = row["Close"] - atr_mult * row["ATR"]
                    risk_per_share = max(row["Close"] - stop, 1e-6)
                    qty = max(int((capital*(fixed_risk/100.0))/risk_per_share), 1)
                    entry_px = row["Close"] * (1 + slippage)
                    stop_px = stop; tp_px = entry_px + take_R * (entry_px - stop_px)
                    in_pos = True
                    trades.append({"open_time": df2.index[i], "open_px": entry_px, "qty": qty,
                                   "stop_px": stop_px, "tp_px": tp_px, "close_time": None, "close_px": None, "ret": None})
                    continue
                if in_pos:
                    low = row["Low"] * (1 - slippage); high = row["High"] * (1 - slippage)
                    exit_px = None
                    if low <= trades[-1]["stop_px"]:
                        exit_px = trades[-1]["stop_px"]
                    elif high >= trades[-1]["tp_px"]:
                        exit_px = trades[-1]["tp_px"]
                    elif row["EXIT"]:
                        exit_px = row["Close"] * (1 - slippage)
                    if exit_px is not None:
                        trades[-1]["close_time"] = df2.index[i]; trades[-1]["close_px"] = exit_px
                        pnl = (exit_px - trades[-1]["open_px"]) * trades[-1]["qty"]
                        trades[-1]["ret"] = pnl / capital; capital *= (1 + trades[-1]["ret"]); in_pos = False

            if trades and trades[-1]["close_time"] is None:
                last_px = df2["Close"].iloc[-1] * (1 - slippage)
                trades[-1]["close_time"] = df2.index[-1]; trades[-1]["close_px"] = last_px
                pnl = (last_px - trades[-1]["open_px"]) * trades[-1]["qty"]
                trades[-1]["ret"] = pnl / capital

            trdf = pd.DataFrame(trades)
            if trdf.empty:
                st.warning("Nenhuma operação gerada."); 
            else:
                st.subheader("Resumo dos Trades")
                st.dataframe(trdf[["open_time","open_px","close_time","close_px","ret"]]
                             .style.format({"open_px":"{:.2f}","close_px":"{:.2f}","ret":"{:.2%}"}),
                             use_container_width=True)
                st.caption(f"Capital final simulado (teórico): **R$ {capital:,.2f}**".replace(",", "X").replace(".", ",").replace("X","."))

                curve = pd.Series(100000.0, index=df2.index, dtype=float); cap = 100000.0
                for _, tr in trdf.iterrows():
                    if pd.notna(tr["close_time"]):
                        cap *= (1 + tr["ret"]); curve.loc[tr["close_time"]:] = cap
                st.plotly_chart(px.line(curve.rename("Equity Curve"), title="Curva de Capital (aprox.)"), use_container_width=True)

# -----------------------------
# TAB 4 — Carteira
# -----------------------------
with tab4:
    st.header("📊 Carteira — Markowitz (Ledoit-Wolf) + Simulação")
    with st.spinner("Coletando dados…"):
        raw2 = fetch_history(tickers, period=period, interval="1d")
        if raw2.empty:
            st.error("Sem dados para os tickers."); st.stop()
        close2, _ = split_close_volume(raw2, fallback_ticker=(tickers[0] if tickers else "TICKER"))
        close2 = close2.dropna(how="all"); rets = close2.pct_change().dropna(how="all")

    mu = rets.mean() * 252
    lw = LedoitWolf().fit(rets.fillna(0.0))
    cov = pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)

    n = len(rets.columns)
    sims = st.slider("Número de simulações de carteira", 200, 5000, 1500, 100)
    Ws = np.random.dirichlet(np.ones(n), size=sims)
    port_mu = Ws @ mu.values
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", Ws, cov.values*252, Ws))
    sharpe = (port_mu - risk_free) / np.where(port_vol==0, np.nan, port_vol)
    st.plotly_chart(px.scatter(pd.DataFrame({"Retorno":port_mu,"Vol":port_vol,"Sharpe":sharpe}),
                               x="Vol", y="Retorno", color="Sharpe", title="Fronteira (simulada)", height=520),
                    use_container_width=True)

    best_idx = np.nanargmax(sharpe); best_w = Ws[best_idx]
    weights = pd.Series(best_w, index=rets.columns).sort_values(ascending=False)
    st.subheader("Pesos sugeridos (máx. Sharpe simulado)")
    st.dataframe((weights*100).to_frame("Peso_%").style.format("{:.2f}"), use_container_width=True)

    buff = io.StringIO(); (weights*100).to_csv(buff, header=["Peso_%"])
    st.download_button("Baixar pesos em CSV", data=buff.getvalue(), file_name="pesos_portfolio.csv", mime="text/csv")

# -----------------------------
# EXTRA — Previsões
# -----------------------------
st.markdown("---")
with st.expander("🔮 Previsões curtas & Monte Carlo (opcional)"):
    sel3 = st.selectbox("Ativo", tickers, key="pred_sel")
    horizon = st.slider("Horizonte (dias úteis)", 5, 120, 30, 5)
    model = st.selectbox("Modelo", ["AR(1) em retornos", "GBM (Monte Carlo)"], index=0)
    paths = st.slider("Caminhos MC", 100, 3000, 500, 100)

    with st.spinner("Calculando…"):
        dfx = yf.download(sel3, period="3y", interval="1d", progress=False)
        if dfx.empty:
            st.warning("Sem dados para previsão.")
        else:
            px0 = float((pick_close_series(dfx) or dfx["Close"]).iloc[-1])
            if model == "AR(1) em retornos":
                r = pick_close_series(dfx).pct_change().dropna()
                try:
                    ar = AutoReg(r, lags=1, old_names=False).fit()
                    mu = ar.params.get("const", r.mean()); phi = ar.params.get("r.L1", 0.0); eps = r.std()
                except Exception:
                    mu, phi, eps = r.mean(), 0.0, r.std()
                sim = np.zeros((horizon, paths))
                sim[0] = mu + phi * r.iloc[-1] + np.random.normal(0, eps, size=paths)
                for t in range(1, horizon):
                    sim[t] = mu + phi * sim[t-1] + np.random.normal(0, eps, size=paths)
                price_paths = px0 * (1 + sim).cumprod(axis=0)
                idx_future = pd.bdate_range(dfx.index[-1], periods=horizon, inclusive="right")
            else:
                r = pick_close_series(dfx).pct_change().dropna()
                mu = r.mean(); sigma = r.std(); dt = 1/252
                Z = np.random.normal(size=(horizon, paths))
                price_paths = np.zeros((horizon+1, paths)); price_paths[0] = px0
                for t in range(1, horizon+1):
                    price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t-1])
                idx_future = pd.bdate_range(dfx.index[-1], periods=horizon+1, inclusive="right")

            df_paths = pd.DataFrame(price_paths, index=idx_future[:price_paths.shape[0]])
            st.plotly_chart(px.line(df_paths.iloc[:, :min(50, paths)], title=f"Simulações — {sel3}"), use_container_width=True)
            q = np.nanpercentile(price_paths[-1], [5,25,50,75,95])
            st.write(pd.DataFrame({"P5":[q[0]],"P25":[q[1]],"P50":[q[2]],"P75":[q[3]],"P95":[q[4]]}).style.format("{:.2f}"))

st.success("Aba 'Ativo & Sinais' corrigida: métricas agora usam valores escalares e caem para 'Adj Close' quando necessário.")
