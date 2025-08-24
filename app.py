# app.py — Screener Probabilístico Multi-Ativos (Yahoo Finance)
# Autor: ChatGPT (para Claudiomar) — v2.0 "Aegis"
# Execução local: streamlit run app.py
# Observação: O app cria 'ativos.json' com listas padrão se o arquivo não existir.

import os
import json
import math
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ==============================
# Configuração geral de página
# ==============================
st.set_page_config(
    page_title="Screener Probabilístico — Yahoo Finance",
    page_icon="📊",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Screener probabilístico multi-ativos com listas dinâmicas, pontuação composta e gestão de risco."
    }
)

VERSION = "2.0 • Aegis"
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

# ==============================
# Utilidades
# ==============================
@st.cache_data(show_spinner=False)
def load_lists():
    # Carrega ou cria o arquivo ativos.json com listas padrão.
    fname = "ativos.json"
    if not os.path.exists(fname):
        default_data = {
            "acoes_br": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "WEGE3.SA", "BBDC4.SA", "ABEV3.SA", "HGLG11.SA"],
            "acoes_us": ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "GOOGL"],
            "etfs": ["SPY", "QQQ", "EEM", "EWZ", "BOVA11.SA"],
            "indices": ["^BVSP", "^GSPC", "^NDX", "^DJI", "^VIX"],
            "fx": ["BRL=X", "EURUSD=X", "JPY=X"],
            "cripto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "futuros": ["GC=F", "SI=F", "CL=F", "NG=F", "ZS=F"],
            "custom": []
        }
        with open(fname, "w") as f:
            json.dump(default_data, f, indent=2)
    with open(fname, "r") as f:
        return json.load(f)

def save_lists(data: dict):
    with open("ativos.json", "w") as f:
        json.dump(data, f, indent=2)
    load_lists.clear()  # invalida o cache

@st.cache_data(show_spinner=False)
def yf_download(ticker: str, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    # Wrapper cacheado para baixar dados do Yahoo.
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

# ==============================
# Indicadores técnicos — sem TA-Lib
# ==============================
def wilder_ewm(series: pd.Series, period: int) -> pd.Series:
    alpha = 1/period
    return series.ewm(alpha=alpha, adjust=False).mean()

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_ewm(gain, period)
    avg_loss = wilder_ewm(loss, period)
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr_wilder(high, low, close, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return wilder_ewm(tr, period)

def adx(high, low, close, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    atr_s = wilder_ewm(tr, period)
    plus_di = 100 * wilder_ewm(pd.Series(plus_dm, index=high.index), period) / atr_s
    minus_di = 100 * wilder_ewm(pd.Series(minus_dm, index=high.index), period) / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = wilder_ewm(dx, period)
    return adx_val

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(series, n)
    std = series.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower

def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series / cummax - 1.0).min()
    return float(dd)

# ==============================
# Engine Probabilístico & Sinal
# ==============================
def composite_probabilities(df: pd.DataFrame, horizon: int = 10):
    # Estima prob_up, prob_down e retorno esperado para horizonte k por método empírico.
    if df.empty or len(df) < max(60, horizon + 20):
        return 0.5, 0.5, 0.0, np.nan, np.nan

    close = df["Close"]
    lookback = min(len(close), 3*252)  # ~3 anos
    c = close.iloc[-lookback:]
    kret = c.pct_change(horizon).dropna()
    slope = ema(c, 50).diff()
    regime = "up" if slope.iloc[-1] > 0 else "down"

    slope_hist = ema(c, 50).diff().dropna()
    mask = slope_hist > 0 if regime == "up" else slope_hist <= 0
    mask = mask.reindex(kret.index, method="nearest").fillna(False)
    kret_reg = kret[mask]

    sample = kret_reg if len(kret_reg) > 100 else kret
    if len(sample) < 20:
        sample = kret

    prob_up = (sample > 0).mean()
    prob_down = 1 - prob_up
    exp_ret = sample.mean()

    if len(sample) > 0:
        var95 = np.nanpercentile(sample, 5)
        cvar95 = sample[sample <= var95].mean() if np.any(sample <= var95) else var95
    else:
        var95, cvar95 = np.nan, np.nan

    return float(prob_up), float(prob_down), float(exp_ret), float(var95), float(cvar95)

def signal_from_indicators(df: pd.DataFrame):
    # Constrói um escore [0,100] combinando tendência, momentum, força e Bollinger.
    close = df["Close"]
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    ema50 = ema(close, 50)
    slope50 = (ema50 - ema50.shift(5)) / 5

    rsi = rsi_wilder(close, 14)
    adxv = adx(df["High"], df["Low"], close, 14)
    bb_u, bb_m, bb_l = bollinger(close, 20, 2)

    score = 50.0
    notes = []

    # Tendência
    if sma20.iloc[-1] > sma50.iloc[-1]:
        score += 15; notes.append("SMA20>SMA50 (tendência de alta)")
    else:
        score -= 15; notes.append("SMA20<SMA50 (tendência de baixa)")

    if slope50.iloc[-1] > 0:
        score += 10; notes.append("EMA50 com inclinação positiva")
    else:
        score -= 10; notes.append("EMA50 com inclinação negativa")

    # Momentum via RSI
    r = rsi.iloc[-1]
    if r < 30:
        score += 15; notes.append("RSI < 30 (sobrevendido)")
    elif r > 70:
        score -= 15; notes.append("RSI > 70 (sobrecomprado)")
    else:
        score += (r - 50) / 2  # de -25 a +25 máx.
        notes.append(f"RSI neutro (ajuste fino {((r-50)/2):.1f})")

    # Força da tendência
    if adxv.iloc[-1] >= 25:
        score += 10; notes.append("ADX ≥ 25 (tendência forte)")
    else:
        score -= 5; notes.append("ADX < 25 (tendência fraca)")

    # Posição em Bollinger
    last_close = close.iloc[-1]
    if not np.isnan(bb_u.iloc[-1]) and not np.isnan(bb_l.iloc[-1]):
        pos = (last_close - bb_l.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1] + 1e-9)
        score += (pos - 0.5) * 20  # -10 a +10
        notes.append(f"Posição nas Bandas: {pos:.2f}")

    score = max(0, min(100, score))

    # Classificação
    if score >= 70:
        label = "📈 COMPRA FORTE"
    elif score <= 30:
        label = "📉 VENDA FORTE"
    else:
        label = "⚖️ NEUTRO"

    return float(score), label, notes

def risk_block(df: pd.DataFrame, atr_mult: float = 2.0):
    atr14 = atr_wilder(df["High"], df["Low"], df["Close"], 14).iloc[-1]
    last = df["Close"].iloc[-1]
    stop = last - atr_mult * atr14
    target = last + atr_mult * atr14
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    mdd = max_drawdown(df["Close"])
    return {
        "ATR14": float(atr14),
        "vol_annual": float(vol) if not np.isnan(vol) else np.nan,
        "max_drawdown": float(mdd),
        "stop": float(stop),
        "target": float(target)
    }

def position_size(capital: float, risk_pct: float, entry: float, stop: float):
    risk_per_share = max(1e-9, entry - stop)
    risk_cash = capital * (risk_pct / 100.0)
    qty = int(risk_cash // risk_per_share) if risk_per_share > 0 else 0
    return max(0, qty)

def historical_hit_rate(df: pd.DataFrame, horizon: int = 10, score_thresh: float = 70.0):
    # Sinal retrospectivo simples: quando score >= thresh, verifica retorno futuro (horizon dias).
    if len(df) < horizon + 60:
        return np.nan, 0
    scores = []
    for i in range(len(df)):
        if i < 60:
            scores.append(np.nan); continue
        dsub = df.iloc[:i+1].copy()
        sc, _, _ = signal_from_indicators(dsub)
        scores.append(sc)
    s = pd.Series(scores, index=df.index)
    signals = s[s >= score_thresh].index
    if len(signals) == 0:
        return np.nan, 0
    rets = []
    for t in signals:
        idx = df.index.get_loc(t)
        if idx + horizon < len(df):
            fwd = df["Close"].iloc[idx + horizon] / df["Close"].iloc[idx] - 1
            rets.append(fwd)
    if len(rets) == 0:
        return np.nan, 0
    hit = (np.array(rets) > 0).mean()
    return float(hit), len(rets)

# ==============================
# UI — Sidebar
# ==============================
lists_data = load_lists()

st.sidebar.title("⚙️ Configurações")
with st.sidebar.expander("⏱ Janela de Dados", expanded=True):
    period = st.selectbox("Período histórico", ["6mo", "1y", "2y", "5y", "max"], index=1)
    interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)
with st.sidebar.expander("🎯 Horizonte & Risco", expanded=True):
    horizon = st.slider("Horizonte de avaliação (dias úteis)", 3, 60, 10)
    risk_pct = st.slider("Risco por trade (%)", 0.5, 5.0, 2.0, 0.5)
    atr_mult = st.slider("Multiplicador ATR para stop/target", 1.0, 4.0, 2.0, 0.5)
with st.sidebar.expander("🧪 Classificação", expanded=True):
    strong_buy = st.slider("Threshold 'Compra Forte' (score)", 60, 90, 70, 1)
    strong_sell = st.slider("Threshold 'Venda Forte' (score)", 10, 40, 30, 1)

st.sidebar.markdown("---")
st.sidebar.caption(f"Versão: {VERSION}")

# ==============================
# Tabs principais
# ==============================
st.title("📊 Screener Probabilístico Multi-Ativos — Yahoo Finance")
st.caption("Sinais compostos + Probabilidades empíricas + Gestão de risco + Backcheck simples")
tabs = st.tabs(["🔎 Screener", "🧠 Detalhe do Ativo", "🗂️ Listas & Consulta", "❓Ajuda"])

# ==============================
# TAB 1 — Screener
# ==============================
with tabs[0]:
    st.subheader("🔎 Screener por Lista")
    cat = st.selectbox("Categoria", list(lists_data.keys()), index=0, key="cat_screener")
    tickers = st.multiselect("Selecione os tickers", lists_data[cat], default=lists_data[cat][:min(10, len(lists_data[cat]))])
    run = st.button("🚀 Rodar Screener", type="primary")
    st.markdown("—")

    if run:
        if not tickers:
            st.warning("Selecione ao menos um ticker."); st.stop()
        progress = st.progress(0, text="Baixando e calculando...")
        rows = []
        for i, tk in enumerate(tickers):
            progress.progress((i+1)/len(tickers), text=f"[{i+1}/{len(tickers)}] {tk}")
            df = yf_download(tk, period=period, interval=interval)
            if df.empty:
                rows.append({"Ticker": tk, "Status": "sem dados"}); continue
            try:
                score, label, notes = signal_from_indicators(df)
                prob_up, prob_down, exp_ret, var95, cvar95 = composite_probabilities(df, horizon=horizon)
                rb = risk_block(df, atr_mult)
                price = float(df["Close"].iloc[-1])
                hit, ntrades = historical_hit_rate(df, horizon=horizon, score_thresh=strong_buy)

                # Escore composto final (0-100): 70% técnico + 30% prob_up
                final_score = 0.7*score + 30*(prob_up)  # prob_up ∈ [0,1] => 0-30 pts

                # Sinal final considerando thresholds customizados
                if score >= strong_buy and prob_up >= 0.55:
                    final_label = "📈 COMPRA FORTE"
                elif score <= strong_sell and prob_up <= 0.45:
                    final_label = "📉 VENDA FORTE"
                else:
                    final_label = "⚖️ NEUTRO"

                rows.append({
                    "Ticker": tk,
                    "Preço": round(price, 4),
                    "Score": round(score, 1),
                    "Prob↑": round(prob_up, 3),
                    "ExpRet(h)": round(exp_ret, 4),
                    "VaR95(h)": round(var95, 4),
                    "ES95(h)": round(cvar95, 4),
                    "ADX14": round(adx(df['High'], df['Low'], df['Close'], 14).iloc[-1], 2),
                    "ATR14": round(rb["ATR14"], 4),
                    "Vol(Anual)": round(rb["vol_annual"], 3) if not math.isnan(rb["vol_annual"]) else None,
                    "MDD(Hist)": round(rb["max_drawdown"], 3),
                    "HitRate(h)": round(hit, 3) if not math.isnan(hit) else None,
                    "N sinais": ntrades,
                    "Sinal": final_label,
                    "Status": "ok"
                })
            except Exception as e:
                rows.append({"Ticker": tk, "Status": f"erro: {e}"})

        progress.empty()
        if rows:
            df_out = pd.DataFrame(rows)

            # Converte colunas numéricas com segurança, mesmo se não existirem
            n = len(df_out)
            def to_num(col):
                series = df_out.get(col, pd.Series([np.nan]*n))
                return pd.to_numeric(series, errors="coerce")

            df_out["Score"] = to_num("Score")
            df_out["Prob↑"] = to_num("Prob↑")

            # Só cria a ordenação se houver ao menos um valor numérico
            if df_out["Score"].notna().any() or df_out["Prob↑"].notna().any():
                df_out["_ord"] = 0.7*df_out["Score"].fillna(0) + 30.0*df_out["Prob↑"].fillna(0)
                df_out = df_out.sort_values(by="_ord", ascending=False).drop(columns=["_ord"], errors="ignore")

            st.dataframe(df_out, use_container_width=True, hide_index=True)

            # Exporta o CSV mesmo se faltarem colunas
            st.download_button(
                "⬇️ Baixar resultados (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"screener_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Nenhuma linha foi gerada. Tente novamente com outros tickers ou período.")


# ==============================
# TAB 2 — Detalhe do Ativo
# ==============================
with tabs[1]:
    st.subheader("🧠 Análise Detalhada")
    colA, colB = st.columns([1,1])
    with colA:
        cat2 = st.selectbox("Categoria", list(lists_data.keys()), index=0, key="cat_detail")
        tk2 = st.selectbox("Ticker", lists_data[cat2], index=0, key="tk_detail")
        show_mc = st.checkbox("Executar Monte Carlo (rápido)", value=True, help="Simula probabilidade de tocar stop/target no horizonte.")
        n_sims = st.slider("N simulações", 200, 3000, 800, 50, key="n_sims_detail")
    with colB:
        capital = st.number_input("Capital (R$ / USD)", min_value=1000.0, value=100000.0, step=1000.0, key="capital_detail")
        # >>>>>>> CORREÇÃO AQUI: adicionamos key único <<<<<<<<
        risk_pct_detail = st.slider("Risco por trade (%)", 0.5, 5.0, 2.0, 0.5, key="risk_pct_detail_slider")

    df2 = yf_download(tk2, period=period, interval=interval)
    if df2.empty:
        st.warning("Sem dados para este ticker.")
    else:
        score, label, notes = signal_from_indicators(df2)
        prob_up, prob_down, exp_ret, var95, cvar95 = composite_probabilities(df2, horizon=horizon)
        rb = risk_block(df2, atr_mult)
        price = float(df2["Close"].iloc[-1])
        qty = position_size(capital, risk_pct_detail, entry=price, stop=rb["stop"])

        # Gráfico — Candlestick com SMA20 & SMA50 e bandas
        bb_u, bb_m, bb_l = bollinger(df2["Close"], 20, 2)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df2.index, open=df2["Open"], high=df2["High"],
                                     low=df2["Low"], close=df2["Close"], name="Preço"))
        fig.add_trace(go.Scatter(x=df2.index, y=sma(df2["Close"],20), name="SMA20", mode="lines"))
        fig.add_trace(go.Scatter(x=df2.index, y=sma(df2["Close"],50), name="SMA50", mode="lines"))
        fig.add_trace(go.Scatter(x=df2.index, y=bb_u, name="BBand Sup", mode="lines"))
        fig.add_trace(go.Scatter(x=df2.index, y=bb_l, name="BBand Inf", mode="lines"))
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h", y=1.1))

        col1, col2 = st.columns([2,1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Preço", f"{price:,.4f}")
            st.metric("Score", f"{score:.1f}", help=label)
            st.metric("Prob↑(h)", f"{prob_up:.2%}")
            st.metric("ExpRet(h)", f"{exp_ret:.2%}")
            st.metric("VaR95(h)", f"{var95:.2%}")
            st.metric("ES95(h)", f"{cvar95:.2%}")
            st.metric("ATR14", f"{rb['ATR14']:.4f}")
            st.metric("Vol(Anual)", f"{rb['vol_annual']:.2%}" if not math.isnan(rb['vol_annual']) else "n/a")
            st.metric("Max DD", f"{rb['max_drawdown']:.2%}")
            st.markdown(f"**Sinal:** {label}")
            st.markdown(f"**Stop (≈ {atr_mult}×ATR):** `{rb['stop']:.4f}`  \n**Target:** `{rb['target']:.4f}`")
            st.markdown(f"**Tamanho de posição (risco {risk_pct_detail:.1f}%):** `{qty}` unidades")

        st.markdown("**Notas do Sinal:**")
        for n in notes:
            st.write("•", n)

        # Monte Carlo simples (bootstrap de retornos diários)
        if show_mc:
            st.markdown("---")
            st.subheader("🎲 Monte Carlo — Prob. de tocar Stop/Target no horizonte")
            daily_ret = df2["Close"].pct_change().dropna()
            if len(daily_ret) >= max(40, horizon+5):
                stop_level = rb["stop"]
                target_level = rb["target"]
                start = price
                hit_stop = 0
                hit_target = 0
                for _ in range(n_sims):
                    path = [start]
                    for _ in range(horizon):
                        r = np.random.choice(daily_ret.values)
                        path.append(path[-1] * (1 + r))
                    path = np.array(path)
                    if path.min() <= stop_level:
                        hit_stop += 1
                    if path.max() >= target_level:
                        hit_target += 1
                p_stop = hit_stop / n_sims
                p_target = hit_target / n_sims
                colA, colB, colC = st.columns(3)
                colA.metric("Prob. Tocar Target", f"{p_target:.1%}")
                colB.metric("Prob. Tocar Stop", f"{p_stop:.1%}")
                rr = (price - stop_level) / (target_level - price) if (target_level - price) != 0 else float("nan")
                colC.metric("Risco/Retorno ATR", f"{rr:.2f}")
                st.caption("Simulação de bootstrap com retornos diários históricos.")
            else:
                st.info("Histórico insuficiente para simular.")

# ==============================
# TAB 3 — Listas & Consulta
# ==============================
with tabs[2]:
    st.subheader("🗂️ Gerenciar Listas de Ativos")
    cat3 = st.selectbox("Categoria", list(lists_data.keys()), index=0, key="cat_list")
    st.write("Atuais:", ", ".join(lists_data[cat3]) if lists_data[cat3] else "—")
    new_tk = st.text_input("Adicionar ticker (ex: HGLG11.SA, AAPL, BTC-USD)")
    colx, coly = st.columns([1,1])
    if colx.button("➕ Adicionar"):
        if new_tk and new_tk not in lists_data[cat3]:
            lists_data[cat3].append(new_tk.strip())
            save_lists(lists_data)
            st.success(f"Adicionado a {cat3}: {new_tk}")
        else:
            st.warning("Ticker vazio ou já existente.")
    rem_tk = st.selectbox("Remover ticker", lists_data[cat3] if lists_data[cat3] else ["—"])
    if coly.button("🗑️ Remover"):
        if lists_data[cat3] and rem_tk in lists_data[cat3]:
            lists_data[cat3].remove(rem_tk)
            save_lists(lists_data)
            st.warning(f"Removido de {cat3}: {rem_tk}")
        else:
            st.info("Nada para remover.")
    st.markdown("---")
    st.subheader("📥 Importar / 📤 Exportar")
    up = st.file_uploader("Importar JSON de listas (substitui o arquivo atual)", type=["json"])
    if up is not None:
        try:
            data = json.loads(up.getvalue().decode("utf-8"))
            assert isinstance(data, dict)
            save_lists(data)
            st.success("Listas importadas com sucesso. Recarregue para ver.")
        except Exception as e:
            st.error(f"Erro ao importar: {e}")

    st.download_button("⬇️ Baixar listas atuais (ativos.json)",
                       data=json.dumps(lists_data, indent=2).encode("utf-8"),
                       file_name="ativos.json", mime="application/json")

    st.markdown("---")
    st.subheader("🔎 Consulta Rápida (um ticker)")
    qtk = st.text_input("Ticker para consulta rápida (ex: URPR11.SA, TSLA, PETR4.SA, BTC-USD)")
    if st.button("Consultar"):
        if qtk.strip():
            d = yf_download(qtk.strip(), period=period, interval=interval)
            if d.empty:
                st.error("Sem dados para este ticker.")
            else:
                st.dataframe(d.tail(10))

# ==============================
# TAB 4 — Ajuda
# ==============================
with tabs[3]:
    st.subheader("❓Como usar")
    st.markdown("""
**Fluxo rápido:**
1) Na aba **Screener**, escolha uma categoria e os tickers, ajuste *período/intervalo*, *horizonte* e *limiares*, e clique **Rodar Screener**.  
2) Veja **Score**, **Prob↑**, **ExpRet(h)**, **VaR/ES**, **HitRate(h)** e **Sinal**. Baixe o CSV.
3) Na aba **Detalhe**, selecione um ticker para ver o gráfico, métricas completas, **stop/target por ATR** e **Monte Carlo**.
4) Em **Listas & Consulta**, adicione/remova ativos, importe/exporte `ativos.json` e faça consultas rápidas.

**Notas de método:**
- *Score* ∈ [0,100] combina tendência (SMA/EMA), momentum (RSI), força (ADX) e posição nas Bandas.  
- *Prob↑(h)* é empírica, baseada na distribuição de retornos k (horizon) dos últimos ~3 anos, condicionada ao regime de inclinação da EMA50.  
- *ExpRet(h)* é a média histórica desses k-retornos. *VaR95* e *ES95* são quantis/ES empíricos.  
- *HitRate(h)* verifica retrospectivamente se **scores ≥ limiar** geraram retornos positivos após *horizon* dias.  
- **Gestão de risco**: stop/target sugeridos por ATR(14)×multiplicador; posição dimensionada por risco fixo do capital.

> **Aviso**: Modelos empíricos não garantem resultados futuros. Combine com análise fundamental e liquidez/estratégia.
""")
    st.info("Dica: use a categoria 'custom' para montar listas personalizadas.")
