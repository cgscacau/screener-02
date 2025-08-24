import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import ta
from scipy import stats
import requests

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sistema Financeiro Completo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert > div {
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

class FinancialAnalyzer:
    def __init__(self):
        self.data = None
        self.ticker = None
    
    @st.cache_data(ttl=1800)  # Cache por 30 minutos
    def get_stock_data(_self, ticker, period="1y"):
        """Obtém dados da ação do Yahoo Finance com cache"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Erro ao obter dados para {ticker}: {e}")
            return None, None
    
    @st.cache_data(ttl=3600)  # Cache por 1 hora
    def get_multiple_tickers(_self, tickers, period="6mo"):
        """Obtém dados de múltiplos tickers"""
        try:
            data = yf.download(tickers, period=period, group_by='ticker', progress=False)
            return data
        except Exception as e:
            st.error(f"Erro ao obter dados múltiplos: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calcula indicadores técnicos usando a biblioteca ta"""
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        # Médias Móveis
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
        
        # Volatilidade
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        return df
    
    def predict_prices(self, data, days_ahead=30):
        """Sistema de previsão com múltiplos modelos"""
        if data is None or len(data) < 100:
            return None, None, None
        
        # Preparar features
        df = self.calculate_technical_indicators(data)
        
        # Features para o modelo
        feature_columns = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 
                          'SMA_20', 'SMA_50', 'EMA_12', 'ATR']
        
        # Remover NaN e preparar dados
        df_clean = df[feature_columns + ['Close']].dropna()
        
        if len(df_clean) < 50:
            return None, None, None
        
        X = df_clean[feature_columns]
        y = df_clean['Close']
        
        # Split dos dados (80% treino, 20% teste)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Linear Regression': LinearRegression()
        }
        
        predictions = {}
        metrics = {}
        test_predictions = {}
        
        for name, model in models.items():
            # Treinamento
            model.fit(X_train_scaled, y_train)
            
            # Predições no conjunto de teste
            y_pred = model.predict(X_test_scaled)
            test_predictions[name] = y_pred
            
            # Métricas
            metrics[name] = {
                'R²': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
            
            # Predição futura
            last_features = X.iloc[-1:].values
            last_features_scaled = scaler.transform(last_features)
            future_pred = model.predict(last_features_scaled)[0]
            predictions[name] = future_pred
        
        return predictions, metrics, (y_test, test_predictions)
    
    def backtest_strategy(self, data, strategy='sma_crossover'):
        """Backtest simples de estratégias"""
        if data is None or len(data) < 100:
            return None
        
        df = self.calculate_technical_indicators(data)
        
        if strategy == 'sma_crossover':
            # Estratégia de cruzamento de médias móveis
            signals = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
            df['Position'] = signals
            df['Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
            
        elif strategy == 'rsi_oversold':
            # Estratégia RSI oversold/overbought
            signals = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, 0, np.nan))
            df['Position'] = pd.Series(signals).fillna(method='ffill').fillna(0)
            df['Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
        
        # Calcular métricas
        total_return = (1 + df['Strategy_Returns']).prod() - 1
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'strategy_data': df[['Close', 'Position', 'Returns', 'Strategy_Returns']].dropna()
        }

def create_advanced_chart(data, ticker, indicators=None):
    """Cria gráfico avançado com indicadores técnicos"""
    if indicators is None:
        indicators = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB']
    
    # Determinar número de subplots
    rows = 1
    if 'RSI' in indicators: rows += 1
    if 'MACD' in indicators: rows += 1
    if 'Volume' in indicators: rows += 1
    
    subplot_titles = ['Preço e Indicadores']
    if 'Volume' in indicators: subplot_titles.append('Volume')
    if 'RSI' in indicators: subplot_titles.append('RSI')
    if 'MACD' in indicators: subplot_titles.append('MACD')
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=[0.6] + [0.2] * (rows - 1) if rows > 1 else [1.0]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Preço',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1
    )
    
    # Médias Móveis
    if 'SMA_20' in indicators and 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], 
                      line=dict(color='#42a5f5', width=2), 
                      name='SMA 20'), row=1, col=1
        )
    
    if 'SMA_50' in indicators and 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], 
                      line=dict(color='#ab47bc', width=2), 
                      name='SMA 50'), row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB' in indicators and all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], 
                      line=dict(color='gray', dash='dash'), 
                      name='BB Superior', opacity=0.7), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], 
                      line=dict(color='gray', dash='dash'), 
                      name='BB Inferior', fill='tonexty', opacity=0.3), row=1, col=1
        )
    
    current_row = 2
    
    # Volume
    if 'Volume' in indicators:
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], 
                   name='Volume', marker_color=colors, opacity=0.7), 
            row=current_row, col=1
        )
        current_row += 1
    
    # RSI
    if 'RSI' in indicators and 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      line=dict(color='purple', width=2), 
                      name='RSI'), row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     row=current_row, col=1, opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     row=current_row, col=1, opacity=0.7)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1
    
    # MACD
    if 'MACD' in indicators and all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      line=dict(color='blue', width=2), 
                      name='MACD'), row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], 
                      line=dict(color='orange', width=2), 
                      name='Signal'), row=current_row, col=1
        )
        
        # Histograma MACD com cores
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_Histogram'], 
                   name='MACD Hist', marker_color=colors, opacity=0.6), 
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    fig.update_layout(
        title=f'Análise Técnica Completa - {ticker}',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    # Adicionar seletor de período
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(step="all", label="Tudo")
            ])
        )
    )
    
    return fig

def stock_screener():
    """Screener avançado de ações"""
    st.subheader("🔍 Screener Avançado de Ações")
    
    # Listas de tickers por categoria
    ticker_lists = {
        'S&P 500 Top 20': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 
                          'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'PFE', 'KO'],
        'Brasileiras': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 
                       'MGLU3.SA', 'WEGE3.SA', 'RENT3.SA', 'LREN3.SA', 'SUZB3.SA'],
        'Tecnologia': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL'],
        'Personalizada': []
    }
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        selected_list = st.selectbox("Selecionar Lista:", list(ticker_lists.keys()))
        
    with col2:
        if selected_list == 'Personalizada':
            custom_tickers = st.text_input("Tickers (separados por vírgula):", "AAPL,MSFT,GOOGL")
            tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
        else:
            tickers = ticker_lists[selected_list]
            st.info(f"Analisando {len(tickers)} ações da lista {selected_list}")
    
    with col3:
        analysis_period = st.selectbox("Período de Análise:", ["1mo", "3mo", "6mo", "1y"], index=2)
    
    # Filtros
    st.subheader("🎛️ Filtros")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_price = st.number_input("Preço Mínimo ($)", value=0.0, step=1.0)
        max_price = st.number_input("Preço Máximo ($)", value=1000.0, step=10.0)
    
    with col2:
        min_volume = st.number_input("Volume Mínimo", value=0, step=100000)
        min_market_cap = st.number_input("Market Cap Mínimo (B)", value=0.0, step=1.0)
    
    with col3:
        max_pe = st.number_input("P/E Máximo", value=50.0, step=5.0)
        min_rsi = st.number_input("RSI Mínimo", value=0.0, step=5.0, max_value=100.0)
        max_rsi = st.number_input("RSI Máximo", value=100.0, step=5.0, max_value=100.0)
    
    with col4:
        min_return_1m = st.number_input("Retorno 1M Mínimo (%)", value=-100.0, step=5.0)
        show_signals = st.checkbox("Mostrar Sinais de Trading", value=True)
    
    if st.button("🚀 Executar Screening", type="primary"):
        screening_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        analyzer = FinancialAnalyzer()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Analisando {ticker}... ({i+1}/{len(tickers)})")
            
            try:
                data, info = analyzer.get_stock_data(ticker, analysis_period)
                
                if data is not None and not data.empty and info:
                    # Calcular indicadores
                    data = analyzer.calculate_technical_indicators(data)
                    
                    # Métricas básicas
                    current_price = data['Close'].iloc[-1]
                    volume_avg = data['Volume'].mean()
                    
                    # Retornos
                    return_1m = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                    
                    # Indicadores técnicos
                    rsi_current = data['RSI'].iloc[-1] if 'RSI' in data.columns else None
                    
                    # Informações fundamentais
                    market_cap = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0  # Em bilhões
                    pe_ratio = info.get('trailingPE', None)
                    
                    # Aplicar filtros
                    if (min_price <= current_price <= max_price and
                        volume_avg >= min_volume and
                        market_cap >= min_market_cap and
                        (pe_ratio is None or pe_ratio <= max_pe) and
                        (rsi_current is None or (min_rsi <= rsi_current <= max_rsi)) and
                        return_1m >= min_return_1m):
                        
                        # Sinais de trading
                        signals = []
                        if show_signals and rsi_current is not None:
                            if rsi_current < 30:
                                signals.append("🟢 RSI Oversold")
                            elif rsi_current > 70:
                                signals.append("🔴 RSI Overbought")
                            
                            # Sinal de média móvel
                            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                                if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
                                    signals.append("📈 SMA Bullish")
                                else:
                                    signals.append("📉 SMA Bearish")
                        
                        screening_results.append({
                            'Ticker': ticker,
                            'Nome': info.get('shortName', 'N/A'),
                            'Preço ($)': round(current_price, 2),
                            'Retorno 1M (%)': round(return_1m, 2),
                            'Volume Médio': int(volume_avg),
                            'Market Cap (B)': round(market_cap, 2),
                            'P/E': round(pe_ratio, 2) if pe_ratio else 'N/A',
                            'RSI': round(rsi_current, 1) if rsi_current else 'N/A',
                            'Sinais': ' | '.join(signals) if signals else 'Neutro',
                            'Setor': info.get('sector', 'N/A')
                        })
                
                progress_bar.progress((i + 1) / len(tickers))
                
            except Exception as e:
                continue
        
        status_text.empty()
        progress_bar.empty()
        
        if screening_results:
            df_results = pd.DataFrame(screening_results)
            
            st.success(f"✅ Encontradas {len(df_results)} ações que atendem aos critérios!")
            
            # Ordenar por retorno
            df_results = df_results.sort_values('Retorno 1M (%)', ascending=False)
            
            # Exibir tabela
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Preço ($)": st.column_config.NumberColumn("Preço ($)", format="%.2f"),
                    "Retorno 1M (%)": st.column_config.NumberColumn("Retorno 1M (%)", format="%.2f"),
                    "Volume Médio": st.column_config.NumberColumn("Volume Médio", format="%d"),
                    "Market Cap (B)": st.column_config.NumberColumn("Market Cap (B)", format="%.2f"),
                }
            )
            
            # Gráficos de análise
            if len(df_results) > 1:
                st.subheader("📊 Análise Visual")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de dispersão Risco vs Retorno
                    fig_scatter = px.scatter(
                        df_results[df_results['RSI'] != 'N/A'], 
                        x='RSI', 
                        y='Retorno 1M (%)',
                        size='Market Cap (B)',
                        color='Setor',
                        hover_name='Ticker',
                        title='RSI vs Retorno 1M'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Gráfico de setores
                    sector_counts = df_results['Setor'].value_counts()
                    fig_pie = px.pie(
                        values=sector_counts.values, 
                        names=sector_counts.index,
                        title='Distribuição por Setor'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.warning("❌ Nenhuma ação encontrada com os critérios especificados. Tente ajustar os filtros.")

def main():
    st.markdown('<h1 class="main-header">📈 Sistema Financeiro Completo</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("🎛️ Navegação")
    
    page = st.sidebar.radio(
        "Selecione o Módulo:",
        ["🏠 Dashboard", "📊 Análise Individual", "🔮 Previsões", "🔍 Screener", "📈 Backtesting"],
        index=0
    )
    
    analyzer = FinancialAnalyzer()
    
    if page == "🏠 Dashboard":
        st.header("🏠 Dashboard Executivo")
        
        # Input para ticker principal
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            main_ticker = st.text_input("Ticker Principal:", value="AAPL", key="main_ticker").upper()
        with col2:
            dashboard_period = st.selectbox("Período:", ["3mo", "6mo", "1y", "2y"], index=2)
        with col3:
            auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            st.rerun()
        
        if main_ticker:
            with st.spinner("Carregando dashboard..."):
                data, info = analyzer.get_stock_data(main_ticker, dashboard_period)
                
                if data is not None and info:
                    data = analyzer.calculate_technical_indicators(data)
                    
                    # Métricas principais
                    current_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2]
                    price_change = current_price - prev_close
                    price_change_pct = (price_change / prev_close) * 100
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Preço Atual", f"${current_price:.2f}", 
                                f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
                    
                    with col2:
                        volume_current = data['Volume'].iloc[-1]
                        volume_avg = data['Volume'].mean()
                        volume_change = ((volume_current - volume_avg) / volume_avg) * 100
                        st.metric("Volume", f"{volume_current:,.0f}", f"{volume_change:+.1f}%")
                    
                    with col3:
                        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 0
                        st.metric("RSI", f"{rsi:.1f}", 
                                "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
                    
                    with col4:
                        market_cap = info.get('marketCap', 0) / 1e9
                        st.metric("Market Cap", f"${market_cap:.1f}B")
                    
                    with col5:
                        pe_ratio = info.get('trailingPE', 0)
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                    
                    # Gráfico principal
                    fig = create_advanced_chart(data, main_ticker, 
                                              ['SMA_20', 'SMA_50', 'BB', 'Volume', 'RSI', 'MACD'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Análise e previsão lado a lado
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Análise Técnica")
                        
                        # Sinais automáticos
                        signals = []
                        
                        if 'RSI' in data.columns:
                            rsi_val = data['RSI'].iloc[-1]
                            if rsi_val < 30:
                                signals.append("🟢 **RSI Oversold** - Possível entrada")
                            elif rsi_val > 70:
                                signals.append("🔴 **RSI Overbought** - Possível saída")
                        
                        if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
                            if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
                                signals.append("📈 **Tendência de Alta** - SMA 20 > SMA 50")
                            else:
                                signals.append("📉 **Tendência de Baixa** - SMA 20 < SMA 50")
                        
                        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                            if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
                                signals.append("🚀 **MACD Bullish** - Momentum positivo")
                            else:
                                signals.append("⚠️ **MACD Bearish** - Momentum negativo")
                        
                        for signal in signals:
                            st.markdown(signal)
                        
                        if not signals:
                            st.info("Nenhum sinal claro identificado no momento.")
                    
                    with col2:
                        st.subheader("🔮 Previsão de Preços")
                        
                        predictions, metrics, _ = analyzer.predict_prices(data)
                        
                        if predictions and metrics:
                            for model_name, pred_price in predictions.items():
                                change_pct = ((pred_price - current_price) / current_price) * 100
                                r2_score = metrics[model_name]['R²']
                                
                                confidence = "Alta" if r2_score > 0.7 else "Média" if r2_score > 0.5 else "Baixa"
                                
                                st.markdown(f"**{model_name}:**")
                                st.markdown(f"- Previsão: ${pred_price:.2f} ({change_pct:+.2f}%)")
                                st.markdown(f"- Confiança: {confidence} (R² = {r2_score:.3f})")
                                st.markdown("---")
                        else:
                            st.warning("Dados insuficientes para previsão confiável.")
    
    elif page == "📊 Análise Individual":
        st.header("📊 Análise Técnica Individual")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ticker = st.text_input("Ticker:", value="AAPL", key="analysis_ticker").upper()
        
        with col2:
            period = st.selectbox("Período:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        with col3:
            st.write("**Indicadores:**")
            show_sma = st.checkbox("SMAs", True)
            show_bb = st.checkbox("Bollinger", True)
            show_volume = st.checkbox("Volume", True)
        
        if ticker:
            with st.spinner("Carregando análise..."):
                data, info = analyzer.get_stock_data(ticker, period)
                
                if data is not None:
                    data = analyzer.calculate_technical_indicators(data)
                    
                    # Informações da empresa
                    if info:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("ℹ️ Informações")
                            st.write(f"**Nome:** {info.get('longName', 'N/A')}")
                            st.write(f"**Setor:** {info.get('sector', 'N/A')}")
                            st.write(f"**Indústria:** {info.get('industry', 'N/A')}")
                        
                        with col2:
                            st.subheader("💰 Financeiro")
                            market_cap = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0
                            st.write(f"**Market Cap:** ${market_cap:.2f}B")
                            st.write(f"**P/E:** {info.get('trailingPE', 'N/A')}")
                            div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                            st.write(f"**Div. Yield:** {div_yield:.2f}%")
                        
                        with col3:
                            st.subheader("📈 Performance")
                            high_52w = info.get('fiftyTwoWeekHigh', 0)
                            low_52w = info.get('fiftyTwoWeekLow', 0)
                            current = data['Close'].iloc[-1]
                            st.write(f"**52W High:** ${high_52w:.2f}")
                            st.write(f"**52W Low:** ${low_52w:.2f}")
                            if high_52w and low_52w:
                                pos_52w = ((current - low_52w) / (high_52w - low_52w)) * 100
                                st.write(f"**Posição 52W:** {pos_52w:.1f}%")
                    
                    # Gráfico
                    indicators = []
                    if show_sma: indicators.extend(['SMA_20', 'SMA_50'])
                    if show_bb: indicators.append('BB')
                    if show_volume: indicators.append('Volume')
                    indicators.extend(['RSI', 'MACD'])
                    
                    fig = create_advanced_chart(data, ticker, indicators)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela de dados recentes
                    st.subheader("📋 Dados Recentes")
                    recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].tail(10)
                    st.dataframe(recent_data, use_container_width=True)
    
    elif page == "🔮 Previsões":
        st.header("🔮 Sistema de Previsões")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pred_ticker = st.text_input("Ticker:", value="AAPL", key="pred_ticker").upper()
        
        with col2:
            pred_period = st.selectbox("Período Histórico:", ["6mo", "1y", "2y", "5y"], index=2)
        
        with col3:
            days_ahead = st.number_input("Dias à Frente:", min_value=1, max_value=60, value=30)
        
        if st.button("🚀 Gerar Previsões", type="primary"):
            with st.spinner("Processando modelos de previsão..."):
                data, info = analyzer.get_stock_data(pred_ticker, pred_period)
                
                if data is not None:
                    predictions, metrics, test_data = analyzer.predict_prices(data, days_ahead)
                    
                    if predictions and metrics:
                        current_price = data['Close'].iloc[-1]
                        
                        st.subheader("🎯 Resultados das Previsões")
                        
                        # Métricas dos modelos
                        col1, col2 = st.columns(2)
                        
                        for i, (model_name, pred_price) in enumerate(predictions.items()):
                            change_pct = ((pred_price - current_price) / current_price) * 100
                            model_metrics = metrics[model_name]
                            
                            with col1 if i == 0 else col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{model_name}</h3>
                                    <h2>${pred_price:.2f}</h2>
                                    <p style="color: {'green' if change_pct > 0 else 'red'};">
                                        {change_pct:+.2f}% vs preço atual
                                    </p>
                                    <p><strong>Métricas:</strong></p>
                                    <p>R²: {model_metrics['R²']:.4f}</p>
                                    <p>RMSE: {model_metrics['RMSE']:.4f}</p>
                                    <p>MAE: {model_metrics['MAE']:.4f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Gráfico de comparação dos modelos
                        if test_data:
                            y_test, test_predictions = test_data
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                y=y_test.values,
                                name='Preço Real',
                                line=dict(color='blue', width=3)
                            ))
                            
                            colors = ['red', 'green', 'orange', 'purple']
                            for i, (model_name, preds) in enumerate(test_predictions.items()):
                                fig.add_trace(go.Scatter(
                                    y=preds,
                                    name=f'Pred. {model_name}',
                                    line=dict(color=colors[i % len(colors)], dash='dash', width=2)
                                ))
                            
                            fig.update_layout(
                                title='Comparação: Previsões vs Preços Reais (Conjunto de Teste)',
                                xaxis_title='Período de Teste',
                                yaxis_title='Preço ($)',
                                height=500,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Resumo e recomendações
                        st.subheader("📋 Resumo e Análise")
                        
                        avg_prediction = np.mean(list(predictions.values()))
                        avg_change = ((avg_prediction - current_price) / current_price) * 100
                        avg_r2 = np.mean([m['R²'] for m in metrics.values()])
                        
                        confidence_level = "Alta" if avg_r2 > 0.7 else "Média" if avg_r2 > 0.5 else "Baixa"
                        
                        st.info(f"""
                        **Previsão Consenso:** ${avg_prediction:.2f} ({avg_change:+.2f}%)
                        
                        **Nível de Confiança:** {confidence_level} (R² médio: {avg_r2:.3f})
                        
                        **Interpretação:**
                        - R² > 0.7: Modelo explica bem a variação dos preços
                        - R² 0.5-0.7: Modelo moderadamente confiável  
                        - R² < 0.5: Modelo com baixa capacidade preditiva
                        
                        ⚠️ **Aviso:** Previsões são baseadas em dados históricos e não garantem resultados futuros.
                        Use apenas como ferramenta de apoio à decisão.
                        """)
                    
                    else:
                        st.error("Não foi possível gerar previsões. Dados insuficientes ou erro no modelo.")
    
    elif page == "🔍 Screener":
        stock_screener()
    
    elif page == "📈 Backtesting":
        st.header("📈 Backtesting de Estratégias")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            backtest_ticker = st.text_input("Ticker:", value="AAPL", key="backtest_ticker").upper()
        
        with col2:
            backtest_period = st.selectbox("Período:", ["6mo", "1y", "2y", "5y"], index=2)
        
        with col3:
            strategy = st.selectbox("Estratégia:", ["sma_crossover", "rsi_oversold"])
        
        if st.button("🧪 Executar Backtest", type="primary"):
            with st.spinner("Executando backtest..."):
                data, _ = analyzer.get_stock_data(backtest_ticker, backtest_period)
                
                if data is not None:
                    result = analyzer.backtest_strategy(data, strategy)
                    
                    if result:
                        strategy_return = result['total_return']
                        buy_hold_return = result['buy_hold_return']
                        strategy_data = result['strategy_data']
                        
                        # Métricas
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Retorno Estratégia", f"{strategy_return*100:.2f}%")
                        
                        with col2:
                            st.metric("Retorno Buy & Hold", f"{buy_hold_return*100:.2f}%")
                        
                        with col3:
                            excess_return = strategy_return - buy_hold_return
                            st.metric("Retorno Excesso", f"{excess_return*100:.2f}%")
                        
                        with col4:
                            sharpe = strategy_data['Strategy_Returns'].mean() / strategy_data['Strategy_Returns'].std() * np.sqrt(252)
                            st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                        
                        # Gráfico de performance
                        strategy_cumulative = (1 + strategy_data['Strategy_Returns']).cumprod()
                        buy_hold_cumulative = (1 + strategy_data['Returns']).cumprod()
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=strategy_data.index,
                            y=strategy_cumulative,
                            name='Estratégia',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=strategy_data.index,
                            y=buy_hold_cumulative,
                            name='Buy & Hold',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f'Performance: {strategy.replace("_", " ").title()} vs Buy & Hold',
                            xaxis_title='Data',
                            yaxis_title='Retorno Acumulado',
                            height=500,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Análise detalhada
                        st.subheader("📊 Análise Detalhada")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Estatísticas da Estratégia:**")
                            st.write(f"- Retorno Total: {strategy_return*100:.2f}%")
                            st.write(f"- Retorno Anualizado: {(strategy_return ** (252/len(strategy_data)) - 1)*100:.2f}%")
                            st.write(f"- Volatilidade: {strategy_data['Strategy_Returns'].std()*np.sqrt(252)*100:.2f}%")
                            st.write(f"- Máximo Drawdown: {(strategy_cumulative / strategy_cumulative.cummax() - 1).min()*100:.2f}%")
                        
                        with col2:
                            st.write("**Comparação:**")
                            win_rate = (strategy_data['Strategy_Returns'] > 0).mean()
                            st.write(f"- Taxa de Acerto: {win_rate*100:.1f}%")
                            
                            trades = (strategy_data['Position'].diff() != 0).sum()
                            st.write(f"- Número de Trades: {trades}")
                            
                            if trades > 0:
                                avg_trade = strategy_data['Strategy_Returns'].mean()
                                st.write(f"- Retorno Médio por Trade: {avg_trade*100:.3f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Sistema Financeiro Completo | Dados: Yahoo Finance | 
        ⚠️ Para fins educacionais - Não constitui recomendação de investimento</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
