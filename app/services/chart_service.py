import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def create_stock_chart(stock_data, indicators, prediction_data=None, chart_type='line', ticker=''):
    """Ana hisse senedi grafiği oluştur."""
    try:
        if stock_data is None or stock_data.empty:
            logger.warning("Grafik için veri bulunamadı")
            return None
        
        # Multi-index sütun kontrolü
        close_col = 'Close'
        if isinstance(stock_data.columns, pd.MultiIndex):
            close_cols = [col for col in stock_data.columns if col[0] == 'Close']
            if close_cols:
                close_col = close_cols[0]
        
        # Subplots oluştur
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(
                f'{ticker} Fiyat Grafiği ({chart_type.capitalize()})',
                'RSI (14)',
                'MACD',
                'Hacim'
            )
        )
        
        # Ana fiyat grafiği (Row 1)
        if chart_type == 'candlestick' and all(col in stock_data.columns or (isinstance(stock_data.columns, pd.MultiIndex) and any(c[0] == col for c in stock_data.columns)) for col in ['Open', 'High', 'Low', 'Close']):
            # Candlestick için sütun adlarını düzelt
            open_col = 'Open'
            high_col = 'High'
            low_col = 'Low'
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                for col_name in ['Open', 'High', 'Low']:
                    matching_cols = [col for col in stock_data.columns if col[0] == col_name]
                    if matching_cols:
                        if col_name == 'Open':
                            open_col = matching_cols[0]
                        elif col_name == 'High':
                            high_col = matching_cols[0]
                        elif col_name == 'Low':
                            low_col = matching_cols[0]
            
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data[open_col],
                    high=stock_data[high_col],
                    low=stock_data[low_col],
                    close=stock_data[close_col],
                    name='Mum Grafiği',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
        else:
            # Line chart
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data[close_col],
                    mode='lines',
                    name='Kapanış Fiyatı',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
        
        # Hareketli ortalamalar ekle
        if indicators and 'SMA_20' in indicators:
            sma_20 = indicators['SMA_20']
            if sma_20 is not None and not sma_20.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sma_20.index,
                        y=sma_20,
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#ff7f0e', width=1.5, dash='dot')
                    ),
                    row=1, col=1
                )
        
        if indicators and 'SMA_50' in indicators:
            sma_50 = indicators['SMA_50']
            if sma_50 is not None and not sma_50.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sma_50.index,
                        y=sma_50,
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='#2ca02c', width=1.5, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands ekle
        if indicators and all(key in indicators for key in ['BB_High', 'BB_Low']):
            bb_high = indicators['BB_High']
            bb_low = indicators['BB_Low']
            
            if bb_high is not None and bb_low is not None and not bb_high.empty and not bb_low.empty:
                # Bollinger band alanı
                fig.add_trace(
                    go.Scatter(
                        x=bb_high.index.tolist() + bb_low.index.tolist()[::-1],
                        y=bb_high.tolist() + bb_low.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(52, 152, 219, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name='Bollinger Bands'
                    ),
                    row=1, col=1
                )
                
                # BB çizgileri
                fig.add_trace(
                    go.Scatter(
                        x=bb_high.index,
                        y=bb_high,
                        mode='lines',
                        name='BB Üst',
                        line=dict(color='rgba(52, 152, 219, 0.7)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=bb_low.index,
                        y=bb_low,
                        mode='lines',
                        name='BB Alt',
                        line=dict(color='rgba(52, 152, 219, 0.7)', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Tahmin verilerini ekle
        if prediction_data is not None and 'predictions' in prediction_data:
            predictions = prediction_data['predictions']
            if not predictions.empty:
                fig.add_trace(
                    go.Scatter(
                        x=predictions.index,
                        y=predictions['predicted_price'],
                        mode='lines',
                        name=f'Tahmin (Güven: %{prediction_data.get("confidence", 0)*100:.0f})',
                        line=dict(color='#e74c3c', width=2, dash='dashdot')
                    ),
                    row=1, col=1
                )
        
        # RSI grafiği (Row 2)
        if indicators and 'RSI' in indicators:
            rsi = indicators['RSI']
            if rsi is not None and not rsi.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rsi.index,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color='#9467bd', width=1.5)
                    ),
                    row=2, col=1
                )
                
                # RSI seviye çizgileri
                fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
                fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor='rgba(255,0,0,0.1)', row=2, col=1)
                fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor='rgba(0,255,0,0.1)', row=2, col=1)
        
        # MACD grafiği (Row 3)
        if indicators and all(key in indicators for key in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            macd = indicators['MACD']
            macd_signal = indicators['MACD_Signal']
            macd_hist = indicators['MACD_Hist']
            
            if all(series is not None and not series.empty for series in [macd, macd_signal, macd_hist]):
                # MACD çizgileri
                fig.add_trace(
                    go.Scatter(
                        x=macd.index,
                        y=macd,
                        mode='lines',
                        name='MACD',
                        line=dict(color='#1f77b4', width=1.5)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=macd_signal.index,
                        y=macd_signal,
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='#ff7f0e', width=1.5)
                    ),
                    row=3, col=1
                )
                
                # MACD histogram
                colors = ['#26a69a' if val >= 0 else '#ef5350' for val in macd_hist]
                fig.add_trace(
                    go.Bar(
                        x=macd_hist.index,
                        y=macd_hist,
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=3, col=1
                )
        
        # Hacim grafiği (Row 4)
        volume_col = 'Volume'
        if isinstance(stock_data.columns, pd.MultiIndex):
            volume_cols = [col for col in stock_data.columns if col[0] == 'Volume']
            if volume_cols:
                volume_col = volume_cols[0]
        
        if volume_col in stock_data.columns:
            volume_data = stock_data[volume_col]
            if not volume_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=volume_data.index,
                        y=volume_data,
                        name='Hacim',
                        marker_color='rgba(158, 158, 158, 0.6)',
                        yaxis='y4'
                    ),
                    row=4, col=1
                )
        
        # Layout düzenlemeleri
        fig.update_layout(
            title=dict(
                text=f'{ticker} Detaylı Teknik Analiz',
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E5E5E5",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=11, color="#333333"),
            margin=dict(l=50, r=50, t=100, b=50),
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # X ve Y eksenleri düzenlemeleri
        for i in range(1, 5):
            fig.update_xaxes(
                showgrid=True,
                gridcolor='#E5E5E5',
                tickformat='%Y-%m-%d',
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor='#E5E5E5',
                row=i, col=1
            )
        
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Grafik oluşturma hatası: {e}")
        return None

def create_comparison_chart(stocks_data, period='1y'):
    """Birden fazla hisse senedini karşılaştır."""
    try:
        if not stocks_data:
            return None
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (ticker, data) in enumerate(stocks_data.items()):
            if data is not None and not data.empty:
                # Close sütununu bul
                close_col = 'Close'
                if isinstance(data.columns, pd.MultiIndex):
                    close_cols = [col for col in data.columns if col[0] == 'Close']
                    if close_cols:
                        close_col = close_cols[0]
                
                # Normalize et (başlangıç değerine göre)
                close_series = data[close_col].dropna()
                if not close_series.empty:
                    normalized = (close_series / close_series.iloc[0] - 1) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=normalized.index,
                            y=normalized,
                            mode='lines',
                            name=ticker,
                            line=dict(color=colors[i % len(colors)], width=2)
                        )
                    )
        
        fig.update_layout(
            title='Hisse Senedi Karşılaştırması (Normalize Edilmiş %)',
            xaxis_title='Tarih',
            yaxis_title='Değişim (%)',
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
        
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Karşılaştırma grafiği hatası: {e}")
        return None

def create_portfolio_chart(portfolio_data):
    """Portföy dağılımı pasta grafiği."""
    try:
        if not portfolio_data:
            return None
        
        # Portföy değerlerini hesapla
        tickers = []
        values = []
        
        for item in portfolio_data:
            ticker = item.get('ticker', 'Unknown')
            quantity = item.get('quantity', 0)
            current_price = item.get('current_price', 0)
            value = quantity * current_price
            
            tickers.append(ticker)
            values.append(value)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=tickers,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Portföy Dağılımı',
            height=500,
            showlegend=True,
            paper_bgcolor='white'
        )
        
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Portföy grafiği hatası: {e}")
        return None

def create_sentiment_chart(sentiment_data, days=30):
    """Duyarlılık trendi grafiği."""
    try:
        if not sentiment_data:
            return None
        
        # Günlük duyarlılık verilerini grupla
        df = pd.DataFrame(sentiment_data)
        if df.empty:
            return None
        
        df['date'] = pd.to_datetime(df['published_at']).dt.date
        daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment = daily_sentiment.tail(days)
        
        fig = go.Figure()
        
        # Duyarlılık çizgisi
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['sentiment_score'],
                mode='lines+markers',
                name='Günlük Ortalama Duyarlılık',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            )
        )
        
        # Pozitif/negatif bölgeler
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig.add_hline(y=0.1, line_dash="dot", line_color="green", line_width=1)
        fig.add_hline(y=-0.1, line_dash="dot", line_color="red", line_width=1)
        
        fig.update_layout(
            title='Haber Duyarlılığı Trendi',
            xaxis_title='Tarih',
            yaxis_title='Duyarlılık Skoru',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(range=[-1, 1])
        )
        
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Duyarlılık grafiği hatası: {e}")
        return None

def create_performance_chart(performance_data):
    """Performans karşılaştırma grafiği."""
    try:
        if not performance_data:
            return None
        
        periods = list(performance_data.keys())
        values = list(performance_data.values())
        
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=periods,
                y=values,
                marker_color=colors,
                text=[f'{v:+.1f}%' for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Performans Özeti',
            xaxis_title='Periyot',
            yaxis_title='Getiri (%)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        # Sıfır çizgisi ekle
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Performans grafiği hatası: {e}")
        return None 