from flask import jsonify, request, current_app
from app.api import bp
from app.services import stock_service, news_service, prediction_service
from app.models import Stock, Analysis
from app import db
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@bp.route('/stocks', methods=['GET'])
def get_stocks():
    """Tüm hisse senetlerini listele."""
    try:
        stocks = stock_service.get_stock_list()
        return jsonify({
            'success': True,
            'data': stocks,
            'count': len(stocks)
        })
    except Exception as e:
        logger.error(f"Hisse listesi API hatası: {e}")
        return jsonify({
            'success': False,
            'error': 'Hisse listesi alınamadı'
        }), 500

@bp.route('/stocks/search', methods=['GET'])
def search_stocks():
    """Hisse senedi ara."""
    try:
        query = request.args.get('q', '').strip()
        if len(query) < 2:
            return jsonify({
                'success': True,
                'data': [],
                'message': 'En az 2 karakter girin'
            })
        
        results = stock_service.search_stocks(query)
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Hisse arama API hatası: {e}")
        return jsonify({
            'success': False,
            'error': 'Arama yapılamadı'
        }), 500

@bp.route('/stocks/<ticker>/data', methods=['GET'])
def get_stock_data(ticker):
    """Hisse senedi verilerini al."""
    try:
        period = request.args.get('period', '1y')
        
        # Hisse verisini çek
        stock_data = stock_service.get_stock_data(ticker.upper(), period)
        if stock_data is None or stock_data.empty:
            return jsonify({
                'success': False,
                'error': 'Veri bulunamadı'
            }), 404
        
        # JSON serializable hale getir
        data_dict = {
            'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
            'open': stock_data['Open'].tolist() if 'Open' in stock_data.columns else [],
            'high': stock_data['High'].tolist() if 'High' in stock_data.columns else [],
            'low': stock_data['Low'].tolist() if 'Low' in stock_data.columns else [],
            'close': stock_data['Close'].tolist() if 'Close' in stock_data.columns else [],
            'volume': stock_data['Volume'].tolist() if 'Volume' in stock_data.columns else []
        }
        
        return jsonify({
            'success': True,
            'data': data_dict,
            'ticker': ticker.upper(),
            'period': period
        })
        
    except Exception as e:
        logger.error(f"Hisse verisi API hatası ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': 'Veri alınamadı'
        }), 500

@bp.route('/stocks/<ticker>/indicators', methods=['GET'])
def get_technical_indicators(ticker):
    """Teknik göstergeleri al."""
    try:
        period = request.args.get('period', '1y')
        
        # Hisse verisini çek
        stock_data = stock_service.get_stock_data(ticker.upper(), period)
        if stock_data is None or stock_data.empty:
            return jsonify({
                'success': False,
                'error': 'Veri bulunamadı'
            }), 404
        
        # Teknik göstergeleri hesapla
        indicators = stock_service.calculate_technical_indicators(stock_data)
        if indicators is None:
            return jsonify({
                'success': False,
                'error': 'Teknik analiz yapılamadı'
            }), 500
        
        # JSON serializable hale getir
        indicators_dict = {}
        for key, series in indicators.items():
            if series is not None and not series.empty:
                indicators_dict[key] = {
                    'current': float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else None,
                    'values': series.dropna().tolist()[-50:],  # Son 50 değer
                    'dates': series.dropna().index.strftime('%Y-%m-%d').tolist()[-50:]
                }
        
        return jsonify({
            'success': True,
            'data': indicators_dict,
            'ticker': ticker.upper(),
            'period': period
        })
        
    except Exception as e:
        logger.error(f"Teknik göstergeler API hatası ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': 'Teknik analiz yapılamadı'
        }), 500

@bp.route('/stocks/<ticker>/news', methods=['GET'])
def get_stock_news(ticker):
    """Hisse senedi haberlerini al."""
    try:
        days_back = int(request.args.get('days', 7))
        days_back = min(max(days_back, 1), 30)  # 1-30 arası sınırla
        
        # Hisse bilgisini al
        stock = Stock.query.filter_by(ticker=ticker.upper()).first()
        if not stock:
            stock_name = ticker.upper()
        else:
            stock_name = stock.name
        
        # Haber analizi yap
        news_analysis = news_service.get_stock_news_analysis(
            stock_name, ticker.upper(), days_back=days_back
        )
        
        return jsonify({
            'success': True,
            'data': {
                'articles': news_analysis['articles'],
                'sentiment': {
                    'average': news_analysis['average_sentiment'],
                    'distribution': news_analysis['sentiment_distribution'],
                    'total_count': news_analysis['total_count']
                }
            },
            'ticker': ticker.upper(),
            'days_analyzed': days_back
        })
        
    except Exception as e:
        logger.error(f"Haber API hatası ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': 'Haber analizi yapılamadı'
        }), 500

@bp.route('/stocks/<ticker>/prediction', methods=['GET'])
def get_price_prediction(ticker):
    """Fiyat tahmini al."""
    try:
        period = request.args.get('period', '1y')
        future_days = int(request.args.get('future_days', 30))
        future_days = min(max(future_days, 7), 90)  # 7-90 arası sınırla
        
        # Hisse verisini çek
        stock_data = stock_service.get_stock_data(ticker.upper(), period)
        if stock_data is None or stock_data.empty:
            return jsonify({
                'success': False,
                'error': 'Veri bulunamadı'
            }), 404
        
        # Close serisini al
        close_col = 'Close'
        if isinstance(stock_data.columns, pd.MultiIndex):
            close_cols = [col for col in stock_data.columns if col[0] == 'Close']
            if close_cols:
                close_col = close_cols[0]
        
        close_series = pd.to_numeric(stock_data[close_col], errors='coerce').dropna()
        
        # Tahmin yap
        prediction_result = prediction_service.get_price_prediction(
            ticker.upper(), close_series, period, future_periods=future_days
        )
        
        if prediction_result is None:
            return jsonify({
                'success': False,
                'error': 'Tahmin yapılamadı'
            }), 500
        
        # JSON serializable hale getir
        predictions = prediction_result['predictions']
        result_data = {
            'predictions': {
                'dates': predictions.index.strftime('%Y-%m-%d').tolist(),
                'prices': predictions['predicted_price'].tolist()
            },
            'confidence': prediction_result['confidence'],
            'model_count': prediction_result['model_count'],
            'last_actual_price': prediction_result['last_actual_price'],
            'horizon_days': prediction_result['prediction_horizon_days']
        }
        
        return jsonify({
            'success': True,
            'data': result_data,
            'ticker': ticker.upper(),
            'period': period
        })
        
    except Exception as e:
        logger.error(f"Tahmin API hatası ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': 'Tahmin yapılamadı'
        }), 500

@bp.route('/market/sentiment', methods=['GET'])
def get_market_sentiment():
    """Genel piyasa duyarlılığını al."""
    try:
        market_sentiment = news_service.get_market_sentiment()
        
        return jsonify({
            'success': True,
            'data': market_sentiment
        })
        
    except Exception as e:
        logger.error(f"Piyasa duyarlılığı API hatası: {e}")
        return jsonify({
            'success': False,
            'error': 'Piyasa duyarlılığı alınamadı'
        }), 500

@bp.route('/analyses', methods=['GET'])
def get_analyses():
    """Son analizleri listele."""
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(max(limit, 1), 100)  # 1-100 arası sınırla
        
        analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(limit).all()
        
        results = []
        for analysis in analyses:
            results.append({
                'id': analysis.id,
                'stock_ticker': analysis.stock.ticker,
                'stock_name': analysis.stock.name,
                'period': analysis.period,
                'chart_type': analysis.chart_type,
                'current_price': analysis.current_price,
                'price_change': analysis.price_change,
                'rsi': analysis.rsi,
                'news_sentiment': analysis.news_sentiment,
                'predicted_price': analysis.predicted_price,
                'created_at': analysis.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Analizler API hatası: {e}")
        return jsonify({
            'success': False,
            'error': 'Analizler alınamadı'
        }), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """API sağlık kontrolü."""
    return jsonify({
        'success': True,
        'message': 'API çalışıyor',
        'version': '2.0'
    })

@bp.errorhandler(404)
def not_found(error):
    """404 hatası."""
    return jsonify({
        'success': False,
        'error': 'Endpoint bulunamadı'
    }), 404

@bp.errorhandler(500)
def internal_error(error):
    """500 hatası."""
    return jsonify({
        'success': False,
        'error': 'Sunucu hatası'
    }), 500 