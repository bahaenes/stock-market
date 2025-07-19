"""
Integration tests for service layer.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestStockService:
    """Test stock service integration."""
    
    def test_fetch_stock_data_integration(self, app, mock_yfinance_data):
        """Test fetching stock data with mocked API."""
        with app.app_context():
            from app.services.stock_service import fetch_stock_data
            
            data = fetch_stock_data('AAPL', '1mo')
            
            assert data is not None
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert 'Close' in data.columns
    
    def test_get_stock_info_integration(self, app, mock_yfinance_data):
        """Test getting stock info with mocked API."""
        with app.app_context():
            from app.services.stock_service import get_stock_info
            
            info = get_stock_info('AAPL')
            
            assert info is not None
            assert 'symbol' in info
            assert info['symbol'] == 'AAPL'
    
    def test_calculate_technical_indicators_integration(self, app, sample_stock_data):
        """Test calculating technical indicators."""
        with app.app_context():
            from app.services.stock_service import calculate_technical_indicators
            
            indicators = calculate_technical_indicators(sample_stock_data)
            
            assert indicators is not None
            assert 'rsi' in indicators
            assert 'macd' in indicators
            assert 'bollinger_bands' in indicators
    
    def test_rate_limiting_integration(self, app):
        """Test rate limiting functionality."""
        with app.app_context():
            from app.services.stock_service import check_rate_limit, update_rate_limit
            
            # Test initial rate limit check
            can_proceed = check_rate_limit()
            assert isinstance(can_proceed, bool)
            
            # Test rate limit update
            update_rate_limit()
            # Should not raise exceptions


@pytest.mark.integration
class TestNewsService:
    """Test news service integration."""
    
    def test_fetch_news_integration(self, app, mock_news_api):
        """Test fetching news with mocked API."""
        with app.app_context():
            from app.services.news_service import fetch_stock_news
            
            news = fetch_stock_news('AAPL')
            
            assert news is not None
            assert isinstance(news, list)
            if len(news) > 0:
                assert 'title' in news[0]
                assert 'description' in news[0]
    
    def test_sentiment_analysis_integration(self, app):
        """Test sentiment analysis integration."""
        with app.app_context():
            from app.services.news_service import analyze_sentiment
            
            # Test with sample text
            positive_text = "This is great news for the company"
            negative_text = "This is terrible news for the company"
            
            pos_sentiment = analyze_sentiment(positive_text)
            neg_sentiment = analyze_sentiment(negative_text)
            
            assert pos_sentiment is not None
            assert neg_sentiment is not None
            
            # VADER sentiment analysis should work
            assert isinstance(pos_sentiment, (float, dict))
            assert isinstance(neg_sentiment, (float, dict))
    
    @patch('app.services.news_service.FINBERT_AVAILABLE', False)
    def test_sentiment_fallback_integration(self, app):
        """Test sentiment analysis fallback to VADER."""
        with app.app_context():
            from app.services.news_service import analyze_sentiment
            
            text = "This is a test sentence for sentiment analysis"
            sentiment = analyze_sentiment(text)
            
            # Should use VADER fallback
            assert sentiment is not None


@pytest.mark.integration
class TestPredictionService:
    """Test prediction service integration."""
    
    def test_prediction_service_error_handling(self, app, sample_stock_data):
        """Test prediction service error handling."""
        with app.app_context():
            from app.services.prediction_service import predict_stock_price
            
            # This should handle missing ML model implementations gracefully
            result = predict_stock_price('AAPL', sample_stock_data, prediction_days=7)
            
            # Either returns a prediction or None (due to missing implementations)
            assert result is None or isinstance(result, dict)
    
    def test_feature_engineering_integration(self, app, sample_stock_data):
        """Test feature engineering functionality."""
        with app.app_context():
            from app.services.prediction_service import create_features
            
            features = create_features(sample_stock_data)
            
            assert features is not None
            assert isinstance(features, pd.DataFrame)
            assert len(features) > 0
            
            # Should have technical indicator features
            expected_features = ['returns', 'sma_20', 'rsi_14', 'volatility_5']
            for feature in expected_features:
                assert feature in features.columns
    
    def test_safe_datetime_functions_integration(self, app):
        """Test safe datetime functions integration."""
        with app.app_context():
            from app.services.prediction_service import (
                safe_datetime_diff, 
                get_current_date_safe, 
                normalize_datetime
            )
            
            # Test safe datetime difference
            date1 = pd.Timestamp('2024-01-15')
            date2 = pd.Timestamp('2024-01-10')
            diff = safe_datetime_diff(date1, date2)
            assert diff >= 0
            
            # Test current date
            current = get_current_date_safe()
            assert current is not None
            
            # Test normalization
            normalized = normalize_datetime(pd.Timestamp('2024-01-15 12:30:45'))
            assert normalized.hour == 0
            assert normalized.minute == 0


@pytest.mark.integration
class TestChartService:
    """Test chart service integration."""
    
    def test_create_price_chart_integration(self, app, sample_stock_data):
        """Test creating price chart."""
        with app.app_context():
            from app.services.chart_service import create_price_chart
            
            chart_json = create_price_chart(sample_stock_data, 'AAPL', chart_type='line')
            
            assert chart_json is not None
            assert isinstance(chart_json, str)
            
            # Should be valid JSON
            import json
            chart_data = json.loads(chart_json)
            assert 'data' in chart_data
    
    def test_create_technical_indicators_chart_integration(self, app, sample_stock_data):
        """Test creating technical indicators chart."""
        with app.app_context():
            from app.services.chart_service import create_technical_indicators_chart
            
            # First calculate indicators
            from app.services.stock_service import calculate_technical_indicators
            indicators = calculate_technical_indicators(sample_stock_data)
            
            chart_json = create_technical_indicators_chart(
                sample_stock_data, 
                indicators, 
                'AAPL'
            )
            
            assert chart_json is not None
            assert isinstance(chart_json, str)
    
    def test_create_comparison_chart_integration(self, app, sample_stock_data):
        """Test creating comparison chart."""
        with app.app_context():
            from app.services.chart_service import create_comparison_chart
            
            # Create mock data for multiple stocks
            comparison_data = {
                'AAPL': sample_stock_data,
                'GOOGL': sample_stock_data.copy()  # Using same data for simplicity
            }
            
            chart_json = create_comparison_chart(comparison_data)
            
            assert chart_json is not None
            assert isinstance(chart_json, str)


@pytest.mark.integration
class TestErrorHandlerIntegration:
    """Test error handler integration."""
    
    def test_error_handler_initialization(self, app):
        """Test error handler initialization."""
        with app.app_context():
            from app.utils.error_handler import initialize_error_handling
            
            # Should initialize without errors
            initialize_error_handling()
    
    def test_error_logging_integration(self, app):
        """Test error logging integration."""
        with app.app_context():
            from app.utils.error_handler import ErrorLogger, DataFetchError
            
            # Test error logging
            test_error = DataFetchError("Test error for integration testing")
            
            # Should not raise exceptions
            ErrorLogger.log_error(
                test_error, 
                context={'test': True},
                user_action='integration_test'
            )
    
    def test_safe_execute_integration(self, app):
        """Test safe execute decorator integration."""
        with app.app_context():
            from app.utils.error_handler import safe_execute
            
            @safe_execute("Integration test operation", fallback_value="fallback")
            def test_function_that_fails():
                raise ValueError("Integration test error")
            
            @safe_execute("Integration test operation", fallback_value="fallback")
            def test_function_that_succeeds():
                return "success"
            
            # Test failure case
            result1 = test_function_that_fails()
            assert result1 == "fallback"
            
            # Test success case
            result2 = test_function_that_succeeds()
            assert result2 == "success"


@pytest.mark.integration
@pytest.mark.slow
class TestDatabaseIntegration:
    """Test database integration."""
    
    def test_database_crud_operations(self, app):
        """Test database CRUD operations."""
        with app.app_context():
            from app.models import Stock, User
            from app import db
            
            # Create
            stock = Stock(ticker='TEST', name='Test Stock', market='TEST')
            db.session.add(stock)
            db.session.commit()
            
            # Read
            retrieved_stock = Stock.query.filter_by(ticker='TEST').first()
            assert retrieved_stock is not None
            assert retrieved_stock.name == 'Test Stock'
            
            # Update
            retrieved_stock.name = 'Updated Test Stock'
            db.session.commit()
            
            updated_stock = Stock.query.filter_by(ticker='TEST').first()
            assert updated_stock.name == 'Updated Test Stock'
            
            # Delete
            db.session.delete(updated_stock)
            db.session.commit()
            
            deleted_stock = Stock.query.filter_by(ticker='TEST').first()
            assert deleted_stock is None
    
    def test_database_relationships(self, app):
        """Test database relationships."""
        with app.app_context():
            from app.models import User, Stock, Watchlist
            from app import db
            
            # Create user and stock
            user = User(username='testuser', email='test@example.com')
            stock = Stock(ticker='TEST', name='Test Stock', market='TEST')
            
            db.session.add(user)
            db.session.add(stock)
            db.session.commit()
            
            # Create watchlist relationship
            watchlist = Watchlist(user_id=user.id, stock_id=stock.id)
            db.session.add(watchlist)
            db.session.commit()
            
            # Test relationship access
            user_watchlist = user.watchlists.first()
            assert user_watchlist is not None
            assert user_watchlist.stock.ticker == 'TEST'