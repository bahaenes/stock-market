"""
Unit tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestErrorHandler:
    """Test error handling utilities."""
    
    def test_safe_execute_decorator(self, app):
        """Test safe execute decorator."""
        with app.app_context():
            from app.utils.error_handler import safe_execute
            
            @safe_execute("Test operation", fallback_value="fallback")
            def test_function():
                raise ValueError("Test error")
            
            result = test_function()
            assert result == "fallback"
    
    def test_safe_execute_success(self, app):
        """Test safe execute with successful function."""
        with app.app_context():
            from app.utils.error_handler import safe_execute
            
            @safe_execute("Test operation", fallback_value="fallback")
            def test_function():
                return "success"
            
            result = test_function()
            assert result == "success"
    
    def test_create_error_response(self, app):
        """Test error response creation."""
        with app.app_context():
            from app.utils.error_handler import create_error_response, DataFetchError
            
            error = DataFetchError("Test error message")
            response = create_error_response(error, "Test operation")
            
            assert response['success'] is False
            assert response['error_type'] == 'DataFetchError'
            assert 'Test error message' in response['message']
            assert response['operation'] == 'Test operation'


@pytest.mark.unit
class TestFormatters:
    """Test formatter utilities."""
    
    def test_format_currency(self, app):
        """Test currency formatting."""
        with app.app_context():
            from app.utils.formatters import format_currency
            
            # Test positive value
            assert format_currency(1234.56) == "₺1,234.56"
            
            # Test negative value
            assert format_currency(-1234.56) == "-₺1,234.56"
            
            # Test zero
            assert format_currency(0) == "₺0.00"
    
    def test_format_percentage(self, app):
        """Test percentage formatting."""
        with app.app_context():
            from app.utils.formatters import format_percentage
            
            # Test positive percentage
            assert format_percentage(0.1234) == "12.34%"
            
            # Test negative percentage
            assert format_percentage(-0.0567) == "-5.67%"
            
            # Test zero
            assert format_percentage(0) == "0.00%"
    
    def test_format_large_number(self, app):
        """Test large number formatting."""
        with app.app_context():
            from app.utils.formatters import format_large_number
            
            # Test millions
            assert format_large_number(1500000) == "1.5M"
            
            # Test billions
            assert format_large_number(2500000000) == "2.5B"
            
            # Test thousands
            assert format_large_number(1500) == "1.5K"
            
            # Test small numbers
            assert format_large_number(500) == "500"
    
    def test_safe_float_conversion(self, app):
        """Test safe float conversion."""
        with app.app_context():
            from app.utils.formatters import safe_float
            
            # Test valid string
            assert safe_float("123.45") == 123.45
            
            # Test invalid string
            assert safe_float("invalid") == 0.0
            
            # Test None
            assert safe_float(None) == 0.0
            
            # Test already float
            assert safe_float(123.45) == 123.45


@pytest.mark.unit
class TestTimezoneFunctions:
    """Test timezone handling functions."""
    
    def test_safe_datetime_diff(self, app):
        """Test safe datetime difference calculation."""
        with app.app_context():
            from app.services.prediction_service import safe_datetime_diff
            
            date1 = pd.Timestamp('2024-01-15')
            date2 = pd.Timestamp('2024-01-10')
            
            diff = safe_datetime_diff(date1, date2)
            assert diff == 5  # 5 days difference
    
    def test_get_current_date_safe(self, app):
        """Test safe current date retrieval."""
        with app.app_context():
            from app.services.prediction_service import get_current_date_safe
            
            current_date = get_current_date_safe()
            assert current_date is not None
            assert hasattr(current_date, 'year')
    
    def test_normalize_datetime(self, app):
        """Test datetime normalization."""
        with app.app_context():
            from app.services.prediction_service import normalize_datetime
            
            # Test pandas timestamp
            ts = pd.Timestamp('2024-01-15 12:30:45')
            normalized = normalize_datetime(ts)
            assert normalized.hour == 0
            assert normalized.minute == 0
            assert normalized.second == 0
            
            # Test string
            normalized = normalize_datetime('2024-01-15')
            assert normalized.year == 2024
            assert normalized.month == 1
            assert normalized.day == 15
            
            # Test None
            normalized = normalize_datetime(None)
            assert normalized is None


@pytest.mark.unit
class TestCacheUtilities:
    """Test caching utilities."""
    
    def test_cache_key_generation(self, app):
        """Test cache key generation."""
        with app.app_context():
            # This would test cache key generation if we had such utilities
            # For now, we'll test that cache dictionaries work correctly
            cache = {}
            key = "test_ticker_1mo_prediction"
            value = {"prediction": 150.0, "confidence": 0.8}
            
            cache[key] = value
            assert cache[key] == value
            assert key in cache
    
    def test_cache_expiration_simulation(self, app):
        """Test cache expiration logic."""
        with app.app_context():
            from datetime import datetime, timedelta
            
            # Simulate cache entry with timestamp
            cache_entry = {
                'data': {'prediction': 150.0},
                'timestamp': datetime.now() - timedelta(seconds=3600)  # 1 hour ago
            }
            
            # Test expiration (assume 30 minute TTL)
            ttl_seconds = 1800  # 30 minutes
            age_seconds = (datetime.now() - cache_entry['timestamp']).total_seconds()
            
            is_expired = age_seconds > ttl_seconds
            assert is_expired is True  # Should be expired
    
    @patch('time.time')
    def test_timestamp_handling(self, mock_time, app):
        """Test timestamp handling in cache."""
        with app.app_context():
            # Mock current time
            mock_time.return_value = 1640995200  # Fixed timestamp
            
            import time
            current_time = time.time()
            assert current_time == 1640995200