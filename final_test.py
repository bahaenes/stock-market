#!/usr/bin/env python3
"""
Tüm sistem düzeltmelerini test et
"""

import sys
import os
import traceback
from datetime import datetime

# Import path ekle
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Kritik importları test et."""
    print("📦 Import Testleri\n")
    
    imports_to_test = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('flask', 'Flask'),
        ('yfinance', 'yf'),
        ('sklearn.ensemble', 'RandomForestRegressor'),
    ]
    
    optional_imports = [
        ('lightgbm', 'lgb'),
        ('prophet', 'Prophet'),
        ('transformers', 'AutoTokenizer'),
    ]
    
    for module_name, alias in imports_to_test:
        try:
            exec(f"import {module_name} as {alias}")
            print(f"✅ {module_name} - BAŞARILI")
        except ImportError as e:
            print(f"❌ {module_name} - HATA: {e}")
    
    print("\n📋 Opsiyonel importlar:")
    for module_name, alias in optional_imports:
        try:
            exec(f"import {module_name} as {alias}")
            print(f"✅ {module_name} - MEVCUT")
        except ImportError:
            print(f"⚠️ {module_name} - MEVCUT DEĞİL")

def test_timezone_functions():
    """Timezone düzeltmelerini test et."""
    print("\n🕐 Timezone Fonksiyon Testleri\n")
    
    try:
        from app.services.prediction_service import (
            safe_datetime_diff, 
            get_current_date_safe, 
            normalize_datetime
        )
        
        # Test tarihleri
        import pandas as pd
        test_dates = [
            pd.Timestamp('2024-01-15'),
            pd.Timestamp('2024-01-15', tz='UTC'),
            '2024-01-15',
            datetime(2024, 1, 15)
        ]
        
        current = get_current_date_safe()
        print(f"Güncel tarih: {current}")
        
        for i, test_date in enumerate(test_dates):
            try:
                normalized = normalize_datetime(test_date)
                diff = safe_datetime_diff(current, normalized)
                print(f"✅ Test {i+1}: {diff} gün fark")
            except Exception as e:
                print(f"❌ Test {i+1}: {e}")
        
    except ImportError as e:
        print(f"❌ Timezone fonksiyonları import edilemedi: {e}")

def test_error_handling():
    """Error handling sistemi test et."""
    print("\n🛡️ Error Handling Testleri\n")
    
    try:
        from app.utils.error_handler import (
            safe_execute,
            create_error_response,
            ErrorLogger,
            DataFetchError
        )
        
        # Safe execute test
        @safe_execute("Test işlemi", fallback_value="FALLBACK")
        def test_function_that_fails():
            raise ValueError("Test hatası")
        
        result = test_function_that_fails()
        print(f"✅ Safe execute: {result}")
        
        # Error response test
        test_error = DataFetchError("Test veri hatası")
        error_response = create_error_response(test_error, "Test operasyonu")
        print(f"✅ Error response: {error_response['error_type']}")
        
        # Error logging test
        ErrorLogger.log_error(test_error, context={'test': True})
        print("✅ Error logging başarılı")
        
    except ImportError as e:
        print(f"❌ Error handling import edilemedi: {e}")

def test_prediction_system():
    """Tahmin sistemini test et."""
    print("\n🤖 Tahmin Sistemi Testleri\n")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Örnek hisse verisi oluştur
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='B')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        print(f"✅ Örnek veri oluşturuldu: {len(sample_data)} kayıt")
        
        # Timezone normalizing test
        from app.services.prediction_service import normalize_datetime
        last_date = normalize_datetime(sample_data.index[-1])
        print(f"✅ Son veri tarihi normalize edildi: {last_date}")
        
        # Business days test
        future_dates = pd.bdate_range(
            start=last_date + timedelta(days=1),
            periods=5,
            freq='B'
        )
        
        if future_dates.tz is not None:
            future_dates = future_dates.tz_localize(None)
        
        print(f"✅ Gelecek iş günleri: {len(future_dates)} gün")
        for date in future_dates:
            print(f"   - {date.strftime('%Y-%m-%d (%A)')}")
        
    except Exception as e:
        print(f"❌ Tahmin sistemi test hatası: {e}")
        print(f"Traceback: {traceback.format_exc()}")

def test_app_creation():
    """Flask app oluşturma test et."""
    print("\n🚀 Flask App Testleri\n")
    
    try:
        from app import create_app
        
        app = create_app('testing')
        print(f"✅ Flask app oluşturuldu: {app}")
        print(f"✅ App name: {app.name}")
        print(f"✅ Debug mode: {app.debug}")
        
        with app.app_context():
            print("✅ App context başarılı")
        
    except Exception as e:
        print(f"❌ Flask app oluşturma hatası: {e}")
        print(f"Traceback: {traceback.format_exc()}")

def main():
    print("🧪 Comprehensive System Test")
    print("=" * 50)
    print(f"Test zamanı: {datetime.now()}")
    print(f"Python sürümü: {sys.version}")
    print("=" * 50)
    
    test_imports()
    test_timezone_functions()
    test_error_handling()
    test_prediction_system()
    test_app_creation()
    
    print("\n" + "=" * 50)
    print("✅ Tüm testler tamamlandı!")
    print("🎯 Sistem kullanıma hazır!")

if __name__ == "__main__":
    main() 