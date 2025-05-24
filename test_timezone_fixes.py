#!/usr/bin/env python3
"""
Timezone düzeltmelerini test et
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Import path ekle
sys.path.append(os.path.dirname(__file__))

def test_timezone_functions():
    print("🕐 Timezone Test Fonksiyonları\n")
    
    # Test için örnek tarihler
    test_cases = [
        # Timezone-naive tarihler
        pd.Timestamp('2024-01-15'),
        datetime(2024, 1, 15),
        '2024-01-15',
        
        # Timezone-aware tarihler
        pd.Timestamp('2024-01-15', tz='UTC'),
        pd.Timestamp('2024-01-15', tz='US/Eastern'),
    ]
    
    # Test fonksiyonları import et
    try:
        from app.services.prediction_service import safe_datetime_diff, get_current_date_safe, normalize_datetime
        
        print("📅 normalize_datetime() testi:")
        for i, test_date in enumerate(test_cases):
            try:
                normalized = normalize_datetime(test_date)
                print(f"✅ Test {i+1}: {type(test_date).__name__} -> {normalized}")
            except Exception as e:
                print(f"❌ Test {i+1} HATA: {e}")
        
        print("\n📊 safe_datetime_diff() testi:")
        current = get_current_date_safe()
        print(f"Şu anki tarih: {current}")
        
        for i, test_date in enumerate(test_cases):
            try:
                diff = safe_datetime_diff(current, test_date)
                print(f"✅ Fark {i+1}: {diff} gün")
            except Exception as e:
                print(f"❌ Fark {i+1} HATA: {e}")
        
        print("\n🎯 get_current_date_safe() testi:")
        try:
            safe_current = get_current_date_safe()
            print(f"✅ Güvenli tarih: {safe_current} (type: {type(safe_current)})")
        except Exception as e:
            print(f"❌ Güvenli tarih HATA: {e}")
            
    except ImportError as e:
        print(f"❌ Import hatası: {e}")

def test_pandas_operations():
    print("\n📊 Pandas Timezone Operations\n")
    
    # Farklı timezone durumları test et
    test_data = {
        'naive_timestamp': pd.Timestamp('2024-01-15'),
        'utc_timestamp': pd.Timestamp('2024-01-15', tz='UTC'),
        'localized_timestamp': pd.Timestamp('2024-01-15').tz_localize('UTC'),
    }
    
    for name, ts in test_data.items():
        print(f"📋 {name}:")
        print(f"   Değer: {ts}")
        print(f"   Timezone: {ts.tz}")
        print(f"   Normalize: {ts.normalize()}")
        
        # Timezone kaldırma testi
        if ts.tz is not None:
            naive = ts.tz_localize(None)
            print(f"   Timezone kaldırılmış: {naive} (tz: {naive.tz})")
        print()

def test_business_days():
    print("📈 İş Günleri Hesaplama Testi\n")
    
    # Farklı başlangıç günleri
    test_dates = [
        pd.Timestamp('2024-01-15'),  # Pazartesi
        pd.Timestamp('2024-01-19'),  # Cuma
        pd.Timestamp('2024-01-20'),  # Cumartesi
        pd.Timestamp('2024-01-21'),  # Pazar
    ]
    
    for start_date in test_dates:
        print(f"📅 Başlangıç: {start_date.strftime('%Y-%m-%d (%A)')}")
        
        try:
            # Business days hesapla
            business_days = pd.bdate_range(
                start=start_date + timedelta(days=1),
                periods=5,
                freq='B'
            )
            
            # Timezone temizle
            if business_days.tz is not None:
                business_days = business_days.tz_localize(None)
            
            print("   İş günleri:")
            for i, bd in enumerate(business_days):
                print(f"   {i+1}. {bd.strftime('%Y-%m-%d (%A)')}")
                
        except Exception as e:
            print(f"   ❌ HATA: {e}")
        print()

if __name__ == "__main__":
    print("🚀 Timezone Düzeltme Testleri\n")
    test_timezone_functions()
    test_pandas_operations()
    test_business_days()
    print("✅ Testler Tamamlandı!") 