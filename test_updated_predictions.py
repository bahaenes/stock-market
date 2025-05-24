#!/usr/bin/env python3
"""
Güncellenmiş tahmin modellerini test et
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Import path ekle
sys.path.append(os.path.dirname(__file__))

def test_date_calculations():
    print("📅 Tarih Hesaplama Testi\n")
    
    # Test verileri
    last_date = pd.Timestamp('2024-01-19')  # Cuma
    print(f"Son veri tarihi: {last_date.strftime('%Y-%m-%d (%A)')}")
    
    # İş günleri hesaplama testi
    future_business_days = []
    current_date = last_date + timedelta(days=1)
    
    # Hafta sonunu atla
    while current_date.weekday() >= 5:
        current_date += timedelta(days=1)
        print(f"Hafta sonu atlandı: {current_date.strftime('%Y-%m-%d (%A)')}")
    
    # 7 iş günü ekle
    prediction_days = 7
    while len(future_business_days) < prediction_days:
        if current_date.weekday() < 5:  # Pazartesi=0, Cuma=4
            future_business_days.append(current_date)
            print(f"✅ İş günü eklendi: {current_date.strftime('%Y-%m-%d (%A)')}")
        else:
            print(f"⏭️  Hafta sonu atlandı: {current_date.strftime('%Y-%m-%d (%A)')}")
        current_date += timedelta(days=1)
    
    print(f"\n📊 Toplam {len(future_business_days)} iş günü tahmin edilecek")
    print(f"🎯 İlk tahmin günü: {future_business_days[0].strftime('%Y-%m-%d (%A)')}")
    print(f"🏁 Son tahmin günü: {future_business_days[-1].strftime('%Y-%m-%d (%A)')}")
    
    # Pandas bdate_range ile karşılaştır
    pd_business_days = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=prediction_days,
        freq='B'
    )
    
    print(f"\n🔄 Pandas bdate_range ile karşılaştırma:")
    for i, (manual, pandas_date) in enumerate(zip(future_business_days, pd_business_days)):
        match = "✅" if manual.date() == pandas_date.date() else "❌"
        print(f"{match} Gün {i+1}: {manual.strftime('%Y-%m-%d')} vs {pandas_date.strftime('%Y-%m-%d')}")

def test_current_date():
    print("\n🕐 Güncel Tarih Testi\n")
    
    current_date = pd.Timestamp.now(tz='UTC').normalize()
    print(f"Şu anki tarih (UTC): {current_date.strftime('%Y-%m-%d (%A)')}")
    
    # Test verileri için farklı tarihler
    test_dates = [
        pd.Timestamp('2024-01-18'),  # Perşembe (bugün simülasyonu)
        pd.Timestamp('2024-01-15'),  # Pazartesi (3 gün önce)
        pd.Timestamp('2024-01-10'),  # Çarşamba (8 gün önce)
    ]
    
    for test_date in test_dates:
        data_age = (current_date - test_date).days
        status = "🟢" if data_age <= 3 else "🟡" if data_age <= 7 else "🔴"
        print(f"{status} Test veri tarihi: {test_date.strftime('%Y-%m-%d (%A)')} - {data_age} gün eski")

if __name__ == "__main__":
    print("🚀 Güncellenmiş Tahmin Sistemi Test\n")
    test_date_calculations()
    test_current_date()
    print("\n✅ Test Tamamlandı!") 