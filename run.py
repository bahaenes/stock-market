#!/usr/bin/env python3
"""
Finans Analiz Aracı - Ana Uygulama
"""

import os
from app import create_app, db
from app.models import Stock, User, Portfolio, Watchlist, Analysis, Alert
from app.services.news_service import initialize_finbert

# Flask uygulamasını oluştur
app = create_app(os.getenv('FLASK_CONFIG', 'development'))

@app.shell_context_processor
def make_shell_context():
    """Flask shell için context ekle."""
    return {
        'db': db,
        'Stock': Stock,
        'User': User,
        'Portfolio': Portfolio,
        'Watchlist': Watchlist,
        'Analysis': Analysis,
        'Alert': Alert
    }

# before_first_request deprecated olduğu için kaldırıldı# FinBERT initialization main başlangıçta yapılacak

if __name__ == '__main__':
    with app.app_context():
        # Veritabanı tablolarını oluştur
        db.create_all()
        
        # FinBERT modelini başlat
        initialize_finbert()
        
        # Varsayılan hisse senetlerini ekle
        from app.main.routes import DEFAULT_STOCKS
        
        for stock_data in DEFAULT_STOCKS:
            existing_stock = Stock.query.filter_by(ticker=stock_data['ticker']).first()
            if not existing_stock:
                stock = Stock(
                    ticker=stock_data['ticker'],
                    name=stock_data['name'],
                    market=stock_data['market']
                )
                db.session.add(stock)
        
        try:
            db.session.commit()
            print("Varsayılan hisse senetleri veritabanına eklendi.")
        except Exception as e:
            db.session.rollback()
            print(f"Hisse senetleri eklenirken hata: {e}")
    
    # Uygulamayı başlat
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug) 