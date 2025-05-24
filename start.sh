#!/bin/bash

echo "============================================="
echo "    Finans Analiz Aracı Başlatılıyor..."
echo "============================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Sanal ortam bulunamadı, oluşturuluyor..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Sanal ortam aktifleştiriliyor..."
source venv/bin/activate

# Check if requirements are installed
if ! pip freeze | grep -q "Flask"; then
    echo "Bağımlılıklar yükleniyor..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo
    echo "UYARI: .env dosyası bulunamadı!"
    echo "Lütfen .env dosyasını oluşturun ve gerekli ayarları yapın."
    echo "Örnek içerik:"
    echo "FLASK_ENV=development"
    echo "SECRET_KEY=your-secret-key"
    echo "NEWS_API_KEY=your-news-api-key"
    echo
    read -p "Devam etmek için Enter'a basın..."
fi

echo
echo "Uygulama başlatılıyor..."
echo "Tarayıcınızda http://localhost:5000 adresine gidin"
echo

python run.py 