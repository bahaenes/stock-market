@echo off
echo.
echo =============================================
echo     Finans Analiz Aracı Başlatılıyor...
echo =============================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Sanal ortam bulunamadı, oluşturuluyor...
    python -m venv venv
)

REM Activate virtual environment
echo Sanal ortam aktifleştiriliyor...
call venv\Scripts\activate

REM Check if requirements are installed
pip freeze | findstr "Flask" >nul
if errorlevel 1 (
    echo Bağımlılıklar yükleniyor...
    pip install -r requirements.txt
)

REM Check for .env file
if not exist ".env" (
    echo.
    echo UYARI: .env dosyası bulunamadı!
    echo Lütfen .env dosyasını oluşturun ve gerekli ayarları yapın.
    echo Örnek içerik:
    echo FLASK_ENV=development
    echo SECRET_KEY=your-secret-key
    echo NEWS_API_KEY=your-news-api-key
    echo.
    pause
)

echo.
echo Uygulama başlatılıyor...
echo Tarayıcınızda http://localhost:5000 adresine gidin
echo.

python run.py

pause 