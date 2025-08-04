@echo off
echo ========================================
echo 🚀 Lity AI - Complete Startup Script
echo ========================================
echo.

echo 📦 Checking Python dependencies...
pip list | findstr "torch transformers flask" >nul
if %errorlevel% neq 0 (
    echo Installing Python packages...
    pip install -r requirements.txt
) else (
    echo ✅ Python dependencies already installed
)
echo.

echo 🤖 Starting Local AI Model Server...
echo 📍 Backend: http://localhost:5000
echo 💡 Keep this window open for the AI model
echo.

start "Lity AI Model Server" cmd /k "C:/Users/user/AppData/Local/Microsoft/WindowsApps/python3.11.exe local_model_server.py"

echo ⏳ Waiting for model server to start...
timeout /t 15 /nobreak >nul

echo.
echo 🌐 Starting React Frontend...
echo 📍 Frontend: http://localhost:3000
echo.

start "Lity AI Frontend" cmd /k "npm start"

echo.
echo ========================================
echo ✅ Lity AI is starting up!
echo ========================================
echo 🤖 AI Model: http://localhost:5000
echo 🌐 Web App: http://localhost:3000
echo 💡 Both windows will stay open
echo ========================================
echo.

pause
