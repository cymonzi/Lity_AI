@echo off
echo 🚀 Starting Lity AI with Local Model...
echo.

echo 📦 Installing Python dependencies...
pip install -r requirements.txt
echo.

echo 🤖 Starting local model server...
echo 📍 Server will be available at: http://localhost:5000
echo 💡 Keep this window open to keep the AI model running
echo.

python local_model_server.py

pause
