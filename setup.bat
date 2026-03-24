@echo off
chcp 65001 >nul
title VN-VoiceClone Pro - Setup

echo.
echo ╔═══════════════════════════════════════════════╗
echo ║         VN-VoiceClone Pro - Cài đặt          ║
echo ╚═══════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LỖI] Không tìm thấy Python. Vui lòng cài Python 3.10+ từ python.org
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [LỖI] Không tìm thấy Node.js. Vui lòng cài Node.js 18+ từ nodejs.org
    pause
    exit /b 1
)

echo [1/4] Đang tạo Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo       ✓ Đã tạo virtual environment
) else (
    echo       ✓ Virtual environment đã tồn tại
)

echo.
echo [2/4] Đang cài đặt Python dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/ --quiet
echo       ✓ Đã cài đặt Python dependencies

echo.
echo [3/4] Đang cài đặt FFmpeg (cho xử lý audio)...
pip install pydub --quiet
echo       ✓ FFmpeg sẽ được tải tự động khi cần

echo.
echo [4/4] Đang cài đặt Frontend dependencies...
cd frontend
call npm install
cd ..
echo       ✓ Đã cài đặt Frontend dependencies

echo.
echo ╔═══════════════════════════════════════════════╗
echo ║            Cài đặt hoàn tất! ✓               ║
echo ║                                               ║
echo ║  Chạy start.bat để khởi động ứng dụng        ║
echo ╚═══════════════════════════════════════════════╝
echo.
pause
