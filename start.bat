@echo off
chcp 65001 >nul
title VN-VoiceClone Pro

echo.
echo ╔═══════════════════════════════════════════════╗
echo ║        VN-VoiceClone Pro - Khởi động          ║
echo ╚═══════════════════════════════════════════════╝
echo.

:: Check if setup has been done
if not exist "venv" (
    echo [INFO] Chưa cài đặt. Đang chạy setup...
    call setup.bat
)

:: Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
if not exist "trimmed" mkdir trimmed

:: Start Backend
echo [1/2] Đang khởi động Backend (FastAPI + VieNeu-TTS)...
echo       API server: http://localhost:8000
start "VN-VoiceClone Backend" cmd /k "call venv\Scripts\activate.bat && python server.py"

:: Wait for backend to initialize
echo       Đang chờ backend khởi động...
timeout /t 5 /nobreak >nul

:: Start Frontend
echo.
echo [2/2] Đang khởi động Frontend (Next.js)...
echo       Web app: http://localhost:3000
start "VN-VoiceClone Frontend" cmd /k "cd frontend && npm run dev"

:: Wait and open browser
timeout /t 5 /nobreak >nul
echo.
echo ╔═══════════════════════════════════════════════╗
echo ║            Ứng dụng đã sẵn sàng! ✓           ║
echo ║                                               ║
echo ║  🌐 Mở trình duyệt: http://localhost:3000    ║
echo ║  📡 API Backend:     http://localhost:8000    ║
echo ║                                               ║
echo ║  Nhấn bất kỳ phím nào để mở trình duyệt...  ║
echo ╚═══════════════════════════════════════════════╝
pause >nul
start http://localhost:3000
