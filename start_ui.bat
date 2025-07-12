@echo off
chcp 65001 >nul
echo 🎯 تشغيل نظام استرجاع المعلومات
echo ================================================

REM فحص وجود Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js غير مثبت. يرجى تثبيته من https://nodejs.org/
    pause
    exit /b 1
)

REM فحص وجود npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm غير مثبت
    pause
    exit /b 1
)

echo ✓ Node.js و npm مثبتان

REM الانتقال إلى مجلد الواجهة
cd ui

REM فحص وجود node_modules
if not exist "node_modules" (
    echo 📦 تثبيت تبعيات المشروع...
    npm install
    if %errorlevel% neq 0 (
        echo ❌ فشل في تثبيت التبعيات
        pause
        exit /b 1
    )
)

REM فحص الخدمات الخلفية
echo 🔍 فحص الخدمات الخلفية...
curl -s http://localhost:8005/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ الخدمات الخلفية غير متاحة
    echo هل تريد تشغيلها الآن؟ (y/n): 
    set /p choice=
    if /i "%choice%"=="y" (
        echo 🚀 تشغيل الخدمات الخلفية...
        start "Backend Services" cmd /k "cd .. && python run_services.py"
        timeout /t 5 /nobreak >nul
    ) else (
        echo ❌ لا يمكن تشغيل واجهة المستخدم بدون الخدمات الخلفية
        pause
        exit /b 1
    )
)

echo 🌐 تشغيل واجهة المستخدم...
start "Frontend" cmd /k "npm start"

echo.
echo 🎉 تم تشغيل النظام بنجاح!
echo.
echo 📋 معلومات الوصول:
echo    • واجهة المستخدم: http://localhost:3000
echo    • API موحد: http://localhost:8005
echo    • وثائق API: http://localhost:8005/docs
echo.
echo ⏹️ للوقف، أغلق نوافذ الأوامر
pause 