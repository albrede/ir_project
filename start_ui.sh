#!/bin/bash

echo "🎯 تشغيل نظام استرجاع المعلومات"
echo "================================================"

# فحص وجود Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js غير مثبت. يرجى تثبيته من https://nodejs.org/"
    exit 1
fi

# فحص وجود npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm غير مثبت"
    exit 1
fi

echo "✓ Node.js و npm مثبتان"

# الانتقال إلى مجلد الواجهة
cd ui

# فحص وجود node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 تثبيت تبعيات المشروع..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ فشل في تثبيت التبعيات"
        exit 1
    fi
fi

# فحص الخدمات الخلفية
echo "🔍 فحص الخدمات الخلفية..."
if ! curl -s http://localhost:8005/health &> /dev/null; then
    echo "⚠️ الخدمات الخلفية غير متاحة"
    echo "هل تريد تشغيلها الآن؟ (y/n): "
    read choice
    if [[ $choice =~ ^[Yy]$ ]]; then
        echo "🚀 تشغيل الخدمات الخلفية..."
        cd ..
        python run_services.py &
        BACKEND_PID=$!
        sleep 5
        cd ui
    else
        echo "❌ لا يمكن تشغيل واجهة المستخدم بدون الخدمات الخلفية"
        exit 1
    fi
fi

echo "🌐 تشغيل واجهة المستخدم..."
npm start &
FRONTEND_PID=$!

echo ""
echo "🎉 تم تشغيل النظام بنجاح!"
echo ""
echo "📋 معلومات الوصول:"
echo "   • واجهة المستخدم: http://localhost:3000"
echo "   • API موحد: http://localhost:8005"
echo "   • وثائق API: http://localhost:8005/docs"
echo ""
echo "⏹️ للوقف، اضغط Ctrl+C"

# انتظار إشارة الإيقاف
trap 'echo ""; echo "🛑 إيقاف النظام..."; kill $FRONTEND_PID $BACKEND_PID 2>/dev/null; echo "✓ تم إيقاف النظام"; exit 0' INT

# انتظار انتهاء العمليات
wait 