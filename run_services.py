#!/usr/bin/env python3
"""
ملف تشغيل جميع خدمات استرجاع المعلومات
يتم تشغيل كل خدمة على منفذ منفصل
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def run_service(service_name, port, script_path):
    """تشغيل خدمة معينة"""
    try:
        print(f"🚀 تشغيل {service_name} على المنفذ {port}...")
        process = subprocess.Popen([
            sys.executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # انتظار قليل للتأكد من بدء الخدمة
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {service_name} يعمل على http://localhost:{port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ فشل في تشغيل {service_name}")
            print(f"الخطأ: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ خطأ في تشغيل {service_name}: {e}")
        return None

def main():
    """الدالة الرئيسية لتشغيل جميع الخدمات"""
    
    # تعريف الخدمات
    services = [
        {
            "name": "Query Processing Service",
            "port": 8001,
            "script": "api_services/query_processing_api.py"
        },
        {
            "name": "Search & Ranking Service", 
            "port": 8002,
            "script": "api_services/search_ranking_api.py"
        },
        {
            "name": "Indexing Service",
            "port": 8003, 
            "script": "api_services/indexing_api.py"
        },
        {
            "name": "Unified API",
            "port": 8005,
            "script": "api_services/query_api.py"
        }
    ]
    
    print("🔧 بدء تشغيل خدمات استرجاع المعلومات...")
    print("=" * 60)
    
    processes = []
    
    # تشغيل كل خدمة
    for service in services:
        script_path = Path(service["script"])
        
        if not script_path.exists():
            print(f"❌ ملف {script_path} غير موجود")
            continue
            
        process = run_service(
            service["name"], 
            service["port"], 
            str(script_path)
        )
        
        if process:
            processes.append((service["name"], process))
    
    if not processes:
        print("❌ لم يتم تشغيل أي خدمة")
        return
    
    print("\n" + "=" * 60)
    print("🎉 تم تشغيل الخدمات التالية:")
    for name, _ in processes:
        print(f"   ✅ {name}")
    
    print("\n📋 روابط الخدمات:")
    print("   🔗 Query Processing: http://localhost:8001")
    print("   🔗 Search & Ranking: http://localhost:8002") 
    print("   🔗 Indexing: http://localhost:8003")
    print("   🔗 Unified API: http://localhost:8005")
    
    print("\n📖 للوصول إلى وثائق API:")
    print("   📚 Query Processing: http://localhost:8001/docs")
    print("   📚 Search & Ranking: http://localhost:8002/docs")
    print("   📚 Indexing: http://localhost:8003/docs")
    print("   📚 Unified API: http://localhost:8005/docs")
    
    print("\n⏹️  اضغط Ctrl+C لإيقاف جميع الخدمات")
    
    try:
        # انتظار حتى يتم إيقاف الخدمات
        while True:
            time.sleep(1)
            
            # فحص إذا كانت أي خدمة توقفت
            for name, process in processes[:]:
                if process.poll() is not None:
                    print(f"⚠️  {name} توقف")
                    processes.remove((name, process))
            
            if not processes:
                print("❌ جميع الخدمات توقفت")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 إيقاف جميع الخدمات...")
        
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ تم إيقاف {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  تم إجبار إيقاف {name}")
            except Exception as e:
                print(f"❌ خطأ في إيقاف {name}: {e}")

if __name__ == "__main__":
    main() 