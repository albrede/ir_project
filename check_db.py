import sqlite3
import os

def check_database():
    db_path = "data/ir_documents.db"
    
    if not os.path.exists(db_path):
        print("❌ قاعدة البيانات غير موجودة")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # فحص الجداول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("📊 الجداول الموجودة في قاعدة البيانات:")
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
            
            # فحص عدد الصفوف في كل جدول
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"    عدد الصفوف: {count}")
            
            # فحص مجموعات البيانات المختلفة
            if table_name == "documents":
                cursor.execute("SELECT DISTINCT dataset FROM documents")
                datasets = cursor.fetchall()
                print(f"    مجموعات البيانات:")
                for dataset in datasets:
                    dataset_name = dataset[0]
                    cursor.execute(f"SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset_name,))
                    dataset_count = cursor.fetchone()[0]
                    print(f"      - {dataset_name}: {dataset_count} وثيقة")
            
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ خطأ في فحص قاعدة البيانات: {e}")

if __name__ == "__main__":
    check_database() 