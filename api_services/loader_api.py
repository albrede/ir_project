import builtins
import sys
import os

# تغيير الترميز الافتراضي للنظام
import locale
import codecs

# إجبار الترميز الافتراضي ليكون utf-8
if hasattr(locale, 'getpreferredencoding'):
    original_getpreferredencoding = locale.getpreferredencoding
    locale.getpreferredencoding = lambda: 'utf-8'

# إجبار codecs لاستخدام utf-8
def strict_error_handler(error):
    return ('', error.end if hasattr(error, 'end') else 0)
codecs.register_error('strict', strict_error_handler)

# تحسين monkey patch ليشمل جميع الحالات
_builtin_open = open

def open_utf8_fallback(*args, **kwargs):
    mode = ''
    if len(args) > 1:
        mode = args[1]
    elif 'mode' in kwargs:
        mode = kwargs['mode']
    
    # إذا كان binary mode، لا تضف encoding
    if 'b' in mode:
        return _builtin_open(*args, **kwargs)
    
    # تجربة utf-8 أولاً
    try:
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        return _builtin_open(*args, **kwargs)
    except UnicodeDecodeError:
        # إذا فشل utf-8، جرب latin1 (لا يرمي خطأ على أي رمز)
        kwargs['encoding'] = 'latin1'
        return _builtin_open(*args, **kwargs)

builtins.open = open_utf8_fallback

# إضافة معالجة خاصة لخطأ الترميز في csv
import csv
_original_field_size_limit = csv.field_size_limit

def safe_field_size_limit(limit):
    try:
        _original_field_size_limit(limit)
    except OverflowError:
        # إذا فشل، استخدم قيمة أصغر
        _original_field_size_limit(min(limit, 2**31 - 1))

csv.field_size_limit = safe_field_size_limit

# إضافة معالجة خاصة لخطأ الترميز في pandas (إذا كان مستخدماً)
try:
    import pandas as pd
    def safe_read_csv(*args, **kwargs):
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        try:
            return pd.read_csv(*args, **kwargs)
        except UnicodeDecodeError:
            kwargs['encoding'] = 'latin1'
            return pd.read_csv(*args, **kwargs)
    pd.read_csv = safe_read_csv
except ImportError:
    pass

from fastapi import FastAPI, Query, HTTPException
from typing import Optional, List, Dict, Any
import ir_datasets
import sqlite3
import logging
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Loader Service API", version="1.0.0")

# قاعدة البيانات
DB_PATH = "data/ir_documents.db"

def init_database():
    """تهيئة قاعدة البيانات"""
    try:
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # إنشاء جدول الوثائق
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                dataset TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # إنشاء فهرس للبحث السريع
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_dataset 
            ON documents(dataset)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_id 
            ON documents(doc_id)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ تم تهيئة قاعدة البيانات بنجاح")
        
    except Exception as e:
        logger.error(f"❌ خطأ في تهيئة قاعدة البيانات: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """تهيئة قاعدة البيانات عند بدء الخدمة"""
    init_database()

@app.get("/")
def read_root():
    """الصفحة الرئيسية للخدمة"""
    return {
        "service": "Data Loader Service",
        "version": "1.0.0",
        "description": "خدمة تحميل مجموعات البيانات للبحث في المعلومات",
        "endpoints": {
            "GET /datasets": "قائمة مجموعات البيانات المتاحة",
            "POST /load-dataset": "تحميل مجموعة بيانات",
            "GET /dataset/{dataset_name}": "معلومات مجموعة بيانات محددة",
            "DELETE /dataset/{dataset_name}": "حذف مجموعة بيانات"
        }
    }

@app.get("/datasets")
def get_available_datasets():
    """الحصول على قائمة مجموعات البيانات المتاحة"""
    try:
        # قائمة مجموعات البيانات المدعومة
        supported_datasets = [
            {
                "name": "beir/trec-covid",
                "description": "TREC-COVID Dataset from BEIR",
                "type": "scientific_papers",
                "size": "~171K documents"
            },
            {
                "name": "beir/arguana",
                "description": "ArguAna Dataset from BEIR",
                "type": "argument_retrieval",
                "size": "~8.7K documents"
            }
        ]
        
        # التحقق من البيانات المحملة مسبقاً
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dataset, COUNT(*) as doc_count
            FROM documents 
            GROUP BY dataset
        """)
        
        loaded_datasets = {}
        for row in cursor.fetchall():
            loaded_datasets[row[0]] = {
                "document_count": row[1]
            }
        
        conn.close()
        
        # دمج المعلومات
        for dataset in supported_datasets:
            if dataset["name"] in loaded_datasets:
                dataset["status"] = "loaded"
                dataset.update(loaded_datasets[dataset["name"]])
            else:
                dataset["status"] = "not_loaded"
        
        return {
            "status": "success",
            "datasets": supported_datasets,
            "total_loaded": len(loaded_datasets)
        }
        
    except Exception as e:
        logger.error(f"خطأ في الحصول على مجموعات البيانات: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-dataset")
def load_dataset(
    dataset_name: str = Query(..., description="اسم مجموعة البيانات"),
    limit: Optional[int] = Query(500000, description="الحد الأقصى لعدد الوثائق"),
    force_reload: bool = Query(False, description="إعادة تحميل حتى لو كانت موجودة مسبقاً")
):
    """تحميل مجموعة بيانات محددة"""
    try:
        # التحقق من صحة اسم مجموعة البيانات
        valid_datasets = ["wikir/en1k/test", "antique/test"]
        if dataset_name not in valid_datasets:
            raise HTTPException(
                status_code=400, 
                detail=f"مجموعة البيانات غير مدعومة. الخيارات المتاحة: {valid_datasets}"
            )
        
        # استخدام الكود المخصص لـ wikir إذا كانت الملفات المحلية متوفرة
        if dataset_name == "wikir/en1k/test":
            local_files_path = "data/wikIR1k/wikIR1k"
            if os.path.exists(local_files_path):
                logger.info(f"استخدام الملفات المحلية لـ {dataset_name}")
                try:
                    from services.wikir_loader import load_wikir_from_local_files
                    result = load_wikir_from_local_files(dataset_name, limit)
                    return result
                except Exception as e:
                    logger.warning(f"فشل في استخدام الملفات المحلية: {e}")
                    logger.info("العودة لاستخدام ir_datasets...")
                # إذا فشل الكود المخصص، يستمر مع ir_datasets
        
        # التحقق من وجود البيانات مسبقاً
        if not force_reload:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset_name,))
            existing_count = cursor.fetchone()[0]
            conn.close()
            
            if existing_count > 0:
                return {
                    "status": "already_exists",
                    "message": f"مجموعة البيانات '{dataset_name}' محملة مسبقاً ({existing_count} وثيقة)",
                    "dataset_name": dataset_name,
                    "document_count": existing_count
                }
        
        # حذف البيانات القديمة إذا طُلب إعادة التحميل
        if force_reload:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE dataset = ?", (dataset_name,))
            conn.commit()
            conn.close()
            logger.info(f"تم حذف البيانات القديمة لمجموعة البيانات: {dataset_name}")
        
        # تحميل مجموعة البيانات مع معالجة خاصة لأخطاء الترميز
        logger.info(f"بدء تحميل مجموعة البيانات: {dataset_name}")
        
        # تنظيف شامل للملفات المؤقتة قبل البدء
        import tempfile
        import shutil
        import gc
        import time
        
        # إجبار garbage collection لإغلاق أي ملفات مفتوحة
        gc.collect()
        
        # تغيير مجلد الملفات المؤقتة إلى مجلد في المشروع لتجنب مشاكل الوصول
        original_tempdir = tempfile.gettempdir()
        project_tempdir = os.path.join(os.getcwd(), 'temp_ir_datasets')
        os.makedirs(project_tempdir, exist_ok=True)
        tempfile.tempdir = project_tempdir
        
        logger.info(f"تم تغيير مجلد الملفات المؤقتة إلى: {project_tempdir}")
        
        # تنظيف شامل للملفات المؤقتة
        try:
            temp_dir = os.path.join(tempfile.gettempdir(), 'ir_datasets')
            if os.path.exists(temp_dir):
                logger.info(f"تنظيف مجلد الملفات المؤقتة: {temp_dir}")
                
                # حذف جميع الملفات المؤقتة
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            logger.info(f"تم حذف الملف: {item}")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            logger.info(f"تم حذف المجلد: {item}")
                    except Exception as e:
                        logger.warning(f"فشل في حذف {item}: {e}")
                
                # انتظار قليل للتأكد من إغلاق الملفات
                time.sleep(1)
        except Exception as e:
            logger.warning(f"فشل في تنظيف الملفات المؤقتة: {e}")
        
        # محاولات متعددة لتحميل مجموعة البيانات مع ترميزات مختلفة
        dataset = None
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"محاولة تحميل {dataset_name} بترميز {encoding}...")
                
                # تغيير الترميز الافتراضي مؤقتاً
                if hasattr(locale, 'getpreferredencoding'):
                    original_encoding = locale.getpreferredencoding()
                    locale.getpreferredencoding = lambda: encoding
                
                # معالجة خاصة لخطأ الوصول للملفات المؤقتة
                try:
                    dataset = ir_datasets.load(dataset_name)
                    logger.info(f"✅ نجح تحميل {dataset_name} بترميز {encoding}")
                    break
                except PermissionError as pe:
                    if "Access is denied" in str(pe) or "WinError 5" in str(pe):
                        logger.warning(f"خطأ في الوصول للملفات المؤقتة: {pe}")
                        
                        # محاولات متعددة لتنظيف الملفات المؤقتة
                        for attempt in range(3):
                            try:
                                logger.info(f"محاولة تنظيف الملفات المؤقتة {attempt + 1}/3...")
                                
                                # تنظيف شامل للملفات المؤقتة
                                temp_dir = os.path.join(tempfile.gettempdir(), 'ir_datasets')
                                if os.path.exists(temp_dir):
                                    for item in os.listdir(temp_dir):
                                        item_path = os.path.join(temp_dir, item)
                                        try:
                                            if os.path.isfile(item_path):
                                                os.remove(item_path)
                                            elif os.path.isdir(item_path):
                                                shutil.rmtree(item_path)
                                        except:
                                            pass
                                
                                # انتظار قليل
                                time.sleep(2)
                                
                                # محاولة تحميل مرة أخرى
                                dataset = ir_datasets.load(dataset_name)
                                logger.info(f"✅ نجح تحميل {dataset_name} بعد تنظيف الملفات المؤقتة (محاولة {attempt + 1})")
                                break
                                
                            except Exception as retry_error:
                                logger.warning(f"فشلت المحاولة {attempt + 1}: {retry_error}")
                                if attempt < 2:  # إذا لم تكن المحاولة الأخيرة
                                    time.sleep(3)  # انتظار أطول
                                    continue
                                else:
                                    logger.error("فشلت جميع محاولات تنظيف الملفات المؤقتة")
                                    continue
                    else:
                        raise pe
                
            except UnicodeDecodeError as e:
                logger.warning(f"فشل تحميل {dataset_name} بترميز {encoding}: {e}")
                continue
            except Exception as e:
                logger.warning(f"خطأ آخر في تحميل {dataset_name} بترميز {encoding}: {e}")
                continue
            finally:
                # إعادة الترميز الأصلي
                if hasattr(locale, 'getpreferredencoding'):
                    locale.getpreferredencoding = lambda: original_encoding
        
        if dataset is None:
            logger.error(f"فشلت جميع محاولات تحميل {dataset_name}")
            
            # محاولة أخيرة بتغيير مجلد الملفات المؤقتة
            try:
                logger.info("محاولة أخيرة بتغيير مجلد الملفات المؤقتة...")
                import tempfile
                
                # تغيير مجلد الملفات المؤقتة مؤقتاً
                original_tempdir = tempfile.gettempdir()
                new_tempdir = os.path.join(os.getcwd(), 'temp_ir_datasets')
                os.makedirs(new_tempdir, exist_ok=True)
                
                # تغيير مجلد temp مؤقتاً
                tempfile.tempdir = new_tempdir
                
                dataset = ir_datasets.load(dataset_name)
                logger.info(f"✅ نجح تحميل {dataset_name} مع مجلد مؤقت جديد")
                
                # إعادة مجلد temp الأصلي
                tempfile.tempdir = original_tempdir
                
            except Exception as final_error:
                logger.error(f"فشلت المحاولة الأخيرة: {final_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"فشل في تحميل مجموعة البيانات {dataset_name}. يرجى إعادة تشغيل الخدمة أو تشغيلها كـ Administrator"
                )
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        loaded_count = 0
        start_time = datetime.now()
        
        for doc in dataset.docs_iter():
            if limit and loaded_count >= limit:
                break
                
            try:
                # استخراج معرف الوثيقة والمحتوى مع معالجة خاصة للترميز
                if hasattr(doc, 'doc_id') and hasattr(doc, 'text'):
                    doc_id = str(doc.doc_id)
                    try:
                        content = str(doc.text)
                    except UnicodeDecodeError:
                        # إذا فشل ترميز النص، استخدم نص بديل
                        content = "محتوى غير قابل للقراءة - خطأ في الترميز"
                        logger.warning(f"خطأ في ترميز محتوى وثيقة {doc_id}")
                elif hasattr(doc, 'doc_id') and hasattr(doc, 'title'):
                    doc_id = str(doc.doc_id)
                    try:
                        content = str(doc.title)
                        if hasattr(doc, 'body'):
                            content += " " + str(doc.body)
                    except UnicodeDecodeError:
                        content = "محتوى غير قابل للقراءة - خطأ في الترميز"
                        logger.warning(f"خطأ في ترميز محتوى وثيقة {doc_id}")
                else:
                    continue
                
                # التحقق من صحة doc_id
                if not doc_id or doc_id.strip() == '':
                    logger.warning(f"تخطي وثيقة بدون معرف صحيح")
                    continue
                
                # إدراج الوثيقة - معالجة doc_id كسلسلة نصية لتجنب مشكلة overflow
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (doc_id, text, dataset)
                    VALUES (?, ?, ?)
                """, (doc_id, content, dataset_name))
                
                loaded_count += 1
                
                if loaded_count % 10000 == 0:
                    logger.info(f"تم تحميل {loaded_count} وثيقة...")
            except UnicodeDecodeError as ude:
                logger.warning(f"تخطي وثيقة بسبب خطأ ترميز: {ude}")
                continue
            except sqlite3.Error as e:
                logger.error(f"خطأ في قاعدة البيانات أثناء تحميل وثيقة: {e}")
                continue
            except Exception as e:
                logger.warning(f"خطأ في تحميل وثيقة: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        # إعادة مجلد temp الأصلي
        tempfile.tempdir = original_tempdir
        logger.info(f"تم إعادة مجلد الملفات المؤقتة إلى: {original_tempdir}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ تم تحميل {loaded_count} وثيقة من مجموعة البيانات '{dataset_name}' في {duration:.2f} ثانية")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "loaded_documents": loaded_count,
            "duration_seconds": duration,
            "database_path": DB_PATH
        }
        
    except Exception as e:
        logger.error(f"خطأ في تحميل مجموعة البيانات: {e}")
        # إعادة مجلد temp الأصلي في حالة الخطأ
        try:
            tempfile.tempdir = original_tempdir
            logger.info(f"تم إعادة مجلد الملفات المؤقتة إلى: {original_tempdir}")
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download-qrel")
def download_qrel(dataset_name: str = Query(..., description="اسم مجموعة البيانات")):
    """تحميل ملف qrel لمجموعة بيانات وحفظه في مجلد evaluation"""
    try:
        valid_datasets = ["beir/trec-covid", "beir/arguana"]
        if dataset_name not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"مجموعة البيانات غير مدعومة. الخيارات المتاحة: {valid_datasets}")
        os.makedirs("evaluation", exist_ok=True)
        dataset = ir_datasets.load(dataset_name)
        qrel_path = f"evaluation/{dataset_name.replace('/', '_')}.qrels"
        count = 0
        with open(qrel_path, "w", encoding="utf-8") as f:
            for qrel in dataset.qrels_iter():
                f.write(f"{qrel.query_id} {qrel.iteration} {qrel.doc_id} {qrel.relevance}\n")
                count += 1
        return {
            "status": "success",
            "message": f"تم تحميل وحفظ ملف qrel ({count} صف) في {qrel_path}",
            "qrel_path": qrel_path
        }
    except Exception as e:
        logger.error(f"خطأ في تحميل qrel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download-queries")
def download_queries(dataset_name: str = Query(..., description="اسم مجموعة البيانات")):
    """تحميل queries لمجموعة بيانات وحفظها في مجلد evaluation"""
    try:
        valid_datasets = ["beir/trec-covid", "beir/arguana"]
        if dataset_name not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"مجموعة البيانات غير مدعومة. الخيارات المتاحة: {valid_datasets}")
        os.makedirs("evaluation", exist_ok=True)
        dataset = ir_datasets.load(dataset_name)
        queries_path = f"evaluation/{dataset_name.replace('/', '_')}.queries.tsv"
        count = 0
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in dataset.queries_iter():
                # بعض الاستعلامات قد تحتوي فقط على title أو text
                text = getattr(query, 'text', None) or getattr(query, 'title', None) or ''
                f.write(f"{query.query_id}\t{text}\n")
                count += 1
        return {
            "status": "success",
            "message": f"تم تحميل وحفظ الاستعلامات ({count} استعلام) في {queries_path}",
            "queries_path": queries_path
        }
    except Exception as e:
        logger.error(f"خطأ في تحميل الاستعلامات: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/{dataset_name:path}")
def get_dataset_info(dataset_name: str):
    """الحصول على معلومات مجموعة بيانات محددة"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # إحصائيات الوثائق
        cursor.execute("""
            SELECT COUNT(*) as total_docs,
                   AVG(LENGTH(text)) as avg_content_length
            FROM documents 
            WHERE dataset = ?
        """, (dataset_name,))
        
        stats = cursor.fetchone()
        
        if not stats or stats[0] == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"مجموعة البيانات '{dataset_name}' غير موجودة"
            )
        
        # عينة من الوثائق
        cursor.execute("""
            SELECT doc_id, text
            FROM documents 
            WHERE dataset = ?
            LIMIT 5
        """, (dataset_name,))
        
        sample_docs = []
        for row in cursor.fetchall():
            sample_docs.append({
                "doc_id": row[0],
                "content_preview": row[1][:200] + "..." if len(row[1]) > 200 else row[1]
            })
        
        conn.close()
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "statistics": {
                "total_documents": stats[0],
                "average_content_length": round(stats[1], 2) if stats[1] else 0
            },
            "sample_documents": sample_docs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في الحصول على معلومات مجموعة البيانات: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/dataset/{dataset_name:path}")
def delete_dataset(dataset_name: str):
    """حذف مجموعة بيانات محددة"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # التحقق من وجود البيانات
        cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset_name,))
        doc_count = cursor.fetchone()[0]
        
        if doc_count == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"مجموعة البيانات '{dataset_name}' غير موجودة"
            )
        
        # حذف البيانات
        cursor.execute("DELETE FROM documents WHERE dataset = ?", (dataset_name,))
        conn.commit()
        conn.close()
        
        logger.info(f"تم حذف {doc_count} وثيقة من مجموعة البيانات '{dataset_name}'")
        
        return {
            "status": "success",
            "message": f"تم حذف مجموعة البيانات '{dataset_name}' بنجاح",
            "deleted_documents": doc_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في حذف مجموعة البيانات: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """فحص صحة الخدمة"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "total_documents": total_docs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
