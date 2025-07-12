#!/usr/bin/env python3
"""
كود مخصص لقراءة بيانات wikir مباشرة من الملفات المحلية
"""

import csv
import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WikirLoader:
    def __init__(self, data_path: str = "data/wikIR1k/wikIR1k"):
        self.data_path = data_path
        self.documents_file = os.path.join(data_path, "documents.csv")
        self.test_queries_file = os.path.join(data_path, "test/queries.csv")
        self.test_qrels_file = os.path.join(data_path, "test/qrels")
        
    def load_documents_to_db(self, dataset_name: str = "wikir/en1k/test", limit: int = None) -> Dict[str, Any]:
        """
        تحميل الوثائق من ملف CSV إلى قاعدة البيانات
        """
        try:
            logger.info(f"بدء تحميل الوثائق من {self.documents_file}")
            
            # التحقق من وجود الملف
            if not os.path.exists(self.documents_file):
                raise FileNotFoundError(f"ملف الوثائق غير موجود: {self.documents_file}")
            
            # فتح قاعدة البيانات
            db_path = "data/ir_documents.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            loaded_count = 0
            start_time = datetime.now()
            
            # قراءة ملف CSV مع معالجة الترميز
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"محاولة قراءة الملف بترميز {encoding}...")
                    
                    with open(self.documents_file, 'r', encoding=encoding, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        
                        # طباعة أسماء الأعمدة للتحقق
                        logger.info(f"أسماء الأعمدة: {reader.fieldnames}")
                        
                        row_count = 0
                        for row in reader:
                            row_count += 1
                            try:
                                # استخراج البيانات - استخدام أسماء الأعمدة الصحيحة
                                doc_id = str(row.get('id_right', ''))
                                text = str(row.get('text_right', ''))
                                
                                # استخدام النص مباشرة
                                content = text.strip()
                                
                                # التحقق من صحة البيانات
                                if not doc_id or not content:
                                    logger.warning(f"تخطي صف {row_count}: doc_id='{doc_id}', content='{content[:50]}...'")
                                    continue
                                
                                # إدراج في قاعدة البيانات
                                cursor.execute("""
                                    INSERT OR REPLACE INTO documents (doc_id, text, dataset)
                                    VALUES (?, ?, ?)
                                """, (doc_id, content, dataset_name))
                                
                                loaded_count += 1
                                
                                # التحقق من الحد
                                if limit and loaded_count >= limit:
                                    break
                                
                                # تسجيل التقدم
                                if loaded_count % 10000 == 0:
                                    logger.info(f"تم تحميل {loaded_count} وثيقة...")
                                    
                            except Exception as e:
                                logger.warning(f"خطأ في معالجة وثيقة: {e}")
                                continue
                    
                    # إذا وصلنا هنا، نجحت القراءة
                    logger.info(f"✅ نجح قراءة الملف بترميز {encoding}")
                    logger.info(f"إجمالي الصفوف المقروءة: {row_count}")
                    break
                    
                except UnicodeDecodeError as e:
                    logger.warning(f"فشل قراءة الملف بترميز {encoding}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"خطأ آخر في قراءة الملف بترميز {encoding}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ تم تحميل {loaded_count} وثيقة في {duration:.2f} ثانية")
            
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "loaded_documents": loaded_count,
                "duration_seconds": duration,
                "source_file": self.documents_file
            }
            
        except Exception as e:
            logger.error(f"خطأ في تحميل الوثائق: {e}")
            raise
    
    def load_test_queries(self) -> List[Dict[str, Any]]:
        """
        تحميل استعلامات الاختبار
        """
        queries = []
        queries_file = self.test_queries_file
        
        if not os.path.exists(queries_file):
            logger.warning(f"ملف الاستعلامات غير موجود: {queries_file}")
            return queries
        
        try:
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(queries_file, 'r', encoding=encoding, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            queries.append({
                                'query_id': row.get('query_id', ''),
                                'query': row.get('query', '')
                            })
                    logger.info(f"تم تحميل {len(queries)} استعلام بترميز {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.error(f"خطأ في تحميل الاستعلامات: {e}")
        
        return queries
    
    def load_test_qrels(self) -> List[Dict[str, Any]]:
        """
        تحميل ملف qrels للاختبار
        """
        qrels = []
        qrels_file = self.test_qrels_file
        
        if not os.path.exists(qrels_file):
            logger.warning(f"ملف qrels غير موجود: {qrels_file}")
            return qrels
        
        try:
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qrels.append({
                            'query_id': parts[0],
                            'doc_id': parts[2],
                            'relevance': int(parts[3])
                        })
            logger.info(f"تم تحميل {len(qrels)} qrel")
        except Exception as e:
            logger.error(f"خطأ في تحميل qrels: {e}")
        
        return qrels

def load_wikir_from_local_files(dataset_name: str = "wikir/en1k/test", limit: int = None) -> Dict[str, Any]:
    """
    دالة رئيسية لتحميل بيانات wikir من الملفات المحلية
    """
    loader = WikirLoader()
    return loader.load_documents_to_db(dataset_name, limit)

if __name__ == "__main__":
    # اختبار الكود
    result = load_wikir_from_local_files(limit=1000)
    print(f"نتيجة التحميل: {result}") 