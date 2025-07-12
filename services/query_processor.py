import logging
from services.preprocessing import preprocess, preprocess_for_vectorization, preprocess_for_indexing
from services.vectorization.tfidf_vectorizer import get_tfidf_vector, get_tfidf_vector_serializable, load_tfidf_model
from services.vectorization.embedding_vectorizer import get_embedding_vector, get_embedding_vector_serializable, load_embedding_model
import joblib
import os
import numpy as np

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, method="tfidf", dataset_name="simple"):
        """
        تهيئة معالج الاستعلامات
        Args:
            method: طريقة التمثيل ("tfidf", "embedding", "hybrid", "hybrid-sequential")
            dataset_name: اسم مجموعة البيانات
        """
        self.method = method
        self.dataset_name = dataset_name
        self.tfidf_model = None
        self.embedding_model = None
        
        # تحميل النماذج المطلوبة
        self._load_models()
    
    def _load_models(self):
        """تحميل النماذج المطلوبة حسب الطريقة"""
        try:
            if self.method in ["tfidf", "hybrid", "hybrid-sequential"]:
                # تحميل نموذج TF-IDF
                try:
                    self.tfidf_model, _, _ = load_tfidf_model(self.dataset_name)
                    logger.info(f"تم تحميل نموذج TF-IDF لـ {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"فشل في تحميل نموذج TF-IDF لـ {self.dataset_name}: {e}")
            
            if self.method in ["embedding", "hybrid", "hybrid-sequential"]:
                # تحميل نموذج Embedding
                try:
                    self.embedding_model, _, _ = load_embedding_model(self.dataset_name)
                    logger.info(f"تم تحميل نموذج Embedding لـ {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"فشل في تحميل نموذج Embedding لـ {self.dataset_name}: {e}")
                    
        except Exception as e:
            logger.error(f"خطأ في تحميل النماذج: {e}")
    
    def process_query(self, query: str, return_vector: bool = False):
        """
        معالجة الاستعلام حسب الطريقة المحددة
        Args:
            query: النص الأصلي للاستعلام
            return_vector: إرجاع المتجه بدلاً من النص المعالج
        Returns:
            النص المعالج أو المتجه حسب return_vector
        """
        if not query or not query.strip():
            raise ValueError("الاستعلام لا يمكن أن يكون فارغاً")
        
        try:
            if self.method == "tfidf":
                return self._process_tfidf_query(query, return_vector)
            elif self.method == "embedding":
                return self._process_embedding_query(query, return_vector)
            elif self.method == "hybrid":
                return self._process_hybrid_query(query, return_vector)
            elif self.method == "hybrid-sequential":
                return self._process_hybrid_sequential_query(query, return_vector)
            else:
                raise ValueError(f"طريقة المعالجة '{self.method}' غير مدعومة")
                
        except Exception as e:
            logger.error(f"خطأ في معالجة الاستعلام: {e}")
            raise
    
    def _process_tfidf_query(self, query: str, return_vector: bool = False):
        """معالجة الاستعلام لـ TF-IDF"""
        # استخدام نفس المعالجة المستخدمة في بناء النموذج
        processed_text = preprocess_for_vectorization(query)
        
        if return_vector and self.tfidf_model:
            # إرجاع المتجه مع معلومات إضافية للوضوح
            vector = get_tfidf_vector_serializable(self.tfidf_model, processed_text)
            
            # الحصول على أسماء الكلمات (features)
            feature_names = self.tfidf_model.get_feature_names_out()
            
            # إنشاء قائمة الكلمات الموجودة مع قيمها
            word_scores = []
            for i, score in enumerate(vector):
                if score > 0:  # فقط الكلمات الموجودة
                    word_scores.append({
                        "word": feature_names[i],
                        "tfidf_score": score
                    })
            
            return {
                "vector": vector,
                "vector_size": len(vector),
                "non_zero_count": len(word_scores),
                "words_with_scores": word_scores,
                "processed_text": processed_text
            }
        else:
            # إرجاع النص المعالج
            return processed_text
    
    def _process_embedding_query(self, query: str, return_vector: bool = False):
        """معالجة الاستعلام لـ Embedding"""
        # للـ embeddings، نستخدم النص الأصلي مع معالجة بسيطة
        # لأن sentence-transformers يقوم بالمعالجة الداخلية
        processed_text = query.strip()
        
        if return_vector and self.embedding_model:
            # إرجاع المتجه مع معلومات إضافية للوضوح
            vector = get_embedding_vector_serializable(self.embedding_model, processed_text)
            
            return {
                "vector": vector,
                "vector_size": len(vector),
                "embedding_dimension": len(vector),
                "processed_text": processed_text,
                "note": "Embedding vectors represent semantic meaning, not individual words"
            }
        else:
            # إرجاع النص المعالج
            return processed_text
    
    def _process_hybrid_query(self, query: str, return_vector: bool = False):
        """معالجة الاستعلام للبحث الهجين المتوازي"""
        tfidf_result = self._process_tfidf_query(query, return_vector)
        embedding_result = self._process_embedding_query(query, return_vector)
        
        if return_vector:
            # إرجاع كلا المتجهين مع معلومات إضافية
            return {
                "tfidf": tfidf_result,
                "embedding": embedding_result,
                "method": "hybrid_parallel",
                "note": "Combines TF-IDF (lexical) and Embedding (semantic) representations"
            }
        else:
            # إرجاع النصوص المعالجة
            return {
                "tfidf_text": tfidf_result,
                "embedding_text": embedding_result
            }
    
    def _process_hybrid_sequential_query(self, query: str, return_vector: bool = False):
        """معالجة الاستعلام للبحث الهجين المتسلسل"""
        tfidf_result = self._process_tfidf_query(query, return_vector)
        embedding_result = self._process_embedding_query(query, return_vector)
        
        if return_vector:
            # إرجاع كلا المتجهين مع معلومات إضافية
            return {
                "tfidf": tfidf_result,
                "embedding": embedding_result,
                "method": "hybrid_sequential",
                "note": "Uses TF-IDF first, then refines with Embedding"
            }
        else:
            # إرجاع النصوص المعالجة
            return {
                "tfidf_text": tfidf_result,
                "embedding_text": embedding_result
            }
    
    def get_processing_info(self, query: str) -> dict:
        """
        الحصول على معلومات حول عملية المعالجة
        Args:
            query: النص الأصلي للاستعلام
        Returns:
            dict: معلومات المعالجة
        """
        try:
            return {
                "original_query": query,
                "query_length": len(query),
                "method": self.method,
                "dataset_name": self.dataset_name,
                "models_loaded": {
                    "tfidf_model": self.tfidf_model is not None,
                    "embedding_model": self.embedding_model is not None
                },
                "processing_steps": self._get_processing_steps()
            }
        except Exception as e:
            logger.error(f"خطأ في الحصول على معلومات المعالجة: {e}")
            return {"error": str(e)}
    
    def _get_processing_steps(self) -> list:
        """الحصول على خطوات المعالجة حسب الطريقة"""
        steps = {
            "tfidf": [
                "Tokenization",
                "Lowercasing", 
                "Stop word removal",
                "Lemmatization",
                "TF-IDF vectorization"
            ],
            "embedding": [
                "Text cleaning",
                "Sentence embedding generation"
            ],
            "hybrid": [
                "TF-IDF processing",
                "Embedding processing", 
                "Parallel combination"
            ],
            "hybrid-sequential": [
                "TF-IDF processing",
                "Embedding processing",
                "Sequential combination"
            ]
        }
        
        return steps.get(self.method, ["Unknown method"])
    
    def get_supported_methods(self) -> list:
        """الحصول على الطرق المدعومة"""
        return ["tfidf", "embedding", "hybrid", "hybrid-sequential"]
    
    def validate_method(self, method: str) -> bool:
        """التحقق من صحة الطريقة"""
        return method in self.get_supported_methods()

    def get_query_vector_for_matching(self, query: str):
        """
        إرجاع تمثيل الاستعلام كمتجه مناسب للمطابقة (numpy array 2D)
        حسب الطريقة المختارة.
        """
        if self.method == "tfidf":
            processed_text = preprocess_for_vectorization(query)
            if self.tfidf_model is None:
                raise ValueError("TF-IDF model not loaded")
            # دائماً numpy array 2D
            return get_tfidf_vector(self.tfidf_model, processed_text)
        elif self.method == "embedding":
            processed_text = query.strip()
            if self.embedding_model is None:
                raise ValueError("Embedding model not loaded")
            return get_embedding_vector(self.embedding_model, processed_text).reshape(1, -1)
        elif self.method in ["hybrid", "hybrid-sequential"]:
            processed_tfidf = preprocess_for_vectorization(query)
            processed_embedding = query.strip()
            tfidf_vec = get_tfidf_vector(self.tfidf_model, processed_tfidf)
            embedding_vec = get_embedding_vector(self.embedding_model, processed_embedding).reshape(1, -1)
            return {"tfidf": tfidf_vec, "embedding": embedding_vec}
        else:
            raise ValueError(f"طريقة المعالجة '{self.method}' غير مدعومة للمطابقة")
