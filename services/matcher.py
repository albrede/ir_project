import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from services.vectorization.tfidf_vectorizer import get_tfidf_vector, load_tfidf_model
from services.vectorization.embedding_vectorizer import get_embedding_vector, load_embedding_model
from services.preprocessing import preprocess_for_vectorization
from services.hybrid import parallel_hybrid_search
from services.hybrid_sequential import sequential_hybrid_search
from services.database.db_handler import get_documents
from services.indexing_service import search_in_index
from services.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class Matcher:
    def __init__(self, doc_vectors, vectorizer=None, method="tfidf", embedding_model=None, hybrid_weights=None, dataset_name="simple", embedding_vectors=None):
        """
        تهيئة مطابق الوثائق
        Args:
            doc_vectors: متجهات الوثائق
            vectorizer: vectorizer للـ TF-IDF
            method: طريقة المطابقة ("tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index")
            embedding_model: نموذج الـ embedding
            hybrid_weights: أوزان للبحث الهجين (tfidf_weight, embedding_weight)
            dataset_name: اسم مجموعة البيانات
        """
        self.doc_vectors = doc_vectors
        self.vectorizer = vectorizer
        self.method = method
        self.embedding_model = embedding_model
        self.hybrid_weights = hybrid_weights or (0.5, 0.5)  # أوزان افتراضية للبحث الهجين
        self.dataset_name = dataset_name
        self.embedding_vectors = embedding_vectors  # جديد
        
        # تحميل النماذج المطلوبة
        self._load_models()
        
        # التحقق من صحة المعاملات
        self._validate_parameters()
        
        self.query_processor = QueryProcessor(method=self.method, dataset_name=self.dataset_name)
    
    def _load_models(self):
        """تحميل النماذج المطلوبة حسب الطريقة"""
        try:
            if self.method in ["tfidf", "hybrid", "hybrid-sequential"] and self.vectorizer is None:
                # تحميل نموذج TF-IDF
                try:
                    self.vectorizer, _, _ = load_tfidf_model(self.dataset_name)
                    logger.info(f"تم تحميل نموذج TF-IDF لـ {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"فشل في تحميل نموذج TF-IDF لـ {self.dataset_name}: {e}")
            
            if self.method in ["embedding", "hybrid", "hybrid-sequential"] and self.embedding_model is None:
                # تحميل نموذج Embedding
                try:
                    self.embedding_model, _, _ = load_embedding_model(self.dataset_name)
                    logger.info(f"تم تحميل نموذج Embedding لـ {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"فشل في تحميل نموذج Embedding لـ {self.dataset_name}: {e}")
                    
        except Exception as e:
            logger.error(f"خطأ في تحميل النماذج: {e}")
    
    def _validate_parameters(self):
        """التحقق من صحة المعاملات"""
        if self.method == "tfidf" and self.vectorizer is None:
            raise ValueError("vectorizer مطلوب لطريقة TF-IDF")
        
        if self.method == "embedding" and self.embedding_model is None:
            raise ValueError("embedding_model مطلوب لطريقة Embedding")
        
        if self.method in ["hybrid", "hybrid-sequential"]:
            if self.vectorizer is None or self.embedding_model is None:
                raise ValueError("كلا vectorizer و embedding_model مطلوبان للبحث الهجين")
            
            if len(self.hybrid_weights) != 2:
                raise ValueError("hybrid_weights يجب أن يحتوي على قيمتين")
            
            if sum(self.hybrid_weights) != 1.0:
                logger.warning("مجموع أوزان البحث الهجين لا يساوي 1.0، سيتم تطبيعها")
                total = sum(self.hybrid_weights)
                self.hybrid_weights = (self.hybrid_weights[0]/total, self.hybrid_weights[1]/total)

    def match(self, query, top_k=None):
        """
        مطابقة الاستعلام مع الوثائق
        Args:
            query: الاستعلام (نص أو متجه)
            top_k: عدد النتائج المطلوبة (None لجميع النتائج)
        Returns:
            قائمة من tuples (doc_index, similarity_score) مرتبة تنازلياً
        """
        try:
            # استخدم QueryProcessor لتمثيل الاستعلام
            if self.method in ["tfidf", "embedding"]:
                query_vec = self.query_processor.get_query_vector_for_matching(query)
                if self.method == "tfidf":
                    return self._match_tfidf(query_vec, top_k, already_vectorized=True)
                else:
                    return self._match_embedding(query_vec, top_k, already_vectorized=True)
            elif self.method == "hybrid":
                query_vecs = self.query_processor.get_query_vector_for_matching(query)
                return self._match_hybrid(query_vecs, top_k, already_vectorized=True)
            elif self.method == "hybrid-sequential":
                query_vecs = self.query_processor.get_query_vector_for_matching(query)
                return self._match_hybrid_sequential(query_vecs, top_k, already_vectorized=True)
            elif self.method == "inverted_index":
                return self._match_inverted_index(query, top_k)
            else:
                raise ValueError(f"طريقة المطابقة '{self.method}' غير مدعومة")
                
        except Exception as e:
            logger.error(f"خطأ في مطابقة الاستعلام: {e}")
            raise
    
    def _match_tfidf(self, query_vec, top_k=None, already_vectorized=False):
        """مطابقة TF-IDF بنفس منطق search_api.py"""
        if not already_vectorized:
            if self.vectorizer is None:
                raise ValueError("vectorizer مطلوب لطريقة TF-IDF")
            if isinstance(query_vec, str):
                processed_query = preprocess_for_vectorization(query_vec)
                query_vec = get_tfidf_vector(self.vectorizer, processed_query)
            else:
                query_vec = query_vec.reshape(1, -1)
        # الآن: query_vec و self.doc_vectors كلاهما sparse
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        if top_k is None:
            top_k = len(sims)
        top_indices = sims.argsort()[-top_k:][::-1]
        results = [(int(idx), float(sims[idx])) for idx in top_indices]
        return results
    
    def _match_embedding(self, query_vec, top_k=None, already_vectorized=False):
        """مطابقة Embedding"""
        if not already_vectorized:
            if self.embedding_model is None:
                raise ValueError("embedding_model مطلوب لطريقة Embedding")
            if isinstance(query_vec, str):
                query_vec = get_embedding_vector(self.embedding_model, query_vec)
                query_vec = query_vec.reshape(1, -1)
            else:
                query_vec = query_vec.reshape(1, -1)
        
        sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
        return self._rank_results(sims, top_k)
    
    def _match_hybrid(self, query_vecs, top_k=None, already_vectorized=False):
        """مطابقة هجينة تجمع بين TF-IDF و Embedding"""
        if not already_vectorized:
            # الاستعلام معالج مسبقاً
            pass
        tfidf_vec = query_vecs["tfidf"]
        embedding_vec = query_vecs["embedding"]
        tfidf_scores = cosine_similarity(tfidf_vec, self.doc_vectors).flatten()
        if self.embedding_vectors is not None:
            embedding_scores = cosine_similarity(embedding_vec, self.embedding_vectors).flatten()
        else:
            embedding_scores = np.zeros_like(tfidf_scores)
        tfidf_weight, embedding_weight = self.hybrid_weights
        combined_scores = tfidf_weight * tfidf_scores + embedding_weight * embedding_scores
        
        return self._rank_results(combined_scores, top_k, doc_count=self.doc_vectors.shape[0])
    
    def _match_hybrid_sequential(self, query_vecs, top_k=None, already_vectorized=False, first_stage='tfidf', top_n=100, doc_ids=None):
        """مطابقة هجينة متسلسلة بنفس منطق sequential_hybrid_search في hybrid_sequential.py"""
        tfidf_vec = query_vecs["tfidf"]
        embedding_vec = query_vecs["embedding"]
        tfidf_matrix = self.doc_vectors
        emb_matrix = self.embedding_vectors
        if first_stage == 'tfidf':
            first_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
            top_n_indices = np.argsort(first_scores)[::-1][:top_n]
            emb_sub_matrix = emb_matrix[top_n_indices]
            emb_scores = cosine_similarity(embedding_vec, emb_sub_matrix)[0]
            final_indices = np.argsort(emb_scores)[::-1][:top_k]
            top_indices = top_n_indices[final_indices]
            top_scores = emb_scores[final_indices]
        else:
            first_scores = cosine_similarity(embedding_vec, emb_matrix)[0]
            top_n_indices = np.argsort(first_scores)[::-1][:top_n]
            tfidf_sub_matrix = tfidf_matrix[top_n_indices]
            tfidf_scores = cosine_similarity(tfidf_vec, tfidf_sub_matrix)[0]
            final_indices = np.argsort(tfidf_scores)[::-1][:top_k]
            top_indices = top_n_indices[final_indices]
            top_scores = tfidf_scores[final_indices]
        results = list(zip(top_indices, top_scores))
        return results
    
    def _match_inverted_index(self, query, top_k=None):
        """مطابقة باستخدام الفهرس المقلوب"""
        if isinstance(query, str):
            # البحث في الفهرس المقلوب
            matching_doc_ids = search_in_index(query, self.dataset_name, top_k or 10, "and")
            
            # تحويل معرفات الوثائق إلى مؤشرات
            results = []
            for i, doc_id in enumerate(matching_doc_ids):
                # البحث عن المؤشر في قائمة معرفات الوثائق (إذا كانت متوفرة)
                try:
                    doc_index = int(doc_id)
                    results.append((doc_index, 1.0))  # درجة ثابتة للفهرس المقلوب
                except ValueError:
                    # إذا كان معرف الوثيقة ليس رقماً، استخدم المؤشر
                    results.append((i, 1.0))
            
            return results
        else:
            raise ValueError("الفهرس المقلوب يتطلب نص استعلام")
    
    def _rank_results(self, similarities, top_k=None, doc_indices=None, doc_count=None):
        """ترتيب النتائج حسب درجة التشابه"""
        if len(similarities) == 0:
            return []
        
        # ترتيب تنازلي
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_scores = similarities[ranked_indices]
        
        # تطبيق top_k إذا تم تحديده
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]
            ranked_scores = ranked_scores[:top_k]
        
        # إذا تم توفير معرفات وثائق مخصصة، استخدمها
        if doc_indices is not None:
            return list(zip([doc_indices[i] for i in ranked_indices], ranked_scores))
        else:
            # استخدم doc_count إذا كان متوفراً بدلاً من len(self.doc_vectors)
            return list(zip(ranked_indices, ranked_scores))
    
    def get_match_info(self, query):
        """الحصول على معلومات حول عملية المطابقة"""
        return {
            "method": self.method,
            "documents_count": self.doc_vectors.shape[0],
            "query_type": type(query).__name__,
            "hybrid_weights": self.hybrid_weights if self.method in ["hybrid", "hybrid-sequential"] else None,
            "dataset_name": self.dataset_name,
            "models_loaded": {
                "vectorizer": self.vectorizer is not None,
                "embedding_model": self.embedding_model is not None
            }
        }
    
    def get_supported_methods(self):
        """الحصول على الطرق المدعومة"""
        return ["tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index"]
    
    def validate_method(self, method):
        """التحقق من صحة الطريقة"""
        return method in self.get_supported_methods()
