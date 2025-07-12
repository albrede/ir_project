from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from services.query_processor import QueryProcessor
from services.matcher import Matcher
from services.vectorization.tfidf_vectorizer import load_tfidf_model
from services.vectorization.embedding_vectorizer import load_embedding_model
from services.database.db_handler import get_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Search & Ranking Service",
    description="خدمة البحث وترتيب النتائج باستخدام مختلف طرق التمثيل",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index"
    dataset_name: str = "simple"
    top_k: int = 10
    alpha: Optional[float] = 0.5  # للبحث الهجين
    first_stage: Optional[str] = "tfidf"  # للبحث المتسلسل
    top_n: Optional[int] = 100  # للبحث المتسلسل
    search_type: Optional[str] = "and"  # للفهرس المقلوب

class RankingRequest(BaseModel):
    query: str
    method: str = "tfidf"
    dataset_name: str = "simple"
    top_k: int = 10
    alpha: Optional[float] = 0.5
    first_stage: Optional[str] = "tfidf"
    top_n: Optional[int] = 100

@app.get("/")
def read_root():
    return {
        "service": "Search & Ranking Service",
        "version": "1.0.0",
        "description": "خدمة البحث وترتيب النتائج باستخدام مختلف طرق التمثيل",
        "endpoints": {
            "POST /search": "البحث في الوثائق",
            "GET /search": "البحث في الوثائق (GET)",
            "POST /rank": "ترتيب النتائج فقط",
            "GET /rank": "ترتيب النتائج فقط (GET)",
            "GET /methods": "طرق البحث المتاحة",
            "GET /health": "فحص صحة الخدمة"
        }
    }

@app.get("/methods")
def get_search_methods():
    return {
        "status": "success",
        "methods": {
            "tfidf": {
                "description": "البحث باستخدام TF-IDF",
                "similarity": "Cosine Similarity",
                "features": "تمثيل معجمي للكلمات"
            },
            "embedding": {
                "description": "البحث باستخدام Sentence Embeddings",
                "similarity": "Cosine Similarity",
                "features": "تمثيل دلالي للجمل"
            },
            "hybrid": {
                "description": "البحث الهجين المتوازي",
                "similarity": "Combined TF-IDF + Embedding",
                "features": "دمج التمثيل المعجمي والدلالي"
            },
            "hybrid-sequential": {
                "description": "البحث الهجين المتسلسل",
                "similarity": "Two-stage ranking",
                "features": "مرحلة أولى ثم تحسين"
            },
            "inverted_index": {
                "description": "البحث باستخدام الفهرس المقلوب",
                "similarity": "Boolean matching",
                "features": "بحث دقيق في النصوص"
            }
        }
    }

def _load_matcher(method: str, dataset_name: str, alpha: float = 0.5):
    try:
        vectorizer, doc_vectors_tfidf, tfidf_doc_ids = load_tfidf_model(dataset_name)
        embedding_model, doc_vectors_embedding, embedding_doc_ids = load_embedding_model(dataset_name)

        if method == "tfidf":
            vectors = doc_vectors_tfidf
            doc_ids = tfidf_doc_ids
            embedding_vectors = None
            hybrid_weights = None
        elif method == "embedding":
            vectors = doc_vectors_embedding
            doc_ids = embedding_doc_ids
            embedding_vectors = None
            hybrid_weights = None
        else:  # hybrid/hybrid-sequential
            vectors = doc_vectors_tfidf
            embedding_vectors = doc_vectors_embedding
            doc_ids = tfidf_doc_ids
            hybrid_weights = (alpha, 1 - alpha)

        matcher = Matcher(
            doc_vectors=vectors,
            vectorizer=vectorizer,
            method=method,
            embedding_model=embedding_model,
            hybrid_weights=hybrid_weights,
            dataset_name=dataset_name,
            embedding_vectors=embedding_vectors
        )

        return matcher, doc_ids

    except Exception as e:
        logger.error(f"خطأ في تحميل المطابق: {e}")
        raise

@app.post("/search")
def search_documents(request: SearchRequest):
    """البحث في الوثائق مع إرجاع المحتوى"""
    try:
        logger.info(f"البحث في الوثائق: {request.query} بطريقة {request.method}")
        
        # تحميل المطابق
        matcher, doc_ids = _load_matcher(
            method=request.method,
            dataset_name=request.dataset_name,
            alpha=request.alpha or 0.5
        )
        
        # معالجة الاستعلام
        query_processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        processed_query = query_processor.process_query(request.query)
        
        # مطابقة الاستعلام مع الوثائق
        matching_results = matcher.match(request.query, top_k=request.top_k)
        
        # تحويل النتائج إلى التنسيق المطلوب
        results = []
        for doc_index, score in matching_results:
            doc_id = doc_ids[doc_index] if doc_ids and doc_index < len(doc_ids) else str(doc_index)
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": len(results) + 1
            })
        
        # الحصول على محتوى الوثائق
        documents = get_documents(request.dataset_name)
        doc_dict = {doc["doc_id"]: doc["text"] for doc in documents}
        
        documents_with_content = []
        for result in results:
            doc_id = result["doc_id"]
            if doc_id in doc_dict:
                documents_with_content.append({
                    "doc_id": doc_id,
                    "text": doc_dict[doc_id],
                    "score": result["score"],
                    "rank": result["rank"]
                })
            else:
                logger.warning(f"الوثيقة {doc_id} غير موجودة")
        
        return {
            "status": "success",
            "search_results": {
                "method": request.method,
                "query": request.query,
                "results": results,
                "total_results": len(results)
            },
            "documents": documents_with_content,
            "total_documents_returned": len(documents_with_content),
            "match_info": matcher.get_match_info(processed_query)
        }
        
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في البحث: {str(e)}"
        )

@app.get("/search")
def search_documents_get(
    query: str = Query(..., description="النص الأصلي للاستعلام"),
    method: str = Query("tfidf", description="طريقة البحث: tfidf, embedding, hybrid, hybrid-sequential, inverted_index"),
    dataset_name: str = Query("simple", description="اسم مجموعة البيانات"),
    top_k: int = Query(10, description="عدد النتائج المطلوبة"),
    alpha: Optional[float] = Query(0.5, description="وزن TF-IDF في البحث الهجين"),
    first_stage: Optional[str] = Query("tfidf", description="المرحلة الأولى في البحث المتسلسل"),
    top_n: Optional[int] = Query(100, description="عدد النتائج المبدئية في البحث المتسلسل"),
    search_type: Optional[str] = Query("and", description="نوع البحث للفهرس المقلوب: and, or")
):
    """البحث في الوثائق مع إرجاع المحتوى (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="الاستعلام لا يمكن أن يكون فارغاً"
            )
        
        # تحميل المطابق
        matcher, doc_ids = _load_matcher(
            method=method,
            dataset_name=dataset_name,
            alpha=alpha or 0.5
        )
        
        # معالجة الاستعلام
        query_processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        processed_query = query_processor.process_query(query)
        
        # مطابقة الاستعلام مع الوثائق
        if method == "inverted_index":
            # للفهرس المقلوب، نستخدم النص الأصلي
            matching_results = matcher.match(query, top_k=top_k)
        else:
            # للطرق الأخرى، نستخدم النص المعالج
            matching_results = matcher.match(processed_query, top_k=top_k)
        
        # تحويل النتائج إلى التنسيق المطلوب
        results = []
        for doc_index, score in matching_results:
            doc_id = doc_ids[doc_index] if doc_ids and doc_index < len(doc_ids) else str(doc_index)
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": len(results) + 1
            })
        
        # الحصول على محتوى الوثائق
        documents = get_documents(dataset_name)
        doc_dict = {doc["doc_id"]: doc["text"] for doc in documents}
        
        documents_with_content = []
        for result in results:
            doc_id = result["doc_id"]
            if doc_id in doc_dict:
                documents_with_content.append({
                    "doc_id": doc_id,
                    "text": doc_dict[doc_id],
                    "score": result["score"],
                    "rank": result["rank"]
                })
            else:
                logger.warning(f"الوثيقة {doc_id} غير موجودة")
        
        return {
            "status": "success",
            "search_results": {
                "method": method,
                "query": query,
                "results": results,
                "total_results": len(results)
            },
            "documents": documents_with_content,
            "total_documents_returned": len(documents_with_content),
            "match_info": matcher.get_match_info(processed_query)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في البحث: {str(e)}"
        )

@app.post("/rank")
def rank_documents(request: RankingRequest):
    """ترتيب النتائج فقط بدون إرجاع المحتوى"""
    try:
        logger.info(f"ترتيب النتائج: {request.query} بطريقة {request.method}")
        
        # تحميل المطابق
        matcher, doc_ids = _load_matcher(
            method=request.method,
            dataset_name=request.dataset_name,
            alpha=request.alpha or 0.5
        )
        
        # معالجة الاستعلام
        query_processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        processed_query = query_processor.process_query(request.query)
        
        # مطابقة الاستعلام مع الوثائق
        matching_results = matcher.match(processed_query, top_k=request.top_k)
        
        # تحويل النتائج إلى التنسيق المطلوب
        results = []
        for doc_index, score in matching_results:
            doc_id = doc_ids[doc_index] if doc_ids and doc_index < len(doc_ids) else str(doc_index)
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": len(results) + 1
            })
        
        return {
            "status": "success",
            "ranking_results": {
                "method": request.method,
                "query": request.query,
                "results": results,
                "total_results": len(results)
            },
            "ranking_info": {
                "method": request.method,
                "dataset": request.dataset_name,
                "total_ranked": len(results),
                "top_score": results[0]["score"] if results else 0,
                "min_score": results[-1]["score"] if results else 0
            },
            # "match_info": matcher.get_match_info(processed_query)
        }
        
    except Exception as e:
        logger.error(f"خطأ في ترتيب النتائج: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في ترتيب النتائج: {str(e)}"
        )

@app.get("/rank")
def rank_documents_get(
    query: str = Query(..., description="النص الأصلي للاستعلام"),
    method: str = Query("tfidf", description="طريقة الترتيب: tfidf, embedding, hybrid, hybrid-sequential"),
    dataset_name: str = Query("simple", description="اسم مجموعة البيانات"),
    top_k: int = Query(10, description="عدد النتائج المطلوبة"),
    alpha: Optional[float] = Query(0.5, description="وزن TF-IDF في البحث الهجين"),
    first_stage: Optional[str] = Query("tfidf", description="المرحلة الأولى في البحث المتسلسل"),
    top_n: Optional[int] = Query(100, description="عدد النتائج المبدئية في البحث المتسلسل")
):
    """ترتيب النتائج فقط بدون إرجاع المحتوى (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="الاستعلام لا يمكن أن يكون فارغاً"
            )
        
        # تحميل المطابق
        matcher, doc_ids = _load_matcher(
            method=method,
            dataset_name=dataset_name,
            alpha=alpha or 0.5
        )
        
        # معالجة الاستعلام
        query_processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        processed_query = query_processor.process_query(query)
        
        # مطابقة الاستعلام مع الوثائق
        matching_results = matcher.match(processed_query, top_k=top_k)
        
        # تحويل النتائج إلى التنسيق المطلوب
        results = []
        for doc_index, score in matching_results:
            doc_id = doc_ids[doc_index] if doc_ids and doc_index < len(doc_ids) else str(doc_index)
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": len(results) + 1
            })
        
        return {
            "status": "success",
            "ranking_results": {
                "method": method,
                "query": query,
                "results": results,
                "total_results": len(results)
            },
            "ranking_info": {
                "method": method,
                "dataset": dataset_name,
                "total_ranked": len(results),
                "top_score": results[0]["score"] if results else 0,
                "min_score": results[-1]["score"] if results else 0
            },
            "match_info": matcher.get_match_info(processed_query)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في ترتيب النتائج: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في ترتيب النتائج: {str(e)}"
        )

@app.get("/health")
def health_check():
    """فحص صحة الخدمة"""
    try:
        # اختبار تحميل النماذج
        query_processor = QueryProcessor(method="tfidf", dataset_name="simple")
        matcher, _ = _load_matcher(method="tfidf", dataset_name="simple")
        
        return {
            "status": "healthy",
            "service": "Search & Ranking Service",
            "version": "1.0.0",
            "supported_methods": matcher.get_supported_methods()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Search & Ranking Service",
            "version": "1.0.0",
            "error": str(e)
        }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002) 