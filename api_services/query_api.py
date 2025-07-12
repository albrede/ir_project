from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from services.query_processor import QueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Information Retrieval API",
    description="API موحد لجميع خدمات استرجاع المعلومات",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf", "embedding", "hybrid", "hybrid-sequential"
    dataset_name: str = "simple"
    return_vector: bool = False

class SearchRequest(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf", "embedding", "hybrid", "hybrid-sequential"
    dataset_name: str = "simple"
    top_k: int = 10
    alpha: Optional[float] = 0.5  # للبحث الهجين
    first_stage: Optional[str] = "tfidf"  # للبحث المتسلسل
    top_n: Optional[int] = 100  # للبحث المتسلسل

@app.get("/")
def read_root():
    return {
        "service": "Unified Information Retrieval Service",
        "version": "1.0.0",
        "description": "API موحد لجميع خدمات استرجاع المعلومات",
        "note": "هذا API يجمع جميع الخدمات. للخدمات المنفصلة، استخدم:",
        "separate_services": {
            "query_processing": "http://localhost:8001",
            "search_ranking": "http://localhost:8002", 
            "indexing": "http://localhost:8003"
        },
        "endpoints": {
            "POST /process-query": "معالجة الاستعلام",
            "GET /process-query": "معالجة الاستعلام (GET)",
            "POST /search": "البحث في الوثائق",
            "GET /search": "البحث في الوثائق (GET)",
            "POST /rank": "ترتيب النتائج فقط",
            "GET /rank": "ترتيب النتائج فقط (GET)",
            "GET /methods": "الطرق المتاحة",
            "GET /health": "فحص صحة الخدمة"
        }
    }

@app.get("/methods")
def get_available_methods():
    return {
        "status": "success",
        "query_processing_methods": {
            "tfidf": {
                "description": "معالجة الاستعلام لـ TF-IDF",
                "preprocessing": "Lemmatization, Tokenization, Lowercasing, إزالة التوقفات"
            },
            "embedding": {
                "description": "معالجة الاستعلام لـ Sentence Embeddings",
                "preprocessing": "معالجة بسيطة (sentence-transformers يتعامل مع المعالجة الداخلية)"
            },
            "hybrid": {
                "description": "معالجة الاستعلام للبحث الهجين المتوازي",
                "preprocessing": "مزيج من TF-IDF و Embedding"
            },
            "hybrid-sequential": {
                "description": "معالجة الاستعلام للبحث الهجين المتسلسل",
                "preprocessing": "مزيج من TF-IDF و Embedding مع مراحل متسلسلة"
            }
        },
        "search_methods": {
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

@app.post("/process-query")
def process_query_post(request: QueryRequest):
    try:
        logger.info(f"معالجة استعلام: {request.query} بطريقة {request.method}")
        
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        processed_result = processor.process_query(
            query=request.query,
            return_vector=request.return_vector
        )
        
        processing_info = processor.get_processing_info(request.query)
        
        # المتجهات الآن قابلة للتسلسل تلقائياً من tfidf_vectorizer
        
        return {
            "status": "success",
            "original_query": request.query,
            "method": request.method,
            "dataset_name": request.dataset_name,
            "processed_result": processed_result,
            "processing_info": processing_info
        }
        
    except Exception as e:
        logger.error(f"خطأ في معالجة الاستعلام: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في معالجة الاستعلام: {str(e)}"
        )

@app.post("/search")
def search_documents(request: SearchRequest):
    """البحث في الوثائق مع إرجاع المحتوى"""
    try:
        logger.info(f"البحث في الوثائق: {request.query} بطريقة {request.method}")
        
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        # البحث في الوثائق
        if request.method == "inverted_index":
            search_results = processor.search_with_index(
                query=request.query,
                top_k=request.top_k,
                search_type="and"
            )
        else:
            search_results = processor.search(
                query=request.query,
                top_k=request.top_k,
                alpha=request.alpha or 0.5,
                first_stage=request.first_stage or "tfidf",
                top_n=request.top_n or 100
            )
        
        # الحصول على محتوى الوثائق
        doc_ids = [result["doc_id"] for result in search_results["results"]]
        documents = processor.get_documents_by_ids(doc_ids)
        
        return {
            "status": "success",
            "search_results": search_results,
            "documents": documents,
            "total_documents_returned": len(documents)
        }
        
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في البحث: {str(e)}"
        )

@app.post("/rank")
def rank_documents(request: SearchRequest):
    """ترتيب النتائج فقط بدون إرجاع المحتوى"""
    try:
        logger.info(f"ترتيب النتائج: {request.query} بطريقة {request.method}")
        
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        # البحث وترتيب النتائج
        search_results = processor.search(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha or 0.5,
            first_stage=request.first_stage or "tfidf",
            top_n=request.top_n or 100
        )
        
        return {
            "status": "success",
            "ranking_results": search_results,
            "ranking_info": {
                "method": request.method,
                "dataset": request.dataset_name,
                "total_ranked": len(search_results["results"]),
                "top_score": search_results["results"][0]["score"] if search_results["results"] else 0,
                "min_score": search_results["results"][-1]["score"] if search_results["results"] else 0
            }
        }
        
    except Exception as e:
        logger.error(f"خطأ في ترتيب النتائج: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في ترتيب النتائج: {str(e)}"
        )

@app.get("/search")
def search_documents_get(
    query: str = Query(..., description="النص الأصلي للاستعلام"),
    method: str = Query("tfidf", description="طريقة البحث: tfidf, embedding, hybrid, hybrid-sequential, inverted_index"),
    dataset_name: str = Query("simple", description="اسم مجموعة البيانات"),
    top_k: int = Query(10, description="عدد النتائج المطلوبة"),
    alpha: Optional[float] = Query(0.5, description="وزن TF-IDF في البحث الهجين"),
    first_stage: Optional[str] = Query("tfidf", description="المرحلة الأولى في البحث المتسلسل"),
    top_n: Optional[int] = Query(100, description="عدد النتائج المبدئية في البحث المتسلسل")
):
    """البحث في الوثائق مع إرجاع المحتوى (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="الاستعلام لا يمكن أن يكون فارغاً"
            )
        
        if method not in ["tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index"]:
            raise HTTPException(
                status_code=400,
                detail=f"طريقة البحث '{method}' غير مدعومة"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        # البحث في الوثائق
        if method == "inverted_index":
            search_results = processor.search_with_index(
                query=query,
                top_k=top_k,
                search_type="and"
            )
        else:
            search_results = processor.search(
                query=query,
                top_k=top_k,
                alpha=alpha or 0.5,
                first_stage=first_stage or "tfidf",
                top_n=top_n or 100
            )
        
        # الحصول على محتوى الوثائق
        doc_ids = [result["doc_id"] for result in search_results["results"]]
        documents = processor.get_documents_by_ids(doc_ids)
        
        return {
            "status": "success",
            "search_results": search_results,
            "documents": documents,
            "total_documents_returned": len(documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في البحث: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في البحث: {str(e)}"
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
        
        if method not in ["tfidf", "embedding", "hybrid", "hybrid-sequential"]:
            raise HTTPException(
                status_code=400,
                detail=f"طريقة الترتيب '{method}' غير مدعومة"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        # البحث وترتيب النتائج
        search_results = processor.search(
            query=query,
            top_k=top_k,
            alpha=alpha or 0.5,
            first_stage=first_stage or "tfidf",
            top_n=top_n or 100
        )
        
        return {
            "status": "success",
            "ranking_results": search_results,
            "ranking_info": {
                "method": method,
                "dataset": dataset_name,
                "total_ranked": len(search_results["results"]),
                "top_score": search_results["results"][0]["score"] if search_results["results"] else 0,
                "min_score": search_results["results"][-1]["score"] if search_results["results"] else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في ترتيب النتائج: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في ترتيب النتائج: {str(e)}"
        )

@app.get("/process-query")
def process_query_get(
    query: str = Query(..., description="النص الأصلي للاستعلام"),
    method: str = Query("tfidf", description="طريقة المعالجة: tfidf, embedding, hybrid"),
    dataset_name: str = Query("simple", description="اسم مجموعة البيانات"),
    return_vector: bool = Query(False, description="إرجاع المتجه بدلاً من النص المعالج")
):
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="الاستعلام لا يمكن أن يكون فارغاً"
            )
        
        if method not in ["tfidf", "embedding", "hybrid", "hybrid-sequential"]:
            raise HTTPException(
                status_code=400,
                detail=f"طريقة المعالجة '{method}' غير مدعومة"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        processed_result = processor.process_query(
            query=query,
            return_vector=return_vector
        )
        
        processing_info = processor.get_processing_info(query)
        
        # المتجهات الآن قابلة للتسلسل تلقائياً من tfidf_vectorizer
        
        return {
            "status": "success",
            "original_query": query,
            "method": method,
            "dataset_name": dataset_name,
            "processed_result": processed_result,
            "processing_info": processing_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ في معالجة الاستعلام: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في معالجة الاستعلام: {str(e)}"
        )

@app.get("/health")
def health_check():
    try:
        processor = QueryProcessor(method="tfidf", dataset_name="simple")
        return {
            "status": "healthy",
            "service": "Query Processing Service",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Query Processing Service",
            "version": "1.0.0",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
