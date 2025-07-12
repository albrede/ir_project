from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from services.query_processor import QueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Query Processing Service",
    description="خدمة معالجة الاستعلامات لتمثيل VSM_TF-IDF و Embeddings",
    version="1.0.0"
)

class QueryProcessingRequest(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf", "embedding", "hybrid", "hybrid-sequential"
    dataset_name: str = "simple"
    return_vector: bool = False

@app.get("/")
def read_root():
    return {
        "service": "Query Processing Service",
        "version": "1.0.0",
        "description": "خدمة معالجة الاستعلامات لتمثيل VSM_TF-IDF و Embeddings",
        "endpoints": {
            "POST /process-query": "معالجة الاستعلام",
            "GET /process-query": "معالجة الاستعلام (GET)",
            "GET /methods": "طرق المعالجة المتاحة",
            "GET /health": "فحص صحة الخدمة"
        }
    }

@app.get("/methods")
def get_processing_methods():
    return {
        "status": "success",
        "methods": {
            "tfidf": {
                "description": "معالجة الاستعلام لـ TF-IDF",
                "preprocessing": "Lemmatization, Tokenization, Lowercasing, إزالة التوقفات",
                "output": "نص معالج أو متجه TF-IDF"
            },
            "embedding": {
                "description": "معالجة الاستعلام لـ Sentence Embeddings",
                "preprocessing": "معالجة بسيطة (sentence-transformers يتعامل مع المعالجة الداخلية)",
                "output": "نص معالج أو متجه Embedding"
            },
            "hybrid": {
                "description": "معالجة الاستعلام للبحث الهجين المتوازي",
                "preprocessing": "مزيج من TF-IDF و Embedding",
                "output": "نصوص معالجة أو متجهات لكلا الطريقتين"
            },
            "hybrid-sequential": {
                "description": "معالجة الاستعلام للبحث الهجين المتسلسل",
                "preprocessing": "مزيج من TF-IDF و Embedding مع مراحل متسلسلة",
                "output": "نصوص معالجة أو متجهات لكلا الطريقتين"
            }
        }
    }

@app.post("/process-query")
def process_query_post(request: QueryProcessingRequest):
    """معالجة الاستعلام (POST)"""
    try:
        logger.info(f"معالجة استعلام: {request.query} بطريقة {request.method}")
        
        # التحقق من صحة الطريقة
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        if not processor.validate_method(request.method):
            raise HTTPException(
                status_code=400,
                detail=f"طريقة المعالجة '{request.method}' غير مدعومة"
            )
        
        processed_result = processor.process_query(
            query=request.query,
            return_vector=request.return_vector
        )
        
        # processing_info = processor.get_processing_info(request.query)
        
        return {
            "status": "success",
            "original_query": request.query,
            "method": request.method,
            "dataset_name": request.dataset_name,
            "processed_result": processed_result,
        }
        
    except Exception as e:
        logger.error(f"خطأ في معالجة الاستعلام: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في معالجة الاستعلام: {str(e)}"
        )

@app.get("/process-query")
def process_query_get(
    query: str = Query(..., description="النص الأصلي للاستعلام"),
    method: str = Query("tfidf", description="طريقة المعالجة: tfidf, embedding, hybrid, hybrid-sequential"),
    dataset_name: str = Query("simple", description="اسم مجموعة البيانات"),
    return_vector: bool = Query(False, description="إرجاع المتجه بدلاً من النص المعالج")
):
    """معالجة الاستعلام (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="الاستعلام لا يمكن أن يكون فارغاً"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        if not processor.validate_method(method):
            raise HTTPException(
                status_code=400,
                detail=f"طريقة المعالجة '{method}' غير مدعومة"
            )
        
        processed_result = processor.process_query(
            query=query,
            return_vector=return_vector
        )
        
        processing_info = processor.get_processing_info(query)
        
        return {
            "status": "success",
            "original_query": query,
            "method": method,
            "dataset_name": dataset_name,
            "processed_result": processed_result,
            "processing_info": processing_info,
            "supported_methods": processor.get_supported_methods()
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
    """فحص صحة الخدمة"""
    try:
        processor = QueryProcessor(method="tfidf", dataset_name="simple")
        return {
            "status": "healthy",
            "service": "Query Processing Service",
            "version": "1.0.0",
            "supported_methods": processor.get_supported_methods()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Query Processing Service",
            "version": "1.0.0",
            "error": str(e)
        }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001) 