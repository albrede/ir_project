from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from services.query_processor import QueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Query Processing Service",
    description="Service for processing queries for VSM_TF-IDF and Embeddings representation",
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
        "description": "Service for processing queries for VSM_TF-IDF and Embeddings representation",
        "endpoints": {
            "POST /process-query": "Process query",
            "GET /process-query": "Process query (GET)",
            "GET /methods": "Available processing methods",
            "GET /health": "Service health check"
        }
    }

@app.get("/methods")
def get_processing_methods():
    return {
        "status": "success",
        "methods": {
            "tfidf": {
                "description": "Query processing for TF-IDF",
                "preprocessing": "Lemmatization, Tokenization, Lowercasing, Stop word removal",
                "output": "Processed text or TF-IDF vector"
            },
            "embedding": {
                "description": "Query processing for Sentence Embeddings",
                "preprocessing": "Simple processing (sentence-transformers handles internal processing)",
                "output": "Processed text or Embedding vector"
            },
            "hybrid": {
                "description": "Query processing for parallel hybrid search",
                "preprocessing": "Combination of TF-IDF and Embedding",
                "output": "Processed texts or vectors for both methods"
            },
            "hybrid-sequential": {
                "description": "Query processing for sequential hybrid search",
                "preprocessing": "Combination of TF-IDF and Embedding with sequential stages",
                "output": "Processed texts or vectors for both methods"
            }
        }
    }

@app.post("/process-query")
def process_query_post(request: QueryProcessingRequest):
    """Process query (POST)"""
    try:
        logger.info(f"Processing query: {request.query} using method {request.method}")
        
        # Validate method
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        if not processor.validate_method(request.method):
            raise HTTPException(
                status_code=400,
                detail=f"Processing method '{request.method}' not supported"
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
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/process-query")
def process_query_get(
    query: str = Query(..., description="Original query text"),
    method: str = Query("tfidf", description="Processing method: tfidf, embedding, hybrid, hybrid-sequential"),
    dataset_name: str = Query("simple", description="Dataset name"),
    return_vector: bool = Query(False, description="Return vector instead of processed text")
):
    """Process query (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        if not processor.validate_method(method):
            raise HTTPException(
                status_code=400,
                detail=f"Processing method '{method}' not supported"
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
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Service health check"""
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