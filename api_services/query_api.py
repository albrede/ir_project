from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from services.query_processor import QueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Information Retrieval API",
    description="Unified API for all information retrieval services",
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
    alpha: Optional[float] = 0.5  # for hybrid search
    first_stage: Optional[str] = "tfidf"  # for sequential search
    top_n: Optional[int] = 100  # for sequential search

@app.get("/")
def read_root():
    return {
        "service": "Unified Information Retrieval Service",
        "version": "1.0.0",
        "description": "Unified API for all information retrieval services",
        "note": "This API combines all services. For separate services, use:",
        "separate_services": {
            "query_processing": "http://localhost:8001",
            "search_ranking": "http://localhost:8002", 
            "indexing": "http://localhost:8003"
        },
        "endpoints": {
            "POST /process-query": "Process query",
            "GET /process-query": "Process query (GET)",
            "POST /search": "Search documents",
            "GET /search": "Search documents (GET)",
            "POST /rank": "Rank results only",
            "GET /rank": "Rank results only (GET)",
            "GET /methods": "Available methods",
            "GET /health": "Service health check"
        }
    }

@app.get("/methods")
def get_available_methods():
    return {
        "status": "success",
        "query_processing_methods": {
            "tfidf": {
                "description": "Query processing for TF-IDF",
                "preprocessing": "Lemmatization, Tokenization, Lowercasing, Stop word removal"
            },
            "embedding": {
                "description": "Query processing for Sentence Embeddings",
                "preprocessing": "Simple processing (sentence-transformers handles internal processing)"
            },
            "hybrid": {
                "description": "Query processing for parallel hybrid search",
                "preprocessing": "Combination of TF-IDF and Embedding"
            },
            "hybrid-sequential": {
                "description": "Query processing for sequential hybrid search",
                "preprocessing": "Combination of TF-IDF and Embedding with sequential stages"
            }
        },
        "search_methods": {
            "tfidf": {
                "description": "Search using TF-IDF",
                "similarity": "Cosine Similarity",
                "features": "Lexical word representation"
            },
            "embedding": {
                "description": "Search using Sentence Embeddings",
                "similarity": "Cosine Similarity",
                "features": "Semantic sentence representation"
            },
            "hybrid": {
                "description": "Parallel hybrid search",
                "similarity": "Combined TF-IDF + Embedding",
                "features": "Combining lexical and semantic representation"
            },
            "hybrid-sequential": {
                "description": "Sequential hybrid search",
                "similarity": "Two-stage ranking",
                "features": "First stage then refinement"
            },
            "inverted_index": {
                "description": "Search using inverted index",
                "similarity": "Boolean matching",
                "features": "Precise text search"
            }
        }
    }

@app.post("/process-query")
def process_query_post(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query} using method {request.method}")
        
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        processed_result = processor.process_query(
            query=request.query,
            return_vector=request.return_vector
        )
        
        processing_info = processor.get_processing_info(request.query)
        
        # Vectors are now automatically serializable from tfidf_vectorizer
        
        return {
            "status": "success",
            "original_query": request.query,
            "method": request.method,
            "dataset_name": request.dataset_name,
            "processed_result": processed_result,
            "processing_info": processing_info
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/search")
def search_documents(request: SearchRequest):
    """Search documents with content return"""
    try:
        logger.info(f"Searching documents: {request.query} using method {request.method}")
        
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        # Search documents
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
        
        # Get document content
        doc_ids = [result["doc_id"] for result in search_results["results"]]
        documents = processor.get_documents_by_ids(doc_ids)
        
        return {
            "status": "success",
            "search_results": search_results,
            "documents": documents,
            "total_documents_returned": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in search: {str(e)}"
        )

@app.post("/rank")
def rank_documents(request: SearchRequest):
    """Rank results only without returning content"""
    try:
        logger.info(f"Ranking results: {request.query} using method {request.method}")
        
        processor = QueryProcessor(
            method=request.method,
            dataset_name=request.dataset_name
        )
        
        # Search and rank results
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
        logger.error(f"Error in ranking: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in ranking: {str(e)}"
        )

@app.get("/search")
def search_documents_get(
    query: str = Query(..., description="Original query text"),
    method: str = Query("tfidf", description="Search method: tfidf, embedding, hybrid, hybrid-sequential, inverted_index"),
    dataset_name: str = Query("simple", description="Dataset name"),
    top_k: int = Query(10, description="Number of results required"),
    alpha: Optional[float] = Query(0.5, description="TF-IDF weight in hybrid search"),
    first_stage: Optional[str] = Query("tfidf", description="First stage in sequential search"),
    top_n: Optional[int] = Query(100, description="Initial number of results in sequential search")
):
    """Search documents with content return (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if method not in ["tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index"]:
            raise HTTPException(
                status_code=400,
                detail=f"Search method '{method}' not supported"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        # Search documents
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
        
        # Get document content
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
        logger.error(f"Error in search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in search: {str(e)}"
        )

@app.get("/rank")
def rank_documents_get(
    query: str = Query(..., description="Original query text"),
    method: str = Query("tfidf", description="Ranking method: tfidf, embedding, hybrid, hybrid-sequential"),
    dataset_name: str = Query("simple", description="Dataset name"),
    top_k: int = Query(10, description="Number of results required"),
    alpha: Optional[float] = Query(0.5, description="TF-IDF weight in hybrid search"),
    first_stage: Optional[str] = Query("tfidf", description="First stage in sequential search"),
    top_n: Optional[int] = Query(100, description="Initial number of results in sequential search")
):
    """Rank results only without returning content (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if method not in ["tfidf", "embedding", "hybrid", "hybrid-sequential"]:
            raise HTTPException(
                status_code=400,
                detail=f"Ranking method '{method}' not supported"
            )
        
        processor = QueryProcessor(
            method=method,
            dataset_name=dataset_name
        )
        
        # Search and rank results
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
        logger.error(f"Error in ranking: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in ranking: {str(e)}"
        )

@app.get("/process-query")
def process_query_get(
    query: str = Query(..., description="Original query text"),
    method: str = Query("tfidf", description="Processing method: tfidf, embedding, hybrid"),
    dataset_name: str = Query("simple", description="Dataset name"),
    return_vector: bool = Query(False, description="Return vector instead of processed text")
):
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if method not in ["tfidf", "embedding", "hybrid", "hybrid-sequential"]:
            raise HTTPException(
                status_code=400,
                detail=f"Processing method '{method}' not supported"
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
        
        # Vectors are now automatically serializable from tfidf_vectorizer
        
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
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
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
