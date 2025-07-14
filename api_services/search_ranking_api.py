from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import requests
import requests
from services.matcher import Matcher
import requests
from services.database.db_handler import get_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tfidf_model_via_api(dataset_name: str):
    """Client function to load TF-IDF model via API"""
    url = "http://localhost:8002/load-tfidf"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            import os
            import joblib
            
            clean_dataset_name = dataset_name.replace('/', '_')
            model_path = os.path.join("models", f"{clean_dataset_name}_tfidf_model.joblib")
            matrix_path = os.path.join("vectors", f"{clean_dataset_name}_tfidf_matrix.joblib")
            
            vectorizer = joblib.load(model_path)
            data = joblib.load(matrix_path)
            tfidf_matrix = data["matrix"]
            doc_ids = data["doc_ids"]
            
            return vectorizer, tfidf_matrix, doc_ids
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load TF-IDF model via API: {e}")

def get_tfidf_vector_via_api(dataset_name: str, query: str):
    """Client function to get TF-IDF vector via API"""
    url = "http://localhost:8002/vectorize"
    
    payload = {
        "dataset_name": dataset_name,
        "query": query,
        "return_serializable": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            import numpy as np
            return np.array(data["vector"]).reshape(1, -1)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get TF-IDF vector via API: {e}")

def get_tfidf_vector_serializable_via_api(dataset_name: str, query: str):
    """Client function to get serializable TF-IDF vector via API"""
    url = "http://localhost:8002/vectorize"
    
    payload = {
        "dataset_name": dataset_name,
        "query": query,
        "return_serializable": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["vector"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get serializable TF-IDF vector via API: {e}")

def load_embedding_model_via_api(dataset_name: str):
    """Client function to load Embedding model via API"""
    url = "http://localhost:8003/load-embedding"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            import os
            import joblib
            
            clean_dataset_name = dataset_name.replace('/', '_')
            model_path = os.path.join("models", f"{clean_dataset_name}_embedding_model.joblib")
            vectors_path = os.path.join("vectors", f"{clean_dataset_name}_embedding_vectors.joblib")
            
            model = joblib.load(model_path)
            data = joblib.load(vectors_path)
            embeddings = data.get("embeddings", data.get("vectors"))
            doc_ids = data["doc_ids"]
            
            return model, embeddings, doc_ids
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load Embedding model via API: {e}")

def process_query_via_api(query: str, method: str, dataset_name: str):
    """Client function to process query via Query Processor API"""
    url = "http://localhost:8006/process-query"
    
    payload = {
        "query": query,
        "method": method,
        "dataset_name": dataset_name,
        "return_vector": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["processed_text"] if "processed_text" in data else data.get("tfidf_text", query)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to process query via Query Processor API: {e}")

app = FastAPI(
    title="Search & Ranking Service",
    description="Search and ranking service using various representation methods",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index"
    dataset_name: str = "simple"
    top_k: int = 10
    alpha: Optional[float] = 0.5  # for hybrid search
    first_stage: Optional[str] = "tfidf"  # for sequential search
    top_n: Optional[int] = 100  # for sequential search
    search_type: Optional[str] = "and"  # for inverted index

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
        "description": "Search and ranking service using various representation methods",
        "endpoints": {
            "POST /search": "Search documents",
            "GET /search": "Search documents (GET)",
            "POST /rank": "Rank results only",
            "GET /rank": "Rank results only (GET)",
            "GET /methods": "Available search methods",
            "GET /health": "Service health check"
        }
    }

@app.get("/methods")
def get_search_methods():
    return {
        "status": "success",
        "methods": {
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

def _load_matcher(method: str, dataset_name: str, alpha: float = 0.5):
    try:
        vectorizer, doc_vectors_tfidf, tfidf_doc_ids = load_tfidf_model_via_api(dataset_name)
        embedding_model, doc_vectors_embedding, embedding_doc_ids = load_embedding_model_via_api(dataset_name)

        # فحص حجم المصفوفات
        tfidf_shape = doc_vectors_tfidf.shape if hasattr(doc_vectors_tfidf, 'shape') else (0, 0)
        emb_shape = doc_vectors_embedding.shape if hasattr(doc_vectors_embedding, 'shape') else (0, 0)
        
        estimated_memory_gb = (tfidf_shape[0] * tfidf_shape[1] * 8) / (1024**3)  # 8 bytes for float64
        if estimated_memory_gb > 1.0:
            logger.warning(f"Matrix too large ({tfidf_shape[0]}x{tfidf_shape[1]}, ~{estimated_memory_gb:.1f}GB). Using local method for hybrid search.")
            # For large matrices, use local method directly
            method = method.replace("hybrid", "local_hybrid")

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
        elif method in ["hybrid", "hybrid-sequential"]:
            vectors = doc_vectors_tfidf
            embedding_vectors = doc_vectors_embedding
            doc_ids = tfidf_doc_ids
            hybrid_weights = (alpha, 1 - alpha)
        elif method in ["local_hybrid", "local_hybrid-sequential"]:
            # For local methods, use the same settings
            vectors = doc_vectors_tfidf
            embedding_vectors = doc_vectors_embedding
            doc_ids = tfidf_doc_ids
            hybrid_weights = (alpha, 1 - alpha)
            # Convert method to original method
            method = method.replace("local_", "")

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
        logger.error(f"Error loading matcher: {e}")
        raise

@app.post("/search")
def search_documents(request: SearchRequest):
    """Search documents with content return"""
    try:
        logger.info(f"Searching documents: {request.query} using method {request.method}")
        
        # Load matcher
        matcher, doc_ids = _load_matcher(
            method=request.method,
            dataset_name=request.dataset_name,
            alpha=request.alpha or 0.5
        )
        
        # Process query via API
        processed_query = process_query_via_api(request.query, request.method, request.dataset_name)
        
        # Match query with documents
        matching_results = matcher.match(request.query, top_k=request.top_k)
        
        # Convert results to required format
        results = []
        for doc_index, score in matching_results:
            doc_id = doc_ids[doc_index] if doc_ids and doc_index < len(doc_ids) else str(doc_index)
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": len(results) + 1
            })
        
        # Get document content
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
                logger.warning(f"Document {doc_id} not found")
        
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
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )

@app.get("/search")
def search_documents_get(
    query: str = Query(..., description="Original query text"),
    method: str = Query("tfidf", description="Search method: tfidf, embedding, hybrid, hybrid-sequential, inverted_index"),
    dataset_name: str = Query("simple", description="Dataset name"),
    top_k: int = Query(10, description="Number of results required"),
    alpha: Optional[float] = Query(0.5, description="TF-IDF weight in hybrid search"),
    first_stage: Optional[str] = Query("tfidf", description="First stage in sequential search"),
    top_n: Optional[int] = Query(100, description="Initial number of results in sequential search"),
    search_type: Optional[str] = Query("and", description="Search type for inverted index: and, or")
):
    """Search documents with content return (GET)"""
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Load matcher
        matcher, doc_ids = _load_matcher(
            method=method,
            dataset_name=dataset_name,
            alpha=alpha or 0.5
        )
        
        # Process query via API
        processed_query = process_query_via_api(query, method, dataset_name)
        
        # Match query with documents
        if method == "inverted_index":
            # For inverted index, use original text
            matching_results = matcher.match(query, top_k=top_k)
        else:
            # For other methods, use processed text
            matching_results = matcher.match(processed_query, top_k=top_k)
        
        # Convert results to required format
        results = []
        for doc_index, score in matching_results:
            doc_id = doc_ids[doc_index] if doc_ids and doc_index < len(doc_ids) else str(doc_index)
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "rank": len(results) + 1
            })
        
        # Get document content
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
                logger.warning(f"Document {doc_id} not found")
        
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
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )

@app.post("/rank")
def rank_documents(request: RankingRequest):
    """Rank results only without returning content"""
    try:
        logger.info(f"Ranking results: {request.query} using method {request.method}")
        
        # Load matcher
        matcher, doc_ids = _load_matcher(
            method=request.method,
            dataset_name=request.dataset_name,
            alpha=request.alpha or 0.5
        )
        
        # Process query via API
        processed_query = process_query_via_api(request.query, request.method, request.dataset_name)
        
        # Match query with documents
        matching_results = matcher.match(processed_query, top_k=request.top_k)
        
        # Convert results to required format
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
        logger.error(f"Error ranking results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ranking results: {str(e)}"
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
        
        # Load matcher
        matcher, doc_ids = _load_matcher(
            method=method,
            dataset_name=dataset_name,
            alpha=alpha or 0.5
        )
        
        # Process query via API
        processed_query = process_query_via_api(query, method, dataset_name)
        
        # Match query with documents
        matching_results = matcher.match(processed_query, top_k=top_k)
        
        # Convert results to required format
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
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ranking results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ranking results: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Service health check"""
    try:
        # Test model loading
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