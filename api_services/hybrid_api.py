from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Hybrid Search Service",
    description="Service for hybrid search operations combining TF-IDF and Embedding methods",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelHybridRequest(BaseModel):
    tfidf_query_vec: List[List[float]]
    tfidf_doc_matrix: List[List[float]]
    emb_query_vec: List[List[float]]
    emb_doc_matrix: List[List[float]]
    alpha: Optional[float] = 0.5
    top_k: Optional[int] = 10
    doc_ids: Optional[List[str]] = None

class HybridWeightsRequest(BaseModel):
    tfidf_query_vec: List[List[float]]
    tfidf_doc_matrix: List[List[float]]
    emb_query_vec: List[List[float]]
    emb_doc_matrix: List[List[float]]
    tfidf_weight: Optional[float] = 0.5
    emb_weight: Optional[float] = 0.5
    top_k: Optional[int] = 10
    doc_ids: Optional[List[str]] = None

class HybridSearchRequest(BaseModel):
    dataset_name: str
    query: str
    method: str = "parallel"  # "parallel" or "weights"
    alpha: Optional[float] = 0.5
    tfidf_weight: Optional[float] = 0.5
    emb_weight: Optional[float] = 0.5
    top_k: Optional[int] = 10

@app.get("/")
def read_root():
    return {
        "service": "Hybrid Search Service",
        "version": "1.0.0",
        "description": "Service for hybrid search operations combining TF-IDF and Embedding methods",
        "endpoints": {
            "POST /parallel-hybrid": "Parallel hybrid search with alpha parameter",
            "POST /hybrid-weights": "Hybrid search with custom weights",
            "POST /hybrid-search": "Complete hybrid search with dataset",
            "GET /health": "Health check"
        }
    }

@app.post("/parallel-hybrid")
def parallel_hybrid_search(request: ParallelHybridRequest):
    """Parallel hybrid search with alpha parameter"""
    try:
        # Check matrix sizes before conversion
        tfidf_matrix_shape = (len(request.tfidf_doc_matrix), len(request.tfidf_doc_matrix[0]) if request.tfidf_doc_matrix else 0)
        emb_matrix_shape = (len(request.emb_doc_matrix), len(request.emb_doc_matrix[0]) if request.emb_doc_matrix else 0)
        
        estimated_memory_gb = (tfidf_matrix_shape[0] * tfidf_matrix_shape[1] * 8) / (1024**3)  # 8 bytes for float64
        if estimated_memory_gb > 1.0:
            logger.warning(f"Matrix too large ({tfidf_matrix_shape[0]}x{tfidf_matrix_shape[1]}, ~{estimated_memory_gb:.1f}GB). Using float32.")
            # Convert to float32 to reduce memory usage
            tfidf_query_vec = np.array(request.tfidf_query_vec, dtype=np.float32)
            tfidf_doc_matrix = np.array(request.tfidf_doc_matrix, dtype=np.float32)
            emb_query_vec = np.array(request.emb_query_vec, dtype=np.float32)
            emb_doc_matrix = np.array(request.emb_doc_matrix, dtype=np.float32)
        else:
            # Convert lists to numpy arrays
            tfidf_query_vec = np.array(request.tfidf_query_vec)
            tfidf_doc_matrix = np.array(request.tfidf_doc_matrix)
            emb_query_vec = np.array(request.emb_query_vec)
            emb_doc_matrix = np.array(request.emb_doc_matrix)
        
        logger.info(f"🔍 Calculating similarity using TF-IDF...")
        tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_doc_matrix)[0]
        
        logger.info(f"🔍 Calculating similarity using Embeddings...")
        emb_scores = cosine_similarity(emb_query_vec, emb_doc_matrix)[0]
        
        logger.info(f"🔄 Combining results (alpha={request.alpha})...")
        final_scores = request.alpha * tfidf_scores + (1 - request.alpha) * emb_scores
        
        top_indices = np.argsort(final_scores)[::-1][:request.top_k]
        top_scores = final_scores[top_indices]
        
        logger.info(f"✅ Found {len(top_indices)} results")
        
        # Prepare response
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            result = {
                "rank": i + 1,
                "index": int(idx),
                "score": float(score),
                "tfidf_score": float(tfidf_scores[idx]),
                "embedding_score": float(emb_scores[idx])
            }
            
            if request.doc_ids and idx < len(request.doc_ids):
                result["doc_id"] = request.doc_ids[idx]
            
            results.append(result)
        
        return {
            "status": "success",
            "method": "parallel_hybrid",
            "alpha": request.alpha,
            "top_k": request.top_k,
            "total_results": len(results),
            "results": results,
            "statistics": {
                "tfidf_scores_range": [float(tfidf_scores.min()), float(tfidf_scores.max())],
                "embedding_scores_range": [float(emb_scores.min()), float(emb_scores.max())],
                "final_scores_range": [float(final_scores.min()), float(final_scores.max())]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in parallel hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid-weights")
def hybrid_search_with_weights(request: HybridWeightsRequest):
    """Hybrid search with custom weights"""
    try:
        # Check matrix sizes before conversion
        tfidf_matrix_shape = (len(request.tfidf_doc_matrix), len(request.tfidf_doc_matrix[0]) if request.tfidf_doc_matrix else 0)
        emb_matrix_shape = (len(request.emb_doc_matrix), len(request.emb_doc_matrix[0]) if request.emb_doc_matrix else 0)
        
        estimated_memory_gb = (tfidf_matrix_shape[0] * tfidf_matrix_shape[1] * 8) / (1024**3)  # 8 bytes for float64
        if estimated_memory_gb > 1.0:
            logger.warning(f"Matrix too large ({tfidf_matrix_shape[0]}x{tfidf_matrix_shape[1]}, ~{estimated_memory_gb:.1f}GB). Using float32.")
            # Convert to float32 to reduce memory usage
            tfidf_query_vec = np.array(request.tfidf_query_vec, dtype=np.float32)
            tfidf_doc_matrix = np.array(request.tfidf_doc_matrix, dtype=np.float32)
            emb_query_vec = np.array(request.emb_query_vec, dtype=np.float32)
            emb_doc_matrix = np.array(request.emb_doc_matrix, dtype=np.float32)
        else:
            # Convert lists to numpy arrays
            tfidf_query_vec = np.array(request.tfidf_query_vec)
            tfidf_doc_matrix = np.array(request.tfidf_doc_matrix)
            emb_query_vec = np.array(request.emb_query_vec)
            emb_doc_matrix = np.array(request.emb_doc_matrix)
        
        # Calculate similarities
        tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_doc_matrix)[0]
        emb_scores = cosine_similarity(emb_query_vec, emb_doc_matrix)[0]
        
        # Normalize scores
        tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
        emb_scores = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-8)
        
        # Combine with weights
        final_scores = request.tfidf_weight * tfidf_scores + request.emb_weight * emb_scores
        
        top_indices = np.argsort(final_scores)[::-1][:request.top_k]
        top_scores = final_scores[top_indices]
        
        # Prepare response
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            result = {
                "rank": i + 1,
                "index": int(idx),
                "score": float(score),
                "tfidf_score": float(tfidf_scores[idx]),
                "embedding_score": float(emb_scores[idx])
            }
            
            if request.doc_ids and idx < len(request.doc_ids):
                result["doc_id"] = request.doc_ids[idx]
            
            results.append(result)
        
        return {
            "status": "success",
            "method": "hybrid_weights",
            "tfidf_weight": request.tfidf_weight,
            "emb_weight": request.emb_weight,
            "top_k": request.top_k,
            "total_results": len(results),
            "results": results,
            "statistics": {
                "tfidf_scores_range": [float(tfidf_scores.min()), float(tfidf_scores.max())],
                "embedding_scores_range": [float(emb_scores.min()), float(emb_scores.max())],
                "final_scores_range": [float(final_scores.min()), float(final_scores.max())]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid weights search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid-search")
def hybrid_search(request: HybridSearchRequest):
    """Complete hybrid search with dataset"""
    try:
        # This endpoint would integrate with TF-IDF and Embedding APIs
        # For now, we'll return a placeholder response
        logger.info(f"Hybrid search requested for dataset: {request.dataset_name}")
        
        return {
            "status": "success",
            "method": request.method,
            "dataset": request.dataset_name,
            "query": request.query,
            "message": "This endpoint requires integration with TF-IDF and Embedding APIs",
            "parameters": {
                "alpha": request.alpha,
                "tfidf_weight": request.tfidf_weight,
                "emb_weight": request.emb_weight,
                "top_k": request.top_k
            }
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check for the Hybrid service"""
    try:
        return {
            "status": "healthy",
            "service": "Hybrid Search Service",
            "version": "1.0.0",
            "dependencies": {
                "numpy": "available",
                "sklearn": "available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Hybrid Search Service",
            "version": "1.0.0",
            "error": str(e)
        }

# Client functions for other services to use
def parallel_hybrid_search_via_api(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, 
                                  alpha=0.5, top_k=10, doc_ids=None):
    """Client function to perform parallel hybrid search via API"""
    url = "http://localhost:8004/parallel-hybrid"
    
    payload = {
        "tfidf_query_vec": tfidf_query_vec.tolist() if hasattr(tfidf_query_vec, 'tolist') else tfidf_query_vec,
        "tfidf_doc_matrix": tfidf_doc_matrix.tolist() if hasattr(tfidf_doc_matrix, 'tolist') else tfidf_doc_matrix,
        "emb_query_vec": emb_query_vec.tolist() if hasattr(emb_query_vec, 'tolist') else emb_query_vec,
        "emb_doc_matrix": emb_doc_matrix.tolist() if hasattr(emb_doc_matrix, 'tolist') else emb_doc_matrix,
        "alpha": alpha,
        "top_k": top_k,
        "doc_ids": doc_ids
    }
    
    try:
        import requests
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Extract results
            results = data["results"]
            top_indices = [result["index"] for result in results]
            top_scores = [result["score"] for result in results]
            
            if doc_ids is not None:
                top_doc_ids = [result.get("doc_id", str(result["index"])) for result in results]
                return top_doc_ids, np.array(top_scores)
            else:
                return np.array(top_indices), np.array(top_scores)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to perform parallel hybrid search via API: {e}")

def hybrid_search_with_weights_via_api(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, 
                                      tfidf_weight=0.5, emb_weight=0.5, top_k=10, doc_ids=None):
    """Client function to perform hybrid search with weights via API"""
    url = "http://localhost:8004/hybrid-weights"
    
    payload = {
        "tfidf_query_vec": tfidf_query_vec.tolist() if hasattr(tfidf_query_vec, 'tolist') else tfidf_query_vec,
        "tfidf_doc_matrix": tfidf_doc_matrix.tolist() if hasattr(tfidf_doc_matrix, 'tolist') else tfidf_doc_matrix,
        "emb_query_vec": emb_query_vec.tolist() if hasattr(emb_query_vec, 'tolist') else emb_query_vec,
        "emb_doc_matrix": emb_doc_matrix.tolist() if hasattr(emb_doc_matrix, 'tolist') else emb_doc_matrix,
        "tfidf_weight": tfidf_weight,
        "emb_weight": emb_weight,
        "top_k": top_k,
        "doc_ids": doc_ids
    }
    
    try:
        import requests
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Extract results
            results = data["results"]
            top_indices = [result["index"] for result in results]
            top_scores = [result["score"] for result in results]
            
            if doc_ids is not None:
                top_doc_ids = [result.get("doc_id", str(result["index"])) for result in results]
                return top_doc_ids, np.array(top_scores)
            else:
                return np.array(top_indices), np.array(top_scores)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to perform hybrid search with weights via API: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 