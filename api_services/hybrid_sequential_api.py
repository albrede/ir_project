from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Hybrid Sequential Search Service",
    description="Service for sequential hybrid search operations combining TF-IDF and Embedding methods in two stages",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequentialHybridRequest(BaseModel):
    tfidf_query_vec: List[List[float]]
    tfidf_doc_matrix: List[List[float]]
    emb_query_vec: List[List[float]]
    emb_doc_matrix: List[List[float]]
    first_stage: Optional[str] = "tfidf"  # "tfidf" or "embedding"
    top_n: Optional[int] = 100
    top_k: Optional[int] = 10
    doc_ids: Optional[List[str]] = None

class SequentialHybridSearchRequest(BaseModel):
    dataset_name: str
    query: str
    first_stage: Optional[str] = "tfidf"
    top_n: Optional[int] = 100
    top_k: Optional[int] = 10

@app.get("/")
def read_root():
    return {
        "service": "Hybrid Sequential Search Service",
        "version": "1.0.0",
        "description": "Service for sequential hybrid search operations combining TF-IDF and Embedding methods in two stages",
        "endpoints": {
            "POST /sequential-hybrid": "Sequential hybrid search with two-stage ranking",
            "POST /sequential-hybrid-search": "Complete sequential hybrid search with dataset",
            "GET /health": "Health check"
        }
    }

@app.post("/sequential-hybrid")
def sequential_hybrid_search(request: SequentialHybridRequest):
    """Sequential hybrid search with two-stage ranking"""
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
        
        logger.info(f"🔍 Starting sequential hybrid search with first stage: {request.first_stage}")
        
        if request.first_stage == 'tfidf':
            logger.info("📊 Stage 1: TF-IDF ranking...")
            first_scores = cosine_similarity(tfidf_query_vec, tfidf_doc_matrix)[0]
            top_n_indices = np.argsort(first_scores)[::-1][:request.top_n]
            
            logger.info(f"📊 Stage 2: Embedding ranking on top {len(top_n_indices)} documents...")
            emb_sub_matrix = emb_doc_matrix[top_n_indices]
            emb_scores = cosine_similarity(emb_query_vec, emb_sub_matrix)[0]
            
            final_indices = np.argsort(emb_scores)[::-1][:request.top_k]
            top_indices = top_n_indices[final_indices]
            top_scores = emb_scores[final_indices]
            
            stage1_scores = first_scores[top_n_indices]
            stage2_scores = emb_scores
            
        else:  # first_stage == 'embedding'
            logger.info("📊 Stage 1: Embedding ranking...")
            first_scores = cosine_similarity(emb_query_vec, emb_doc_matrix)[0]
            top_n_indices = np.argsort(first_scores)[::-1][:request.top_n]
            
            logger.info(f"📊 Stage 2: TF-IDF ranking on top {len(top_n_indices)} documents...")
            tfidf_sub_matrix = tfidf_doc_matrix[top_n_indices]
            tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_sub_matrix)[0]
            
            final_indices = np.argsort(tfidf_scores)[::-1][:request.top_k]
            top_indices = top_n_indices[final_indices]
            top_scores = tfidf_scores[final_indices]
            
            stage1_scores = first_scores[top_n_indices]
            stage2_scores = tfidf_scores
        
        logger.info(f"✅ Found {len(top_indices)} results after two-stage ranking")
        
        # Prepare response
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            result = {
                "rank": i + 1,
                "index": int(idx),
                "score": float(score),
                "stage1_score": float(stage1_scores[final_indices[i]] if i < len(final_indices) else 0),
                "stage2_score": float(stage2_scores[final_indices[i]] if i < len(final_indices) else 0)
            }
            
            if request.doc_ids and idx < len(request.doc_ids):
                result["doc_id"] = request.doc_ids[idx]
            
            results.append(result)
        
        return {
            "status": "success",
            "method": "sequential_hybrid",
            "first_stage": request.first_stage,
            "top_n": request.top_n,
            "top_k": request.top_k,
            "total_results": len(results),
            "results": results,
            "statistics": {
                "stage1_scores_range": [float(stage1_scores.min()), float(stage1_scores.max())],
                "stage2_scores_range": [float(stage2_scores.min()), float(stage2_scores.max())],
                "final_scores_range": [float(top_scores.min()), float(top_scores.max())],
                "documents_processed_stage1": len(top_n_indices),
                "documents_processed_stage2": len(top_indices)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in sequential hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sequential-hybrid-search")
def sequential_hybrid_search_with_dataset(request: SequentialHybridSearchRequest):
    """Complete sequential hybrid search with dataset"""
    try:
        # This endpoint would integrate with TF-IDF and Embedding APIs
        # For now, we'll return a placeholder response
        logger.info(f"Sequential hybrid search requested for dataset: {request.dataset_name}")
        
        return {
            "status": "success",
            "method": "sequential_hybrid",
            "dataset": request.dataset_name,
            "query": request.query,
            "message": "This endpoint requires integration with TF-IDF and Embedding APIs",
            "parameters": {
                "first_stage": request.first_stage,
                "top_n": request.top_n,
                "top_k": request.top_k
            }
        }
        
    except Exception as e:
        logger.error(f"Error in sequential hybrid search with dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check for the Hybrid Sequential service"""
    try:
        return {
            "status": "healthy",
            "service": "Hybrid Sequential Search Service",
            "version": "1.0.0",
            "dependencies": {
                "numpy": "available",
                "sklearn": "available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Hybrid Sequential Search Service",
            "version": "1.0.0",
            "error": str(e)
        }

# Client functions for other services to use
def sequential_hybrid_search_via_api(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, 
                                   first_stage='tfidf', top_n=100, top_k=10, doc_ids=None):
    """Client function to perform sequential hybrid search via API"""
    url = "http://localhost:8005/sequential-hybrid"
    
    payload = {
        "tfidf_query_vec": tfidf_query_vec.tolist() if hasattr(tfidf_query_vec, 'tolist') else tfidf_query_vec,
        "tfidf_doc_matrix": tfidf_doc_matrix.tolist() if hasattr(tfidf_doc_matrix, 'tolist') else tfidf_doc_matrix,
        "emb_query_vec": emb_query_vec.tolist() if hasattr(emb_query_vec, 'tolist') else emb_query_vec,
        "emb_doc_matrix": emb_doc_matrix.tolist() if hasattr(emb_doc_matrix, 'tolist') else emb_doc_matrix,
        "first_stage": first_stage,
        "top_n": top_n,
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
        raise Exception(f"Failed to perform sequential hybrid search via API: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 