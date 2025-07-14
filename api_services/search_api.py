from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import os
import logging
from services.vectorization.tfidf_vectorizer import get_tfidf_vector
from services.vectorization.embedding_vectorizer import get_embedding_vector
from services.hybrid import parallel_hybrid_search, hybrid_search_with_weights
from services.hybrid_sequential import sequential_hybrid_search
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Search Service API", version="1.0.0")

# Model and matrix paths
MODELS_DIR = "models"
VECTORS_DIR = "vectors"

class SearchRequest(BaseModel):
    query: str
    method: str = "hybrid"  # "tfidf", "embeddings", "hybrid", "hybrid-sequential"
    alpha: Optional[float] = 0.5  # TF-IDF weight in hybrid search
    top_k: Optional[int] = 10
    dataset_name: Optional[str] = "simple"  # Dataset name
    # Additional options for sequential hybrid search
    first_stage: Optional[str] = "tfidf"  # "tfidf" or "emb"
    top_n: Optional[int] = 100

class HybridSearchRequest(BaseModel):
    query: str
    alpha: Optional[float] = 0.5
    top_k: Optional[int] = 10
    dataset_name: Optional[str] = "simple"

class HybridSequentialSearchRequest(BaseModel):
    query: str
    first_stage: Optional[str] = "tfidf"  # "tfidf" أو "emb"
    top_n: Optional[int] = 100
    top_k: Optional[int] = 10
    dataset_name: Optional[str] = "simple"

def load_models(dataset_name: str = "simple", method: str = "hybrid"):
    """Load models and matrices based on the search method"""
    try:
        # Clean dataset name - replace / with _
        clean_dataset_name = dataset_name.replace('/', '_')
        
        # Correct file paths after new standardization
        tfidf_model_path = os.path.join(MODELS_DIR, f"{clean_dataset_name}_tfidf_model.joblib")
        tfidf_matrix_path = os.path.join(VECTORS_DIR, f"{clean_dataset_name}_tfidf_matrix.joblib")
        
        possible_embedding_model_paths = [
            os.path.join(VECTORS_DIR, f"{clean_dataset_name}_embedding_model.joblib"),
            os.path.join(MODELS_DIR, f"{clean_dataset_name}_embedding_model.joblib"),
            os.path.join(VECTORS_DIR, f"{dataset_name}_embedding_model.joblib"),
            os.path.join(MODELS_DIR, f"{dataset_name}_embedding_model.joblib")
        ]
        
        possible_embedding_matrix_paths = [
            os.path.join(VECTORS_DIR, f"{clean_dataset_name}_embedding_vectors.joblib"),
            os.path.join(VECTORS_DIR, f"{clean_dataset_name}_embedding.joblib"),
            os.path.join(VECTORS_DIR, f"{dataset_name}_embedding_vectors.joblib"),
            os.path.join(VECTORS_DIR, f"{dataset_name}_embedding.joblib")
        ]
        
        # Search for existing files
        embedding_model_path = None
        embedding_matrix_path = None
        
        for path in possible_embedding_model_paths:
            if os.path.exists(path):
                embedding_model_path = path
                break
                
        for path in possible_embedding_matrix_paths:
            if os.path.exists(path):
                embedding_matrix_path = path
                break
        
        # Check for required files based on search method
        missing_files = []
        
        # TF-IDF is required for all methods
        if not os.path.exists(tfidf_model_path):
            missing_files.append(f"TF-IDF model for {dataset_name}")
        if not os.path.exists(tfidf_matrix_path):
            missing_files.append(f"TF-IDF matrix for {dataset_name}")
        
        # Embedding is required only for embeddings and hybrid
        if method in ["embeddings", "hybrid"]:
            if not embedding_model_path:
                missing_files.append(f"Embedding model for {dataset_name}")
            if not embedding_matrix_path:
                missing_files.append(f"Embedding matrix for {dataset_name}")
            
        if missing_files:
            raise FileNotFoundError(f"The following files are missing: {', '.join(missing_files)}")
        
        # Load only required models
        tfidf_model = joblib.load(tfidf_model_path)
        tfidf_matrix = joblib.load(tfidf_matrix_path)
        
        embedding_model = None
        embedding_matrix = None
        
        if method in ["embeddings", "hybrid"]:
            embedding_model = joblib.load(embedding_model_path)
            embedding_matrix = joblib.load(embedding_matrix_path)
        
        logger.info(f"Models loaded successfully for {dataset_name} (method: {method})")
        logger.info(f"TF-IDF Model: {tfidf_model_path}")
        logger.info(f"TF-IDF Matrix: {tfidf_matrix_path}")
        
        if embedding_model:
            logger.info(f"Embedding Model: {embedding_model_path}")
            logger.info(f"Embedding Matrix: {embedding_matrix_path}")
        
        return tfidf_model, tfidf_matrix, embedding_model, embedding_matrix
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@app.get("/")
def read_root():
    return {
        "service": "Search Service",
        "version": "1.0.0",
        "description": "Service for searching information using different methods",
        "endpoints": {
            "POST /search": "Search using a specified method",
            "POST /hybrid-search": "Hybrid search",
            "GET /methods": "Available search methods",
            "GET /health": "Service health check"
        }
    }

@app.get("/methods")
def get_search_methods():
    """Get available search methods"""
    return {
        "status": "success",
        "methods": {
            "tfidf": {
                "description": "Search using TF-IDF",
                "parameters": ["query", "top_k", "dataset_name"]
            },
            "embeddings": {
                "description": "Search using Embeddings",
                "parameters": ["query", "top_k", "dataset_name"]
            },
            "hybrid": {
                "description": "Hybrid search (TF-IDF + Embeddings)",
                "parameters": ["query", "alpha", "top_k", "dataset_name"]
            },
            "hybrid-sequential": {
                "description": "Sequential hybrid search (initial filtering then reordering)",
                "parameters": ["query", "first_stage", "top_n", "top_k", "dataset_name"]
            }
        }
    }

@app.post("/search")
def search(request: SearchRequest):
    """Search using a specified method"""
    try:
        dataset_name = request.dataset_name if request.dataset_name is not None else "simple"
        tfidf_model, tfidf_matrix_data, embedding_model, embedding_matrix_data = load_models(dataset_name, request.method)
        # Support doc_ids with tfidf/embedding
        if isinstance(tfidf_matrix_data, dict):
            tfidf_matrix = tfidf_matrix_data["matrix"]
            tfidf_doc_ids = tfidf_matrix_data["doc_ids"]
        else:
            tfidf_matrix = tfidf_matrix_data
            tfidf_doc_ids = None
        if embedding_matrix_data is not None and isinstance(embedding_matrix_data, dict):
            embedding_matrix = embedding_matrix_data["vectors"]
            embedding_doc_ids = embedding_matrix_data["doc_ids"]
        else:
            embedding_matrix = embedding_matrix_data
            embedding_doc_ids = None
        tfidf_query_vec = get_tfidf_vector(tfidf_model, request.query)
        embedding_query_vec = None
        if embedding_model is not None:
            embedding_query_vec = get_embedding_vector(embedding_model, request.query)
        top_k = request.top_k if request.top_k is not None else 10
        results = []
        scores = []
        if request.method == "tfidf":
            tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]
            top_indices = tfidf_scores.argsort()[-top_k:][::-1]
            results = [tfidf_doc_ids[idx] if tfidf_doc_ids else int(idx) for idx in top_indices]
            scores = tfidf_scores[top_indices].tolist()
        elif request.method == "embeddings":
            if embedding_query_vec is None:
                raise HTTPException(status_code=400, detail="Embedding model not available")
            emb_scores = cosine_similarity(embedding_query_vec, embedding_matrix)[0]
            top_indices = emb_scores.argsort()[-top_k:][::-1]
            results = [embedding_doc_ids[idx] if embedding_doc_ids else int(idx) for idx in top_indices]
            scores = emb_scores[top_indices].tolist()
        elif request.method == "hybrid":
            if embedding_query_vec is None:
                raise HTTPException(status_code=400, detail="Embedding model not available for hybrid search")
            alpha = request.alpha if request.alpha is not None else 0.5
            indices, hybrid_scores = parallel_hybrid_search(
                tfidf_query_vec, tfidf_matrix,
                embedding_query_vec, embedding_matrix,
                alpha=alpha, top_k=top_k,
                doc_ids=tfidf_doc_ids if tfidf_doc_ids else None
            )
            results = indices if isinstance(indices[0], str) else [tfidf_doc_ids[idx] if tfidf_doc_ids else int(idx) for idx in indices]
            scores = hybrid_scores.tolist()
        elif request.method == "hybrid-sequential":
            if embedding_query_vec is None:
                raise HTTPException(status_code=400, detail="Embedding model not available for sequential hybrid search")
            first_stage = request.first_stage if request.first_stage is not None else "tfidf"
            top_n = request.top_n if request.top_n is not None else 100
            indices, seq_scores = sequential_hybrid_search(
                tfidf_query_vec, tfidf_matrix,
                embedding_query_vec, embedding_matrix,
                first_stage=first_stage, top_n=top_n, top_k=top_k,
                doc_ids=tfidf_doc_ids if tfidf_doc_ids else None
            )
            results = indices if isinstance(indices[0], str) else [tfidf_doc_ids[idx] if tfidf_doc_ids else int(idx) for idx in indices]
            scores = seq_scores.tolist()
        else:
            raise HTTPException(status_code=400, detail=f"Search method '{request.method}' not supported")
        return {
            "status": "success",
            "query": request.query,
            "method": request.method,
            "dataset": dataset_name,
            "results_count": len(results),
            "results": [
                {
                    "document_id": doc_id,
                    "score": float(score)
                }
                for doc_id, score in zip(results, scores)
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid-search")
def hybrid_search(request: HybridSearchRequest):
    """Hybrid search with advanced options"""
    try:
        dataset_name = request.dataset_name if request.dataset_name is not None else "simple"
        tfidf_model, tfidf_matrix_data, embedding_model, embedding_matrix_data = load_models(dataset_name, "hybrid")
        if isinstance(tfidf_matrix_data, dict):
            tfidf_matrix = tfidf_matrix_data["matrix"]
            tfidf_doc_ids = tfidf_matrix_data["doc_ids"]
        else:
            tfidf_matrix = tfidf_matrix_data
            tfidf_doc_ids = None
        if embedding_matrix_data is not None and isinstance(embedding_matrix_data, dict):
            embedding_matrix = embedding_matrix_data["vectors"]
            embedding_doc_ids = embedding_matrix_data["doc_ids"]
        else:
            embedding_matrix = embedding_matrix_data
            embedding_doc_ids = None
        tfidf_query_vec = get_tfidf_vector(tfidf_model, request.query)
        embedding_query_vec = None
        if embedding_model is not None:
            embedding_query_vec = get_embedding_vector(embedding_model, request.query)
        if embedding_query_vec is None:
            raise HTTPException(status_code=400, detail="Embedding model not available for hybrid search")
        alpha = request.alpha if request.alpha is not None else 0.5
        top_k = request.top_k if request.top_k is not None else 10
        indices, scores = parallel_hybrid_search(
            tfidf_query_vec, tfidf_matrix,
            embedding_query_vec, embedding_matrix,
            alpha=alpha, top_k=top_k,
            doc_ids=tfidf_doc_ids if tfidf_doc_ids else None
        )
        results = indices if isinstance(indices[0], str) else [tfidf_doc_ids[idx] if tfidf_doc_ids else int(idx) for idx in indices]
        scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]
        emb_scores = cosine_similarity(embedding_query_vec, embedding_matrix)[0]
        tfidf_top = tfidf_scores.argsort()[-1]
        emb_top = emb_scores.argsort()[-1]
        comparison = {
            "tfidf_only": {
                "top_document": tfidf_doc_ids[tfidf_top] if tfidf_doc_ids else int(tfidf_top),
                "score": float(tfidf_scores[tfidf_top])
            },
            "embeddings_only": {
                "top_document": embedding_doc_ids[emb_top] if embedding_doc_ids else int(emb_top),
                "score": float(emb_scores[emb_top])
            },
            "hybrid": {
                "top_document": results[0],
                "score": float(scores[0])
            }
        }
        return {
            "status": "success",
            "query": request.query,
            "dataset": dataset_name,
            "alpha": alpha,
            "results_count": len(results),
            "results": [
                {
                    "document_id": doc_id,
                    "score": float(score)
                }
                for doc_id, score in zip(results, scores)
            ],
            "comparison": comparison
        }
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid-sequential-search")
def hybrid_sequential_search_api(request: HybridSequentialSearchRequest):
    """Sequential hybrid search with advanced options"""
    try:
        dataset_name = request.dataset_name if request.dataset_name is not None else "simple"
        tfidf_model, tfidf_matrix_data, embedding_model, embedding_matrix_data = load_models(dataset_name, "hybrid")
        if isinstance(tfidf_matrix_data, dict):
            tfidf_matrix = tfidf_matrix_data["matrix"]
            tfidf_doc_ids = tfidf_matrix_data["doc_ids"]
        else:
            tfidf_matrix = tfidf_matrix_data
            tfidf_doc_ids = None
        if embedding_matrix_data is not None and isinstance(embedding_matrix_data, dict):
            embedding_matrix = embedding_matrix_data["vectors"]
            embedding_doc_ids = embedding_matrix_data["doc_ids"]
        else:
            embedding_matrix = embedding_matrix_data
            embedding_doc_ids = None
        tfidf_query_vec = get_tfidf_vector(tfidf_model, request.query)
        embedding_query_vec = None
        if embedding_model is not None:
            embedding_query_vec = get_embedding_vector(embedding_model, request.query)
        if embedding_query_vec is None:
            raise HTTPException(status_code=400, detail="Embedding model not available for sequential hybrid search")
        first_stage = request.first_stage if request.first_stage is not None else "tfidf"
        top_n = request.top_n if request.top_n is not None else 100
        top_k = request.top_k if request.top_k is not None else 10
        indices, scores = sequential_hybrid_search(
            tfidf_query_vec, tfidf_matrix,
            embedding_query_vec, embedding_matrix,
            first_stage=first_stage, top_n=top_n, top_k=top_k,
            doc_ids=tfidf_doc_ids if tfidf_doc_ids else None
        )
        results = indices if isinstance(indices[0], str) else [tfidf_doc_ids[idx] if tfidf_doc_ids else int(idx) for idx in indices]
        scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        return {
            "status": "success",
            "query": request.query,
            "dataset": dataset_name,
            "first_stage": first_stage,
            "top_n": top_n,
            "results_count": len(results),
            "results": [
                {
                    "document_id": doc_id,
                    "score": float(score)
                }
                for doc_id, score in zip(results, scores)
            ]
        }
    except Exception as e:
        logger.error(f"Error in sequential hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Service health check"""
    try:
        # Check for model existence
        models_status = {}
        for dataset in ["simple", "msmarco", "trec-covid", "beir_arguana"]:
            try:
                # Clean dataset name
                clean_dataset = dataset.replace('/', '_')
                
                # Correct file paths after new standardization
                tfidf_model_path = os.path.join(MODELS_DIR, f"{clean_dataset}_tfidf_model.joblib")
                tfidf_matrix_path = os.path.join(VECTORS_DIR, f"{clean_dataset}_tfidf_matrix.joblib")
                
                possible_embedding_model_paths = [
                    os.path.join(VECTORS_DIR, f"{clean_dataset}_embedding_model.joblib"),
                    os.path.join(MODELS_DIR, f"{clean_dataset}_embedding_model.joblib"),
                    os.path.join(VECTORS_DIR, f"{dataset}_embedding_model.joblib"),
                    os.path.join(MODELS_DIR, f"{dataset}_embedding_model.joblib")
                ]
                
                possible_embedding_matrix_paths = [
                    os.path.join(VECTORS_DIR, f"{clean_dataset}_embedding_vectors.joblib"),
                    os.path.join(VECTORS_DIR, f"{clean_dataset}_embedding.joblib"),
                    os.path.join(VECTORS_DIR, f"{dataset}_embedding_vectors.joblib"),
                    os.path.join(VECTORS_DIR, f"{dataset}_embedding.joblib")
                ]
                
                # Search for existing files
                tfidf_model_exists = os.path.exists(tfidf_model_path)
                tfidf_matrix_exists = os.path.exists(tfidf_matrix_path)
                embedding_model_exists = any(os.path.exists(path) for path in possible_embedding_model_paths)
                embedding_matrix_exists = any(os.path.exists(path) for path in possible_embedding_matrix_paths)
                
                models_status[dataset] = {
                    "tfidf_model": tfidf_model_exists,
                    "tfidf_matrix": tfidf_matrix_exists,
                    "embedding_model": embedding_model_exists,
                    "embedding_matrix": embedding_matrix_exists,
                    "ready_for_search": tfidf_model_exists and tfidf_matrix_exists and embedding_model_exists and embedding_matrix_exists
                }
            except Exception as e:
                models_status[dataset] = {"error": f"Error checking files: {str(e)}"}
        
        return {
            "status": "healthy",
            "models": models_status,
            "available_datasets": [dataset for dataset, status in models_status.items() 
                                 if isinstance(status, dict) and status.get("ready_for_search", False)]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }