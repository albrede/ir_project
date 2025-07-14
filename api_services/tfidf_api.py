from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import sqlite3
import os
import joblib
import numpy as np
from services.vectorization.tfidf_vectorizer import build_tfidf_model, load_tfidf_model, get_tfidf_vector, get_tfidf_vector_serializable

app = FastAPI(
    title="TF-IDF Model Service",
    description="Service for TF-IDF operations including building, loading, and vectorization",
    version="1.0.0"
)

DB_PATH = "data/ir_documents.db"
VECTORS_DIR = "vectors"
MODELS_DIR = "models"

class BuildTfidfRequest(BaseModel):
    dataset_name: str
    limit: Optional[int] = None

class LoadTfidfRequest(BaseModel):
    dataset_name: str

class VectorizeRequest(BaseModel):
    dataset_name: str
    query: str
    return_serializable: Optional[bool] = False

class GetVectorRequest(BaseModel):
    dataset_name: str
    query: str

@app.get("/")
def read_root():
    return {
        "service": "TF-IDF Model Service",
        "version": "1.0.0",
        "description": "Service for TF-IDF operations including building, loading, and vectorization",
        "endpoints": {
            "POST /build-tfidf": "Build TF-IDF model for a dataset",
            "POST /load-tfidf": "Load TF-IDF model and matrix",
            "POST /vectorize": "Vectorize a query using TF-IDF",
            "POST /get-vector": "Get TF-IDF vector for a query",
            "GET /health": "Health check"
        }
    }

@app.post("/build-tfidf")
def build_tfidf(request: BuildTfidfRequest):
    """Build TF-IDF model for a specific dataset"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = "SELECT doc_id, text FROM documents WHERE dataset = ?"
        params = [request.dataset_name]
        if request.limit:
            query += " LIMIT ?"
            params.append(str(request.limit))
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            raise HTTPException(status_code=404, detail=f"No documents found for dataset '{request.dataset_name}'")
        
        doc_ids = [str(row[0]) for row in rows]
        texts = [str(row[1]) for row in rows]
        
        vectorizer, tfidf_matrix, doc_ids = build_tfidf_model(texts, request.dataset_name, doc_ids)
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "documents_count": len(texts),
            "matrix_shape": tfidf_matrix.shape,
            "model_files": [
                f"models/{request.dataset_name.replace('/', '_')}_tfidf_model.joblib",
                f"vectors/{request.dataset_name.replace('/', '_')}_tfidf_matrix.joblib"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error building TF-IDF model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-tfidf")
def load_tfidf(request: LoadTfidfRequest):
    """Load TF-IDF model and matrix for a dataset"""
    try:
        vectorizer, tfidf_matrix, doc_ids = load_tfidf_model(request.dataset_name)
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "matrix_shape": tfidf_matrix.shape,
            "doc_ids_count": len(doc_ids) if doc_ids else 0,
            "feature_names_count": len(vectorizer.get_feature_names_out()),
            "model_loaded": True
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"TF-IDF model not found for dataset '{request.dataset_name}': {str(e)}")
    except Exception as e:
        logging.error(f"Error loading TF-IDF model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize")
def vectorize_query(request: VectorizeRequest):
    """Vectorize a query using TF-IDF model"""
    try:
        # Load the TF-IDF model
        vectorizer, tfidf_matrix, doc_ids = load_tfidf_model(request.dataset_name)
        
        # Vectorize the query
        if request.return_serializable:
            query_vector = get_tfidf_vector_serializable(vectorizer, request.query)
        else:
            query_vector = get_tfidf_vector(vectorizer, request.query)
        
        # Get feature names for non-zero values
        feature_names = vectorizer.get_feature_names_out()
        non_zero_indices = np.nonzero(query_vector.flatten())[0]
        word_scores = []
        
        for idx in non_zero_indices:
            word_scores.append({
                "word": feature_names[idx],
                "tfidf_score": float(query_vector.flatten()[idx])
            })
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "query": request.query,
            "vector_shape": query_vector.shape,
            "non_zero_count": len(non_zero_indices),
            "word_scores": word_scores,
            "vector": query_vector.tolist() if request.return_serializable else query_vector.flatten().tolist()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"TF-IDF model not found for dataset '{request.dataset_name}': {str(e)}")
    except Exception as e:
        logging.error(f"Error vectorizing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-vector")
def get_vector(request: GetVectorRequest):
    """Get TF-IDF vector for a query (simplified version)"""
    try:
        # Load the TF-IDF model
        vectorizer, tfidf_matrix, doc_ids = load_tfidf_model(request.dataset_name)
        
        # Get vector
        query_vector = get_tfidf_vector(vectorizer, request.query)
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "query": request.query,
            "vector": query_vector.tolist(),
            "vector_shape": query_vector.shape
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"TF-IDF model not found for dataset '{request.dataset_name}': {str(e)}")
    except Exception as e:
        logging.error(f"Error getting vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check for the TF-IDF service"""
    try:
        # Test if models directory exists
        models_exist = os.path.exists(MODELS_DIR)
        vectors_exist = os.path.exists(VECTORS_DIR)
        
        return {
            "status": "healthy",
            "service": "TF-IDF Model Service",
            "version": "1.0.0",
            "directories": {
                "models": models_exist,
                "vectors": vectors_exist
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "TF-IDF Model Service",
            "version": "1.0.0",
            "error": str(e)
        }

# Client functions for other services to use
def load_tfidf_model_via_api(dataset_name: str):
    """Client function to load TF-IDF model via API"""
    url = "http://localhost:8002/load-tfidf"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        import requests
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            clean_dataset_name = dataset_name.replace('/', '_')
            model_path = os.path.join(MODELS_DIR, f"{clean_dataset_name}_tfidf_model.joblib")
            matrix_path = os.path.join(VECTORS_DIR, f"{clean_dataset_name}_tfidf_matrix.joblib")
            
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
        import requests
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
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
        import requests
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["vector"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get serializable TF-IDF vector via API: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 