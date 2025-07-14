from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import sqlite3
import os
import joblib
import numpy as np
from services.vectorization.embedding_vectorizer import build_embedding_model, load_embedding_model, get_embedding_vector, get_embedding_vector_serializable

app = FastAPI(
    title="Embedding Model Service",
    description="Service for Embedding operations including building, loading, and vectorization",
    version="1.0.0"
)

DB_PATH = "data/ir_documents.db"
VECTORS_DIR = "vectors"
MODELS_DIR = "models"

class BuildEmbeddingRequest(BaseModel):
    dataset_name: str
    limit: Optional[int] = None
    model_name: Optional[str] = 'paraphrase-MiniLM-L3-v2'

class LoadEmbeddingRequest(BaseModel):
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
        "service": "Embedding Model Service",
        "version": "1.0.0",
        "description": "Service for Embedding operations including building, loading, and vectorization",
        "endpoints": {
            "POST /build-embedding": "Build Embedding model for a dataset",
            "POST /load-embedding": "Load Embedding model and vectors",
            "POST /vectorize": "Vectorize a query using Embedding",
            "POST /get-vector": "Get Embedding vector for a query",
            "GET /health": "Health check"
        }
    }

@app.post("/build-embedding")
def build_embedding(request: BuildEmbeddingRequest):
    """Build Embedding model for a specific dataset"""
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
        model_name = request.model_name or 'paraphrase-MiniLM-L3-v2'
        
        model, embeddings, doc_ids = build_embedding_model(
            texts, request.dataset_name, doc_ids, model_name=model_name
        )
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "documents_count": len(texts),
            "embeddings_shape": embeddings.shape,
            "model_name": model_name,
            "model_files": [
                f"models/{request.dataset_name.replace('/', '_')}_embedding_model.joblib",
                f"vectors/{request.dataset_name.replace('/', '_')}_embedding_vectors.joblib"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error building Embedding model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-embedding")
def load_embedding(request: LoadEmbeddingRequest):
    """Load Embedding model and vectors for a dataset"""
    try:
        model, embeddings, doc_ids = load_embedding_model(request.dataset_name)
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "embeddings_shape": embeddings.shape,
            "doc_ids_count": len(doc_ids) if doc_ids else 0,
            "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
            "model_loaded": True
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Embedding model not found for dataset '{request.dataset_name}': {str(e)}")
    except Exception as e:
        logging.error(f"Error loading Embedding model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize")
def vectorize_query(request: VectorizeRequest):
    """Vectorize a query using Embedding model"""
    try:
        # Load the Embedding model
        model, embeddings, doc_ids = load_embedding_model(request.dataset_name)
        
        # Vectorize the query
        if request.return_serializable:
            query_vector = get_embedding_vector_serializable(model, request.query)
        else:
            query_vector = get_embedding_vector(model, request.query)
        
        # Get vector statistics
        vector_norm = np.linalg.norm(query_vector)
        vector_mean = np.mean(query_vector)
        vector_std = np.std(query_vector)
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "query": request.query,
            "vector_shape": query_vector.shape,
            "vector_norm": float(vector_norm),
            "vector_mean": float(vector_mean),
            "vector_std": float(vector_std),
            "vector": query_vector.tolist() if request.return_serializable else query_vector.flatten().tolist()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Embedding model not found for dataset '{request.dataset_name}': {str(e)}")
    except Exception as e:
        logging.error(f"Error vectorizing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-vector")
def get_vector(request: GetVectorRequest):
    """Get Embedding vector for a query (simplified version)"""
    try:
        # Load the Embedding model
        model, embeddings, doc_ids = load_embedding_model(request.dataset_name)
        
        # Get vector
        query_vector = get_embedding_vector(model, request.query)
        
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "query": request.query,
            "vector": query_vector.tolist(),
            "vector_shape": query_vector.shape
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Embedding model not found for dataset '{request.dataset_name}': {str(e)}")
    except Exception as e:
        logging.error(f"Error getting vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check for the Embedding service"""
    try:
        # Test if models directory exists
        models_exist = os.path.exists(MODELS_DIR)
        vectors_exist = os.path.exists(VECTORS_DIR)
        
        return {
            "status": "healthy",
            "service": "Embedding Model Service",
            "version": "1.0.0",
            "directories": {
                "models": models_exist,
                "vectors": vectors_exist
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Embedding Model Service",
            "version": "1.0.0",
            "error": str(e)
        }

# Client functions for other services to use
def load_embedding_model_via_api(dataset_name: str):
    """Client function to load Embedding model via API"""
    url = "http://localhost:8003/load-embedding"
    
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
            model_path = os.path.join(MODELS_DIR, f"{clean_dataset_name}_embedding_model.joblib")
            vectors_path = os.path.join(VECTORS_DIR, f"{clean_dataset_name}_embedding_vectors.joblib")
            
            model = joblib.load(model_path)
            data = joblib.load(vectors_path)
            embeddings = data["embeddings"]
            doc_ids = data["doc_ids"]
            
            return model, embeddings, doc_ids
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load Embedding model via API: {e}")

def get_embedding_vector_via_api(dataset_name: str, query: str):
    """Client function to get Embedding vector via API"""
    url = "http://localhost:8003/vectorize"
    
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
        raise Exception(f"Failed to get Embedding vector via API: {e}")

def get_embedding_vector_serializable_via_api(dataset_name: str, query: str):
    """Client function to get serializable Embedding vector via API"""
    url = "http://localhost:8003/vectorize"
    
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
        raise Exception(f"Failed to get serializable Embedding vector via API: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 