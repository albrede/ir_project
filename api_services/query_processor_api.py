from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import requests
import numpy as np
import joblib
import os

app = FastAPI(
    title="Query Processor Service",
    description="Service for processing queries and loading models",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessRequest(BaseModel):
    text: str
    return_tokens: bool = False
    min_length: int = 2
    remove_stopwords: bool = True
    use_lemmatization: bool = True

class LoadTfidfRequest(BaseModel):
    dataset_name: str

class LoadEmbeddingRequest(BaseModel):
    dataset_name: str

class GetTfidfVectorRequest(BaseModel):
    dataset_name: str
    query: str
    return_serializable: bool = False

class GetEmbeddingVectorRequest(BaseModel):
    dataset_name: str
    query: str
    return_serializable: bool = False

class ProcessQueryRequest(BaseModel):
    query: str
    method: str = "tfidf"
    dataset_name: str = "simple"
    return_vector: bool = False

class GetQueryVectorRequest(BaseModel):
    query: str
    method: str = "tfidf"
    dataset_name: str = "simple"

@app.get("/")
def read_root():
    return {
        "service": "Query Processor Service",
        "version": "1.0.0",
        "description": "Service for processing queries and loading models",
        "endpoints": {
            "POST /preprocess": "Preprocess text",
            "POST /load-tfidf": "Load TF-IDF model",
            "POST /load-embedding": "Load Embedding model",
            "POST /get-tfidf-vector": "Get TF-IDF vector",
            "POST /get-embedding-vector": "Get Embedding vector",
            "POST /process-query": "Process query",
            "POST /get-query-vector": "Get query vector for matching",
            "GET /health": "Health check"
        }
    }

@app.post("/preprocess")
def preprocess_text(request: PreprocessRequest):
    """Preprocess text via preprocessing API"""
    try:
        url = "http://localhost:8001/preprocess"
        
        payload = {
            "text": request.text,
            "return_tokens": request.return_tokens,
            "min_length": request.min_length,
            "remove_stopwords": request.remove_stopwords,
            "use_lemmatization": request.use_lemmatization
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return {
                "status": "success",
                "result": data["result"]
            }
        else:
            raise Exception(f"Preprocessing API returned error: {data}")
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-tfidf")
def load_tfidf_model(request: LoadTfidfRequest):
    """Load TF-IDF model via TF-IDF API"""
    try:
        url = "http://localhost:8002/load-tfidf"
        
        payload = {
            "dataset_name": request.dataset_name
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            clean_dataset_name = request.dataset_name.replace('/', '_')
            model_path = os.path.join("models", f"{clean_dataset_name}_tfidf_model.joblib")
            matrix_path = os.path.join("vectors", f"{clean_dataset_name}_tfidf_matrix.joblib")
            
            vectorizer = joblib.load(model_path)
            data = joblib.load(matrix_path)
            tfidf_matrix = data["matrix"]
            doc_ids = data["doc_ids"]
            
            return {
                "status": "success",
                "vectorizer": vectorizer,
                "matrix": tfidf_matrix,
                "doc_ids": doc_ids
            }
        else:
            raise Exception(f"TF-IDF API returned error: {data}")
            
    except Exception as e:
        logger.error(f"Error loading TF-IDF model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-embedding")
def load_embedding_model(request: LoadEmbeddingRequest):
    """Load Embedding model via Embedding API"""
    try:
        url = "http://localhost:8003/load-embedding"
        
        payload = {
            "dataset_name": request.dataset_name
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            clean_dataset_name = request.dataset_name.replace('/', '_')
            model_path = os.path.join("models", f"{clean_dataset_name}_embedding_model.joblib")
            vectors_path = os.path.join("vectors", f"{clean_dataset_name}_embedding_vectors.joblib")
            
            model = joblib.load(model_path)
            data = joblib.load(vectors_path)
            # Handle both "vectors" and "embeddings" keys for compatibility
            if "vectors" in data:
                embeddings = data["vectors"]
            elif "embeddings" in data:
                embeddings = data["embeddings"]
            else:
                raise KeyError("Neither 'vectors' nor 'embeddings' key found in data")
            doc_ids = data["doc_ids"]
            
            return {
                "status": "success",
                "model": model,
                "embeddings": embeddings,
                "doc_ids": doc_ids
            }
        else:
            raise Exception(f"Embedding API returned error: {data}")
            
    except Exception as e:
        logger.error(f"Error loading Embedding model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-tfidf-vector")
def get_tfidf_vector(request: GetTfidfVectorRequest):
    """Get TF-IDF vector via TF-IDF API"""
    try:
        url = "http://localhost:8002/vectorize"
        
        payload = {
            "dataset_name": request.dataset_name,
            "query": request.query,
            "return_serializable": request.return_serializable
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            if request.return_serializable:
                return {
                    "status": "success",
                    "vector": data["vector"]
                }
            else:
                return {
                    "status": "success",
                    "vector": np.array(data["vector"]).reshape(1, -1).tolist()
                }
        else:
            raise Exception(f"TF-IDF API returned error: {data}")
            
    except Exception as e:
        logger.error(f"Error getting TF-IDF vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-embedding-vector")
def get_embedding_vector(request: GetEmbeddingVectorRequest):
    """Get Embedding vector via Embedding API"""
    try:
        url = "http://localhost:8003/vectorize"
        
        payload = {
            "dataset_name": request.dataset_name,
            "query": request.query,
            "return_serializable": request.return_serializable
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            if request.return_serializable:
                return {
                    "status": "success",
                    "vector": data["vector"]
                }
            else:
                return {
                    "status": "success",
                    "vector": np.array(data["vector"]).reshape(1, -1).tolist()
                }
        else:
            raise Exception(f"Embedding API returned error: {data}")
            
    except Exception as e:
        logger.error(f"Error getting Embedding vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-query")
def process_query(request: ProcessQueryRequest):
    """Process query according to method"""
    try:
        if not request.query or not request.query.strip():
            raise ValueError("Query cannot be empty")
        
        # Preprocess text for vectorization
        preprocess_url = "http://localhost:8001/preprocess"
        preprocess_payload = {
            "text": request.query,
            "return_tokens": False,
            "min_length": 2,
            "remove_stopwords": True,
            "use_lemmatization": True
        }
        
        preprocess_response = requests.post(preprocess_url, json=preprocess_payload, timeout=30)
        preprocess_response.raise_for_status()
        preprocess_data = preprocess_response.json()
        
        if preprocess_data["status"] != "success":
            raise Exception(f"Preprocessing failed: {preprocess_data}")
        
        processed_text = preprocess_data["result"]
        
        if request.method == "tfidf":
            # Get TF-IDF vector
            tfidf_url = "http://localhost:8002/vectorize"
            tfidf_payload = {
                "dataset_name": request.dataset_name,
                "query": processed_text,
                "return_serializable": request.return_vector
            }
            
            tfidf_response = requests.post(tfidf_url, json=tfidf_payload, timeout=30)
            tfidf_response.raise_for_status()
            tfidf_data = tfidf_response.json()
            
            if tfidf_data["status"] == "success":
                return {
                    "status": "success",
                    "method": "tfidf",
                    "processed_text": processed_text,
                    "vector": tfidf_data["vector"] if request.return_vector else None
                }
            else:
                raise Exception(f"TF-IDF vectorization failed: {tfidf_data}")
                
        elif request.method == "embedding":
            # Get Embedding vector
            emb_url = "http://localhost:8003/vectorize"
            emb_payload = {
                "dataset_name": request.dataset_name,
                "query": request.query.strip(),
                "return_serializable": request.return_vector
            }
            
            emb_response = requests.post(emb_url, json=emb_payload, timeout=30)
            emb_response.raise_for_status()
            emb_data = emb_response.json()
            
            if emb_data["status"] == "success":
                return {
                    "status": "success",
                    "method": "embedding",
                    "processed_text": request.query.strip(),
                    "vector": emb_data["vector"] if request.return_vector else None
                }
            else:
                raise Exception(f"Embedding vectorization failed: {emb_data}")
                
        elif request.method in ["hybrid", "hybrid-sequential"]:
            # Get both TF-IDF and Embedding vectors
            tfidf_url = "http://localhost:8002/vectorize"
            tfidf_payload = {
                "dataset_name": request.dataset_name,
                "query": processed_text,
                "return_serializable": request.return_vector
            }
            
            emb_url = "http://localhost:8003/vectorize"
            emb_payload = {
                "dataset_name": request.dataset_name,
                "query": request.query.strip(),
                "return_serializable": request.return_vector
            }
            
            tfidf_response = requests.post(tfidf_url, json=tfidf_payload, timeout=30)
            emb_response = requests.post(emb_url, json=emb_payload, timeout=30)
            
            tfidf_response.raise_for_status()
            emb_response.raise_for_status()
            
            tfidf_data = tfidf_response.json()
            emb_data = emb_response.json()
            
            if tfidf_data["status"] == "success" and emb_data["status"] == "success":
                return {
                    "status": "success",
                    "method": request.method,
                    "tfidf_text": processed_text,
                    "embedding_text": request.query.strip(),
                    "tfidf_vector": tfidf_data["vector"] if request.return_vector else None,
                    "embedding_vector": emb_data["vector"] if request.return_vector else None
                }
            else:
                raise Exception(f"Hybrid processing failed: TF-IDF={tfidf_data}, Embedding={emb_data}")
        else:
            raise ValueError(f"Unsupported method: {request.method}")
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-query-vector")
def get_query_vector_for_matching(request: GetQueryVectorRequest):
    """Get query vector for matching"""
    try:
        if request.method == "tfidf":
            # Preprocess and get TF-IDF vector
            preprocess_url = "http://localhost:8001/preprocess"
            preprocess_payload = {
                "text": request.query,
                "return_tokens": False,
                "min_length": 2,
                "remove_stopwords": True,
                "use_lemmatization": True
            }
            
            preprocess_response = requests.post(preprocess_url, json=preprocess_payload, timeout=30)
            preprocess_response.raise_for_status()
            preprocess_data = preprocess_response.json()
            
            if preprocess_data["status"] != "success":
                raise Exception(f"Preprocessing failed: {preprocess_data}")
            
            processed_text = preprocess_data["result"]
            
            # Get TF-IDF vector
            tfidf_url = "http://localhost:8002/vectorize"
            tfidf_payload = {
                "dataset_name": request.dataset_name,
                "query": processed_text,
                "return_serializable": False
            }
            
            tfidf_response = requests.post(tfidf_url, json=tfidf_payload, timeout=30)
            tfidf_response.raise_for_status()
            tfidf_data = tfidf_response.json()
            
            if tfidf_data["status"] == "success":
                return {
                    "status": "success",
                    "method": "tfidf",
                    "vector": tfidf_data["vector"]
                }
            else:
                raise Exception(f"TF-IDF vectorization failed: {tfidf_data}")
                
        elif request.method == "embedding":
            # Get Embedding vector
            emb_url = "http://localhost:8003/vectorize"
            emb_payload = {
                "dataset_name": request.dataset_name,
                "query": request.query.strip(),
                "return_serializable": False
            }
            
            emb_response = requests.post(emb_url, json=emb_payload, timeout=30)
            emb_response.raise_for_status()
            emb_data = emb_response.json()
            
            if emb_data["status"] == "success":
                return {
                    "status": "success",
                    "method": "embedding",
                    "vector": emb_data["vector"]
                }
            else:
                raise Exception(f"Embedding vectorization failed: {emb_data}")
                
        elif request.method in ["hybrid", "hybrid-sequential"]:
            # Get both TF-IDF and Embedding vectors
            preprocess_url = "http://localhost:8001/preprocess"
            preprocess_payload = {
                "text": request.query,
                "return_tokens": False,
                "min_length": 2,
                "remove_stopwords": True,
                "use_lemmatization": True
            }
            
            preprocess_response = requests.post(preprocess_url, json=preprocess_payload, timeout=30)
            preprocess_response.raise_for_status()
            preprocess_data = preprocess_response.json()
            
            if preprocess_data["status"] != "success":
                raise Exception(f"Preprocessing failed: {preprocess_data}")
            
            processed_text = preprocess_data["result"]
            
            # Get TF-IDF vector
            tfidf_url = "http://localhost:8002/vectorize"
            tfidf_payload = {
                "dataset_name": request.dataset_name,
                "query": processed_text,
                "return_serializable": False
            }
            
            # Get Embedding vector
            emb_url = "http://localhost:8003/vectorize"
            emb_payload = {
                "dataset_name": request.dataset_name,
                "query": request.query.strip(),
                "return_serializable": False
            }
            
            tfidf_response = requests.post(tfidf_url, json=tfidf_payload, timeout=30)
            emb_response = requests.post(emb_url, json=emb_payload, timeout=30)
            
            tfidf_response.raise_for_status()
            emb_response.raise_for_status()
            
            tfidf_data = tfidf_response.json()
            emb_data = emb_response.json()
            
            if tfidf_data["status"] == "success" and emb_data["status"] == "success":
                return {
                    "status": "success",
                    "method": request.method,
                    "tfidf": tfidf_data["vector"],
                    "embedding": emb_data["vector"]
                }
            else:
                raise Exception(f"Hybrid vectorization failed: TF-IDF={tfidf_data}, Embedding={emb_data}")
        else:
            raise ValueError(f"Unsupported method: {request.method}")
            
    except Exception as e:
        logger.error(f"Error getting query vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check for the Query Processor service"""
    try:
        return {
            "status": "healthy",
            "service": "Query Processor Service",
            "version": "1.0.0",
            "dependencies": {
                "requests": "available",
                "numpy": "available",
                "joblib": "available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Query Processor Service",
            "version": "1.0.0",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006) 