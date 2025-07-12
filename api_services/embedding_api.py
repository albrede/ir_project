from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import sqlite3
import os
from services.vectorization.embedding_vectorizer import build_embedding_model

app = FastAPI(
    title="Embedding Model Service",
    description="Service for building Embedding model for a specific dataset",
    version="1.0.0"
)

DB_PATH = "data/ir_documents.db"

class BuildEmbeddingRequest(BaseModel):
    dataset_name: str
    limit: Optional[int] = None
    model_name: Optional[str] = 'paraphrase-MiniLM-L3-v2'

@app.post("/build-embedding")
def build_embedding(request: BuildEmbeddingRequest):
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