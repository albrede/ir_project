from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import sqlite3
import os
from services.vectorization.tfidf_vectorizer import build_tfidf_model

app = FastAPI(
    title="TF-IDF Model Service",
    description="Service for building TF-IDF model for a specific dataset",
    version="1.0.0"
)

DB_PATH = "data/ir_documents.db"

class BuildTfidfRequest(BaseModel):
    dataset_name: str
    limit: Optional[int] = None

@app.post("/build-tfidf")
def build_tfidf(request: BuildTfidfRequest):
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
            "model_files": [
                f"models/{request.dataset_name.replace('/', '_')}_tfidf_model.joblib",
                f"models/{request.dataset_name.replace('/', '_')}_vectorizer.joblib",
                f"models/{request.dataset_name.replace('/', '_')}_tfidf_matrix.joblib"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error building TF-IDF model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 