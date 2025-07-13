from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from services.indexing_service import build_inverted_index, save_index, load_index, search_in_index, get_index_statistics, inverted_index_tfidf_search
from services.vectorization.tfidf_vectorizer import load_tfidf_model
from services.preprocessing import preprocess_for_indexing, preprocess_for_vectorization
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlite3
import logging
from services.database.db_handler import get_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indexing Service",
    description="Indexing and search service for inverted indexes",
    version="1.0.0"
)

class IndexSearchRequest(BaseModel):
    query: str
    dataset_name: str = "simple"
    top_k: int = 10
    search_type: str = "and"

class IndexStatsRequest(BaseModel):
    dataset_name: str = "simple"

DB_PATH = "data/ir_documents.db"

def fetch_documents_from_db(dataset_name: str, limit: Optional[int] = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = "SELECT doc_id, text FROM documents WHERE dataset = ?"
        params = [dataset_name]
        
        if limit:
            query += " LIMIT ?"
            params.append(str(limit))
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        docs = []
        for row in rows:
            if len(row) >= 2 and row[0] and row[1]:
                docs.append({
                    'doc_id': str(row[0]), 
                    'content': str(row[1])
                })
        
        return docs
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "service": "Indexing Service",
        "version": "1.0.0",
        "description": "Indexing and search service for inverted indexes",
        "endpoints": {
            "POST /search": "Search in inverted index",
            "GET /search": "Search in inverted index (GET)",
            "POST /stats": "Index statistics",
            "GET /stats": "Index statistics (GET)",
            "GET /health": "Service health check"
        }
    }

@app.post("/build-index")
def build_index(
    dataset_name: str = Query(..., description="Dataset name"),
    limit: Optional[int] = Query(500000, description="Number of documents to index (optional)"),
    min_token_length: int = Query(2, description="Minimum word length")
):
    try:
        docs = fetch_documents_from_db(dataset_name, limit)
        
        if not docs:
            raise HTTPException(
                status_code=404, 
                detail=f"No documents found for dataset '{dataset_name}'"
            )
        
        logger.info(f"Starting index building for {len(docs)} documents...")
        
        index = build_inverted_index(docs, min_token_length)
        
        if not index:
            raise HTTPException(
                status_code=500, 
                detail="Failed to build index"
            )
        
        index_path = save_index(index, dataset_name)
        
        return {
            "status": "success",
            "dataset": dataset_name,
            "indexed_documents": len(docs),
            "index_terms": len(index),
            "index_file": index_path
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_index(request: IndexSearchRequest):
    try:
        logger.info(f"Searching index: {request.query} in {request.dataset_name}")
        
        matching_doc_ids = search_in_index(
            query=request.query,
            dataset_name=request.dataset_name,
            max_results=request.top_k,
            search_type=request.search_type
        )
        
        documents = get_documents(request.dataset_name)
        doc_dict = {doc["doc_id"]: doc["text"] for doc in documents}
        
        matched_documents = []
        for doc_id in matching_doc_ids:
            if doc_id in doc_dict:
                matched_documents.append({
                    "doc_id": doc_id,
                    "text": doc_dict[doc_id]
                })
            else:
                logger.warning(f"Document {doc_id} not found in database")
        
        return {
            "status": "success",
            "query": request.query,
            "dataset_name": request.dataset_name,
            "search_type": request.search_type,
            "matching_doc_ids": matching_doc_ids,
            "documents": matched_documents,
            "total_matches": len(matching_doc_ids),
            "documents_returned": len(matched_documents)
        }
        
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching index: {str(e)}"
        )

@app.get("/search")
def search_index_get(
    query: str = Query(..., description="Original query text"),
    dataset_name: str = Query("simple", description="Dataset name"),
    top_k: int = Query(10, description="Number of results required"),
    search_type: str = Query("and", description="Search type: and, or")
):
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if search_type not in ["and", "or"]:
            raise HTTPException(
                status_code=400,
                detail="Search type must be 'and' or 'or'"
            )
        
        logger.info(f"Searching index: {query} in {dataset_name}")
        
        matching_doc_ids = search_in_index(
            query=query,
            dataset_name=dataset_name,
            max_results=top_k,
            search_type=search_type
        )
        
        documents = get_documents(dataset_name)
        doc_dict = {doc["doc_id"]: doc["text"] for doc in documents}
        
        matched_documents = []
        for doc_id in matching_doc_ids:
            if doc_id in doc_dict:
                matched_documents.append({
                    "doc_id": doc_id,
                    "text": doc_dict[doc_id]
                })
            else:
                logger.warning(f"Document {doc_id} not found in database")
        
        return {
            "status": "success",
            "query": query,
            "dataset_name": dataset_name,
            "search_type": search_type,
            "matching_doc_ids": matching_doc_ids,
            "documents": matched_documents,
            "total_matches": len(matching_doc_ids),
            "documents_returned": len(matched_documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching index: {str(e)}"
        )

@app.post("/stats")
def get_index_stats_post(request: IndexStatsRequest):
    try:
        logger.info(f"Getting index statistics for {request.dataset_name}")
        
        stats = get_index_statistics(request.dataset_name)
        
        return {
            "status": "success",
            "dataset_name": request.dataset_name,
            "index_statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting index statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting index statistics: {str(e)}"
        )

@app.get("/stats")
def get_index_stats_get(
    dataset_name: str = Query("simple", description="Dataset name")
):
    try:
        logger.info(f"Getting index statistics for {dataset_name}")
        
        stats = get_index_statistics(dataset_name)
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "index_statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting index statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting index statistics: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Indexing Service",
        "version": "1.0.0"
    }

class TfidfSearchRequest(BaseModel):
    query: str
    dataset_name: str
    max_results: Optional[int] = 10
    limit_candidates: Optional[int] = 1000
    search_type: Optional[str] = 'and'

@app.post("/tfidf-search")
def tfidf_search(request: TfidfSearchRequest):
    try:
        logger.info(f"TF-IDF search: {request.query} in {request.dataset_name}")
        
        results = inverted_index_tfidf_search(
            query=request.query,
            dataset_name=request.dataset_name,
            max_results=request.max_results if request.max_results is not None else 10,
            limit_candidates=request.limit_candidates if request.limit_candidates is not None else 1000,
            search_type=request.search_type if request.search_type else 'and'
        )
        
        return {
            "status": "success",
            "query": request.query,
            "dataset_name": request.dataset_name,
            "search_type": request.search_type,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in TF-IDF search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in TF-IDF search: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
