from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import requests

router = APIRouter(prefix="/faiss", tags=["FAISS Index Management"])
logger = logging.getLogger(__name__)

class BuildFaissRequest(BaseModel):
    dataset_name: str

class BuildFaissResponse(BaseModel):
    success: bool
    message: str
    faiss_path: Optional[str] = None
    error: Optional[str] = None

class CheckFaissRequest(BaseModel):
    dataset_name: str

class CheckFaissResponse(BaseModel):
    exists: bool
    dataset_name: str
    message: str

def build_faiss_index_from_existing_via_api(dataset_name: str):
    """Client function to build FAISS index via API"""
    url = "http://localhost:8003/build-faiss"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)  # Longer timeout for building
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["faiss_path"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to build FAISS index via API: {e}")

def check_faiss_index_exists_via_api(dataset_name: str):
    """Client function to check FAISS index via API"""
    url = "http://localhost:8003/check-faiss"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["exists"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to check FAISS index via API: {e}")

@router.post("/build", response_model=BuildFaissResponse)
async def build_faiss_index(request: BuildFaissRequest):
    """
    Build FAISS index from existing files
    """
    try:
        dataset_name = request.dataset_name
        
        # Check if FAISS index exists
        if check_faiss_index_exists_via_api(dataset_name):
            return BuildFaissResponse(
                success=True,
                message=f"FAISS index already exists for dataset: {dataset_name}",
                faiss_path=f"vectors/{dataset_name.replace('/', '_')}_faiss.index"
            )
        
        # Build FAISS index
        logger.info(f"Building FAISS index for dataset: {dataset_name}")
        faiss_path = build_faiss_index_from_existing_via_api(dataset_name)
        
        return BuildFaissResponse(
            success=True,
            message=f"Successfully built FAISS index for dataset: {dataset_name}",
            faiss_path=faiss_path
        )
        
    except Exception as e:
        logger.error(f"Error building FAISS index for {request.dataset_name}: {e}")
        return BuildFaissResponse(
            success=False,
            message=f"Failed to build FAISS index for dataset: {request.dataset_name}",
            error=str(e)
        )

@router.post("/check", response_model=CheckFaissResponse)
async def check_faiss_index(request: CheckFaissRequest):
    """
    Check if FAISS index exists
    """
    try:
        dataset_name = request.dataset_name
        exists = check_faiss_index_exists_via_api(dataset_name)
        
        if exists:
            return CheckFaissResponse(
                exists=True,
                dataset_name=dataset_name,
                message=f"FAISS index exists for dataset: {dataset_name}"
            )
        else:
            return CheckFaissResponse(
                exists=False,
                dataset_name=dataset_name,
                message=f"FAISS index does not exist for dataset: {dataset_name}"
            )
            
    except Exception as e:
        logger.error(f"Error checking FAISS index for {request.dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking FAISS index: {str(e)}")

@router.get("/list-available")
async def list_available_datasets():
    """
    List available datasets with FAISS index status
    """
    try:
        # List of expected datasets
        datasets = [
            "antique/test",
            "beir_arguana", 
            "beir_trec-covid",
            "msmarco",
            "robust04"
        ]
        
        results = []
        for dataset in datasets:
            exists = check_faiss_index_exists_via_api(dataset)
            results.append({
                "dataset_name": dataset,
                "faiss_index_exists": exists,
                "status": "✅ Exists" if exists else "❌ Not found"
            })
        
        return {
            "datasets": results,
            "total": len(results),
            "with_faiss": sum(1 for r in results if r["faiss_index_exists"]),
            "without_faiss": sum(1 for r in results if not r["faiss_index_exists"])
        }
        
    except Exception as e:
        logger.error(f"Error listing available datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@router.post("/build-all")
async def build_faiss_for_all_datasets():
    """
    Build FAISS index for all available datasets
    """
    try:
        datasets = [
            "antique/test",
            "beir_arguana", 
            "beir_trec-covid",
            "msmarco",
            "robust04"
        ]
        
        results = []
        success_count = 0
        
        for dataset in datasets:
            try:
                if check_faiss_index_exists_via_api(dataset):
                    results.append({
                        "dataset": dataset,
                        "status": "skipped",
                        "message": "FAISS index already exists"
                    })
                    success_count += 1
                else:
                    faiss_path = build_faiss_index_from_existing_via_api(dataset)
                    results.append({
                        "dataset": dataset,
                        "status": "built",
                        "message": "Successfully built FAISS index",
                        "faiss_path": faiss_path
                    })
                    success_count += 1
            except Exception as e:
                results.append({
                    "dataset": dataset,
                    "status": "failed",
                    "message": f"Failed to build FAISS index: {str(e)}"
                })
        
        return {
            "results": results,
            "summary": {
                "total": len(datasets),
                "successful": success_count,
                "failed": len(datasets) - success_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error building FAISS for all datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error building FAISS for all datasets: {str(e)}") 