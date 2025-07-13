from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from services.vectorization.embedding_vectorizer import (
    build_faiss_index_from_existing, 
    check_faiss_index_exists
)

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

@router.post("/build", response_model=BuildFaissResponse)
async def build_faiss_index(request: BuildFaissRequest):
    """
    بناء FAISS index من الملفات الموجودة
    """
    try:
        dataset_name = request.dataset_name
        
        # التحقق من وجود FAISS index
        if check_faiss_index_exists(dataset_name):
            return BuildFaissResponse(
                success=True,
                message=f"FAISS index موجود بالفعل للداتا ست: {dataset_name}",
                faiss_path=f"vectors/{dataset_name.replace('/', '_')}_faiss.index"
            )
        
        # بناء FAISS index
        logger.info(f"Building FAISS index for dataset: {dataset_name}")
        faiss_path = build_faiss_index_from_existing(dataset_name)
        
        return BuildFaissResponse(
            success=True,
            message=f"تم بناء FAISS index بنجاح للداتا ست: {dataset_name}",
            faiss_path=faiss_path
        )
        
    except Exception as e:
        logger.error(f"Error building FAISS index for {request.dataset_name}: {e}")
        return BuildFaissResponse(
            success=False,
            message=f"فشل في بناء FAISS index للداتا ست: {request.dataset_name}",
            error=str(e)
        )

@router.post("/check", response_model=CheckFaissResponse)
async def check_faiss_index(request: CheckFaissRequest):
    """
    التحقق من وجود FAISS index
    """
    try:
        dataset_name = request.dataset_name
        exists = check_faiss_index_exists(dataset_name)
        
        if exists:
            return CheckFaissResponse(
                exists=True,
                dataset_name=dataset_name,
                message=f"FAISS index موجود للداتا ست: {dataset_name}"
            )
        else:
            return CheckFaissResponse(
                exists=False,
                dataset_name=dataset_name,
                message=f"FAISS index غير موجود للداتا ست: {dataset_name}"
            )
            
    except Exception as e:
        logger.error(f"Error checking FAISS index for {request.dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking FAISS index: {str(e)}")

@router.get("/list-available")
async def list_available_datasets():
    """
    قائمة الداتا ست المتوفرة مع حالة FAISS index
    """
    try:
        # قائمة الداتا ست المتوقعة
        datasets = [
            "antique/test",
            "beir_arguana", 
            "beir_trec-covid",
            "msmarco",
            "robust04"
        ]
        
        results = []
        for dataset in datasets:
            exists = check_faiss_index_exists(dataset)
            results.append({
                "dataset_name": dataset,
                "faiss_index_exists": exists,
                "status": "✅ موجود" if exists else "❌ غير موجود"
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
    بناء FAISS index لجميع الداتا ست المتوفرة
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
                if check_faiss_index_exists(dataset):
                    results.append({
                        "dataset": dataset,
                        "status": "skipped",
                        "message": "FAISS index موجود بالفعل"
                    })
                    success_count += 1
                else:
                    faiss_path = build_faiss_index_from_existing(dataset)
                    results.append({
                        "dataset": dataset,
                        "status": "built",
                        "message": "تم بناء FAISS index بنجاح",
                        "faiss_path": faiss_path
                    })
                    success_count += 1
            except Exception as e:
                results.append({
                    "dataset": dataset,
                    "status": "failed",
                    "message": f"فشل في بناء FAISS index: {str(e)}"
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