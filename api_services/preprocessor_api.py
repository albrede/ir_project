from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from services.preprocessing import preprocess, preprocess_for_indexing, get_preprocessing_stats

app = FastAPI(title="Preprocessor Service API", version="1.0.0")

class PreprocessRequest(BaseModel):
    text: str
    return_tokens: Optional[bool] = False
    min_length: Optional[int] = 2
    remove_stopwords: Optional[bool] = True
    use_lemmatization: Optional[bool] = True

@app.get("/")
def read_root():
    return {
        "service": "Preprocessor Service",
        "version": "1.0.0",
        "description": "Text processing service (cleaning, stopwords removal, lemmatization, etc.)",
        "endpoints": {
            "POST /preprocess": "Process text and return text or word list",
            "POST /stats": "Return preprocessing statistics for text"
        }
    }

@app.post("/preprocess")
def preprocess_text(request: PreprocessRequest):
    result = preprocess(
        text=request.text,
        return_tokens=request.return_tokens if request.return_tokens is not None else False,
        min_length=request.min_length if request.min_length is not None else 2,
        remove_stopwords=request.remove_stopwords if request.remove_stopwords is not None else True,
        use_lemmatization=request.use_lemmatization if request.use_lemmatization is not None else True
    )
    return {
        "status": "success",
        "result": result
    }

class StatsRequest(BaseModel):
    text: str

@app.post("/stats")
def preprocessing_stats(request: StatsRequest):
    stats = get_preprocessing_stats(request.text)
    return {
        "status": "success",
        "statistics": stats
    }

