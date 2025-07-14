from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from services.preprocessing import preprocess, preprocess_for_indexing, get_preprocessing_stats
import requests

app = FastAPI(title="Preprocessor Service API", version="1.0.0")

class PreprocessRequest(BaseModel):
    text: str
    return_tokens: Optional[bool] = False
    min_length: Optional[int] = 2
    remove_stopwords: Optional[bool] = True
    use_lemmatization: Optional[bool] = True

class TokenizeRequest(BaseModel):
    text: str
    return_tokens: Optional[bool] = True
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
            "POST /tokenize": "Tokenize text and return tokens list",
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

@app.post("/tokenize")
def tokenize_text(request: TokenizeRequest):
    """Endpoint specifically for tokenization used by TF-IDF vectorizer"""
    tokens = preprocess(
        text=request.text,
        return_tokens=True,
        min_length=request.min_length if request.min_length is not None else 2,
        remove_stopwords=request.remove_stopwords if request.remove_stopwords is not None else True,
        use_lemmatization=request.use_lemmatization if request.use_lemmatization is not None else True
    )
    return {
        "status": "success",
        "tokens": tokens
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

# Client function for other services to use
def preprocess_via_api(text, return_tokens=False, min_length=2, remove_stopwords=True, use_lemmatization=True):
    """Client function to call preprocessing API"""
    url = "http://localhost:8001/tokenize" if return_tokens else "http://localhost:8001/preprocess"
    
    payload = {
        "text": text,
        "return_tokens": return_tokens,
        "min_length": min_length,
        "remove_stopwords": remove_stopwords,
        "use_lemmatization": use_lemmatization
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            if return_tokens:
                return data["tokens"]
            else:
                return data["result"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to preprocessing API: {e}")
    except Exception as e:
        raise Exception(f"Error processing text via API: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

