import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

def preprocess_via_api(text, return_tokens=True):
    """Call preprocessing API for tokenization"""
    url = "http://localhost:8001/tokenize"
    
    payload = {
        "text": text,
        "return_tokens": True,
        "min_length": 2,
        "remove_stopwords": True,
        "use_lemmatization": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["tokens"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to preprocessing API: {e}")
    except Exception as e:
        raise Exception(f"Error processing text via API: {e}")

def custom_tokenizer(text):
    return preprocess_via_api(text, return_tokens=True)

VECTORS_DIR = "vectors"
MODELS_DIR = "models"

def build_tfidf_model(texts, save_name, doc_ids=None):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        lowercase=False,
        max_features=500000
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    os.makedirs(VECTORS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    clean_save_name = save_name.replace('/', '_')

    matrix_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_tfidf_matrix.joblib")
    data = {"doc_ids": doc_ids, "matrix": tfidf_matrix}
    joblib.dump(data, matrix_path)
    model_path = os.path.join(MODELS_DIR, f"{clean_save_name}_tfidf_model.joblib")
    joblib.dump(vectorizer, model_path)

    print(f"[✓] TF-IDF model created: {save_name}")
    print(f"[✓] Model saved in: {model_path}")
    print(f"[✓] Matrix with doc_ids saved in: {matrix_path}")
    return vectorizer, tfidf_matrix, doc_ids

def get_tfidf_vector(vectorizer, query):
    query_vector = vectorizer.transform([query])
    return query_vector.toarray()  # دائماً numpy array 2D

def get_tfidf_vector_serializable(vectorizer, query):
    query_vector = vectorizer.transform([query])
    return [float(x) for x in query_vector.toarray().flatten()]

def load_tfidf_model(save_name):
    clean_save_name = save_name.replace('/', '_')
    model_path = os.path.join(MODELS_DIR, f"{clean_save_name}_tfidf_model.joblib")
    matrix_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_tfidf_matrix.joblib")
    if not os.path.exists(model_path) or not os.path.exists(matrix_path):
        raise FileNotFoundError(f"TF-IDF model not found: {save_name}")
    vectorizer = joblib.load(model_path)
    data = joblib.load(matrix_path)
    tfidf_matrix = data["matrix"]
    doc_ids = data["doc_ids"]
    print(f"[✓] TF-IDF model loaded: {save_name}")
    return vectorizer, tfidf_matrix, doc_ids
