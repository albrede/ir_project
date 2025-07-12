import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from services.preprocessing import preprocess

def custom_tokenizer(text):
    return preprocess(text, return_tokens=True)

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
