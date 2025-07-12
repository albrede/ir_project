import os
import joblib
import time
import numpy as np
from sentence_transformers import SentenceTransformer

VECTORS_DIR = "vectors"
MODELS_DIR = "models"

def build_embedding_model(texts, save_name, doc_ids=None, model_name='paraphrase-MiniLM-L3-v2', max_retries=3):
    print(f"[⏳] Loading model: {model_name}")
    for attempt in range(max_retries):
        try:
            model = SentenceTransformer(model_name, device='cpu')
            break
        except Exception as e:
            print(f"[⚠️] Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"[⏳] Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"[❌] Failed to load model after {max_retries} attempts")
                raise e
    print(f"[⏳] Creating vectors for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    os.makedirs(VECTORS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    clean_save_name = save_name.replace('/', '_')
    vectors_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_embedding_vectors.joblib")
    data = {"doc_ids": doc_ids, "vectors": embeddings}
    joblib.dump(data, vectors_path)
    model_path = os.path.join(MODELS_DIR, f"{clean_save_name}_embedding_model.joblib")
    joblib.dump(model, model_path)
    print(f"[✓] Documents represented using Embedding and results stored.")
    print(f"[📁] Model saved in: {model_path}")
    print(f"[📁] Vectors with doc_ids saved in: {vectors_path}")
    
    # ---FAISS index ---
    try:
        import faiss
        faiss_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_faiss.index")
        emb_float32 = np.atleast_2d(embeddings.astype(np.float32))
        dim = emb_float32.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(n=emb_float32.shape[0], x=emb_float32)
        faiss.write_index(index, faiss_path)
        print(f"[✓] FAISS index saved at: {faiss_path}")
    except Exception as e:
        print(f"[⚠️] Could not build FAISS index: {e}")
    
    return model, embeddings, doc_ids

def get_embedding_vector(model, query):
    query_vector = model.encode([query], convert_to_numpy=True)
    return query_vector.reshape(1, -1)

def get_embedding_vector_serializable(model, query):
    query_vector = model.encode([query], convert_to_numpy=True)
    return [float(x) for x in query_vector.flatten()]

def load_embedding_model(save_name):
    clean_save_name = save_name.replace('/', '_')
    model_path = os.path.join(MODELS_DIR, f"{clean_save_name}_embedding_model.joblib")
    vectors_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_embedding_vectors.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(f"Vectors not found: {vectors_path}")
    print(f"[⏳] Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"[⏳] Loading vectors from: {vectors_path}")
    data = joblib.load(vectors_path)
    embeddings = data["vectors"]
    doc_ids = data["doc_ids"]
    print(f"[✓] Model and vectors loaded successfully")
    return model, embeddings, doc_ids

def get_similarity_scores(query_vector, document_vectors):
    similarities = np.dot(document_vectors, query_vector.T).flatten()
    return similarities

def search_faiss_index(query_embedding, dataset_name, top_k=5):

    try:
        import faiss
        clean_save_name = dataset_name.replace('/', '_')
        faiss_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_faiss.index")
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        
        index = faiss.read_index(faiss_path)
        
        query_float32 = np.atleast_2d(query_embedding.astype(np.float32))
        
        D, I = index.search(query_float32, top_k)
        
        print(f"[✓] FAISS search completed, found {len(I[0])} results")
        return I[0]  
        
    except Exception as e:
        print(f"[❌] Error in FAISS search: {e}")
        return []

def load_faiss_index(dataset_name):

    try:
        import faiss
        clean_save_name = dataset_name.replace('/', '_')
        faiss_path = os.path.join(VECTORS_DIR, f"{clean_save_name}_faiss.index")
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        
        index = faiss.read_index(faiss_path)
        print(f"[✓] FAISS index loaded from: {faiss_path}")
        return index
        
    except Exception as e:
        print(f"[❌] Error loading FAISS index: {e}")
        return None
