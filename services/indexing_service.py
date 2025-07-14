import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import os
import logging
import requests
import requests
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Loading required NLTK resources...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

_load_nltk_resources()

try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.error(f"Error loading NLTK resources: {e}")
    stop_words = set()
    lemmatizer = None

def preprocess_for_indexing_via_api(text, min_token_length=2):
    """Call preprocessing API for indexing"""
    url = "http://localhost:8001/preprocess"
    
    payload = {
        "text": text,
        "return_tokens": True,
        "min_length": min_token_length,
        "remove_stopwords": True,
        "use_lemmatization": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["result"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to preprocessing API: {e}")
    except Exception as e:
        raise Exception(f"Error processing text via API: {e}")

def get_preprocessing_stats_via_api(text):
    """Call preprocessing API for statistics"""
    url = "http://localhost:8001/stats"
    
    payload = {
        "text": text
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["statistics"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to preprocessing API: {e}")
    except Exception as e:
        raise Exception(f"Error getting statistics via API: {e}")

def preprocess_for_vectorization_via_api(text):
    """Call preprocessing API for vectorization"""
    url = "http://localhost:8001/preprocess"
    
    payload = {
        "text": text,
        "return_tokens": False,
        "min_length": 2,
        "remove_stopwords": True,
        "use_lemmatization": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["result"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to preprocessing API: {e}")
    except Exception as e:
        raise Exception(f"Error processing text via API: {e}")

def build_inverted_index(docs, min_token_length=2):
    if not docs:
        logger.warning("No documents to build index")
        return {}
    
    inverted_index = defaultdict(set)
    total_docs = len(docs)
    
    logger.info(f"Starting index building for {total_docs} documents...")
    
    total_tokens = 0
    total_stopwords_removed = 0
    total_short_words_removed = 0
    
    for i, doc in enumerate(docs):
        if i % 10000 == 0:
            logger.info(f"Processing document {i+1}/{total_docs}")
        
        try:
            doc_id = doc.get('doc_id')
            content = doc.get('content', '')
            
            if not doc_id or not content:
                continue
                
            tokens = preprocess_for_indexing_via_api(content, min_token_length)
            
            stats = get_preprocessing_stats_via_api(content)
            total_tokens += stats.get('tokens_count', 0)
            total_stopwords_removed += stats.get('stopwords_removed', 0)
            total_short_words_removed += stats.get('short_words_removed', 0)
            
            for token in set(tokens):
                inverted_index[token].add(doc_id)
                    
        except Exception as e:
            logger.error(f"Error processing document {doc.get('doc_id', 'unknown')}: {e}")
            continue

    logger.info("Converting index to storage format...")
    result_index = {}
    for token, doc_ids in inverted_index.items():
        result_index[token] = list(doc_ids)
    
    logger.info(f"📊 Processing statistics:")
    logger.info(f"   - Total processed tokens: {total_tokens}")
    logger.info(f"   - Removed stopwords: {total_stopwords_removed}")
    logger.info(f"   - Removed short words: {total_short_words_removed}")
    logger.info(f"Index built successfully. Number of terms: {len(result_index)}")
    
    return result_index

def save_index(index, dataset_name):
    try:
        safe_name = dataset_name.replace('/', '_')
        os.makedirs("indexes", exist_ok=True)
        path = os.path.join("indexes", f"{safe_name}_inverted_index.joblib")
        
        joblib.dump(index, path)
        
        total_terms = len(index)
        total_postings = sum(len(doc_ids) for doc_ids in index.values())
        
        logger.info(f"✅ Index saved to: {path}")
        logger.info(f"📊 Index statistics: {total_terms} terms, {total_postings} postings")
        
        return path
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise

def load_index(dataset_name):
    try:
        safe_name = dataset_name.replace('/', '_')
        path = os.path.join("indexes", f"{safe_name}_inverted_index.joblib")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        index = joblib.load(path)
        logger.info(f"✅ Index loaded from: {path}")
        return index
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise

def search_in_index(query, dataset_name, max_results=10, search_type='and'):
    try:
        index = load_index(dataset_name)
        query_tokens = preprocess_for_indexing_via_api(query)
        if not query_tokens:
            logger.warning("No valid terms in query")
            return []
        if search_type == 'and':
            matching_docs = set()
            first_token = True
            for token in query_tokens:
                if token in index:
                    doc_ids = set(index[token])
                    if first_token:
                        matching_docs = doc_ids
                        first_token = False
                    else:
                        matching_docs = matching_docs.intersection(doc_ids)
                else:
                    return []
        elif search_type == 'or':
            matching_docs = set()
            for token in query_tokens:
                if token in index:
                    matching_docs.update(index[token])
            if not matching_docs:
                return []
        else:
            raise ValueError("search_type must be 'and' or 'or'")
        result = list(matching_docs)[:max_results]
        logger.info(f"Found {len(result)} matching documents")
        return result
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

def get_index_statistics(dataset_name):
    try:
        safe_name = dataset_name.replace('/', '_')
        path = os.path.join("indexes", f"{safe_name}_inverted_index.joblib")
        index = load_index(dataset_name)
        
        total_terms = len(index)
        total_postings = sum(len(doc_ids) for doc_ids in index.values())
        avg_postings = total_postings / total_terms if total_terms > 0 else 0
        
        sorted_terms = sorted(index.items(), key=lambda x: len(x[1]), reverse=True)
        top_terms = [(term, len(doc_ids)) for term, doc_ids in sorted_terms[:10]]
        
        bottom_terms = [(term, len(doc_ids)) for term, doc_ids in sorted_terms[-10:]]
        
        return {
            'total_terms': total_terms,
            'total_postings': total_postings,
            'average_postings_per_term': avg_postings,
            'top_terms': top_terms,
            'bottom_terms': bottom_terms,
            'index_size_mb': os.path.getsize(path) / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error calculating index statistics: {e}")
        return {}

def inverted_index_tfidf_search(query, dataset_name, max_results=10, limit_candidates=1000, search_type='and'):
    try:
        index = load_index(dataset_name)
        query_tokens = preprocess_for_indexing_via_api(query)
        
        if not query_tokens:
            logger.warning("No valid terms in query")
            return []
        
        candidate_docs = set()
        
        if search_type == 'and':
            first_token = True
            for token in query_tokens:
                if token in index:
                    doc_ids = set(index[token])
                    if first_token:
                        candidate_docs = doc_ids
                        first_token = False
                    else:
                        candidate_docs = candidate_docs.intersection(doc_ids)
                else:
                    return []
        elif search_type == 'or':
            for token in query_tokens:
                if token in index:
                    candidate_docs.update(index[token])
        else:
            raise ValueError("search_type must be 'and' or 'or'")
        
        if not candidate_docs:
            return []
        
        candidate_list = list(candidate_docs)[:limit_candidates]
        
        try:
            vectorizer, tfidf_matrix, doc_ids = load_tfidf_model_via_api(dataset_name)
            query_vector = vectorizer.transform([query])
            
            candidate_indices = []
            for doc_id in candidate_list:
                if doc_id in doc_ids:
                    candidate_indices.append(doc_ids.index(doc_id))
            
            if not candidate_indices:
                return []
            
            candidate_matrix = tfidf_matrix[candidate_indices]
            similarities = cosine_similarity(query_vector, candidate_matrix)[0]
            
            sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
            
            results = []
            for idx in sorted_indices[:max_results]:
                doc_id = candidate_list[candidate_indices[idx]]
                score = similarities[idx]
                results.append({
                    'doc_id': doc_id,
                    'score': float(score)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Error in inverted index TF-IDF search: {e}")
        return []

def embedding_search(query, dataset_name, max_results=10):
    try:
        model, embedding_matrix, doc_ids = load_embedding_model_via_api(dataset_name)
        query_vector = get_embedding_vector_via_api(dataset_name, query)
        # Normalize vectors for cosine similarity
        embedding_matrix_norm = embedding_matrix / (np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10)
        query_vector_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        similarities = np.dot(embedding_matrix_norm, query_vector_norm.T).flatten()
        sorted_indices = np.argsort(similarities)[::-1][:max_results]
        results = []
        for idx in sorted_indices:
            results.append({
                'doc_id': doc_ids[idx],
                'score': float(similarities[idx])
            })
        return results
    except Exception as e:
        logger.error(f"Error in embedding search: {e}")
        return []

def load_tfidf_model_via_api(dataset_name: str):
    """Client function to load TF-IDF model via API"""
    url = "http://localhost:8002/load-tfidf"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            import os
            import joblib
            
            clean_dataset_name = dataset_name.replace('/', '_')
            model_path = os.path.join("models", f"{clean_dataset_name}_tfidf_model.joblib")
            matrix_path = os.path.join("vectors", f"{clean_dataset_name}_tfidf_matrix.joblib")
            
            vectorizer = joblib.load(model_path)
            data = joblib.load(matrix_path)
            tfidf_matrix = data["matrix"]
            doc_ids = data["doc_ids"]
            
            return vectorizer, tfidf_matrix, doc_ids
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load TF-IDF model via API: {e}")

def load_embedding_model_via_api(dataset_name: str):
    """Client function to load Embedding model via API"""
    url = "http://localhost:8003/load-embedding"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Load the actual model files
            import os
            import joblib
            
            clean_dataset_name = dataset_name.replace('/', '_')
            model_path = os.path.join("models", f"{clean_dataset_name}_embedding_model.joblib")
            vectors_path = os.path.join("vectors", f"{clean_dataset_name}_embedding_vectors.joblib")
            
            model = joblib.load(model_path)
            data = joblib.load(vectors_path)
            embeddings = data["embeddings"]
            doc_ids = data["doc_ids"]
            
            return model, embeddings, doc_ids
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load Embedding model via API: {e}")

def get_embedding_vector_via_api(dataset_name: str, query: str):
    """Client function to get Embedding vector via API"""
    url = "http://localhost:8003/vectorize"
    
    payload = {
        "dataset_name": dataset_name,
        "query": query,
        "return_serializable": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return np.array(data["vector"]).reshape(1, -1)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get Embedding vector via API: {e}")
