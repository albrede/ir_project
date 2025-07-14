import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests
from services.preprocessing import preprocess_for_vectorization
from services.database.db_handler import get_documents
from services.indexing_service import search_in_index
import requests

logger = logging.getLogger(__name__)

def preprocess_for_vectorization_via_api(text):
    """Call preprocessing API for vectorization"""
    url = "http://localhost:8006/preprocess"
    
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
        raise Exception(f"Failed to connect to query processor API: {e}")
    except Exception as e:
        raise Exception(f"Error processing text via API: {e}")

def load_tfidf_model_via_query_processor_api(dataset_name: str):
    """Client function to load TF-IDF model via Query Processor API"""
    url = "http://localhost:8006/load-tfidf"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["vectorizer"], data["matrix"], data["doc_ids"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load TF-IDF model via Query Processor API: {e}")

def get_tfidf_vector_via_query_processor_api(dataset_name: str, query: str):
    """Client function to get TF-IDF vector via Query Processor API"""
    url = "http://localhost:8006/get-tfidf-vector"
    
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
            import numpy as np
            return np.array(data["vector"]).reshape(1, -1)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get TF-IDF vector via Query Processor API: {e}")

def load_embedding_model_via_query_processor_api(dataset_name: str):
    """Client function to load Embedding model via Query Processor API"""
    url = "http://localhost:8006/load-embedding"
    
    payload = {
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["model"], data["embeddings"], data["doc_ids"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to load Embedding model via Query Processor API: {e}")

def get_embedding_vector_via_query_processor_api(dataset_name: str, query: str):
    """Client function to get Embedding vector via Query Processor API"""
    url = "http://localhost:8006/get-embedding-vector"
    
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
            import numpy as np
            return np.array(data["vector"]).reshape(1, -1)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get Embedding vector via Query Processor API: {e}")

def get_query_vector_for_matching_via_api(query: str, method: str, dataset_name: str):
    """Client function to get query vector for matching via Query Processor API"""
    url = "http://localhost:8006/get-query-vector"
    
    payload = {
        "query": query,
        "method": method,
        "dataset_name": dataset_name
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            if method in ["tfidf", "embedding"]:
                import numpy as np
                return np.array(data["vector"]).reshape(1, -1)
            elif method in ["hybrid", "hybrid-sequential"]:
                import numpy as np
                return {
                    "tfidf": np.array(data["tfidf"]).reshape(1, -1),
                    "embedding": np.array(data["embedding"]).reshape(1, -1)
                }
            else:
                raise ValueError(f"Unsupported method: {method}")
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get query vector via Query Processor API: {e}")

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

def get_tfidf_vector_via_api(dataset_name: str, query: str):
    """Client function to get TF-IDF vector via API"""
    url = "http://localhost:8002/vectorize"
    
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
            import numpy as np
            return np.array(data["vector"]).reshape(1, -1)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get TF-IDF vector via API: {e}")

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
            import numpy as np
            return np.array(data["vector"]).reshape(1, -1)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get Embedding vector via API: {e}")

def parallel_hybrid_search_via_api(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, 
                                  alpha=0.5, top_k=10, doc_ids=None):
    """Client function to perform parallel hybrid search via API"""
    url = "http://localhost:8004/parallel-hybrid"

    # --- Modification: Convert matrices to dense and float32 with size check ---
    import numpy as np
    from scipy.sparse import issparse
    def to_dense_float32(arr):
        if issparse(arr):
            shape = arr.shape
            estimated_memory_gb = (shape[0] * shape[1] * 4) / (1024**3)  # 4 bytes for float32
            if estimated_memory_gb > 1.0:
                logger.warning(f"Matrix too large ({shape[0]}x{shape[1]}, ~{estimated_memory_gb:.1f}GB). Using sparse matrix.")
                return arr
            arr = arr.toarray()
        return np.asarray(arr, dtype=np.float32)

    tfidf_query_vec = to_dense_float32(tfidf_query_vec)
    tfidf_doc_matrix = to_dense_float32(tfidf_doc_matrix)
    emb_query_vec = to_dense_float32(emb_query_vec)
    emb_doc_matrix = to_dense_float32(emb_doc_matrix)
    # ---------------------------------------------------

    # Convert matrices to list for JSON payload
    def matrix_to_list(matrix):
        if hasattr(matrix, 'tolist'):
            return matrix.tolist()
        elif hasattr(matrix, 'toarray'):
            return matrix.toarray().tolist()
        else:
            return matrix.tolist() if hasattr(matrix, 'tolist') else matrix

    payload = {
        "tfidf_query_vec": matrix_to_list(tfidf_query_vec),
        "tfidf_doc_matrix": matrix_to_list(tfidf_doc_matrix),
        "emb_query_vec": matrix_to_list(emb_query_vec),
        "emb_doc_matrix": matrix_to_list(emb_doc_matrix),
        "alpha": alpha,
        "top_k": top_k,
        "doc_ids": doc_ids
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Extract results
            results = data["results"]
            top_indices = [result["index"] for result in results]
            top_scores = [result["score"] for result in results]
            
            if doc_ids is not None:
                top_doc_ids = [result.get("doc_id", str(result["index"])) for result in results]
                return top_doc_ids, np.array(top_scores)
            else:
                return np.array(top_indices), np.array(top_scores)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to perform parallel hybrid search via API: {e}")


def sequential_hybrid_search_via_api(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, 
                                   first_stage='tfidf', top_n=100, top_k=10, doc_ids=None):
    """Client function to perform sequential hybrid search via API"""
    url = "http://localhost:8005/sequential-hybrid"

    # --- Modification: Convert matrices to dense and float32 with size check ---
    import numpy as np
    from scipy.sparse import issparse
    def to_dense_float32(arr):
        if issparse(arr):
            shape = arr.shape
            estimated_memory_gb = (shape[0] * shape[1] * 4) / (1024**3)  # 4 bytes for float32
            if estimated_memory_gb > 1.0:
                logger.warning(f"Matrix too large ({shape[0]}x{shape[1]}, ~{estimated_memory_gb:.1f}GB). Using sparse matrix.")
                return arr
            arr = arr.toarray()
        return np.asarray(arr, dtype=np.float32)

    tfidf_query_vec = to_dense_float32(tfidf_query_vec)
    tfidf_doc_matrix = to_dense_float32(tfidf_doc_matrix)
    emb_query_vec = to_dense_float32(emb_query_vec)
    emb_doc_matrix = to_dense_float32(emb_doc_matrix)
    # ---------------------------------------------------

    # Convert matrices to list for JSON payload
    def matrix_to_list(matrix):
        if hasattr(matrix, 'tolist'):
            return matrix.tolist()
        elif hasattr(matrix, 'toarray'):
            return matrix.toarray().tolist()
        else:
            return matrix.tolist() if hasattr(matrix, 'tolist') else matrix

    payload = {
        "tfidf_query_vec": matrix_to_list(tfidf_query_vec),
        "tfidf_doc_matrix": matrix_to_list(tfidf_doc_matrix),
        "emb_query_vec": matrix_to_list(emb_query_vec),
        "emb_doc_matrix": matrix_to_list(emb_doc_matrix),
        "first_stage": first_stage,
        "top_n": top_n,
        "top_k": top_k,
        "doc_ids": doc_ids
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            # Extract results
            results = data["results"]
            top_indices = [result["index"] for result in results]
            top_scores = [result["score"] for result in results]
            
            if doc_ids is not None:
                top_doc_ids = [result.get("doc_id", str(result["index"])) for result in results]
                return top_doc_ids, np.array(top_scores)
            else:
                return np.array(top_indices), np.array(top_scores)
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to perform sequential hybrid search via API: {e}")

class Matcher:
    def __init__(self, doc_vectors, vectorizer=None, method="tfidf", embedding_model=None, hybrid_weights=None, dataset_name="simple", embedding_vectors=None):
        """
        Document matcher initialization
        Args:
            doc_vectors: Document vectors
            vectorizer: TF-IDF vectorizer
            method: Matching method ("tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index")
            embedding_model: Embedding model
            hybrid_weights: Weights for hybrid search (tfidf_weight, embedding_weight)
            dataset_name: Dataset name
        """
        self.doc_vectors = doc_vectors
        self.vectorizer = vectorizer
        self.method = method
        self.embedding_model = embedding_model
        self.hybrid_weights = hybrid_weights or (0.5, 0.5)  # Default weights for hybrid search
        self.dataset_name = dataset_name
        self.embedding_vectors = embedding_vectors  # New
        
        # Load required models
        self._load_models()
        
        # Validate parameters
        self._validate_parameters()
        
        # QueryProcessor is now handled via API
        pass
    
    def _load_models(self):
        """Load required models based on the method"""
        try:
            if self.method in ["tfidf", "hybrid", "hybrid-sequential"] and self.vectorizer is None:
                # Load TF-IDF model
                try:
                    self.vectorizer, _, _ = load_tfidf_model_via_query_processor_api(self.dataset_name)
                    logger.info(f"TF-IDF model loaded for {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to load TF-IDF model for {self.dataset_name}: {e}")
            
            if self.method in ["embedding", "hybrid", "hybrid-sequential"] and self.embedding_model is None:
                # Load Embedding model
                try:
                    self.embedding_model, _, _ = load_embedding_model_via_query_processor_api(self.dataset_name)
                    logger.info(f"Embedding model loaded for {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to load Embedding model for {self.dataset_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _validate_parameters(self):
        """Validate parameters"""
        if self.method == "tfidf" and self.vectorizer is None:
            raise ValueError("vectorizer is required for TF-IDF method")
        
        if self.method == "embedding" and self.embedding_model is None:
            raise ValueError("embedding_model is required for Embedding method")
        
        if self.method in ["hybrid", "hybrid-sequential"]:
            if self.vectorizer is None or self.embedding_model is None:
                raise ValueError("Both vectorizer and embedding_model are required for hybrid search")
            
            if len(self.hybrid_weights) != 2:
                raise ValueError("hybrid_weights must contain two values")
            
            if sum(self.hybrid_weights) != 1.0:
                logger.warning("Sum of hybrid search weights does not equal 1.0, they will be normalized")
                total = sum(self.hybrid_weights)
                self.hybrid_weights = (self.hybrid_weights[0]/total, self.hybrid_weights[1]/total)

    def match(self, query, top_k=None):
        """
        Match query against documents
        Args:
            query: Query (string or vector)
            top_k: Number of results to return (None for all results)
        Returns:
            List of tuples (doc_index, similarity_score) sorted descending
        """
        try:
            # Use Query Processor API to represent the query
            if self.method in ["tfidf", "embedding"]:
                query_vec = get_query_vector_for_matching_via_api(query, self.method, self.dataset_name)
                if self.method == "tfidf":
                    return self._match_tfidf(query_vec, top_k, already_vectorized=True)
                else:
                    return self._match_embedding(query_vec, top_k, already_vectorized=True)
            elif self.method == "hybrid":
                query_vecs = get_query_vector_for_matching_via_api(query, self.method, self.dataset_name)
                return self._match_hybrid(query_vecs, top_k, already_vectorized=True)
            elif self.method == "hybrid-sequential":
                query_vecs = get_query_vector_for_matching_via_api(query, self.method, self.dataset_name)
                return self._match_hybrid_sequential(query_vecs, top_k, already_vectorized=True)
            elif self.method == "inverted_index":
                return self._match_inverted_index(query, top_k)
            else:
                raise ValueError(f"Matching method '{self.method}' is not supported")
                
        except Exception as e:
            logger.error(f"Error matching query: {e}")
            raise
    
    def _match_tfidf(self, query_vec, top_k=None, already_vectorized=False):
        """Match TF-IDF using the same logic as search_api.py"""
        if not already_vectorized:
            if self.vectorizer is None:
                raise ValueError("vectorizer is required for TF-IDF method")
            if isinstance(query_vec, str):
                processed_query = preprocess_for_vectorization_via_api(query_vec)
                query_vec = get_tfidf_vector_via_api(self.dataset_name, processed_query)
            else:
                query_vec = query_vec.reshape(1, -1)
        # Now: query_vec and self.doc_vectors are both sparse
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        if top_k is None:
            top_k = len(sims)
        top_indices = sims.argsort()[-top_k:][::-1]
        results = [(int(idx), float(sims[idx])) for idx in top_indices]
        return results
    
    def _match_embedding(self, query_vec, top_k=None, already_vectorized=False):
        """Match Embedding"""
        if not already_vectorized:
            if self.embedding_model is None:
                raise ValueError("embedding_model is required for Embedding method")
            if isinstance(query_vec, str):
                query_vec = get_embedding_vector_via_api(self.dataset_name, query_vec)
                query_vec = query_vec.reshape(1, -1)
            else:
                query_vec = query_vec.reshape(1, -1)
        
        sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
        return self._rank_results(sims, top_k)
    
    def _match_hybrid(self, query_vecs, top_k=None, already_vectorized=False):
        """Hybrid matching combining TF-IDF and Embedding using Hybrid API"""
        if not already_vectorized:
            # Query is pre-processed
            pass
        
        tfidf_vec = query_vecs["tfidf"]
        embedding_vec = query_vecs["embedding"]
        
        # Check matrix sizes
        tfidf_shape = self.doc_vectors.shape if hasattr(self.doc_vectors, 'shape') else (0, 0)
        estimated_memory_gb = (tfidf_shape[0] * tfidf_shape[1] * 8) / (1024**3)  # 8 bytes for float64
        
        if estimated_memory_gb > 1.0:
            logger.info(f"Matrix too large ({tfidf_shape[0]}x{tfidf_shape[1]}, ~{estimated_memory_gb:.1f}GB). Using local hybrid method.")
            # Use local method directly for large matrices
            tfidf_scores = cosine_similarity(tfidf_vec, self.doc_vectors if self.doc_vectors is not None else np.zeros((1, 1))).flatten()
            if self.embedding_vectors is not None:
                embedding_scores = cosine_similarity(embedding_vec, self.embedding_vectors).flatten()
            else:
                embedding_scores = np.zeros_like(tfidf_scores)
            tfidf_weight, embedding_weight = self.hybrid_weights
            combined_scores = tfidf_weight * tfidf_scores + embedding_weight * embedding_scores
            
            return self._rank_results(combined_scores, top_k, doc_count=self.doc_vectors.shape[0] if self.doc_vectors is not None else 1)
        
        # Use Hybrid API for small matrices
        try:
            tfidf_weight, embedding_weight = self.hybrid_weights
            
            # Call Hybrid API
            top_indices, top_scores = parallel_hybrid_search_via_api(
                tfidf_query_vec=tfidf_vec,
                tfidf_doc_matrix=self.doc_vectors if self.doc_vectors is not None else np.zeros((1, 1)),
                emb_query_vec=embedding_vec,
                emb_doc_matrix=self.embedding_vectors if self.embedding_vectors is not None else np.zeros((1, 1)),
                alpha=tfidf_weight,
                top_k=top_k or 10,
                doc_ids=None
            )
            
            # Convert results to the required format
            results = list(zip(top_indices, top_scores))
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search via API: {e}")
            # Fallback to local method
            logger.info("Using local method as fallback")
            tfidf_scores = cosine_similarity(tfidf_vec, self.doc_vectors if self.doc_vectors is not None else np.zeros((1, 1))).flatten()
            if self.embedding_vectors is not None:
                embedding_scores = cosine_similarity(embedding_vec, self.embedding_vectors).flatten()
            else:
                embedding_scores = np.zeros_like(tfidf_scores)
            tfidf_weight, embedding_weight = self.hybrid_weights
            combined_scores = tfidf_weight * tfidf_scores + embedding_weight * embedding_scores
            
            return self._rank_results(combined_scores, top_k, doc_count=self.doc_vectors.shape[0] if self.doc_vectors is not None else 1)
    
    def _match_hybrid_sequential(self, query_vecs, top_k=None, already_vectorized=False, first_stage='tfidf', top_n=100, doc_ids=None):
        """Sequential hybrid matching using Hybrid Sequential API"""
        if not already_vectorized:
            # Query is pre-processed
            pass
        
        tfidf_vec = query_vecs["tfidf"]
        embedding_vec = query_vecs["embedding"]
        
        # Check matrix sizes
        tfidf_shape = self.doc_vectors.shape if hasattr(self.doc_vectors, 'shape') else (0, 0)
        estimated_memory_gb = (tfidf_shape[0] * tfidf_shape[1] * 8) / (1024**3)  # 8 bytes for float64
        
        if estimated_memory_gb > 1.0:
            logger.info(f"Matrix too large ({tfidf_shape[0]}x{tfidf_shape[1]}, ~{estimated_memory_gb:.1f}GB). Using local hybrid sequential method.")
            # Use local method directly for large matrices
            tfidf_matrix = self.doc_vectors if self.doc_vectors is not None else np.zeros((1, 1))
            emb_matrix = self.embedding_vectors if self.embedding_vectors is not None else np.zeros((1, 1))
            
            if first_stage == 'tfidf':
                first_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
                top_n_indices = np.argsort(first_scores)[::-1][:top_n]
                emb_sub_matrix = emb_matrix[top_n_indices]
                emb_scores = cosine_similarity(embedding_vec, emb_sub_matrix)[0]
                final_indices = np.argsort(emb_scores)[::-1][:top_k]
                top_indices = top_n_indices[final_indices]
                top_scores = emb_scores[final_indices]
            else:
                first_scores = cosine_similarity(embedding_vec, emb_matrix)[0]
                top_n_indices = np.argsort(first_scores)[::-1][:top_n]
                tfidf_sub_matrix = tfidf_matrix[top_n_indices]
                tfidf_scores = cosine_similarity(tfidf_vec, tfidf_sub_matrix)[0]
                final_indices = np.argsort(tfidf_scores)[::-1][:top_k]
                top_indices = top_n_indices[final_indices]
                top_scores = tfidf_scores[final_indices]
            
            results = list(zip(top_indices, top_scores))
            return results
        
        # Use Hybrid Sequential API for small matrices
        try:
            # Call Hybrid Sequential API
            top_indices, top_scores = sequential_hybrid_search_via_api(
                tfidf_query_vec=tfidf_vec,
                tfidf_doc_matrix=self.doc_vectors if self.doc_vectors is not None else np.zeros((1, 1)),
                emb_query_vec=embedding_vec,
                emb_doc_matrix=self.embedding_vectors if self.embedding_vectors is not None else np.zeros((1, 1)),
                first_stage=first_stage,
                top_n=top_n,
                top_k=top_k or 10,
                doc_ids=doc_ids
            )
            
            # Convert results to the required format
            results = list(zip(top_indices, top_scores))
            return results
            
        except Exception as e:
            logger.error(f"Error in sequential hybrid search via API: {e}")
            # Fallback to local method
            logger.info("Using local method as fallback")
            
            tfidf_matrix = self.doc_vectors if self.doc_vectors is not None else np.zeros((1, 1))
            emb_matrix = self.embedding_vectors if self.embedding_vectors is not None else np.zeros((1, 1))
            
            if first_stage == 'tfidf':
                first_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
                top_n_indices = np.argsort(first_scores)[::-1][:top_n]
                emb_sub_matrix = emb_matrix[top_n_indices]
                emb_scores = cosine_similarity(embedding_vec, emb_sub_matrix)[0]
                final_indices = np.argsort(emb_scores)[::-1][:top_k]
                top_indices = top_n_indices[final_indices]
                top_scores = emb_scores[final_indices]
            else:
                first_scores = cosine_similarity(embedding_vec, emb_matrix)[0]
                top_n_indices = np.argsort(first_scores)[::-1][:top_n]
                tfidf_sub_matrix = tfidf_matrix[top_n_indices]
                tfidf_scores = cosine_similarity(tfidf_vec, tfidf_sub_matrix)[0]
                final_indices = np.argsort(tfidf_scores)[::-1][:top_k]
                top_indices = top_n_indices[final_indices]
                top_scores = tfidf_scores[final_indices]
            
            results = list(zip(top_indices, top_scores))
            return results
    
    def _match_inverted_index(self, query, top_k=None):
        """Match using inverted index"""
        if isinstance(query, str):
            # Search in inverted index
            matching_doc_ids = search_in_index(query, self.dataset_name, top_k or 10, "and")
            
            # Convert document IDs to indices
            results = []
            for i, doc_id in enumerate(matching_doc_ids):
                # Search for the index in the document ID list (if available)
                try:
                    doc_index = int(doc_id)
                    results.append((doc_index, 1.0))  # Constant score for inverted index
                except ValueError:
                    # If document ID is not a number, use the index
                    results.append((i, 1.0))
            
            return results
        else:
            raise ValueError("Inverted index requires a query string")
    
    def _rank_results(self, similarities, top_k=None, doc_indices=None, doc_count=None):
        """Rank results by similarity score"""
        if len(similarities) == 0:
            return []
        
        # Sort descending
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_scores = similarities[ranked_indices]
        
        # Apply top_k if specified
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]
            ranked_scores = ranked_scores[:top_k]
        
        # If custom document indices are provided, use them
        if doc_indices is not None:
            return list(zip([doc_indices[i] for i in ranked_indices], ranked_scores))
        else:
            # Use doc_count if available instead of len(self.doc_vectors)
            return list(zip(ranked_indices, ranked_scores))
    
    def get_match_info(self, query):
        """Get information about the matching process"""
        return {
            "method": self.method,
            "documents_count": self.doc_vectors.shape[0],
            "query_type": type(query).__name__,
            "hybrid_weights": self.hybrid_weights if self.method in ["hybrid", "hybrid-sequential"] else None,
            "dataset_name": self.dataset_name,
            "models_loaded": {
                "vectorizer": self.vectorizer is not None,
                "embedding_model": self.embedding_model is not None
            }
        }
    
    def get_supported_methods(self):
        """Get supported methods"""
        return ["tfidf", "embedding", "hybrid", "hybrid-sequential", "inverted_index"]
    
    def validate_method(self, method):
        """Validate method"""
        return method in self.get_supported_methods()
