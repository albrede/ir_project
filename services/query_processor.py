import logging
import requests
import requests
import requests
import joblib
import os
import numpy as np

logger = logging.getLogger(__name__)

def preprocess_via_api(text, return_tokens=False, min_length=2, remove_stopwords=True, use_lemmatization=True):
    """Call preprocessing API"""
    url = "http://localhost:8001/preprocess"
    
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
            return data["result"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to preprocessing API: {e}")
    except Exception as e:
        raise Exception(f"Error processing text via API: {e}")

def preprocess_for_vectorization_via_api(text):
    """Call preprocessing API for vectorization"""
    return preprocess_via_api(text, return_tokens=False, min_length=2, remove_stopwords=True, use_lemmatization=True)

def preprocess_for_indexing_via_api(text, min_token_length=2):
    """Call preprocessing API for indexing"""
    return preprocess_via_api(text, return_tokens=True, min_length=min_token_length, remove_stopwords=True, use_lemmatization=True)

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

def get_tfidf_vector_serializable_via_api(dataset_name: str, query: str):
    """Client function to get serializable TF-IDF vector via API"""
    url = "http://localhost:8002/vectorize"
    
    payload = {
        "dataset_name": dataset_name,
        "query": query,
        "return_serializable": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["vector"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get serializable TF-IDF vector via API: {e}")

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
            # Handle both "vectors" and "embeddings" keys for compatibility
            if "vectors" in data:
                embeddings = data["vectors"]
            elif "embeddings" in data:
                embeddings = data["embeddings"]
            else:
                raise KeyError("Neither 'vectors' nor 'embeddings' key found in data")
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

def get_embedding_vector_serializable_via_api(dataset_name: str, query: str):
    """Client function to get serializable Embedding vector via API"""
    url = "http://localhost:8003/vectorize"
    
    payload = {
        "dataset_name": dataset_name,
        "query": query,
        "return_serializable": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["vector"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to get serializable Embedding vector via API: {e}")

class QueryProcessor:
    def __init__(self, method="tfidf", dataset_name="simple", embedding_model=None, tfidf_model=None):
        """
        Query processor initialization
        Args:
            method: Representation method ("tfidf", "embedding", "hybrid", "hybrid-sequential")
            dataset_name: Dataset name
            embedding_model: Embedding model (optional)
            tfidf_model: TF-IDF model (optional)
        """
        self.method = method
        self.dataset_name = dataset_name
        self.tfidf_model = tfidf_model
        self.embedding_model = embedding_model
        
        # Load required models if not passed
        self._load_models()
    
    def _load_models(self):
        """Load required models based on the method"""
        try:
            if self.method in ["tfidf", "hybrid", "hybrid-sequential"] and self.tfidf_model is None:
                # Load TF-IDF model
                try:
                    self.tfidf_model, _, _ = load_tfidf_model_via_api(self.dataset_name)
                    logger.info(f"TF-IDF model loaded for {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to load TF-IDF model for {self.dataset_name}: {e}")
            
            if self.method in ["embedding", "hybrid", "hybrid-sequential"] and self.embedding_model is None:
                # Load Embedding model
                try:
                    self.embedding_model, _, _ = load_embedding_model_via_api(self.dataset_name)
                    logger.info(f"Embedding model loaded for {self.dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to load Embedding model for {self.dataset_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def process_query(self, query: str, return_vector: bool = False):
        """
        Process query according to the specified method
        Args:
            query: Original query text
            return_vector: Return vector instead of processed text
        Returns:
            Processed text or vector based on return_vector
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            if self.method == "tfidf":
                return self._process_tfidf_query(query, return_vector)
            elif self.method == "embedding":
                return self._process_embedding_query(query, return_vector)
            elif self.method == "hybrid":
                return self._process_hybrid_query(query, return_vector)
            elif self.method == "hybrid-sequential":
                return self._process_hybrid_sequential_query(query, return_vector)
            else:
                raise ValueError(f"Processing method '{self.method}' is not supported")
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _process_tfidf_query(self, query: str, return_vector: bool = False):
        """Process query for TF-IDF"""
        # Use the same preprocessing used in model building
        processed_text = preprocess_for_vectorization_via_api(query)
        
        if return_vector and self.tfidf_model:
            # Return vector with additional information for clarity
            vector = get_tfidf_vector_serializable_via_api(self.dataset_name, processed_text)
            
            # Get feature names (words)
            feature_names = self.tfidf_model.get_feature_names_out()
            
            # Create list of existing words with their values
            word_scores = []
            for i, score in enumerate(vector):
                if score > 0:  # Only existing words
                    word_scores.append({
                        "word": feature_names[i],
                        "tfidf_score": score
                    })
            
            return {
                "vector": vector,
                "vector_size": len(vector),
                "non_zero_count": len(word_scores),
                "words_with_scores": word_scores,
                "processed_text": processed_text
            }
        else:
            # Return processed text
            return processed_text
    
    def _process_embedding_query(self, query: str, return_vector: bool = False):
        """Process query for Embedding"""
        # For embeddings, we use the original text with simple processing
        # because sentence-transformers handles internal processing
        processed_text = query.strip()
        
        if return_vector and self.embedding_model:
            # Return vector with additional information for clarity
            vector = get_embedding_vector_serializable_via_api(self.dataset_name, processed_text)
            
            return {
                "vector": vector,
                "vector_size": len(vector),
                "embedding_dimension": len(vector),
                "processed_text": processed_text,
                "note": "Embedding vectors represent semantic meaning, not individual words"
            }
        else:
            # Return processed text
            return processed_text
    
    def _process_hybrid_query(self, query: str, return_vector: bool = False):
        """Process query for parallel hybrid search"""
        tfidf_result = self._process_tfidf_query(query, return_vector)
        embedding_result = self._process_embedding_query(query, return_vector)
        
        if return_vector:
            # Return both vectors with additional information
            return {
                "tfidf": tfidf_result,
                "embedding": embedding_result,
                "method": "hybrid_parallel",
                "note": "Combines TF-IDF (lexical) and Embedding (semantic) representations"
            }
        else:
            # Return processed texts
            return {
                "tfidf_text": tfidf_result,
                "embedding_text": embedding_result
            }
    
    def _process_hybrid_sequential_query(self, query: str, return_vector: bool = False):
        """Process query for sequential hybrid search"""
        tfidf_result = self._process_tfidf_query(query, return_vector)
        embedding_result = self._process_embedding_query(query, return_vector)
        
        if return_vector:
            # Return both vectors with additional information
            return {
                "tfidf": tfidf_result,
                "embedding": embedding_result,
                "method": "hybrid_sequential",
                "note": "Uses TF-IDF first, then refines with Embedding"
            }
        else:
            # Return processed texts
            return {
                "tfidf_text": tfidf_result,
                "embedding_text": embedding_result
            }
    
    def get_processing_info(self, query: str) -> dict:
        """
        Get information about the processing process
        Args:
            query: Original query text
        Returns:
            dict: Processing information
        """
        try:
            return {
                "original_query": query,
                "query_length": len(query),
                "method": self.method,
                "dataset_name": self.dataset_name,
                "models_loaded": {
                    "tfidf_model": self.tfidf_model is not None,
                    "embedding_model": self.embedding_model is not None
                },
                "processing_steps": self._get_processing_steps()
            }
        except Exception as e:
            logger.error(f"Error getting processing information: {e}")
            return {"error": str(e)}
    
    def _get_processing_steps(self) -> list:
        """Get processing steps based on the method"""
        steps = {
            "tfidf": [
                "Tokenization",
                "Lowercasing", 
                "Stop word removal",
                "Lemmatization",
                "TF-IDF vectorization"
            ],
            "embedding": [
                "Simple text cleaning",
                "Sentence embedding generation"
            ],
            "hybrid": [
                "TF-IDF processing",
                "Embedding processing",
                "Combination of both"
            ],
            "hybrid-sequential": [
                "TF-IDF processing",
                "Embedding processing",
                "Sequential combination"
            ]
        }
        
        return steps.get(self.method, ["Unknown method"])
    
    def get_supported_methods(self) -> list:
        """Get supported methods"""
        return ["tfidf", "embedding", "hybrid", "hybrid-sequential"]
    
    def validate_method(self, method: str) -> bool:
        """Validate method"""
        return method in self.get_supported_methods()

    def get_query_vector_for_matching(self, query: str):
        """
        Return query representation as a suitable vector (numpy array 2D)
        based on the selected method.
        """
        if self.method == "tfidf":
            processed_text = preprocess_for_vectorization_via_api(query)
            if self.tfidf_model is None:
                raise ValueError("TF-IDF model not loaded")
            # Always numpy array 2D
            return get_tfidf_vector_via_api(self.dataset_name, processed_text)
        elif self.method == "embedding":
            processed_text = query.strip()
            if self.embedding_model is None:
                raise ValueError("Embedding model not loaded")
            return get_embedding_vector_via_api(self.dataset_name, processed_text)
        elif self.method in ["hybrid", "hybrid-sequential"]:
            processed_tfidf = preprocess_for_vectorization_via_api(query)
            processed_embedding = query.strip()
            tfidf_vec = get_tfidf_vector_via_api(self.dataset_name, processed_tfidf)
            embedding_vec = get_embedding_vector_via_api(self.dataset_name, processed_embedding)
            return {"tfidf": tfidf_vec, "embedding": embedding_vec}
        else:
            raise ValueError(f"Processing method '{self.method}' is not supported for matching")
