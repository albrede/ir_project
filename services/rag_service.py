import os
import logging
import requests
from services.database.db_handler import get_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def search_faiss_index_via_api(query_embedding, dataset_name: str, top_k: int = 5):
    """Client function to search FAISS index via API"""
    url = "http://localhost:8003/search-faiss"
    
    payload = {
        "dataset_name": dataset_name,
        "query_embedding": query_embedding.tolist(),
        "top_k": top_k
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "success":
            return data["similar_indices"]
        else:
            raise Exception(f"API returned error: {data}")
            
    except Exception as e:
        raise Exception(f"Failed to search FAISS index via API: {e}")

class RAGService:
    def __init__(self, dataset_name, local_model_name="microsoft/DialoGPT-medium"):
        """
        RAG service initialization
        Args:
            dataset_name: Dataset name
            local_model_name: Local model name (optional)
        """
        self.dataset_name = dataset_name
        self.model = None
        self.doc_ids = None
        self.local_llm = None
        self.local_model_name = local_model_name
        self.tokenizer = None
        self._load_models()
    
    def _load_models(self):
        """Load required models"""
        try:
            self.model, _, self.doc_ids = load_embedding_model_via_api(self.dataset_name)
            logger.info(f"✅ Models loaded for dataset: {self.dataset_name}")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _load_local_llm(self):
        """Load local model (loaded only when needed)"""
        if self.local_llm is None:
            try:
                # Check for transformers
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                except ImportError:
                    logger.error("❌ transformers library not installed. Please install it with: pip install transformers torch")
                    raise ImportError("transformers library not installed")
                
                logger.info(f"⏳ Loading local LLM: {self.local_model_name}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
                
                # Add special token for padding if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.local_llm = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True
                )
                
                logger.info("✅ Local LLM loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Error loading local LLM: {e}")
                # Reset variables in case of failure
                self.local_llm = None
                self.tokenizer = None
                raise
    
    def search_similar_documents(self, query, top_k=5):
        """
        Search for documents most similar to the query
        """
        try:
            # Convert query to embedding
            query_embedding = get_embedding_vector_via_api(self.dataset_name, query)
            
            # Search in FAISS index
            similar_indices = search_faiss_index_via_api(query_embedding, self.dataset_name, top_k)
            
            if not similar_indices:
                logger.warning("No similar documents found")
                return []
            
            # Check for doc_ids
            if self.doc_ids is None:
                logger.error("doc_ids is None")
                return []
            
            # Retrieve document IDs
            similar_doc_ids = [self.doc_ids[idx] for idx in similar_indices if idx < len(self.doc_ids)]
            
            # Retrieve document texts from database
            documents = get_documents(similar_doc_ids)
            
            logger.info(f"✅ Found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error in document search: {e}")
            return []
    
    def generate_rag_answer(self, query, top_k=5, llm_provider="local"):
        """
        Generate answer using RAG
        Args:
            query: Query
            top_k: Number of documents to retrieve
            llm_provider: LLM service provider ("openai", "local", etc.)
        """
        try:
            # Search for similar documents
            similar_docs = self.search_similar_documents(query, top_k)
            
            if not similar_docs:
                return "Sorry, I couldn't find relevant documents to answer your question."
            
            # Build context from documents
            context = self._build_context(similar_docs)
            
            # Generate answer using LLM
            answer = self._call_llm(query, context, llm_provider)
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error in RAG generation: {e}")
            return f"An error occurred while generating the answer: {str(e)}"
    
    def _build_context(self, documents):
        """Build context from documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            if content:
                # Shorten content to avoid exceeding model limits
                shortened_content = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"Document {i}:\n{shortened_content}\n")
        
        return "\n".join(context_parts)
    
    def _call_llm(self, query, context, provider="local"):
        """
        Call LLM to generate answer
        """
        if provider == "openai":
            return self._call_openai_llm(query, context)
        elif provider == "local":
            return self._call_local_llm(query, context)
        else:
            return f"LLM service provider '{provider}' is not currently supported."
    
    def _call_local_llm(self, query, context):
        """
        Call local LLM using transformers
        """
        try:
            # Load local model if not loaded
            self._load_local_llm()
            
            if self.local_llm is None or self.tokenizer is None:
                return "Sorry, the local model was not loaded successfully."
            
            # Build prompt
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
            
            # Encode text
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True
            )
            
            # Generate answer
            import torch
            with torch.no_grad():
                outputs = self.local_llm.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 150,  # Add 150 tokens for answer
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode answer
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer only (after "Answer:")
            answer_start = generated_text.find("Answer:")
            if answer_start != -1:
                answer = generated_text[answer_start + 7:].strip()
            else:
                # If we don't find "Answer:", take text after the prompt
                answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I couldn't generate a proper answer."
            
        except Exception as e:
            logger.error(f"❌ Error calling local LLM: {e}")
            return f"Error generating answer: {str(e)}"

def rag_answer(query, dataset_name, top_k=5, llm_provider="local"):
    """
    دالة مساعدة لاستخدام RAG بسرعة
    """
    try:
        rag_service = RAGService(dataset_name)
        return rag_service.generate_rag_answer(query, top_k, llm_provider)
    except Exception as e:
        logger.error(f"❌ Error in RAG answer: {e}")
        return f"حدث خطأ: {str(e)}" 