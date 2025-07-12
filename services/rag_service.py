import os
import logging
from services.vectorization.embedding_vectorizer import (
    load_embedding_model, 
    get_embedding_vector, 
    search_faiss_index
)
from services.database.db_handler import get_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, dataset_name, local_model_name="microsoft/DialoGPT-medium"):
        """
        تهيئة خدمة RAG
        Args:
            dataset_name: اسم الداتا ست
            local_model_name: اسم النموذج المحلي (اختياري)
        """
        self.dataset_name = dataset_name
        self.model = None
        self.doc_ids = None
        self.local_llm = None
        self.local_model_name = local_model_name
        self.tokenizer = None
        self._load_models()
    
    def _load_models(self):
        """تحميل النماذج المطلوبة"""
        try:
            self.model, _, self.doc_ids = load_embedding_model(self.dataset_name)
            logger.info(f"✅ Models loaded for dataset: {self.dataset_name}")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _load_local_llm(self):
        """تحميل النموذج المحلي (يتم تحميله عند الحاجة فقط)"""
        if self.local_llm is None:
            try:
                # التحقق من وجود transformers
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                except ImportError:
                    logger.error("❌ transformers library not installed. Please install it with: pip install transformers torch")
                    raise ImportError("transformers library not installed")
                
                logger.info(f"⏳ Loading local LLM: {self.local_model_name}")
                
                # تحميل الـ tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
                
                # إضافة token خاص للـ padding إذا لم يكن موجود
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # تحميل النموذج
                self.local_llm = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True
                )
                
                logger.info("✅ Local LLM loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Error loading local LLM: {e}")
                # إعادة تعيين المتغيرات في حالة الفشل
                self.local_llm = None
                self.tokenizer = None
                raise
    
    def search_similar_documents(self, query, top_k=5):
        """
        البحث عن الوثائق الأكثر تشابهًا مع الاستعلام
        """
        try:
            # تحويل الاستعلام إلى embedding
            query_embedding = get_embedding_vector(self.model, query)
            
            # البحث في FAISS index
            similar_indices = search_faiss_index(query_embedding, self.dataset_name, top_k)
            
            if not similar_indices:
                logger.warning("No similar documents found")
                return []
            
            # التحقق من وجود doc_ids
            if self.doc_ids is None:
                logger.error("doc_ids is None")
                return []
            
            # استرجاع معرفات الوثائق
            similar_doc_ids = [self.doc_ids[idx] for idx in similar_indices if idx < len(self.doc_ids)]
            
            # استرجاع نصوص الوثائق من قاعدة البيانات
            documents = get_documents(similar_doc_ids)
            
            logger.info(f"✅ Found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error in document search: {e}")
            return []
    
    def generate_rag_answer(self, query, top_k=5, llm_provider="local"):
        """
        توليد إجابة باستخدام RAG
        Args:
            query: الاستعلام
            top_k: عدد الوثائق المراد استرجاعها
            llm_provider: مزود خدمة LLM ("openai", "local", etc.)
        """
        try:
            # البحث عن الوثائق المشابهة
            similar_docs = self.search_similar_documents(query, top_k)
            
            if not similar_docs:
                return "عذرًا، لم أجد وثائق ذات صلة للإجابة على سؤالك."
            
            # بناء السياق من الوثائق
            context = self._build_context(similar_docs)
            
            # توليد الإجابة باستخدام LLM
            answer = self._call_llm(query, context, llm_provider)
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error in RAG generation: {e}")
            return f"حدث خطأ أثناء توليد الإجابة: {str(e)}"
    
    def _build_context(self, documents):
        """بناء السياق من الوثائق"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            if content:
                # تقصير المحتوى لتجنب تجاوز حدود النموذج
                shortened_content = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"Document {i}:\n{shortened_content}\n")
        
        return "\n".join(context_parts)
    
    def _call_llm(self, query, context, provider="local"):
        """
        استدعاء LLM لتوليد الإجابة
        """
        if provider == "openai":
            return self._call_openai_llm(query, context)
        elif provider == "local":
            return self._call_local_llm(query, context)
        else:
            return f"مزود خدمة LLM '{provider}' غير مدعوم حاليًا."
    
    def _call_local_llm(self, query, context):
        """
        استدعاء LLM محلي باستخدام transformers
        """
        try:
            # تحميل النموذج المحلي إذا لم يكن محمل
            self._load_local_llm()
            
            if self.local_llm is None or self.tokenizer is None:
                return "عذرًا، لم يتم تحميل النموذج المحلي بنجاح."
            
            # بناء الـ prompt
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
            
            # ترميز النص
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True
            )
            
            # توليد الإجابة
            import torch
            with torch.no_grad():
                outputs = self.local_llm.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 150,  # إضافة 150 token للإجابة
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # فك ترميز الإجابة
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # استخراج الإجابة فقط (بعد "Answer:")
            answer_start = generated_text.find("Answer:")
            if answer_start != -1:
                answer = generated_text[answer_start + 7:].strip()
            else:
                # إذا لم نجد "Answer:"، نأخذ النص بعد الـ prompt
                answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "عذرًا، لم أتمكن من توليد إجابة مناسبة."
            
        except Exception as e:
            logger.error(f"❌ Error in local LLM: {e}")
            return f"حدث خطأ في النموذج المحلي: {str(e)}"
    
    def _call_openai_llm(self, query, context):
        """
        استدعاء OpenAI API
        """
        try:
            import openai
            
            prompt = f"""
            السياق التالي يحتوي على معلومات للإجابة على السؤال:
            
            {context}
            
            السؤال: {query}
            
            الإجابة:
            """
            
            # استدعاء OpenAI API (للإصدار 1.x)
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except ImportError:
            return "OpenAI library not installed. Please install it with: pip install openai"
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"

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