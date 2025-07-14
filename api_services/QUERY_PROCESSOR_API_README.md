# Query Processor API

## نظرة عامة
Query Processor API هو خدمة تقوم بمعالجة الاستعلامات وتحميل النماذج عبر API endpoints. تعمل على port 8006.

## التشغيل
```bash
python run_query_processor_api.py
```

## الـ Endpoints

### 1. `POST /preprocess`
معالجة النص عبر preprocessing API

**Request Body:**
```json
{
    "text": "النص المراد معالجته",
    "return_tokens": false,
    "min_length": 2,
    "remove_stopwords": true,
    "use_lemmatization": true
}
```

**Response:**
```json
{
    "status": "success",
    "result": "النص المعالج"
}
```

### 2. `POST /load-tfidf`
تحميل نموذج TF-IDF

**Request Body:**
```json
{
    "dataset_name": "beir/trec-covid"
}
```

**Response:**
```json
{
    "status": "success",
    "vectorizer": "TF-IDF vectorizer object",
    "matrix": "TF-IDF matrix",
    "doc_ids": ["doc1", "doc2", ...]
}
```

### 3. `POST /load-embedding`
تحميل نموذج Embedding

**Request Body:**
```json
{
    "dataset_name": "beir/trec-covid"
}
```

**Response:**
```json
{
    "status": "success",
    "model": "Embedding model object",
    "embeddings": "Embedding vectors",
    "doc_ids": ["doc1", "doc2", ...]
}
```

### 4. `POST /get-tfidf-vector`
الحصول على متجه TF-IDF للاستعلام

**Request Body:**
```json
{
    "dataset_name": "beir/trec-covid",
    "query": "what is COVID-19",
    "return_serializable": false
}
```

**Response:**
```json
{
    "status": "success",
    "vector": [[0.1, 0.2, 0.3, ...]]
}
```

### 5. `POST /get-embedding-vector`
الحصول على متجه Embedding للاستعلام

**Request Body:**
```json
{
    "dataset_name": "beir/trec-covid",
    "query": "what is COVID-19",
    "return_serializable": false
}
```

**Response:**
```json
{
    "status": "success",
    "vector": [[0.1, 0.2, 0.3, ...]]
}
```

### 6. `POST /process-query`
معالجة الاستعلام حسب الطريقة المحددة

**Request Body:**
```json
{
    "query": "what is COVID-19",
    "method": "hybrid",
    "dataset_name": "beir/trec-covid",
    "return_vector": false
}
```

**Response:**
```json
{
    "status": "success",
    "method": "hybrid",
    "tfidf_text": "معالج tfidf",
    "embedding_text": "what is COVID-19",
    "tfidf_vector": null,
    "embedding_vector": null
}
```

### 7. `POST /get-query-vector`
الحصول على متجه الاستعلام للمطابقة

**Request Body:**
```json
{
    "query": "what is COVID-19",
    "method": "hybrid",
    "dataset_name": "beir/trec-covid"
}
```

**Response:**
```json
{
    "status": "success",
    "method": "hybrid",
    "tfidf": [[0.1, 0.2, 0.3, ...]],
    "embedding": [[0.4, 0.5, 0.6, ...]]
}
```

### 8. `GET /health`
فحص صحة الخدمة

**Response:**
```json
{
    "status": "healthy",
    "service": "Query Processor Service",
    "version": "1.0.0",
    "dependencies": {
        "requests": "available",
        "numpy": "available",
        "joblib": "available"
    }
}
```

## الطرق المدعومة
- `tfidf`: معالجة TF-IDF
- `embedding`: معالجة Embedding
- `hybrid`: معالجة هجينة
- `hybrid-sequential`: معالجة هجينة متسلسلة

## التكامل مع الخدمات الأخرى
- يستخدم Preprocessor API (port 8001) لمعالجة النصوص
- يستخدم TF-IDF API (port 8002) لتحميل النماذج والمتجهات
- يستخدم Embedding API (port 8003) لتحميل النماذج والمتجهات

## الاستخدام في الخدمات الأخرى
يتم استخدام Query Processor API في:
- `matcher.py`: للحصول على متجهات الاستعلام
- `search_ranking_api.py`: لمعالجة الاستعلامات 