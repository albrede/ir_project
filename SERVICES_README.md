# خدمات استرجاع المعلومات المنفصلة

## نظرة عامة

تم فصل نظام استرجاع المعلومات إلى خدمات منفصلة وفقاً لمتطلبات المشروع، حيث كل خدمة مسؤولة عن وظيفة محددة:

## بنية الخدمات

### 1. خدمة معالجة الاستعلامات (Query Processing Service)
- **المنفذ**: 8001
- **الملف**: `api_services/query_processing_api.py`
- **المسؤولية**: معالجة الاستعلامات وتحويلها إلى تمثيلات مختلفة

#### Endpoints:
- `POST /process-query` - معالجة الاستعلام
- `GET /process-query` - معالجة الاستعلام (GET)
- `GET /methods` - طرق المعالجة المتاحة
- `GET /health` - فحص صحة الخدمة

#### الطرق المدعومة:
- **TF-IDF**: معالجة معجمية مع Lemmatization و Tokenization
- **Embedding**: معالجة بسيطة لـ Sentence Embeddings
- **Hybrid**: مزيج من TF-IDF و Embedding
- **Hybrid-Sequential**: بحث هجين متسلسل

---

### 2. خدمة البحث وترتيب النتائج (Search & Ranking Service)
- **المنفذ**: 8002
- **الملف**: `api_services/search_ranking_api.py`
- **المسؤولية**: البحث في الوثائق وترتيب النتائج

#### Endpoints:
- `POST /search` - البحث في الوثائق مع إرجاع المحتوى
- `GET /search` - البحث في الوثائق (GET)
- `POST /rank` - ترتيب النتائج فقط
- `GET /rank` - ترتيب النتائج فقط (GET)
- `GET /methods` - طرق البحث المتاحة
- `GET /health` - فحص صحة الخدمة

#### طرق البحث:
- **TF-IDF**: Cosine Similarity
- **Embedding**: Cosine Similarity للتمثيل الدلالي
- **Hybrid**: دمج TF-IDF و Embedding
- **Hybrid-Sequential**: ترتيب متسلسل
- **Inverted Index**: بحث دقيق في النصوص

---

### 3. خدمة الفهرسة (Indexing Service)
- **المنفذ**: 8003
- **الملف**: `api_services/indexing_api.py`
- **المسؤولية**: إدارة الفهارس المقلوبة والبحث فيها

#### Endpoints:
- `POST /search` - البحث في الفهرس المقلوب
- `GET /search` - البحث في الفهرس المقلوب (GET)
- `POST /stats` - إحصائيات الفهرس
- `GET /stats` - إحصائيات الفهرس (GET)
- `GET /health` - فحص صحة الخدمة

#### الميزات:
- البحث البولياني (AND/OR)
- إحصائيات مفصلة للفهرس
- إرجاع محتوى الوثائق المطابقة

---

### 4. API موحد (Unified API)
- **المنفذ**: 8005
- **الملف**: `api_services/query_api.py`
- **المسؤولية**: تجميع جميع الخدمات في API واحد

#### Endpoints:
- جميع endpoints من الخدمات المنفصلة
- واجهة موحدة للوصول لجميع الوظائف

## تشغيل الخدمات

### تشغيل جميع الخدمات مرة واحدة:
```bash
python run_services.py
```

### تشغيل كل خدمة على حدة:

#### خدمة معالجة الاستعلامات:
```bash
python api_services/query_processing_api.py
```

#### خدمة البحث وترتيب النتائج:
```bash
python api_services/search_ranking_api.py
```

#### خدمة الفهرسة:
```bash
python api_services/indexing_api.py
```

#### API موحد:
```bash
python api_services/query_api.py
```

## أمثلة الاستخدام

### 1. معالجة استعلام TF-IDF:
```bash
curl -X POST "http://localhost:8001/process-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Being vegetarian reduces risks",
    "method": "tfidf",
    "dataset_name": "beir/arguana",
    "return_vector": true
  }'
```

### 2. البحث في الوثائق:
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vegetarian diet benefits",
    "method": "hybrid",
    "dataset_name": "beir/arguana",
    "top_k": 10,
    "alpha": 0.6
  }'
```

### 3. البحث في الفهرس المقلوب:
```bash
curl -X GET "http://localhost:8003/search?query=vegetarian&dataset_name=beir/arguana&top_k=5&search_type=and"
```

### 4. ترتيب النتائج فقط:
```bash
curl -X GET "http://localhost:8002/rank?query=health benefits&method=embedding&top_k=10"
```

## وثائق API

يمكن الوصول إلى وثائق Swagger لكل خدمة عبر:
- Query Processing: http://localhost:8001/docs
- Search & Ranking: http://localhost:8002/docs
- Indexing: http://localhost:8003/docs
- Unified API: http://localhost:8005/docs

## بنية SOA (Service Oriented Architecture)

### مزايا التصميم:
1. **فصل المسؤوليات**: كل خدمة مسؤولة عن وظيفة محددة
2. **الاستقلالية**: يمكن اختبار كل خدمة على حدة
3. **القابلية للتوسع**: يمكن إضافة خدمات جديدة بسهولة
4. **المرونة**: يمكن تشغيل الخدمات على خوادم مختلفة

### التواصل بين الخدمات:
- كل خدمة مستقلة تماماً
- لا توجد تبعيات مباشرة بين الخدمات
- يمكن استدعاء أي خدمة من أي مكان
- كل خدمة لها قاعدة بيانات خاصة بها

### اختبار الخدمات:
- يمكن اختبار كل خدمة باستخدام Postman
- كل endpoint له وثائق واضحة
- يمكن حفظ أمثلة الاستجابة في Postman Collection

## ملاحظات مهمة

1. **المنافذ**: تأكد من أن المنافذ 8001, 8002, 8003, 8005 غير مستخدمة
2. **قاعدة البيانات**: تأكد من وجود البيانات المطلوبة
3. **النماذج**: تأكد من وجود ملفات النماذج المحفوظة
4. **البيئة**: تأكد من تفعيل البيئة الافتراضية

## استكشاف الأخطاء

### مشاكل شائعة:
1. **خطأ في المنفذ**: تأكد من عدم استخدام المنفذ من قبل تطبيق آخر
2. **خطأ في تحميل النماذج**: تأكد من وجود ملفات joblib
3. **خطأ في قاعدة البيانات**: تأكد من وجود ملف قاعدة البيانات

### سجلات الأخطاء:
- جميع الخدمات تسجل الأخطاء في console
- يمكن مراجعة السجلات لتحديد المشكلة 