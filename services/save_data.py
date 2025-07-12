# services/save_data.py
import json

def save_preprocessed_data(docs, output_path, preprocess_func):
    data = []
    for doc in docs[:1000]:
        processed = preprocess_func(doc.text)
        data.append({
            "doc_id": doc.doc_id,
            "original_text": doc.text,
            "processed_text": processed
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
