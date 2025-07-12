from services.database.db_handler import get_documents
from services.vectorization.tfidf_vectorizer import build_tfidf_model

def run_tfidf():
    dataset_name = "beir/arguana"
    docs = get_documents(dataset_name)
    texts = [doc["text"] for doc in docs]
    doc_ids = [doc["doc_id"] for doc in docs]

    build_tfidf_model(texts, save_name=dataset_name, doc_ids=doc_ids)

if __name__ == "__main__":
    run_tfidf()
