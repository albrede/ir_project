from services.database.db_handler import get_documents
from services.vectorization.embedding_vectorizer import build_embedding_model

def run_embeddings():
    dataset_name = "beir/arguana"
    docs = get_documents(dataset_name)
    texts = [doc["text"] for doc in docs]
    doc_ids = [doc["doc_id"] for doc in docs]

    build_embedding_model(texts, save_name=dataset_name, doc_ids=doc_ids)

if __name__ == "__main__":
    run_embeddings()
