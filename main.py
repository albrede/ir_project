from services.loader import load_dataset
from services.database.db_handler import create_table, insert_documents

def main():
    create_table()
    dataset_name = "trec-covid"
    raw_docs = load_dataset("beir/trec-covid")
    docs = []
    for i, doc in enumerate(raw_docs):
        if i >= 1000:
            break
        docs.append({"doc_id": doc.doc_id, "text": doc.text})
    insert_documents(dataset_name, docs)
    print(f"[✓] تم تحميل وتخزين {len(docs)} مستند في قاعدة البيانات تحت مجموعة {dataset_name}")


if __name__ == "__main__":
    main() 