import sqlite3

def create_connection(db_file="data/ir_documents.db"):
    return sqlite3.connect(db_file)

def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            dataset TEXT,
            text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_documents(dataset_name, documents):
    conn = create_connection()
    c = conn.cursor()
    for doc in documents:
        c.execute('''
            INSERT OR REPLACE INTO documents (doc_id, dataset, text)
            VALUES (?, ?, ?)
        ''', (doc["doc_id"], dataset_name, doc["text"]))
    conn.commit()
    conn.close()

def get_documents(dataset_name):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT doc_id, text FROM documents WHERE dataset = ?', (dataset_name,))
    rows = c.fetchall()
    conn.close()
    return [{"doc_id": row[0], "text": row[1]} for row in rows]
