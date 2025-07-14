import builtins
import sys
import os

# Change default system encoding
import locale
import codecs

# Force default encoding to be utf-8
if hasattr(locale, 'getpreferredencoding'):
    original_getpreferredencoding = locale.getpreferredencoding
    locale.getpreferredencoding = lambda: 'utf-8'

# Force codecs to use utf-8
def strict_error_handler(error):
    return ('', error.end if hasattr(error, 'end') else 0)
codecs.register_error('strict', strict_error_handler)

# Improve monkey patch to cover all cases
_builtin_open = open

def open_utf8_fallback(*args, **kwargs):
    mode = ''
    if len(args) > 1:
        mode = args[1]
    elif 'mode' in kwargs:
        mode = kwargs['mode']
    
    # If binary mode, don't add encoding
    if 'b' in mode:
        return _builtin_open(*args, **kwargs)
    
    # Try utf-8 first
    try:
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        return _builtin_open(*args, **kwargs)
    except UnicodeDecodeError:
        # If utf-8 fails, try latin1 (doesn't throw error on any character)
        kwargs['encoding'] = 'latin1'
        return _builtin_open(*args, **kwargs)

builtins.open = open_utf8_fallback

# Add special handling for encoding error in csv
import csv
_original_field_size_limit = csv.field_size_limit

def safe_field_size_limit(limit):
    try:
        _original_field_size_limit(limit)
    except OverflowError:
        # If it fails, use a smaller value
        _original_field_size_limit(min(limit, 2**31 - 1))

csv.field_size_limit = safe_field_size_limit

# Add special handling for encoding error in pandas (if used)
try:
    import pandas as pd
    def safe_read_csv(*args, **kwargs):
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        try:
            return pd.read_csv(*args, **kwargs)
        except UnicodeDecodeError:
            kwargs['encoding'] = 'latin1'
            return pd.read_csv(*args, **kwargs)
    pd.read_csv = safe_read_csv
except ImportError:
    pass

from fastapi import FastAPI, Query, HTTPException
from typing import Optional, List, Dict, Any
import ir_datasets
import sqlite3
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Loader Service API", version="1.0.0")

# Database
DB_PATH = "data/ir_documents.db"

def init_database():
    """Initialize the database"""
    try:
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                dataset TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create fast search index
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_dataset 
            ON documents(dataset)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_doc_id 
            ON documents(doc_id)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the database on service startup"""
    init_database()

@app.get("/")
def read_root():
    """Main page for the service"""
    return {
        "service": "Data Loader Service",
        "version": "1.0.0",
        "description": "Service for loading datasets for information retrieval",
        "endpoints": {
            "GET /datasets": "List available datasets",
            "POST /load-dataset": "Load a dataset",
            "GET /dataset/{dataset_name}": "Get information about a specific dataset",
            "DELETE /dataset/{dataset_name}": "Delete a dataset"
        }
    }

@app.get("/datasets")
def get_available_datasets():
    """Get a list of available datasets"""
    try:
        # List of supported datasets
        supported_datasets = [
            {
                "name": "beir/trec-covid",
                "description": "TREC-COVID Dataset from BEIR",
                "type": "scientific_papers",
                "size": "~171K documents"
            },
            {
                "name": "beir/arguana",
                "description": "ArguAna Dataset from BEIR",
                "type": "argument_retrieval",
                "size": "~8.7K documents"
            }
        ]
        
        # Check for pre-loaded data
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dataset, COUNT(*) as doc_count
            FROM documents 
            GROUP BY dataset
        """)
        
        loaded_datasets = {}
        for row in cursor.fetchall():
            loaded_datasets[row[0]] = {
                "document_count": row[1]
            }
        
        conn.close()
        
        # Merge information
        for dataset in supported_datasets:
            if dataset["name"] in loaded_datasets:
                dataset["status"] = "loaded"
                dataset.update(loaded_datasets[dataset["name"]])
            else:
                dataset["status"] = "not_loaded"
        
        return {
            "status": "success",
            "datasets": supported_datasets,
            "total_loaded": len(loaded_datasets)
        }
        
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-dataset")
def load_dataset(
    dataset_name: str = Query(..., description="Dataset name"),
    limit: Optional[int] = Query(500000, description="Maximum number of documents"),
    force_reload: bool = Query(False, description="Force reload even if already loaded")
):
    """Load a specific dataset"""
    try:
        # Validate dataset name
        valid_datasets = ["wikir/en1k/test", "antique/test"]
        if dataset_name not in valid_datasets:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset not supported. Available options: {valid_datasets}"
            )
        
        # Use custom code for wikir if local files are available
        if dataset_name == "wikir/en1k/test":
            local_files_path = "data/wikIR1k/wikIR1k"
            if os.path.exists(local_files_path):
                logger.info(f"Using local files for {dataset_name}")
                try:
                    from services.wikir_loader import load_wikir_from_local_files
                    result = load_wikir_from_local_files(dataset_name, limit)
                    return result
                except Exception as e:
                    logger.warning(f"Failed to use local files: {e}")
                    logger.info("Returning to ir_datasets...")
                # If custom code fails, proceed with ir_datasets
        
        # Check if data is already pre-loaded
        if not force_reload:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset_name,))
            existing_count = cursor.fetchone()[0]
            conn.close()
            
            if existing_count > 0:
                return {
                    "status": "already_exists",
                    "message": f"Dataset '{dataset_name}' already loaded ({existing_count} documents)",
                    "dataset_name": dataset_name,
                    "document_count": existing_count
                }
        
        # Delete old data if force_reload is True
        if force_reload:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE dataset = ?", (dataset_name,))
            conn.commit()
            conn.close()
            logger.info(f"Deleted old data for dataset: {dataset_name}")
        
        # Load dataset with special handling for encoding errors
        logger.info(f"Starting dataset loading: {dataset_name}")
        
        # Clean up temporary files before starting
        import tempfile
        import shutil
        import gc
        import time
        
        # Force garbage collection to close any open files
        gc.collect()
        
        # Change temporary file directory to project directory to avoid access issues
        original_tempdir = tempfile.gettempdir()
        project_tempdir = os.path.join(os.getcwd(), 'temp_ir_datasets')
        os.makedirs(project_tempdir, exist_ok=True)
        tempfile.tempdir = project_tempdir
        
        logger.info(f"Changed temporary file directory to: {project_tempdir}")
        
        # Clean up temporary files
        try:
            temp_dir = os.path.join(tempfile.gettempdir(), 'ir_datasets')
            if os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary files: {temp_dir}")
                
                # Delete all temporary files
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            logger.info(f"Deleted file: {item}")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            logger.info(f"Deleted directory: {item}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {item}: {e}")
                
                # Wait a little to ensure files are closed
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")
        
        # Multiple attempts to load dataset with different encodings
        dataset = None
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Attempting to load {dataset_name} with encoding {encoding}...")
                
                # Temporarily change default encoding
                if hasattr(locale, 'getpreferredencoding'):
                    original_encoding = locale.getpreferredencoding()
                    locale.getpreferredencoding = lambda: encoding
                
                # Special handling for access issues to temporary files
                try:
                    dataset = ir_datasets.load(dataset_name)
                    logger.info(f"✅ Successfully loaded {dataset_name} with encoding {encoding}")
                    break
                except PermissionError as pe:
                    if "Access is denied" in str(pe) or "WinError 5" in str(pe):
                        logger.warning(f"Access issue to temporary files: {pe}")
                        
                        # Multiple attempts to clean up temporary files
                        for attempt in range(3):
                            try:
                                logger.info(f"Attempting to clean up temporary files {attempt + 1}/3...")
                                
                                # Clean up temporary files
                                temp_dir = os.path.join(tempfile.gettempdir(), 'ir_datasets')
                                if os.path.exists(temp_dir):
                                    for item in os.listdir(temp_dir):
                                        item_path = os.path.join(temp_dir, item)
                                        try:
                                            if os.path.isfile(item_path):
                                                os.remove(item_path)
                                            elif os.path.isdir(item_path):
                                                shutil.rmtree(item_path)
                                        except:
                                            pass
                                
                                # Wait a little
                                time.sleep(2)
                                
                                # Try loading again
                                dataset = ir_datasets.load(dataset_name)
                                logger.info(f"✅ Successfully loaded {dataset_name} after cleaning up temporary files (attempt {attempt + 1})")
                                break
                                
                            except Exception as retry_error:
                                logger.warning(f"Attempt {attempt + 1} failed: {retry_error}")
                                if attempt < 2:  # If not the last attempt
                                    time.sleep(3)  # Wait longer
                                    continue
                                else:
                                    logger.error("Failed to clean up temporary files")
                                    continue
                    else:
                        raise pe
                
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to load {dataset_name} with encoding {encoding}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Other error loading {dataset_name} with encoding {encoding}: {e}")
                continue
            finally:
                # Revert default encoding
                if hasattr(locale, 'getpreferredencoding'):
                    locale.getpreferredencoding = lambda: original_encoding
        
        if dataset is None:
            logger.error(f"Failed to load {dataset_name} after all attempts")
            
            # Final attempt by changing temporary file directory
            try:
                logger.info("Final attempt by changing temporary file directory...")
                import tempfile
                
                # Temporarily change temporary directory
                original_tempdir = tempfile.gettempdir()
                new_tempdir = os.path.join(os.getcwd(), 'temp_ir_datasets')
                os.makedirs(new_tempdir, exist_ok=True)
                
                # Temporarily change temp directory
                tempfile.tempdir = new_tempdir
                
                dataset = ir_datasets.load(dataset_name)
                logger.info(f"✅ Successfully loaded {dataset_name} with new temporary directory")
                
                # Revert temp directory
                tempfile.tempdir = original_tempdir
                
            except Exception as final_error:
                logger.error(f"Final attempt failed: {final_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to load dataset {dataset_name}. Please restart the service or run it as Administrator"
                )
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        loaded_count = 0
        start_time = datetime.now()
        
        for doc in dataset.docs_iter():
            if limit and loaded_count >= limit:
                break
                
            try:
                # Extract document ID and content with special handling for encoding
                if hasattr(doc, 'doc_id') and hasattr(doc, 'text'):
                    doc_id = str(doc.doc_id)
                    try:
                        content = str(doc.text)
                    except UnicodeDecodeError:
                        # If text encoding fails, use a fallback
                        content = "Unreadable content - encoding error"
                        logger.warning(f"Encoding error in document content {doc_id}")
                elif hasattr(doc, 'doc_id') and hasattr(doc, 'title'):
                    doc_id = str(doc.doc_id)
                    try:
                        content = str(doc.title)
                        if hasattr(doc, 'body'):
                            content += " " + str(doc.body)
                    except UnicodeDecodeError:
                        content = "Unreadable content - encoding error"
                        logger.warning(f"Encoding error in document content {doc_id}")
                else:
                    continue
                
                # Validate doc_id
                if not doc_id or doc_id.strip() == '':
                    logger.warning(f"Skipping document with invalid ID")
                    continue
                
                # Insert document - handle doc_id as string to avoid overflow
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (doc_id, text, dataset)
                    VALUES (?, ?, ?)
                """, (doc_id, content, dataset_name))
                
                loaded_count += 1
                
                if loaded_count % 10000 == 0:
                    logger.info(f"Loaded {loaded_count} documents...")
            except UnicodeDecodeError as ude:
                logger.warning(f"Skipping document due to encoding error: {ude}")
                continue
            except sqlite3.Error as e:
                logger.error(f"Database error during document loading: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error loading document: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        # Revert temporary directory
        tempfile.tempdir = original_tempdir
        logger.info(f"Reverted temporary file directory to: {original_tempdir}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Loaded {loaded_count} documents from dataset '{dataset_name}' in {duration:.2f} seconds")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "loaded_documents": loaded_count,
            "duration_seconds": duration,
            "database_path": DB_PATH
        }
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Revert temporary directory in case of error
        try:
            tempfile.tempdir = original_tempdir
            logger.info(f"Reverted temporary file directory to: {original_tempdir}")
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download-qrel")
def download_qrel(dataset_name: str = Query(..., description="Dataset name")):
    """Download qrel file for a dataset and save it to evaluation folder"""
    try:
        valid_datasets = ["antique/test", "wikir/en1k/test"]
        if dataset_name not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"Dataset not supported. Available options: {valid_datasets}")
        os.makedirs("evaluation", exist_ok=True)
        dataset = ir_datasets.load(dataset_name)
        qrel_path = f"evaluation/{dataset_name.replace('/', '_')}.qrels"
        count = 0
        with open(qrel_path, "w", encoding="utf-8") as f:
            for qrel in dataset.qrels_iter():
                f.write(f"{qrel.query_id} {qrel.iteration} {qrel.doc_id} {qrel.relevance}\n")
                count += 1
        return {
            "status": "success",
            "message": f"Successfully downloaded and saved qrel ({count} rows) to {qrel_path}",
            "qrel_path": qrel_path
        }
    except Exception as e:
        logger.error(f"Error downloading qrel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download-queries")
def download_queries(dataset_name: str = Query(..., description="Dataset name")):
    """Download queries for a dataset and save them to evaluation folder"""
    try:
        valid_datasets = ["antique/test", "wikir/en1k/test"]
        if dataset_name not in valid_datasets:
            raise HTTPException(status_code=400, detail=f"Dataset not supported. Available options: {valid_datasets}")
        os.makedirs("evaluation", exist_ok=True)
        dataset = ir_datasets.load(dataset_name)
        queries_path = f"evaluation/{dataset_name.replace('/', '_')}.queries.tsv"
        count = 0
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in dataset.queries_iter():
                # Some queries might only contain title or text
                text = getattr(query, 'text', None) or getattr(query, 'title', None) or ''
                f.write(f"{query.query_id}\t{text}\n")
                count += 1
        return {
            "status": "success",
            "message": f"Successfully downloaded and saved queries ({count} queries) to {queries_path}",
            "queries_path": queries_path
        }
    except Exception as e:
        logger.error(f"Error downloading queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/{dataset_name:path}")
def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Document statistics
        cursor.execute("""
            SELECT COUNT(*) as total_docs,
                   AVG(LENGTH(text)) as avg_content_length
            FROM documents 
            WHERE dataset = ?
        """, (dataset_name,))
        
        stats = cursor.fetchone()
        
        if not stats or stats[0] == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Sample documents
        cursor.execute("""
            SELECT doc_id, text
            FROM documents 
            WHERE dataset = ?
            LIMIT 5
        """, (dataset_name,))
        
        sample_docs = []
        for row in cursor.fetchall():
            sample_docs.append({
                "doc_id": row[0],
                "content_preview": row[1][:200] + "..." if len(row[1]) > 200 else row[1]
            })
        
        conn.close()
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "statistics": {
                "total_documents": stats[0],
                "average_content_length": round(stats[1], 2) if stats[1] else 0
            },
            "sample_documents": sample_docs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset information: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/dataset/{dataset_name:path}")
def delete_dataset(dataset_name: str):
    """Delete a specific dataset"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check for data
        cursor.execute("SELECT COUNT(*) FROM documents WHERE dataset = ?", (dataset_name,))
        doc_count = cursor.fetchone()[0]
        
        if doc_count == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Delete data
        cursor.execute("DELETE FROM documents WHERE dataset = ?", (dataset_name,))
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted {doc_count} documents from dataset '{dataset_name}'")
        
        return {
            "status": "success",
            "message": f"Successfully deleted dataset '{dataset_name}'",
            "deleted_documents": doc_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Check service health"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "total_documents": total_docs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
