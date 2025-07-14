#!/usr/bin/env python3
"""
Custom code for reading wikir data directly from local files
"""

import csv
import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WikirLoader:
    def __init__(self, data_path: str = "data/wikIR1k/wikIR1k"):
        self.data_path = data_path
        self.documents_file = os.path.join(data_path, "documents.csv")
        self.test_queries_file = os.path.join(data_path, "test/queries.csv")
        self.test_qrels_file = os.path.join(data_path, "test/qrels")
        
    def load_documents_to_db(self, dataset_name: str = "wikir/en1k/test", limit: int = None) -> Dict[str, Any]:
        """
        Load documents from CSV file to database
        """
        try:
            logger.info(f"Starting to load documents from {self.documents_file}")
            
            # Check if file exists
            if not os.path.exists(self.documents_file):
                raise FileNotFoundError(f"Documents file not found: {self.documents_file}")
            
            # Open database
            db_path = "data/ir_documents.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            loaded_count = 0
            start_time = datetime.now()
            
            # Read CSV file with encoding handling
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Attempting to read file with encoding {encoding}...")
                    
                    with open(self.documents_file, 'r', encoding=encoding, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        
                        # Print column names for verification
                        logger.info(f"Column names: {reader.fieldnames}")
                        
                        row_count = 0
                        for row in reader:
                            row_count += 1
                            try:
                                # Extract data - use correct column names
                                doc_id = str(row.get('id_right', ''))
                                text = str(row.get('text_right', ''))
                                
                                # Use text directly
                                content = text.strip()
                                
                                # Validate data
                                if not doc_id or not content:
                                    logger.warning(f"Skipping row {row_count}: doc_id='{doc_id}', content='{content[:50]}...'")
                                    continue
                                
                                # Insert into database
                                cursor.execute("""
                                    INSERT OR REPLACE INTO documents (doc_id, text, dataset)
                                    VALUES (?, ?, ?)
                                """, (doc_id, content, dataset_name))
                                
                                loaded_count += 1
                                
                                # Check limit
                                if limit and loaded_count >= limit:
                                    break
                                
                                # Log progress
                                if loaded_count % 10000 == 0:
                                    logger.info(f"Loaded {loaded_count} documents...")
                                    
                            except Exception as e:
                                logger.warning(f"Error processing document: {e}")
                                continue
                    
                    # If we reach here, reading was successful
                    logger.info(f"✅ Successfully read file with encoding {encoding}")
                    logger.info(f"Total rows read: {row_count}")
                    break
                    
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed to read file with encoding {encoding}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Other error reading file with encoding {encoding}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Loaded {loaded_count} documents in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "dataset_name": dataset_name,
                "loaded_documents": loaded_count,
                "duration_seconds": duration,
                "source_file": self.documents_file
            }
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def load_test_queries(self) -> List[Dict[str, Any]]:
        """
        Load test queries
        """
        queries = []
        queries_file = self.test_queries_file
        
        if not os.path.exists(queries_file):
            logger.warning(f"Queries file not found: {queries_file}")
            return queries
        
        try:
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(queries_file, 'r', encoding=encoding, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            queries.append({
                                'query_id': row.get('query_id', ''),
                                'query': row.get('query', '')
                            })
                    logger.info(f"Loaded {len(queries)} queries with encoding {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
        
        return queries
    
    def load_test_qrels(self) -> List[Dict[str, Any]]:
        """
        Load qrels file for testing
        """
        qrels = []
        qrels_file = self.test_qrels_file
        
        if not os.path.exists(qrels_file):
            logger.warning(f"Qrels file not found: {qrels_file}")
            return qrels
        
        try:
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qrels.append({
                            'query_id': parts[0],
                            'doc_id': parts[2],
                            'relevance': int(parts[3])
                        })
            logger.info(f"Loaded {len(qrels)} qrels")
        except Exception as e:
            logger.error(f"Error loading qrels: {e}")
        
        return qrels

def load_wikir_from_local_files(dataset_name: str = "wikir/en1k/test", limit: int = None) -> Dict[str, Any]:
    """
    Main function to load wikir data from local files
    """
    loader = WikirLoader()
    return loader.load_documents_to_db(dataset_name, limit)

if __name__ == "__main__":
    # Test the code
    result = load_wikir_from_local_files(limit=1000)
    print(f"Loading result: {result}") 