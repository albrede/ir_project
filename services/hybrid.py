import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def parallel_hybrid_search(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, alpha=0.5, top_k=10, doc_ids=None):
    print(f"🔍 Calculating similarity using TF-IDF...")
    tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_doc_matrix)[0]
    
    print(f"🔍 Calculating similarity using Embeddings...")
    emb_scores = cosine_similarity(emb_query_vec, emb_doc_matrix)[0]
    
    print(f"🔄 Combining results (alpha={alpha})...")
    final_scores = alpha * tfidf_scores + (1 - alpha) * emb_scores
    
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    top_scores = final_scores[top_indices]
    
    print(f"✅ Found {len(top_indices)} results")
    
    if doc_ids is not None:
        top_doc_ids = [doc_ids[idx] for idx in top_indices]
        return top_doc_ids, top_scores
    return top_indices, top_scores

def hybrid_search_with_weights(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, 
                              tfidf_weight=0.5, emb_weight=0.5, top_k=10, doc_ids=None):
    tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_doc_matrix)[0]
    emb_scores = cosine_similarity(emb_query_vec, emb_doc_matrix)[0]
    
    tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
    emb_scores = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-8)
    
    final_scores = tfidf_weight * tfidf_scores + emb_weight * emb_scores
    
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    top_scores = final_scores[top_indices]
    
    if doc_ids is not None:
        top_doc_ids = [doc_ids[idx] for idx in top_indices]
        return top_doc_ids, top_scores
    return top_indices, top_scores
