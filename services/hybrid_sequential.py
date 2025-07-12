import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def sequential_hybrid_search(tfidf_query_vec, tfidf_doc_matrix, emb_query_vec, emb_doc_matrix, first_stage='tfidf', top_n=100, top_k=10, doc_ids=None):
    if first_stage == 'tfidf':
        first_scores = cosine_similarity(tfidf_query_vec, tfidf_doc_matrix)[0]
        top_n_indices = np.argsort(first_scores)[::-1][:top_n]
        emb_sub_matrix = emb_doc_matrix[top_n_indices]
        emb_scores = cosine_similarity(emb_query_vec, emb_sub_matrix)[0]
        final_indices = np.argsort(emb_scores)[::-1][:top_k]
        top_indices = top_n_indices[final_indices]
        top_scores = emb_scores[final_indices]
    else:
        first_scores = cosine_similarity(emb_query_vec, emb_doc_matrix)[0]
        top_n_indices = np.argsort(first_scores)[::-1][:top_n]
        tfidf_sub_matrix = tfidf_doc_matrix[top_n_indices]
        tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_sub_matrix)[0]
        final_indices = np.argsort(tfidf_scores)[::-1][:top_k]
        top_indices = top_n_indices[final_indices]
        top_scores = tfidf_scores[final_indices]
    if doc_ids is not None:
        top_doc_ids = [doc_ids[idx] for idx in top_indices]
        return top_doc_ids, top_scores
    return top_indices, top_scores 