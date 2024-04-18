import numpy as np
cimport numpy as np

cpdef xtr_inference(np.ndarray[np.float64_t, ndim=2] scores, 
                    np.ndarray[np.int64_t, ndim=2] docs):
        
    cdef np.ndarray[np.int64_t, ndim=1] uniq_docs = np.unique(docs)
    cdef int num_uniq_docs = uniq_docs.size
    cdef dict doc2ids = {doc: i for i, doc in enumerate(uniq_docs)}
    del uniq_docs
    
    cdef np.ndarray[np.float64_t, ndim=2] doc_scores = np.zeros((scores.shape[0], num_uniq_docs), dtype=np.float64) + scores[:, -1:]

    cdef np.int64_t[:, :] docs_view = docs

    cdef int token_id, doc_id
    cdef double score
    
    # iterate over query tokens, each token retrieves k doc tokens
    for token_id in range(scores.shape[0]):  # qxk
        for i in range(scores.shape[1]):
            score = scores[token_id, i]
            doc_id = docs_view[token_id, i]
            doc_scores[token_id, doc2ids[doc_id]] = max(doc_scores[token_id, doc2ids[doc_id]], score)
    
    cdef np.ndarray[np.float64_t, ndim=1] sum_over_tokens = doc_scores.sum(0)
    cdef dict doc_scores_dict = {doc: sum_over_tokens[docid] for doc, docid in doc2ids.items()}    
    return doc_scores_dict

