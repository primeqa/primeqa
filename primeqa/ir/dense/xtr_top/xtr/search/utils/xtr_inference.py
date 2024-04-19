import numpy as np

def xtr_inference(scores, docs):
        
    uniq_docs = np.unique(docs)
    num_uniq_docs = uniq_docs.size
    doc2ids = {doc: i for i, doc in enumerate(uniq_docs)}
    del uniq_docs
    
    doc_scores = np.zeros((scores.shape[0], num_uniq_docs), dtype=np.float64) + scores[:, -1:]

    # iterate over query tokens, each token retrieves k doc tokens
    for token_id in range(scores.shape[0]):  # qxk
        for i in range(scores.shape[1]):
            score = scores[token_id, i]
            doc_id = docs[token_id, i]
            doc_scores[token_id, doc2ids[doc_id]] = max(doc_scores[token_id, doc2ids[doc_id]], score)
    
    sum_over_tokens = doc_scores.sum(0)
    doc_scores_dict = {doc: sum_over_tokens[docid] for doc, docid in doc2ids.items()}    
    return doc_scores_dict
