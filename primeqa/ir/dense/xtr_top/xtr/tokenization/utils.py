import torch


def tensorize_triples(tokenizer, queries, passages, config):
    nway = config.nway
    bsize = config.bsize 
    if nway > 2:
        nway += 1

    Q_encodings = tokenizer.tensorize(queries, text_type='query')
            
    D_encodings = tokenizer.tensorize(passages, text_type='passage')

    query_batches = _split_into_batches(Q_encodings, bsize)
    doc_batches = _split_into_batches(D_encodings, bsize * nway)

    batches = []
    for Q, D in zip(query_batches, doc_batches):
        batches.append((Q, D))

    return batches

def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices
    
    return ids[indices], mask[indices], reverse_indices

def _split_into_batches(encodings, bsize):
    ids = encodings['input_ids']
    mask = encodings['attention_mask']

    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))
    return batches
