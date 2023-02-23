import torch

def tensorize_triples(query_tokenizer, doc_tokenizer, queries, passages, scores, bsize, nway):
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(passages)

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    doc_batches = _split_into_batches(D_ids, D_mask, bsize * nway)

    if len(scores):
        score_batches = _split_into_batches2(scores, bsize * nway)
    else:
        score_batches = [[] for _ in doc_batches]

    batches = []
    for Q, D, S in zip(query_batches, doc_batches, score_batches):
        batches.append((Q, D, S))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches


def _split_into_batches2(scores, bsize):
    batches = []
    for offset in range(0, len(scores), bsize):
        batches.append(scores[offset:offset+bsize])

    return batches
