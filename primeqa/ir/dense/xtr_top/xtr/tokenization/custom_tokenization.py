import os
import json
import copy
import numpy as np
from collections import Counter

import torch

from transformers import AutoTokenizer

from primeqa.ir.dense.xtr_top.xtr.tokenization.utils import _split_into_batches, _sort_by_length


class XTRTokenizer():
    def __init__(self, config):
        self.config = config
        self.tok = AutoTokenizer.from_pretrained(config.model_name_or_path)
        
    @classmethod
    def subword_pooling(self, encodings, nway=1):
        batch_size, seq_len = encodings.input_ids.shape
        batch_size = batch_size // nway 
        attention_mask = torch.zeros(batch_size, seq_len)
        pooler_mask = torch.zeros(batch_size, seq_len, seq_len)

        max_length = float('-inf')
        positive_docs = torch.arange(0, batch_size) * (nway)
        for b, bid in enumerate(positive_docs):
            counts = Counter(encodings.word_ids(bid))
            counts.pop(None)
            num_words = len(counts)
            attention_mask[b, :num_words] = 1
            max_length = max(num_words, max_length)

            cum_count = 0
            for sw, count in counts.items():
                pooler_mask[b, sw, cum_count:cum_count+count] = 1 #/count
                cum_count += count

        attention_mask = attention_mask[:, :max_length]
        pooler_mask = pooler_mask[:, :max_length, :]

        return pooler_mask, attention_mask
 
    def tensorize(self, batch_text, bsize=None, text_type=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        if text_type == 'query':
            maxlen = self.config.query_maxlen
            batch_text = [f"Query: {x}" for x in batch_text]
        elif text_type == 'passage':
            maxlen = self.config.doc_maxlen
            batch_text = [f"Document: {x}" for x in batch_text]
        else:
            maxlen = self.config.doc_maxlen + self.config.query_maxlen
            batch_text_ref = [x for x in batch_text]

        encodings = self.tok(batch_text, truncation=True, padding='max_length',
                        return_tensors='pt', max_length=maxlen)

        if bsize:
            if text_type == 'passage':
                ids, mask, reverse_indices = _sort_by_length(encodings['input_ids'], encodings['attention_mask'], bsize)
                encodings['input_ids'] = ids
                encodings['attention_mask'] = mask
                batches = _split_into_batches(encodings, bsize)
                return batches, reverse_indices
            else:
                batches = _split_into_batches(encodings, bsize)
                return batches

        return encodings
