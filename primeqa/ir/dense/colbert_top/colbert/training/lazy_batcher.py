import random
import os
import ujson

from functools import partial
from primeqa.ir.dense.colbert_top.colbert.infra.config.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message, zipstar
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization import tensorize_triples
from primeqa.ir.dense.colbert_top.colbert.modeling.factory import get_query_tokenizer, get_doc_tokenizer
from primeqa.ir.dense.colbert_top.colbert.evaluation.loaders import load_collection

from primeqa.ir.dense.colbert_top.colbert.data.collection import Collection
from primeqa.ir.dense.colbert_top.colbert.data.queries import Queries
from primeqa.ir.dense.colbert_top.colbert.data.examples import Examples

# from colbert.utils.runs import Run



class LazyBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        # self.query_tokenizer = QueryTokenizer(config)
        # self.doc_tokenizer = DocTokenizer(config)
        self.query_tokenizer = get_query_tokenizer(config.model_type, config.query_maxlen, config.attend_to_mask_tokens)
        self.doc_tokenizer = get_doc_tokenizer(config.model_type, config.doc_maxlen)

        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[:self.nway]

            query = self.queries[query]

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            passages = [self.collection[pid] for pid in pids]



            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)
        
        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, passages, scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)

    # adding shuffle
    def shuffle(self):
        print_message("#> Shuffling triples...")
        random.shuffle(self.triples)

    # adding for training loop logic
    def skip_to_batch(self, batch_idx, intended_batch_size):
        print_message(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
