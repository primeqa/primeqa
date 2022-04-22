import random

from functools import partial
from oneqa.ir.dense.colbert.colbert import ColBERTConfig
from oneqa.ir.dense.colbert.colbert import print_message, zipstar
from oneqa.ir.dense.colbert.colbert import tensorize_triples
from oneqa.ir.dense.colbert.colbert import get_query_tokenizer, get_doc_tokenizer

from oneqa.ir.dense.colbert.colbert.data.collection import Collection
from oneqa.ir.dense.colbert.colbert.data.queries import Queries
from oneqa.ir.dense.colbert.colbert import Examples

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
