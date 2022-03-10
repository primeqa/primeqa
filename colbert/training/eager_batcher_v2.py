
import random
import os
import ujson

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar, remove_first_and_last_quote
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples



class EagerBatcher():
    def __init__(self, config: ColBERTConfig, triples, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.rank, self.nranks = rank, nranks
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = self._load_triples(triples, rank, nranks)
        self.reader = open(triples, mode='r', encoding="utf-8")
        self.length = len(self.reader.readlines())

    def shuffle(self):
        print_message("#> Shuffling triples...")
        random.shuffle(self.triples)

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    query, pos, neg = line.strip().split('\t')

                    triples.append((remove_first_and_last_quote(query), remove_first_and_last_quote(pos), remove_first_and_last_quote(neg)))

        return triples

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        queries, positives, negatives = [], [], []
        passages = []
        scores = []

        for line_idx in range(self.bsize * self.nranks):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            real_line_idx = (self.position + line_idx) % len(self.triples)
            query, pos, neg = self.triples[real_line_idx]
            pas = [ pos, neg ]
            sco = []

            queries.append(query)
            passages.extend(pas)
            scores.extend(sco)

        self.position += line_idx + 1

        return self.collate(queries, passages, scores)

    def collate(self, queries, passages, scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)
