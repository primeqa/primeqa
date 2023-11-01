
import csv
import random
import os
import ujson

from functools import partial
from primeqa.ir.dense.colbert_top.colbert.infra.config.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message, zipstar, remove_first_and_last_quote
from primeqa.ir.dense.colbert_top.colbert.modeling.tokenization import tensorize_triples
from primeqa.ir.dense.colbert_top.colbert.modeling.factory import get_query_tokenizer, get_doc_tokenizer
from primeqa.ir.dense.colbert_top.colbert.evaluation.loaders import load_collection

from primeqa.ir.dense.colbert_top.colbert.data.collection import Collection
from primeqa.ir.dense.colbert_top.colbert.data.queries import Queries
from primeqa.ir.dense.colbert_top.colbert.data.examples import Examples



class EagerBatcher():
    def __init__(self, config: ColBERTConfig, triples, rank=0, nranks=1, is_teacher=False):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.rank, self.nranks = rank, nranks
        self.nway = config.nway

        # self.query_tokenizer = QueryTokenizer(config)
        # self.doc_tokenizer = DocTokenizer(config)
        self.query_tokenizer = get_query_tokenizer(\
                    config.checkpoint if not is_teacher \
                    else config.teacher_checkpoint, config)
        self.doc_tokenizer = get_doc_tokenizer(\
                    config.checkpoint if not is_teacher \
                    else config.teacher_checkpoint, config, is_teacher=is_teacher)

        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = self._load_triples(triples, rank, nranks)
        self.length = len(self.triples)

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
            csv_reader = csv.DictReader(f, fieldnames=["query", "positive", "negative"], delimiter="\t")
            for line_idx, row in enumerate(csv_reader):
                if line_idx % nranks == rank:
                    query = row["query"]
                    pos = row["positive"]
                    neg = row["negative"]
                    triples.append((query, pos, neg))

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

    # adding for training loop logic
    def skip_to_batch(self, batch_idx, intended_batch_size):
        print_message(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
