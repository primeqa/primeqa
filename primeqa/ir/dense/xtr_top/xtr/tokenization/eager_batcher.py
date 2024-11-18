import os
import csv
import gzip
import json
import ujson
import random
from tqdm import tqdm
from functools import partial

from primeqa.ir.dense.xtr_top.xtr.tokenization.utils import tensorize_triples
from primeqa.ir.dense.xtr_top.xtr.tokenization.custom_tokenization import XTRTokenizer


class EagerBatcher():
    def __init__(self, config, triples, rank=0, nranks=1):
        self.config = config 
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.rank, self.nranks = rank, nranks
        self.nway = config.nway

        self.tokenizer = XTRTokenizer(self.config)

        self.tensorize_triples = partial(tensorize_triples, self.tokenizer)
        self.position = 0

        self.triples = self._load_triples(triples, rank, nranks)
        #self.shuffle()
        #self.reader = open(triples, mode='r', encoding="utf-8")
        #self.length = len(self.reader.readlines())
        self.length = len(self.triples)

    def shuffle(self):
        print("#> Shuffling triples...")
        random.shuffle(self.triples)

    def _from_jsonl(self, path):
        triples = []
        with gzip.open(path) as fp:
            for lid, line in tqdm(enumerate(fp), desc="Reading json lines"):
                #if lid > 1000: break
                sample_list = json.loads(line)
                query = sample_list[0]
                positive = sample_list[1]
                negatives = sample_list[2:] #first 5/10 are minor variations of true document only, so ignore them
                triples.append((query, positive, negatives))
        print(f"No. negatives used: {len(triples[0][-1])}")
        return triples

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print("#> Loading triples...")
        if path.endswith("jsonl") or path.endswith("jsonl.gz"):
            return self._from_jsonl(path)

        triples = []

        with open(path) as f:
            csv_reader = csv.DictReader(f, fieldnames=["query", "positive", "negative"], delimiter="\t")
            for line_idx, row in enumerate(csv_reader):
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
        queries, passages = [], []

        for line_idx in range(self.bsize * self.nranks):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            real_line_idx = (self.position + line_idx) % len(self.triples)
            query, pos, negs = self.triples[real_line_idx]
            if self.nway == 2:
                pas = [pos, random.choice(negs[20:])]
            elif self.nway > 2:
                pas = [pos] + random.choices(negs[20:], k=self.config.nway)
            else:
                pas = [pos]

            queries.append(query)
            passages.extend(pas)

        self.position += line_idx + 1

        return self.collate(queries, passages)

    def collate(self, queries, passages):
        return self.tensorize_triples(queries, passages, self.config)

    # adding for training loop logic
    def skip_to_batch(self, batch_idx, intended_batch_size):
        print(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
