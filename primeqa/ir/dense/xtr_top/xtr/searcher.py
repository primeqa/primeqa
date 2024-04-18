import os
import torch
from tqdm import tqdm
from typing import Union, List, Dict

from primeqa.ir.dense.colbert_top.colbert.data import Queries
from primeqa.ir.dense.xtr_top.xtr.modeling.checkpoint import Checkpoint
from primeqa.ir.dense.xtr_top.xtr.search.index_storage import IndexScorer


class Searcher:
    def __init__(self, index_name, checkpoint=None, collection=None, config=None):
        self.config = config
        self.index = index_name

        self.collection = collection
        self.checkpoint = Checkpoint(self.config)
        use_gpu = torch.cuda.is_available()
        self.retriever = IndexScorer(self.index, config=config, use_gpu=use_gpu)

    def configure(self, **kw_args):
        for key, value in kw_args.items():
            setattr(self.config, key, value)

    def encode(self, text):
        queries = text if isinstance(text, list) else [text]
        Q, mask = self.checkpoint.queryFromText(queries)

        return Q, mask

    def search_all(self, text, k=10):
        queries = Queries.cast(text)
        query_text = list(queries.values())
        all_scored_pids = []
        for b in tqdm(range(0, len(query_text), self.config.bsize), desc='Retrieving:'):
            Q, mask = self.encode(query_text[b:b+self.config.bsize])
            batch_scored_pids = self.dense_search(Q, mask=mask, k=k)
            all_scored_pids.extend(batch_scored_pids)
        results = {qid: preds for qid, preds in zip(queries.keys(), all_scored_pids)}
        return results

    def dense_search(self, Q, mask=None, k=10):
        if self.config.ncells is None:
            self.configure(ncells=10)
        if self.config.ndocs is None:
            self.configure(ndocs=1000)

        outputs = self.retriever.retrieve(self.config, Q, mask=mask, k=k)
        return outputs
