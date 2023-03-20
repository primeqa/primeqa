import os
import torch

from tqdm import tqdm
from typing import Union, List, Dict


from primeqa.ir.dense.colbert_top.colbert.data import Collection, Queries, Ranking

from primeqa.ir.dense.colbert_top.colbert.modeling.checkpoint import Checkpoint
from primeqa.ir.dense.colbert_top.colbert.search.index_storage import IndexScorer

from primeqa.ir.dense.colbert_top.colbert.infra.provenance import Provenance
from primeqa.ir.dense.colbert_top.colbert.infra.run import Run
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.infra.launcher import print_memory_stats

TextQueries = Union[str, List[str], Dict[int, str], Queries]
TextDocuments = Union[List[str]]

class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None, rescore_only=False):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(Run().config, config)
        self.rescore_only = rescore_only

        if self.rescore_only:
            self.checkpoint = checkpoint
            self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
            self.config = ColBERTConfig.from_existing(self.checkpoint_config, None, initial_config)

            self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config)
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                self.checkpoint = self.checkpoint.cuda()

            return

        if initial_config.index_location is not None:
            self.index = initial_config.index_location
        else:
            self.index = (
                initial_config.index_path
                if initial_config.index_path
                else (
                    os.path.join(initial_config.index_root, index)
                    if initial_config.index_root
                    else os.path.join(initial_config.index_root_, index)
                )
            )
 
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = None
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()

        self.ranker = IndexScorer(self.index, use_gpu)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries):
        queries = text if isinstance(text, list) else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True)

        return Q

    def encode_documents(self, docs: TextDocuments):
        self.checkpoint.doc_tokenizer.doc_maxlen = self.config.doc_maxlen

        input_ids, attention_mask = self.checkpoint.doc_tokenizer.tensorize(docs)        # as in colbert/modeling/checkpoint.py:112
        D = self.checkpoint.doc(input_ids, attention_mask, keep_dims=True, to_cpu=False) # colbert/modeling/checkpoint.py:113

        return D, attention_mask # .sum(1)   # mask contains doc lengths

    def rescore(self, text_queries, text_documents):
        from primeqa.ir.dense.colbert_top.colbert.modeling.colbert import colbert_score, colbert_score_packed, colbert_score_reduce

        Q = self.encode(text_queries)
        D, attention_mask = self.encode_documents(text_documents)

        scores = colbert_score(Q, D, attention_mask, self.config)
        return scores

    def search(self, text: str, k=10):
        assert not self.rescore_only,  f"It looks like the engine was initialized for rescoring only."
        return self.dense_search(self.encode(text), k)

    def search_all(self, queries: TextQueries, k=10):
        assert not self.rescore_only,  f"It looks like the engine was initialized for rescoring only."
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_)

        return self._search_all_Q(queries, Q, k)

    def _search_all_Q(self, queries, Q, k):
        all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx+1], k=k)))
                           for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, k)

        return pids[:k], list(range(1, k+1)), scores[:k]
