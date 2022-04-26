from pyserini.search import LuceneSearcher
from typing import Optional, List
import logging
import json
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

class BaseRetriever(metaclass=ABCMeta):
    """ 
        Base class for Retriever
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = 10):
        pass

    def batch_retrieve(self,  queries: List[str], qids: List[str], k: int = 10, threads: int = 1):
        pass

class PyseriniRetriever(BaseRetriever):
    def __init__(self, index_path: str, use_bm25=True, k1=float(0.9), b=float(0.4)):
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25()
        if use_bm25:
            self.searcher.set_bm25(k1=k1,b=b)
        self.top_k = 10
        logger.info(f'Initialized LuceneSearcher index_dir: {self.searcher.index_dir}  num_docs: {self.searcher.num_docs} k1: {k1} b: {b}')

    def retrieve(self, query: str, top_k: Optional[int] = 10):
        """
        Return documents that are most relevant to the query.

        :param query: The query
        :param top_k: How many documents to return per query.
        """

        hits = self.searcher.search(query, top_k)
        passage_hits = []
        for i, hit in enumerate(hits):
            title, text = json.loads(hit.raw)['contents'].split("\t")
            title = title.replace('\n',' ')
            text = text.replace('\n',' ')
            docid = hit.docid
            passage_hit = {
                "rank": i,
                "score": hit.score,
                "passage_id": docid,
                "doc_id": docid,
                "title": title,
                "text": text 
            }
            passage_hits.append(passage_hit)
        return passage_hits

    def batch_retrieve(self,  queries: List[str], qids: List[str], k: int = 10, threads: int = 1):
        query_hits = self.searcher.batch_retrieve(queries, qids, k=k, threads=threads)
        query_to_passage_hits = {}
        for q, hits in query_hits.item():
            query_to_passage_hits[q] = self._collect_hits(hits)
        return query_to_passage_hits


    def _collect_hits(hits):
        passage_hits = []
        for i, hit in enumerate(hits):
            title, text = json.loads(hit.raw)['contents'].split("\t")
            title = title.replace('\n',' ')
            text = text.replace('\n',' ')
            docid = hit.docid
            passage_hit = {
                "rank": i,
                "score": hit.score,
                "passage_id": docid,
                "doc_id": docid,
                "title": title,
                "text": text 
            }
            passage_hits.append(passage_hit)
        return passage_hits

