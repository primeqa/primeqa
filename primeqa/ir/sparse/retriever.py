from pyserini.search import LuceneSearcher
from typing import Optional, List
import logging
import json
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

class PyseriniRetriever:
    def __init__(self, index_location: str, use_bm25: bool = True, k1: float = float(0.9), b: float = float(0.4)):
        """
        Initialize Pyserini retriever

        Args:
            index_path (str): Path to a Pyserini index
            use_bm25 (bool, optional): set BM25 as the scoring function. Defaults to True.
            k1 (float, optional): bm25 parameter to tune impact of term frequency Defaults to float(0.9).
            b (float, optional): bm25 constant to fine tune the effect of document length   Defaults to float(0.4).
        """
        self.index_location = index_location
        self.searcher = LuceneSearcher(index_location)
        self.searcher.set_bm25()
        if use_bm25:
            self.searcher.set_bm25(k1=k1,b=b)
        self.topK = 10
        logger.info(f'Initialized LuceneSearcher index_dir: {self.searcher.index_dir}  num_docs: {self.searcher.num_docs} use_bm25: {use_bm25} k1: {k1} b: {b}')

    def retrieve(self, query: str, topK: Optional[int] = 10):
        """

        Run queries against the index to retrieve ranked list of documents
        Return documents that are most relevant to the query.

        Args:
             query: search
             top_k: number of hits to return, defaults to 10


        Returns:
             List of hits, each hit is a dict containing :
             {
                "rank": i,
                "score": hit.score,
                "doc_id": docid,
                "title": title,
                "text": text 
            }
                

        """

        hits = self.searcher.search(query, topK)
        search_results = self._collect_hits(hits)
        return search_results


    def batch_retrieve(self,  queries: List[str], qids: List[str], topK: int = 10, threads: int = 1):

        """
           Run a batch of queries 

           Args:
                queries:  list of query strings
                qids:     list of qid strings corresponding to queries
                top_k:    number of hits to return, defaults to 10
                threads:  maximum number of threads to use
                
            Returns:
                Dict of qid to hits

                
        """

        hits = self.searcher.batch_search(queries, qids, k=topK, threads=threads)
        query_to_hits = {}
        for q, hits in hits.items():
            query_to_hits[q] = self._collect_hits(hits)
        return query_to_hits


    def _collect_hits(self, hits: List):
        search_results = []
        for i, hit in enumerate(hits):
            title, text = json.loads(hit.raw)['contents'].split("\t")
            title = title.replace('\n',' ')
            text = text.replace('\n',' ')
            docid = hit.docid
            search_result = {
                "rank": i,
                "score": hit.score,
                "doc_id": docid,
                "title": title,
                "text": text 
            }
            search_results.append(search_result)
        return search_results


