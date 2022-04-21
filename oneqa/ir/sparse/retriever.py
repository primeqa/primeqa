from pyserini.search import LuceneSearcher
from typing import Optional
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

class PyseriniRetriever(BaseRetriever):
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
        self.top_k = 10
        logger.info(f'Initialized LuceneSearcher index_dir: {self.searcher.index_dir}  num_docs: {self.searcher.num_docs}')

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

