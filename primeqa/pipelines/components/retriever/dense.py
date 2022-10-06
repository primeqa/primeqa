import logging
from typing import List
from dataclasses import dataclass

from primeqa.pipelines.components.base import RetrieverComponent
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher


@dataclass
class ColBERTRetrieverComponent(RetrieverComponent):
    """_summary_

    Args:
        index_root: str
        index_name: str
        max_num_documents (int, optional): Maximum number of document. Defaults to 100.
        ncells (int, optional): Defaults to None.
        ndocs (int, optional): Defaults to None.
        logger (logging.Logger, optional): logger object. Defaults to logging.getLogger(ColBERTRetrieverComponent).

    Returns:
        _type_: _description_
    """

    index_root: str
    index_name: str
    max_num_documents: int = 100
    ncells: int = None
    centroid_score_threshold: float = None
    ndocs: int = None
    logger: logging.Logger = logging.getLogger("ColBERTRetrieverComponent")

    def __post_init__(self):
        self.name = "ColBERT Reader"
        self.type = RetrieverComponent.__name__

        self._config = ColBERTConfig(
            index_root=self.index_root,
            ncells=self.ncells,
            centroid_score_threshold=self.centroid_score_threshold,
            ndocs=self.ndocs,
        )

        # Placeholder variables
        self._searcher = None

    def load(self, *args, **kwargs):
        self._searcher = Searcher(self.index_name, config=self._config)

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        # TODO: Add kwarg defining return format (List[List[Tuple(pids, score)]], List[List[<document>]])
        ranking_results = self._searcher.search_all(
            {idx: str(input_text) for idx, input_text in enumerate(input_texts)},
            k=self.max_num_documents,
        )
        return [
            [(result[0], result[-1]) for result in results_per_query]
            for results_per_query in ranking_results.data.values()
        ]
