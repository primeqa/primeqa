import logging
from typing import Union, List

from primeqa.pipelines.components.base import RetrieverComponent
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher


class ColBERTRetrieverComponent(RetrieverComponent):
    def __init__(
        self,
        index_root: str,
        index_name: str,
        max_num_documents: int = 100,
        ncells: Union[int, None] = None,
        centroid_score_threshold: Union[float, None] = None,
        ndocs: Union[int, None] = None,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger

        # Default object variables
        self.name = "ColBERT Retriever"
        self.type = RetrieverComponent.__name__

        # Custom object variable
        self.index_root = index_root
        self.index_name = index_name
        self.max_num_documents = max_num_documents
        self.ncells = ncells
        self.centroid_score_threshold = centroid_score_threshold
        self.ndocs = ndocs

        # Create configuration
        self.config = ColBERTConfig(
            index_root=self.index_root,
            ncells=self.ncells,
            centroid_score_threshold=self.centroid_score_threshold,
            ndocs=self.ndocs,
        )

        # Placeholder variables
        self.searcher = None

    def load(self, *args, **kwargs):
        self.searcher = Searcher(self.index_name, config=self.config)

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        # TODO: Add kwarg defining return format (List[List[Tuple(pids, score)]], List[List[<document>]])
        ranking_results = self.searcher.search_all(
            {idx: str(input_text) for idx, input_text in enumerate(input_texts)},
            k=self.max_num_documents,
        )
        return [
            [(result[0], result[-1]) for result in results_per_query]
            for results_per_query in ranking_results.data.values()
        ]
