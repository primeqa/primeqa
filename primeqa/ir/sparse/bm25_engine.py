import os
import logging

from primeqa.ir.sparse.retriever import PyseriniRetriever
from primeqa.ir.sparse.indexer import PyseriniIndexer
from primeqa.ir.sparse.utils import load_queries, write_colbert_ranking_tsv
from primeqa.ir.sparse.config import BM25Config

logger = logging.getLogger(__name__)


class BM25Engine:
    def __init__(self, config: BM25Config):
        self.config = config
        logger.info("Running BM25")
        logger.info(config)

    def do_index(self):
        logger.info("Running BM25 indexing")
        indexer = PyseriniIndexer()
        rc = indexer.index_collection(
            self.config.collection,
            self.config.index_location,
            self.config.fieldnames,
            self.config.overwrite,
            self.config.threads,
            self.config.additional_indexing_args,
        )
        logger.info("BM25 Indexing finished with rc: %s", rc)

    def do_search(self):
        logger.info("Running BM25 search with uniform parameters")
        queries = load_queries(self.config.queries)
        logger.info("Loaded queries num %d", len(queries))
        logger.info("Loaded index from %s", self.config.index_location)
        searcher = PyseriniRetriever(
            self.config.index_location,
            use_bm25=self.config.use_bm25,
            k1=self.config.k1,
            b=self.config.b,
        )
        logger.info(
            "Running search num queries: %d topK: %d threads: %d",
            len(queries),
            {self.config.topK},
            {self.config.threads},
        )
        search_results = searcher.batch_retrieve(
            list(queries.values()),
            list(queries.keys()),
            topK=self.config.topK,
            threads=self.config.threads,
        )

        if self.config.output_dir:
            logger.info("Writing ranked results to %s", self.config.output_dir)
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)
            write_colbert_ranking_tsv(self.config.output_dir, search_results)
        logger.info("BM25 Search finished")
