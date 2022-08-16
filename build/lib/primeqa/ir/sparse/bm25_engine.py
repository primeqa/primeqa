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
        logger.info(f"Running BM25")
        
    def do_index(self):
        logger.info("Running BM25 indexing")
        indexer = PyseriniIndexer()
        rc = indexer.index_collection(self.config.corpus_path, self.config.index_path, 
                    self.config.fieldnames, self.config.overwrite, 
                    self.config.threads, self.config.additional_indexing_args )
        logger.info(f"BM25 Indexing finished with rc: {rc}")

    def do_search(self):
            logger.info("Running BM25 search")
            queries = load_queries(self.config.queries_path)
            logger.info(f"Loaded queries num {len(queries)}")
            logger.info(f"Loaded index from {self.config.index_path}")
            searcher = PyseriniRetriever(self.config.index_path,use_bm25=self.config.use_bm25,k1=self.config.k1,b=self.config.b)
            logger.info(f"Running search num queries: {len(queries)} top_k: {self.config.nhits} threads: {self.config.threads}")
            search_results = searcher.batch_retrieve(list(queries.values()),list(queries.keys()),
                        top_k=self.config.nhits,threads=self.config.threads)

            if self.config.output_dir != None:
                logger.info(f"Writing ranked results to {self.config.output_dir}")
                if not os.path.exists(self.config.output_dir):
                    os.makedirs(self.config.output_dir)
                write_colbert_ranking_tsv(self.config.output_dir, search_results)
            logger.info("BM25 Search finished")