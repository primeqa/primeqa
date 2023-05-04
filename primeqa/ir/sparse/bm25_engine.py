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
        logger.info(config)
        
    def do_index(self):
        logger.info("Running BM25 indexing")
        indexer = PyseriniIndexer()
        rc = indexer.index_collection(self.config.collection, self.config.index_location, 
                    self.config.fieldnames, self.config.overwrite, 
                    self.config.threads, self.config.additional_indexing_args )
        logger.info(f"BM25 Indexing finished with rc: {rc}")

    def do_search(self):
            logger.info("Running BM25 search with uniform parameters")
            queries = load_queries(self.config.queries)
            logger.info(f"Loaded queries num {len(queries)}")
            logger.info(f"Loaded index from {self.config.index_location}")
            searcher = PyseriniRetriever(self.config.index_location,use_bm25=self.config.use_bm25,k1=self.config.k1,b=self.config.b)
            logger.info(f"Running search num queries: {len(queries)} topK: {self.config.topK} threads: {self.config.threads}")
            
            all_results = {}
            all_queries = list(queries.values())
            all_keys = list(queries.keys())
            step = 1000
            
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)
            output_file = os.path.join(self.config.output_dir, "ranked_passages.tsv")
            with open(output_file,'w',encoding='utf-8') as f:
                for x in range(0, len(queries), step):
                    
                    logger.info(f"Running queries {x} to {x+step} of {len(queries)}")
                    id_to_hits = searcher.batch_retrieve(all_queries[x:x+step], all_keys[x:x+step],
                            topK=self.config.topK,threads=self.config.threads)
                    logger.info(f"Search Done {len(all_results)}")
                    
                    lines = []
                    for id in id_to_hits:
                        for i, hit in enumerate(id_to_hits[id]):
                            lines.append(f"{id}\t{hit[2]}\t{hit[0]}\t{hit[1]}")

                    f.writelines([f'{l}\n' for l in lines])
                    f.flush()
                    logger.info(f"Wrote {output_file}")
                
            # if self.config.output_dir != None:
            #     logger.info(f"Writing ranked results to {self.config.output_dir}")
            #     if not os.path.exists(self.config.output_dir):
            #         os.makedirs(self.config.output_dir)
            #     write_colbert_ranking_tsv(self.config.output_dir, search_results, json_format=False)
            logger.info("BM25 Search finished")