import numpy as np
import os
import torch
import tqdm
import time
from tqdm import tqdm

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from colbert import Indexer, Searcher

print("GPU Availability")
print(torch.cuda.is_available())
print(torch.cuda.device_count())

#################################

CKPT = "msmarco.psg.kldR2.nway64.ib__colbert-400000"

#chosen_collection = "collection.tsv" 
#chosen_queries = "queries.dev.small.tsv" 
#chosen_dataset = "MS_MARCO"

chosen_collection = "psgs_w100.tsv" 
chosen_queries = "xorqa_dev_gmt.tsv" 
chosen_dataset = "XOR_TyDi"

def evaluate(index=True):
    split = "dev"
    nbits = 2
    k = 10
    experiment = (f"msmarco.nbits={nbits}",)
    collection = chosen_collection
    
    if not os.path.exists(collection):
        print(f"No data found for {dataset} at {collection}, skipping...")
        return
    with Run().context(RunConfig(nranks=1)):
        INDEX_NAME = f"msmarco.nbits={nbits}.latest_" + chosen_dataset

        if index:
            config = ColBERTConfig(
                doc_maxlen=300,
                nbits=nbits,
                kmeans_niters=4,
                root="/lfs/1/keshav2/colbert/experiments",
                experiment=experiment,
            )
            indexer = Indexer(CKPT, config=config)
            indexer.index(name=INDEX_NAME, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root="/lfs/1/keshav2/colbert/experiments",
            experiment=experiment,
        )
        searcher = Searcher(
            index=INDEX_NAME,
            config=config,

        )
        queries = chosen_queries
        queries = Queries(path=queries)
        if torch.cuda.device_count() < 1:
            print(f"No GPU detected, setting num_threads to 1...")
            torch.set_num_threads(1)
            device = "cpu"
        else:
            device = "gpu"
        
        #ranking = searcher.search_all(queries, k=k)
        #ranking.save(f"msmarco.k={k}.device={device}.ranking.tsv")

        # Warmup
        print("Performing warmup!") 
        for query, count in zip(queries, range(10)):
            ranking = searcher.search(query[1], k=k)

        print("Beginning experiment...")

        total_times = []
        for query in tqdm(queries):
           start_time = time.time()
           ranking = searcher.search(query[1], k=k)
           if torch.cuda.is_available():
               torch.cuda.synchronize()
           total_times.append(time.time() - start_time)

        print("Average Query Time: " + str(sum(total_times) / len(total_times)))

def main():
    evaluate(index=False)


if __name__ == "__main__":
    main()
