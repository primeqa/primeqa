
import numpy as np
import os
import torch
import tqdm
import time

from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig, ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.data import Queries

from primeqa.ir.dense.colbert_top.colbert import Indexer, Searcher

#############################################################################

def evaluate(CKPT, experiment, chosen_collection, chosen_queries, index=True):
    split = "dev"
    nbits = 2
    k = 1000
    collection = chosen_collection
    queries = chosen_queries
    experiment = (f"{experiment}.nbits={nbits}",)

    if not os.path.exists(collection):
        print(f"No data found for {dataset} at {collection}, skipping...")
        return
    with Run().context(RunConfig(nranks=1)):
        INDEX_NAME = f"{experiment}.nbits={nbits}.latest"

        if index:
            config = ColBERTConfig(
                doc_maxlen=220,
                query_maxlen=32,
                nbits=nbits,
                kmeans_niters=4,
                root="primeqa/ir/dense/colbert_top/colbert/experiments",
                experiment=experiment,
            )
            indexer = Indexer(CKPT, config=config)
            indexer.index(name=INDEX_NAME, collection=collection, overwrite=True)

        config = ColBERTConfig(
            root="primeqa/ir/dense/colbert_top/colbert/experiments",
            query_maxlen=32,
            experiment=experiment,
        )
        searcher = Searcher(
            index=INDEX_NAME,
            config=config,
        )
        queries = Queries(path=queries)
        if torch.cuda.device_count() < 1:
            print(f"No GPU detected, setting num_threads to 1...")
            torch.set_num_threads(1)
            device = "cpu"
        else:
            device = "gpu"
        ranking = searcher.search_all(queries, k=k)
        ranking_output_path = ranking.save(
            f"{experiment}.k=1000.device={device}.ranking.tsv"
        )
        return ranking_output_path

#############################################################################

def generate_ColBERTv2_zeroshot_results(synthetic_queries_filename, CKPT, chosen_split, chosen_type, chosen_set, given_process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder, re_index=True):

	if LoTTE_or_BEIR == "LoTTE":
		experiment = "ColBERTv2_Zeroshot:_FLAN_XXL_" + chosen_split + "_" + chosen_type  + "_" + chosen_set +  "_" + str(given_process_number)
		chosen_collection = downloads_folder + "/lotte/" + chosen_split + "/" + chosen_set + "/collection.tsv" 
		chosen_queries = downloads_folder + "/lotte/" + chosen_split + "/" + chosen_set + "/questions." + chosen_type + ".tsv"
	elif LoTTE_or_BEIR == "BEIR":
		experiment = "ColBERTv2_Zeroshot:_FLAN_XXL_" + LoTTE_or_BEIR + "_" + chosen_BEIR_set  + "_" + chosen_BEIR_type +  "_" + str(given_process_number)
		chosen_collection = downloads_folder + "/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/collection.tsv" 
		chosen_queries = downloads_folder + "/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/questions.tsv"

	# Use synthetic queries if given
	if synthetic_queries_filename != None:
		chosen_queries = synthetic_queries_filename

	#############################################################################

	print("About to start indexing function!")

	ranking_output_path = evaluate(CKPT, experiment, chosen_collection, chosen_queries, index=re_index)
	return ranking_output_path


