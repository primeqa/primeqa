#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import pickle
import random
import re
from tqdm.auto import tqdm
import sys
import tempfile
import time
from typing import List, Any
from unittest.mock import patch
import torch.nn as nn

import numpy as np
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
import chromadb
import faiss
from primeqa.ir.dense.dpr_top.dpr.dpr_util import queries_to_vectors
from primeqa.util.searchable_corpus import SearchableCorpus
from libSIRE.timer import timer

def handle_args():
    usage = 'usage'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('--input_passage_vectors', '-p', default=None)
    parser.add_argument('--input_query_vectors', '-q', default=None)
    parser.add_argument('--input_passages', '-s', default=None)
    parser.add_argument('--input_queries', '-r', default=None)

    parser.add_argument('--db_engine', '-e', default='pinecone',
                        choices=['pinecone', 'pqa', 'pqa_colbert', 'chromadb', 'milvus',
                                 'faiss', 'es', 'es-esre', 'es-elser', 'bm25'], required=False)
    parser.add_argument('--output_ranks', '-o', default="", help="The output rank file.")

    parser.add_argument('--num_embeddings_deleted', '-n', type=int, default=100, )
    parser.add_argument('--top_k', '-t', type=int, default=10, )

    parser.add_argument('--create_own_embeddings', '-w', default=False, action='store_true')
    parser.add_argument('--model_name', '-m', default='all-MiniLM-L6-v2')
    parser.add_argument('--dimension', '-d', type=int, default=768, )
    parser.add_argument('--actions', default="cridIrR",
                        help="The actions that can be done: c(create), i(ingest), d(delete), "
                             "I(insert), r(retrieve), R(recreate)")
    parser.add_argument("--normalize_embs", action="store_true", help="If present, the embeddings are normalized.")
    parser.add_argument("--evaluate", action="store_true",
                        help="If present, evaluates the results based on test data, at the provided ranks.")
    parser.add_argument("--ranks", default="1,5,10,100", help="Defines the R@i evaluation ranks.")
    parser.add_argument("--colbert_root", default="", help="The root dir for the ColBERT model.")
    parser.add_argument("--index_dir", default=None, help="The index directory, if the db supports it.")
    parser.add_argument("--max_doc_length", default=None, type=int, help="Trim the passages to the given length, if provided")
    parser.add_argument('--data', default=None, type=str, help="The directory containing the data to use. The passage "
                                                               "file is assumed to be args.data/psgs.tsv and "
                                                               "the question file is args.data/questions.tsv.")
    parser.add_argument("--ingestion_batch_size", default=40, type=int, help="For elastic search only, sets the ingestion batch "
                                                                            "size (default 40).")
    parser.add_argument("--max_num_docs", default=-1, type=int,
                        help="If defined, only the given number of "
                        "documents will be ingested (first <max_num_docs> documents).")

    args = parser.parse_args()
    if args.output_ranks == "":
        args.output_ranks = f"tmp/{args.db_engine}_top{args.top_k}_{args.model_name}"
        if os.path.exists(args.output_ranks):
            i=0
            while True:
                n = args.output_ranks + f"v{i}"
                if not os.path.exists(n):
                    args.output_ranks = n
                    break
                i += 1
        print(f"Saving rank output to: {args.output_ranks}")
    if args.db_engine == "es-esre":
        args.db_engine = "es-elser"

    if args.data is not None:
        if args.input_passages is None:
            args.input_passages = os.path.join(args.data, "psgs.tsv")
        if args.input_queries is None:
            args.input_queries = os.path.join(args.data, "questions.tsv")
    else:
        args.data = os.path.basename(os.path.dirname(args.inpu_passages))

    if args.input_queries is None or args.input_passages is None:
        print("You need to define either the data dir (with --data) or both the passage file (using --input_passages) "
              "and the question file (using --input-queries")
        print(parser.usage)
        sys.exit(10)

    return args


def time_str(seconds: float) -> str:
    if seconds > 60 * 60:
        return f'{seconds / (60.0 * 60.0):.1f} hours'
    if seconds > 60:
        return f'{seconds / 60.0:.1f} minutes'
    return f'{seconds:.1f} seconds'


def report_time(last_time, count=None):
    now = time.time()
    elapsed = now - last_time
    res = f'took {time_str(elapsed)}'
    if count is not None:
        res += f" for {count} items - {count*1.0/elapsed:.1f} items/s"
    print(res)
    return now


def normalize(passage_vectors):
    return [v / np.linalg.norm(v) for v in passage_vectors if np.linalg.norm(v) > 0]


class MyChromaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, name, batch_size=128, model_type='pqa'):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
        self.pqa = False
        self.batch_size = batch_size
        if model_type=='pqa':
            from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoConfig
            self.queries_to_vectors = queries_to_vectors
            self.model = DPRQuestionEncoder.from_pretrained(
                pretrained_model_name_or_path=name,
                from_tf = False,
                cache_dir=None,)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(name)
            self._model_config = AutoConfig.from_pretrained(name)
            self.model.eval()
            self.model = self.model.half()
            self.model.to(device)
            self.pqa = True
        elif model_type == "chromadb":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name, device=device).half()
            # self.model = nn.DataParallel(self.model)
        print('=== done initializing model')

    def get_sentence_embedding_dimension(self):
        return self._model_config.hidden_size

    def __call__(self, texts: Documents) -> Embeddings:
        return self.encode(texts)

    def encode(self, texts:Documents, batch_size=-1, normalize_embeddings=False) -> Embeddings:
        # embed the documents somehow
        if not self.pqa:
            embs = self.model.encode(texts,
                                     normalize_embeddings,
                                     show_progress_bar=False \
                                         if isinstance(texts, str) or\
                                            max(len(texts), batch_size) <= 1 \
                                         else True
                                     )
        else:
            if batch_size < 0:
                batch_size = self.batch_size
            if len(texts) > batch_size:
                embs = []
                for i in tqdm(range(0, len(texts), batch_size)):
                    i_end = min(i + batch_size, len(texts))
                    tems = self.queries_to_vectors(self.tokenizer,
                                                   self.model,
                                                   texts[i:i_end],
                                                   max_query_length=500).tolist()
                    embs.extend(tems)
            else:
                embs = self.queries_to_vectors(self.tokenizer, self.model, texts, max_query_length=500).tolist()
            if normalize_embeddings:
                normalize(embs)
        return embs


def trim_passages(input_passages, max_doc_length):
    out = []
    for passage in input_passages:
        txt = passage['text']
        a = txt.split(" ")
        if len(a) > max_doc_length:
            passage['text'] = " ".join(a[:max_doc_length])
    return input_passages


def main():
    random.seed(12345)

    last_time = time.time()
    # === read input files
    args = handle_args()
    t = timer(f"VectorDB:{args.db_engine}")
    working_dir = None
    output_dir = None
    if args.db_engine in ["pqa", "pqa_colbert", "bm25"]:
        working_dir = tempfile.TemporaryDirectory().name
        output_dir = os.path.join(working_dir, 'output_dir')

    if args.input_passage_vectors is not None:
        with open(args.input_passage_vectors, 'rb') as in_file:
            passage_vectors = pickle.load(in_file)

    if args.input_query_vectors is not None:
        with open(args.input_query_vectors, 'rb') as in_file:
            query_vectors = pickle.load(in_file)

    input_passages = read_data(args.input_passages, fields=["id", "text", "title"], max_num_entries=args.max_num_docs)
    if args.max_doc_length is not None:
        input_passages = trim_passages(input_passages, args.max_doc_length)
    if args.evaluate:
        input_queries = read_data(args.input_queries, fields=["id", "text", "relevant", "answers"])
    else:
        input_queries = read_data(args.input_queries, fields=["id", "text"])

    # with open(args.input_queries) as in_file:
    #     csv_reader = csv.reader(in_file, delimiter="\t")
    #     for row in csv_reader:
    #         assert len(row) == 2, f'Invalid .tsv record (has to contain 2 fields): {row}'
    #         input_queries.append({'id': row[0], 'text': row[1]})

    print('=== done reading')
    last_time = report_time(last_time)
    create_db = 'c' in args.actions
    insert_db = 'i' in args.actions
    delete_items_db = 'd' in args.actions
    insert_deleted_items = 'I' in args.actions
    retrieve_items = 'r' in args.actions
    index_name = (f"{args.data}_{args.db_engine}_{args.model_name}_index").lower()
    index_name = re.sub('[^a-z0-9]', '-', index_name)
    if args.db_engine == "milvus":
        index_name = index_name.replace("-", "_")

    milvus_default_search_params = {
        "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
        "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
        "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
        "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
        "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
        "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
        "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
        "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
        "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
        "AUTOINDEX": {"metric_type": "L2", "params": {}},
    }
    milvus_default_index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64},
    }
    milvus_hnsw_index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 8, "efConstruction": 64},
    }
    if args.db_engine.startswith("es"):
        import logging
        logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

    query_vectors = []
    already_ingested = False
    # === initialize index and engine
    t.mark()
    if create_db:
        if args.db_engine == 'pinecone':
            import pinecone

            PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
            PINECONE_ENV = os.environ.get('PINECONE_ENV')

            pinecone.init(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENV
            )

            # pinecone.delete_index('test-10k-v1')
            if 'R' in args.actions:
                if index_name in pinecone.list_indexes():
                    pinecone.delete_index(index_name)

            # only create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=args.dimension,
                    metric='cosine'
                )
            # connect to the index
            index = pinecone.GRPCIndex(index_name)
        elif args.db_engine.startswith("pqa"):
            collection = SearchableCorpus(args.model_name, top_k=args.top_k, batch_size=64)
        elif args.db_engine == "chromadb":
            # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            #     model_name=args.model_name, device="cuda")
            if args.index_dir is not None:
                _client_settings = chromadb.config.Settings()
                _client_settings = chromadb.config.Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=args.index_dir,
                    )
                client = chromadb.Client(_client_settings)
                already_ingested = True
            else:
                client = chromadb.Client()
            # if os.path.exists(args.model_name):
            sentence_transformer_ef = MyChromaEmbeddingFunction(args.model_name, batch_size=64,
                                                                model_type=args.db_engine)
            collection = client.create_collection("vectordbtest",
                                                  embedding_function=sentence_transformer_ef)
            # else:
            #     collection = client.create_collection("vectordbtest")
        elif args.db_engine == "milvus":
            create_milvusdb()
        elif args.db_engine == "faiss":
            pass
        elif args.db_engine.startswith("es"):
            from elasticsearch import Elasticsearch
            ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
            client = Elasticsearch("https://cloud.elastic.co/",# "https://localhost:9200",
                                   # ca_certs = "/home/raduf/sandbox2/primeqa/ES-8.8.1/elasticsearch-8.8.1/config/certs/http_ca.crt",
                                   basic_auth = ("stefan.diederichs@sap.com", ELASTIC_PASSWORD)
                                   )
        elif args.db_engine == "bm25":
            pass
    else:
        if args.db_engine == 'pinecone':
            import pinecone
            # connect to the index
            index = pinecone.GRPCIndex(index_name)
        elif args.db_engine.startswith("es"):
            from elasticsearch import Elasticsearch
            ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
            client = Elasticsearch("https://cloud.elastic.co/", # https://localhost:9200",
                                   ca_certs = "/home/raduf/sandbox2/primeqa/ES-8.8.1/elasticsearch-8.8.1/config/certs/http_ca.crt",
                                   basic_auth = ("elastic", ELASTIC_PASSWORD)
                                   )

    model = None
    # === create embeddings
    if insert_db:
        if args.create_own_embeddings and args.db_engine not in ['pqa',"chromadb"]:
            # if args.db_engine == 'pinecone' or args.db_engine == 'milvus':
            from sentence_transformers import SentenceTransformer
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device != 'cuda':
                print(f"You are using {device}. This is much slower than using "
                      "a CUDA-enabled GPU. If on Colab you can change this by "
                      "clicking Runtime > Change runtime type > GPU.")

            batch_size = args.ingestion_batch_size
            model = MyChromaEmbeddingFunction(args.model_name, model_type=args.db_engine)

            # hidden_dim = model.get_sentence_embedding_dimension()

            print('=== done initializing model')
            report_time(last_time)

            passage_vectors = model.encode([passage['text'] for passage in input_passages],
                                           normalize_embeddings=args.normalize_embs,
                                           batch_size=batch_size)
            # for i in tqdm(range(0, len(input_passages), batch_size)):
            #     # find end of batch
            #     i_end = min(i + batch_size, len(input_passages))
            #     passage_vectors.extend(model.encode([passage['text'] for passage in input_passages[i:i_end]]))
            hidden_dim = len(passage_vectors[0])
            # if args.normalize_embs:
            #     passage_vectors = normalize(passage_vectors)

            print('=== done creating passage embeddings')
            report_time(last_time)
            # last_time = report_time(last_time)

            # if args.db_engine in ['pinecone', 'milvus']:

            # print('=== done creating query embeddings')
            # last_time = report_time(last_time)

        # === update index
        if args.db_engine == 'pinecone':
            batch_size = 128

            for i in tqdm(range(0, len(input_passages), batch_size)):
                # find end of batch
                i_end = min(i + batch_size, len(input_passages))
                # create IDs batch, IDs are just positions of the passages in input
                ids = [str(x) for x in range(i, i_end)]
                # create metadata batch
                metadatas = input_passages[i:i_end]
                # create embeddings
                vectors = passage_vectors[i:i_end]
                # create records list for upsert
                records = zip(ids, vectors, metadatas)
                # upsert to Pinecone
                index.upsert(vectors=records)

            # check number of records in the index
            index.describe_index_stats()
        elif args.db_engine == "pqa" or args.db_engine == "pqa_colbert":
            collection.add(texts=[row['text'] for row in input_passages],
                           titles=[row['title'] for row in input_passages],
                           ids=[row['id'] for row in input_passages])
            # indexing_args = [
            #     "prog",
            #     "--ctx_encoder_path_name_or_path", os.path.join(args.model_name, "ctx_encoder"),
            #     "--embed", "1of1",
            #     "--sharded_index",
            #     "--bsize", "1",
            #     "--collection", args.input_passages,
            #     "--output_dir", output_dir]
            # with patch.object(sys, 'argv', indexing_args):
            #     parser = HfArgumentParser(DPRIndexingArguments)
            #     (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            #     indexer = DPRIndexer(dpr_args)
            #     indexer.index()
        elif args.db_engine == "chromadb":
            # create IDs batch, IDs are just positions of the passages in input
            # ids = [str(x) for x in range(i, i_end)]
            ids = [row['id'] for row in input_passages]
            # create metadata batch
            metadatas = [{"title": row["title"]} for row in input_passages]
            # create embeddings
            if args.create_own_embeddings:
                # mp = {row['id']: passage_vectors[int(row['id'])] for row in input_passages}
                # emb = MyChromaEmbeddingFunction(mp)
                vectors = [row['text'] for row in input_passages]
                collection.add(documents=vectors, metadatas=metadatas, ids=ids)
            else:
                vectors = [row['text'] for row in input_passages]
                collection.add(documents=vectors, metadatas=metadatas, ids=ids)
        elif args.db_engine == "milvus":
            fmt = "\n=== {:30} ===\n"
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=hidden_dim)
            ]
            collection_loaded = utility.has_collection(index_name)
            if collection_loaded:
                print(fmt.format(f"Collection {index_name} exists, dropping"))
                utility.drop_collection(index_name)

            schema = CollectionSchema(fields, "Test for speed.")
            print(fmt.format(f"Create collection `f{index_name}`"))
            milvus1k = Collection(index_name, schema, consistency_level="Strong")

            ids = [int(row['id']) for row in input_passages]
            # create metadata batch
            text_vectors = [row['text'] for row in input_passages]
            # create embeddings
            batch_size = 1000
            for i in tqdm(range(0, len(text_vectors), batch_size), desc="Milvus index docs:"):
                milvus1k.insert([ids[i:i+batch_size],
                                 text_vectors[i:i+batch_size],
                                 passage_vectors[i:i+batch_size]]
                                )
            # index = {"index_type": "IVF_FLAT",
            #          "metric_type": "IP",
            #          "params": {"nlist": 128}}
            index_params = milvus_default_index_params
            milvus1k.create_index("embeddings",
                                  index_params)
            milvus1k.load()
        elif args.db_engine == "faiss":
            ids = [row['id'] for row in input_passages]
            # create metadata batch
            metadatas = [{"title": row["title"]} for row in input_passages]
            # index = faiss.IndexFlatL2(hidden_dim)
            index = faiss.IndexHNSWFlat(hidden_dim, 128, faiss.METRIC_INNER_PRODUCT)
            to_index = np.array(passage_vectors)
            index.train(to_index)
            index.add(to_index)
        elif args.db_engine == "es":
            mappings = {
                "properties": {
                    "title": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                    "vector": {"type": "dense_vector", "dims": hidden_dim,
                               "similarity": "cosine", "index": "true"},
                }
            }
            if client.indices.exists(index=index_name):
                client.options(ignore_status=[400,404]).indices.delete(index=index_name)
            client.indices.create(index=index_name, mappings=mappings)
            import logging
            logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
            for ri, row in enumerate(input_passages):
                doc = {'text': row['text'],
                       'title': row['title'],
                       'vector': passage_vectors[ri]}
                client.index(index=index_name, id=row['id'], document=doc)
        elif args.db_engine == "es-elser":
            mappings = {
                "properties": {
                    "ml.tokens": {
                        "type": "rank_features"
                    },
                    "title": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                }
            }
            mappings1 = {
                "properties": {
                    "title": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                }
            }
            processors = [
                {
                "inference": {
                "model_id": ".elser_model_1",
                "target_field": "ml",
                "field_map": {
                    "text": "text_field"
                },
                "inference_config": {
                    "text_expansion": {
                    "results_field": "tokens"
                    }
                }
                }}
            ]
            bulk_batch = args.ingestion_batch_size
            from elasticsearch.helpers import bulk
            if client.indices.exists(index=index_name):
                client.options(ignore_status=[400, 404]).indices.delete(index=index_name)
            if client.indices.exists(index=f"{index_name}-tmp"):
                client.options(ignore_status=[400, 404]).indices.delete(index=f"{index_name}-tmp")
            client.indices.create(index=f"{index_name}-tmp", mappings=mappings1)
            client.indices.create(index=f"{index_name}", mappings=mappings)
            client.ingest.put_pipeline(processors=processors, id='elser-v1-test')
            actions = []
            for ri, row in tqdm(enumerate(input_passages), total=len(input_passages), desc="Indexing passages"):
                actions.append({
                    "_index": index_name,
                    "_id": row['id'],
                    "_source": {
                        'text': row['text'],
                        'title': row['title']
                    }
                }
                )
                if ri % bulk_batch == bulk_batch-1:
                    try:
                        res = bulk(client=client, actions=actions, pipeline="elser-v1-test")
                    except Exception as e:
                        print(f"Got an error in indexing: {e}, {len(actions)} {res}")
                    actions = []
            try:
                bulk(client=client, actions=actions, pipeline="elser-v1-test")
            except Exception as e:
                print(f"Got an error in indexing: {e}, {len(actions)}")
        elif args.db_engine == "bm25":
            from ..sparse.indexer import PyseriniIndexer
            from ..sparse.retriever import PyseriniRetriever
            indexer = PyseriniIndexer()

            indexer.index_collection()

        print('=== done creating index')
        last_time = report_time(last_time)
        t.add_timing("Create&Load")
    # === test index items deletion
    if delete_items_db:
        deleted_embedding_ids = [id for id in random.sample(range(len(input_passages)), args.num_embeddings_deleted)]

        if args.db_engine == 'pinecone':
            deleted_embedding_ids_str = [str(id) for id in deleted_embedding_ids]
            index.delete(ids=deleted_embedding_ids_str)  # , namespace='example-namespace')
        elif args.db_engine == 'es':
            for id in deleted_embedding_ids:
                client.delete(index=index_name, id=id)

        print('=== done testing index items deletion')
        last_time = report_time(last_time)
        t.add_timing("Remove")
        # === test index items insertion
        if insert_deleted_items:
            if args.db_engine == 'pinecone':
                vectors = [passage_vectors[id] for id in deleted_embedding_ids]
                metadatas = [input_passages[id] for id in deleted_embedding_ids]
                records = zip(deleted_embedding_ids_str, vectors, metadatas)
                index.upsert(vectors=records)
            elif args.db_engine == 'es':
                for id in deleted_embedding_ids:
                    client.index(index=index_name,
                                 id=id,
                                 document={'text': input_passages[id]['text'],
                                           'title': input_passages[id]['title'],
                                           'vector': passage_vectors[id]})

            print('=== done testing index items insertion')
            last_time = report_time(last_time)
            t.add_timing("Reinsert")

    # === run retrieval
    out_ranks = []

    # query_vectors = []
    # for query_number in range(len(input_queries)):
    #     query_vectors.append(model.encode(input_queries[query_number]['text']))
    #
    # if args.normalize_embs:
    #     normalize(query_vectors)

    if retrieve_items:
        print(f"**** Retrieving with {args.db_engine}")
        if args.db_engine == 'pinecone':
            for query_number in tqdm(range(len(query_vectors))):
                # for query_number in range(len(query_vectors)):
                query_vector = compute_embedding(model, input_queries[query_number]['text'], args.normalize_embs)
                response = index.query(query_vector, top_k=args.top_k, include_metadata=True)
                for rank, match in enumerate(response['matches']):
                    out_ranks.append([input_queries[query_number]['id'], match["id"], rank + 1, match["score"]])
        elif args.db_engine == "chromadb":
            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                response = collection.query(
                    query_texts=[input_queries[query_number]['text']],
                    n_results=args.top_k
                )
                for rank, match in enumerate(response['ids'][0]):
                    out_ranks.append(
                        [input_queries[query_number]['id'], match, rank + 1, response['distances'][0][rank]])
        elif args.db_engine == 'pqa':
            # searcher = create_pqa_searcher(args, output_dir)
            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                q_ids, response = collection.search(
                    input_queries=[input_queries[query_number]['text']],
                    top_k=args.top_k
                )
                for rank, match in enumerate(q_ids[0]):
                    out_ranks.append([input_queries[query_number]['id'], match, rank + 1, response[0][rank]])
        elif args.db_engine == 'pqa_colbert':
            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                q_ids, response = collection.search(
                    input_qureries=[input_queries[query_number]['text']],
                    top_k=args.top_k
                )
                for rank, match in enumerate(q_ids[0]):
                    out_ranks.append([input_queries[query_number]['id'], match, rank + 1, response[0][rank]])
        elif args.db_engine == 'milvus':
            # search_params = {
            #     "metric_type": "L2",
            #     "params": {"nprobe": 10},
            # }
            # search_params = {
            #     "metric_type": "L2",
            #     "params": {"ef": 10},
            # }
            search_params = milvus_default_search_params["IVF_FLAT"]

            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                query_vector = compute_embedding(model, input_queries[query_number]['text'], args.normalize_embs)

                result = milvus1k.search(
                    [query_vector],
                    "embeddings",
                    search_params,
                    limit=args.top_k,
                    output_fields=["id", "text"]
                )
                for rank, hit in enumerate(result[0]):
                    out_ranks.append([input_queries[query_number]['id'], hit.entity.get('id'), rank + 1, hit.score])
        elif args.db_engine == 'faiss':
            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                query_vector = compute_embedding(model, input_queries[query_number]['text'], args.normalize_embs)
                if len(query_vector) == hidden_dim:
                    query_vector = [query_vector]
                distances, ann = index.search(
                    np.array(query_vector),
                    k = args.top_k
                )
                for rank, hit in enumerate(ann[0]):
                    out_ranks.append([input_queries[query_number]['id'], ids[hit], rank + 1, distances[0][rank]])
        elif args.db_engine == 'es':
            for query_number in tqdm(range(len(input_queries))):
                query_vector = compute_embedding(model, input_queries[query_number]['text'], args.normalize_embs)
                query = {
                        "field": "vector",
                        "query_vector": query_vector,
                        "k": args.top_k,
                        "num_candidates": 1000,
                }
                res = client.search(
                    index=index_name,
                    knn=query,
                    source_excludes=['vector']
                )
                for rank, r in enumerate(res._body['hits']['hits']):
                    out_ranks.append([input_queries[query_number]['id'], r['_id'], rank+1, r['_score']])
        elif args.db_engine == 'es-elser':
            for query_number in tqdm(range(len(input_queries))):
                # query_vector = compute_embedding(model, input_queries[query_number]['text'], args.normalize_embs)
                query = {
                        "text_expansion": {
                            "ml.tokens": {
                                "model_id": ".elser_model_1",
                                "model_text": input_queries[query_number]['text']
                            }
                        }
                }
                res = client.search(
                    index=index_name,
                    query=query,
                    size=args.top_k,
                )
                for rank, r in enumerate(res._body['hits']['hits']):
                    out_ranks.append([input_queries[query_number]['id'], r['_id'], rank+1, r['_score']])

        print('=== done running retrieval')
        last_time = report_time(last_time, len(input_queries))

        with open(args.output_ranks, 'w') as out_file:
            tsv_writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
            tsv_writer.writerows(out_ranks)
        t.add_timing("Retrieve")
        if args.evaluate:
            if "relevant" not in input_queries[0] or input_queries[0]['relevant'] is None:
                print("The input question file does not contain answers. Please fix that and restart.")
                sys.exit(12)
            ranks = [int(r) for r in args.ranks.split(",")]
            scores = {r:0 for r in ranks}
            lscores = {r:0 for r in ranks}
            gt = {-1: -1}
            for q in input_queries:
                gt[q['id']] = {id: 1 for id in q['relevant'].split(",")}

            def skip(out_ranks, record, rid):
                qid = record[0]
                while rid < len(out_ranks) and out_ranks[rid][0] == qid:
                    rid += 1
                return rid

            def update_scores(ranks, rnk, scores):
                j = 0
                while j<len(ranks) and ranks[j] < rnk:
                    j += 1
                for k in ranks[j:]:
                    scores[k] += 1

            rid = 0
            out_ranks1 = []
            # while rid < len(out_ranks):
            with_answers = False
            if 'answers' in input_queries[0]:
                rq_map = reverse_map(input_queries)
                rp_map = reverse_map(input_passages)
                with_answers = True
            tmp_scores = scores.copy()
            tmp_lscores = lscores.copy()
            prev_id = -1
            num_eval_questions = 0
            for rid, record in enumerate(out_ranks):
                outr = record[0:3]
                if prev_id != record[0]:
                    if prev_id != -1 and '-1' not in gt[prev_id]:
                        num_eval_questions += 1
                        for r in ranks:
                            scores[r]  += int(tmp_scores[r]  >= 1)
                            lscores[r] += int(tmp_lscores[r] >= 1)
                        tmp_scores = {r:0 for r in ranks}
                        tmp_lscores = {r:0 for r in ranks}
                    prev_id = record[0]
                if str(record[1]) in gt[record[0]]: # Great, we found a match.
                    update_scores(ranks, record[2], tmp_scores)
                    outr.append(1)
                else:
                    outr.append(0)
                if with_answers:
                    qid = rq_map[record[0]]
                    did = rp_map[str(record[1])]
                    inputq = input_queries[qid]
                    txt = input_passages[did]['text'].lower()
                    found = False
                    for s in inputq['answers']:
                        if txt.find(s.lower()) >= 0:
                            found = True
                            break
                    if (found):
                        outr.append(1)
                        update_scores(ranks, record[2], tmp_lscores)
                    else:
                        outr.append(0)
                out_ranks1.append(outr)

            if gt[prev_id] != -1:
                num_eval_questions += 1
                for r in ranks:
                    scores[r] += int(tmp_scores[r] >= 1)
                    lscores[r] += int(tmp_lscores[r] >= 1)
            res = {"num_ranked_queries": num_eval_questions,
                   "num_judged_queries": num_eval_questions,
                   "success__WARNING":
                       {r:int(1000*scores[r]/num_eval_questions)/1000.0 for r in ranks},
                   "lienient_success__WARNING":
                       {r:int(1000*lscores[r]/num_eval_questions)/1000.0 for r in ranks}
                   }
            with open(args.output_ranks+".annotated.deb","w") as out_file:
                tsv_writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
                tsv_writer.writerows(out_ranks1)

            with open(args.output_ranks+".metrics","w") as out:
                out.write(json.dumps(res, indent=2)+"\n")
    # if args.db_engine == 'pinecone':
    #     pinecone.delete_index(index_name)
        print('=== done computing scores')
        last_time = report_time(last_time)
        # t.add_timing("Evaluate")
    import io
    from contextlib import redirect_stdout

    # with open(args.output_ranks+".timing", "w") as sys.stdout:
    buf = io.StringIO()
    with redirect_stdout(buf):
        timer.display_timing(t.milliseconds_since_beginning(), num_words=len(input_passages), num_chars=len(input_queries))
    # timer.display_timing(t.milliseconds_since_beginning(), num_words=len(input_passages), num_chars=len(input_queries))
    with open(args.output_ranks+".timing", "w") as out:
        out.write(buf.getvalue())
    print(buf.getvalue())


def reverse_map(input_queries):
    rq_map = {}
    for i, q in enumerate(input_queries):
        rq_map[q['id']] = i
    return rq_map


def compute_embedding(model, input_query, normalize_embs):
    query_vector = model.encode(input_query, normalize_embeddings=normalize_embs)
    return query_vector

def create_pqa_searcher(args, output_dir):
    from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher
    from primeqa.ir.dense.dpr_top.dpr.config import   DPRSearchArguments

    search_args = [
        "prog",
        "--engine_type", "DPR",
        "--model_name_or_path", os.path.join(args.model_name, "qry_encoder"),
        "--bsize", "1",
        "--index_location", os.path.join(output_dir, "index"),
        "--top_k", str(args.top_k),
    ]
    with patch.object(sys, 'argv', search_args):
        parser = HfArgumentParser(DPRSearchArguments)
        (dpr_args, remaining_args) = \
            parser.parse_args_into_dataclasses(return_remaining_strings=True)
        searcher = DPRSearcher(dpr_args)
    return searcher


def create_milvusdb():
    connections.connect("default", host="localhost", port="19530")


def read_data(input_file, fields=None, max_num_entries: int=-1):
    passages = []
    if fields is None:
        num_args = 3
    else:
        num_args = len(fields)
    with open(input_file) as in_file:
        if input_file.endswith(".tsv"):
            # We'll assume this is the PrimeQA standard format
            csv_reader = \
                csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                    if fields is not None \
                    else csv.DictReader(in_file, delimiter="\t")
            next(csv_reader)
            for i, row in enumerate(csv_reader):
                if 0 <= max_num_entries <= i:
                    break
                assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
                itm = {'text': (row["title"] + ' '  if 'title' in row else '') + row["text"],
                                 'id': row['id']}
                if 'title'in row:
                    itm['title'] = row['title']
                if 'relevant' in row:
                    itm['relevant'] = row['relevant']
                if 'answers' in row:
                    itm['answers'] = row['answers'].split("::")
                passages.append(itm)
        elif input_file.endswith('.json'):
            # This should be the SAP json format
            data = json.load(in_file)
            for doc in data:
                doc_id = doc['document_id']
                title = doc['title']
                for passage in doc['passages']:
                    itm = {'title': passage['title']}
                    itm['id'] = f"{doc_id}:{passage['passage_id']}"
                    itm['text'] = passage['text']
                    passages.append(itm)
        elif input_file.endswith(".csv"):
            import pandas as pd
            queries = pd.read_csv(input_file, header=0)


    return passages

# do main
if __name__ == '__main__':
    main()
