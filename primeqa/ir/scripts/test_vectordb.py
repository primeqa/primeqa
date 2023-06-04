#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import pickle
import random
import re
import sys
import tempfile
import time
from typing import List, Any
from unittest.mock import patch

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


def handle_args():
    usage = 'usage'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('--input_passage_vectors', '-p', required=True)
    parser.add_argument('--input_query_vectors', '-q', required=True)
    parser.add_argument('--input_passages', '-s', required=True)
    parser.add_argument('--input_queries', '-r', required=True)

    parser.add_argument('--db_engine', '-e', default='pinecone',
                        choices=['pinecone', 'pqa', 'chromadb', 'milvus'], required=False)
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
    return args


def time_str(seconds: float) -> str:
    if seconds > 60 * 60:
        return f'{seconds / (60.0 * 60.0):.1f} hours'
    if seconds > 60:
        return f'{seconds / 60.0:.1f} minutes'
    return f'{seconds:.1f} seconds'


def report_time(last_time):
    now = time.time()
    elapsed = now - last_time

    print(f'took {time_str(elapsed)}')
    return now


def normalize(passage_vectors):
    passage_vectors = passage_vectors / np.linalg.norm(passage_vectors)

class MyChromaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, name, batch_size=128):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print(f"You are using {device}. This is much slower than using "
                  "a CUDA-enabled GPU. If on Colab you can change this by "
                  "clicking Runtime > Change runtime type > GPU.")
        self.pqa = False
        if os.path.exists(name):
            from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoConfig
            from primeqa.ir.dense.dpr_top.dpr.dpr_util import queries_to_vectors
            self.queries_to_vectors = queries_to_vectors
            # self.config = DPRConfig(name)
            # if self.config.model_type in {"gpt2", "roberta"}:
            #     self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            #         name,
            #         use_fast=True,
            #         add_prefix_space=True,
            #     )
            # else:
            #     self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            #         name,
            #         use_fast=True,
            #     )
            #
            self.model = DPRQuestionEncoder.from_pretrained(
                pretrained_model_name_or_path=name,
                from_tf = False,
                cache_dir=None,)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(name)
            self.config = AutoConfig.from_pretrained(name)
            self.model.eval()
            self.batch_size = batch_size
            self.model = self.model.half()
            self.model.to(device)
            self.pqa = True
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(name, device=device)
        print('=== done initializing model')

    def get_sentence_embedding_dimension(self):
        return self.config.hidden_size

    def __call__(self, texts: Documents) -> Embeddings:
        return self.encode(texts)

    def encode(self, texts:Documents) -> Embeddings:
        # embed the documents somehow
        if not self.pqa:
            embs = self.model(texts)
        else:
            # tokt = self.tokenizer(texts)
            # ems = self.model.run(tokt)
            if len(texts) > self.batch_size:
                embs = []
                for i in tqdm(range(0, len(texts), self.batch_size)):
                    i_end = min(i + self.batch_size, len(texts))
                    tems = self.queries_to_vectors(self.tokenizer,
                                                   self.model,
                                                   texts[i:i_end],
                                                   max_query_length=500).tolist()
                    embs.extend(tems)
            else:
                embs = self.queries_to_vectors(self.tokenizer, self.model, texts, max_query_length=500).tolist()
        return embs


def main():
    random.seed(12345)

    last_time = time.time()

    # === read input files
    args = handle_args()

    with open(args.input_passage_vectors, 'rb') as in_file:
        passage_vectors = pickle.load(in_file)

    with open(args.input_query_vectors, 'rb') as in_file:
        query_vectors = pickle.load(in_file)

    input_passages = []  # {'text': text}
    input_passages = read_data(args.input_passages, fields=["id", "text", "title"])

    input_queries: list[Any] = []  # {'text': text}
    input_queries = read_data(args.input_queries, num_args=2, fields=["id", "text"])
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
    index_name = (args.model_name + '__index').lower()
    index_name = re.sub('[^a-z0-9]', '-', index_name)
    if args.db_engine == "milvus":
        index_name = index_name.replace("-", "_")

    # === initialize index and engine
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
        elif args.db_engine == "pqa":
            from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
            from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher
            from primeqa.ir.dense.dpr_top.dpr.config import DPRIndexingArguments, \
                DPRSearchArguments
            if args.model_name is None:
                print("You need to specify the encoder if you're going to use PrimeQA: use --model_name|-m argument")
                sys.exit(1)

            working_dir = tempfile.TemporaryDirectory().name
            output_dir = os.path.join(working_dir, 'output_dir')
            os.makedirs(output_dir, exist_ok=True)
            print(output_dir)

            search_args = [
                "prog",
                "--engine_type", "DPR",
                "--do_index",
                "--bsize", "16",
                "--top_k", str(args.top_k),
                "--ctx_encoder_name_or_path", args.model_name+"/ctx_encoder",
                "--embed", "1of1",
                "--output_dir", output_dir+"/index",
                "--shared_index"
            ]

            with patch.object(sys, 'argv', search_args):
                parser = HfArgumentParser(DPRIndexingArguments)
                (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
                searcher = DPRIndexer(dpr_args)
            search_args = [
                
            ]
        elif args.db_engine == "chromadb":
            # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            #     model_name=args.model_name, device="cuda")
            sentence_transformer_ef = MyChromaEmbeddingFunction(args.model_name, batch_size=64)
            client = chromadb.Client()
            collection = client.create_collection("vectordbtest",
                                                  embedding_function=sentence_transformer_ef)
        elif args.db_engine == "milvus":
            create_milvusdb()
    else:
        if args.db_engine == 'pinecone':
            import pinecone
            # connect to the index
            index = pinecone.GRPCIndex(index_name)

    # === create embeddings
    if insert_db:
        if args.create_own_embeddings and args.db_engine != "chromadb":
            # if args.db_engine == 'pinecone' or args.db_engine == 'milvus':
            from sentence_transformers import SentenceTransformer
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device != 'cuda':
                print(f"You are using {device}. This is much slower than using "
                      "a CUDA-enabled GPU. If on Colab you can change this by "
                      "clicking Runtime > Change runtime type > GPU.")

            if args.db_engine == 'chromadb':
                model = sentence_transformer_ef
            else:
                model = SentenceTransformer(args.model_name, device=device)

            hidden_dim = model.get_sentence_embedding_dimension()

            print('=== done initializing model')
            last_time = report_time(last_time)

            passage_vectors = []
            batch_size = 128
            for i in tqdm(range(0, len(input_passages), batch_size)):
                # find end of batch
                i_end = min(i + batch_size, len(input_passages))
                passage_vectors.extend(model.encode([passage['text'] for passage in input_passages[i:i_end]]))

            if args.normalize_embs:
                normalize(passage_vectors)

            print('=== done creating passage embeddings')
            last_time = report_time(last_time)

            # if args.db_engine in ['pinecone', 'milvus']:
            query_vectors = []
            for query_number in range(len(input_queries)):
                query_vectors.append(model.encode(input_queries[query_number]['text']))

            if args.normalize_embs:
                normalize(query_vectors)

            print('=== done creating query embeddings')
            last_time = report_time(last_time)

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
        elif args.db_engine == "pqa":
            indexing_args = [
                "prog",
                "--ctx_encoder_path_name_or_path", os.path.join(output_dir, "ctx_encoder"),
                "--embed", "1of1",
                "--sharded_index",
                "--bsize", "1",
                "--collection", args.input_passages,
                "--output_dir", output_dir]
            with patch.object(sys, 'argv', indexing_args):
                parser = HfArgumentParser(DPRIndexingArguments)
                (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
                indexer = DPRIndexer(dpr_args)
                indexer.index()
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

            milvus1k.insert([ids, text_vectors, passage_vectors])
            # index = {"index_type": "IVF_FLAT",
            #          "metric_type": "IP",
            #          "params": {"nlist": 128}}
            index_params = {
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {"M": 8, "efConstruction": 64},
                }
            milvus1k.create_index("embeddings",
                                  index_params)
            milvus1k.load()
        print('=== done creating index')
        last_time = report_time(last_time)

    # === test index items deletion
    if delete_items_db:
        deleted_embedding_ids = [id for id in random.sample(range(len(input_passages)), args.num_embeddings_deleted)]

        if args.db_engine == 'pinecone':
            deleted_embedding_ids_str = [str(id) for id in deleted_embedding_ids]
            index.delete(ids=deleted_embedding_ids_str)  # , namespace='example-namespace')

        print('=== done testing index items deletion')
        last_time = report_time(last_time)

        # === test index items insertion
        if insert_deleted_items:
            if args.db_engine == 'pinecone':
                vectors = [passage_vectors[id] for id in deleted_embedding_ids]
                metadatas = [input_passages[id] for id in deleted_embedding_ids]
                records = zip(deleted_embedding_ids_str, vectors, metadatas)
                index.upsert(vectors=records)

            print('=== done testing index items insertion')
            last_time = report_time(last_time)

    # === run retrieval
    out_ranks = []

    if retrieve_items:
        print(f"**** Retrieving with {args.db_engine}")
        if args.db_engine == 'pinecone':
            for query_number in tqdm(range(len(query_vectors))):
                # for query_number in range(len(query_vectors)):
                response = index.query(query_vectors[query_number], top_k=args.top_k, include_metadata=True)
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
            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                q_ids, response = searcher.search(
                    query_batch=[input_queries[query_number]['text']],
                    top_k=args.top_k,
                    mode="query_list"
                )
                for rank, match in enumerate(q_ids):
                    out_ranks.append([input_queries[query_number]['id'], match, rank + 1, response['scores'][0][rank]])
        elif args.db_engine == 'milvus':
            # search_params = {
            #     "metric_type": "L2",
            #     "params": {"nprobe": 10},
            # }
            search_params = {
                "metric_type": "L2",
                "params": {"ef": 10},
            }

            for query_number in tqdm(range(len(input_queries))):
                # for query_number in range(len(query_vectors)):
                result = milvus1k.search(
                    [query_vectors[query_number]],
                    "embeddings",
                    search_params,
                    limit=args.top_k,
                    output_fields=["id", "text"]
                )
                for rank, hit in enumerate(result[0]):
                    out_ranks.append([input_queries[query_number]['id'], hit.entity.get('id'), rank + 1, hit.score])

        print('=== done running retrieval')
        last_time = report_time(last_time)

        with open(args.output_ranks, 'w') as out_file:
            tsv_writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
            tsv_writer.writerows(out_ranks)

    # if args.db_engine == 'pinecone':
    #     pinecone.delete_index(index_name)


def create_milvusdb():
    connections.connect("default", host="localhost", port="19530")


def read_data(input_file, fields=None, num_args=3):
    passages = []
    with open(input_file) as in_file:
        csv_reader = \
            csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                if fields is not None \
                else csv.DictReader(in_file, delimiter="\t")
        next(csv_reader)
        for row in csv_reader:
            assert len(row) == num_args or len(row) == 2, f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
            itm = {'text': (row["title"] if len(row) == 3 else ' ') + ' ' + row["text"],
                             'id': row['id']}
            if 'title'in row:
                itm['title'] = row['title']
            passages.append(itm)
    return passages

# do main
if __name__ == '__main__':
    main()
